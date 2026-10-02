from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import warnings
import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .augment import (
    corrupt,
    frame_mask_from_lengths,
)
from .data import PoseData, PoseDataset, resolve_dataset_path, development_split
from .model import encoder_from_config
from .provenance import configuration_fingerprint, source_fingerprint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-checkout"


def _merge_configuration(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_configuration(target[key], value)
        else:
            target[key] = value


def _load_recipe(project_root: Path, preset: str, parents=()) -> dict:
    if preset in parents:
        chain = " -> ".join((*parents, preset))
        raise ValueError(f"cyclic recipe inheritance: {chain}")
    path = project_root / "configs/recipes" / f"{preset}.yaml"
    with path.open() as handle:
        recipe = yaml.safe_load(handle) or {}
    parent = recipe.pop("extends", None)
    merged = {}
    if parent is not None:
        names = [parent] if isinstance(parent, str) else list(parent)
        for name in names:
            _merge_configuration(merged, _load_recipe(project_root, name, (*parents, preset)))
    _merge_configuration(merged, recipe)
    return merged


def load_configuration(project_root: Path, dataset: str, method: str, preset: str | None = None) -> dict:
    with (project_root / "configs/base.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    with (project_root / f"configs/datasets/{dataset}.yaml").open() as handle:
        cfg["dataset"] = yaml.safe_load(handle)
    with (project_root / "configs/methods.yaml").open() as handle:
        methods = yaml.safe_load(handle)
    if method not in methods:
        raise ValueError(f"unknown method {method}; choose from {', '.join(methods)}")
    cfg["method_name"] = method
    cfg["method"] = methods[method]
    if preset is not None:
        _merge_configuration(cfg, _load_recipe(project_root, preset))
    return cfg


# Kept as public aliases for existing scripts and notebooks.
from .training import dataset_options as _dataset_options, train_model as _train, save_checkpoint


GLOBAL_DESCRIPTOR = "global"
PART_DESCRIPTOR = "global_parts"
RELIABLE_DESCRIPTOR = "reliable_parts"
RAW_RELIABLE_DESCRIPTOR = "raw_reliable"
RAW_UNIFORM_DESCRIPTOR = "raw_uniform"


def _structured_descriptor(
    embedding: torch.Tensor,
    parts: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pack global and local vectors for reliability-aware score fusion.

    Raw local pools need not have the same width as the global embedding on
    every backbone. Zero-padding both to their shared maximum width preserves
    every within-type cosine score while allowing one compact tensor format.
    """
    if embedding.ndim != 2 or parts.ndim != 3 or weights.ndim != 2:
        raise ValueError("structured descriptors require [N,D], [N,P,D], and [N,P] tensors")
    if embedding.shape[0] != parts.shape[0] or weights.shape != parts.shape[:2]:
        raise ValueError("structured descriptor batch and part dimensions must agree")
    width = max(embedding.shape[-1], parts.shape[-1])
    embedding = F.pad(embedding, (0, width - embedding.shape[-1]))
    parts = F.pad(parts, (0, width - parts.shape[-1]))
    vectors = torch.cat([embedding[:, None], parts], dim=1)
    all_weights = torch.cat([torch.ones_like(weights[:, :1]), weights], dim=1)
    return torch.cat([vectors, all_weights[..., None]], dim=-1)


def _descriptors(
    outputs: dict,
    include_reliable: bool = False,
    include_raw: bool = False,
) -> dict[str, torch.Tensor]:
    """Retrieval descriptors built from one forward pass.

    `global` is the sequence embedding the original code matched on. It discards
    the part heads, which for the part-aware methods is where the anatomy-level
    evidence lives; `global_parts` concatenates the global embedding with the
    L2-normalised part projections. Completely absent parts are zero vectors;
    the concatenation is normalized over the blocks that remain. Reporting this
    for every method does not replace a trained part-head control.
    """
    embedding, parts = outputs["embedding"], outputs["parts"]
    stacked = torch.cat([embedding] + [parts[:, k] for k in range(parts.shape[1])], dim=-1)
    result = {
        GLOBAL_DESCRIPTOR: embedding,
        PART_DESCRIPTOR: F.normalize(stacked, dim=-1),
    }
    if include_reliable:
        # [N, 1 + parts, embedding_dim + 1]; final channel is reliability.
        # This structured descriptor is scored pairwise, never by a plain dot.
        result[RELIABLE_DESCRIPTOR] = _structured_descriptor(
            embedding, parts, outputs["reliability"]
        )
    if include_raw:
        reliability = outputs["reliability"]
        presence = (reliability > 0).to(reliability.dtype)
        result[RAW_RELIABLE_DESCRIPTOR] = _structured_descriptor(
            embedding, outputs["raw_parts"], reliability
        )
        result[RAW_UNIFORM_DESCRIPTOR] = _structured_descriptor(
            embedding, outputs["raw_parts"], presence
        )
    return result


def _descriptor_builder(descriptors: list[str]):
    requested = set(descriptors)
    include_reliable = RELIABLE_DESCRIPTOR in requested
    include_raw = bool(
        requested.intersection({RAW_RELIABLE_DESCRIPTOR, RAW_UNIFORM_DESCRIPTOR})
    )
    return lambda output: _descriptors(output, include_reliable, include_raw)


def descriptor_similarity(probe, gallery, descriptor=GLOBAL_DESCRIPTOR, part_weight=0.5):
    # Structured part descriptors have shape [sample, global + parts,
    # feature + reliability].  Detect them by representation rather than by a
    # single hard-coded name so checkpoint-only descriptor controls can reuse
    # the exact retrieval implementation.
    structured = np.ndim(probe) == 3 and np.ndim(gallery) == 3
    if not structured:
        return probe @ gallery.T
    if not 0 <= part_weight <= 1:
        raise ValueError("part_weight must lie in [0,1]")
    global_score = probe[:, 0, :-1] @ gallery[:, 0, :-1].T
    weighted_score, total = np.zeros_like(global_score), np.zeros_like(global_score)
    for p in range(1, probe.shape[1]):
        weight = probe[:, p, -1, None] * gallery[None, :, p, -1]
        weighted_score += weight * (probe[:, p, :-1] @ gallery[:, p, :-1].T)
        total += weight
    local = weighted_score / np.maximum(total, 1e-8)
    combined = (1 - part_weight) * global_score + part_weight * local
    return np.where(total > 1e-8, combined, global_score)


@torch.no_grad()
def _embeddings(
    model,
    data: PoseData,
    indices: np.ndarray,
    device,
    corruption: str | None = None,
    severity: float = 0,
    seed: int = 0,
    dataset_options: dict | None = None,
    batch_size: int = 32,
    descriptor_builder=None,
):
    loader = DataLoader(
        PoseDataset(data, indices, **(dataset_options or {})), batch_size=batch_size, shuffle=False
    )
    outputs: dict[str, list[torch.Tensor]] = {}
    output_labels, output_indices = [], []
    model.eval()
    for x, visibility, lengths, labels, original_indices in loader:
        # Trim the batch to its longest real sequence. Padding is excluded from
        # pooling but still runs through the convolutional trunk, and carrying
        # the dataset-wide maximum costs three times the work on CASIA-B for a
        # tail that only bleeds across the real/padding boundary.
        usable = int(lengths.max())
        x, visibility = x[:, :usable], visibility[:, :usable]
        if corruption and severity > 0:
            # CPU, per-sequence masks stay identical across devices, batch sizes,
            # backbones and training seeds. Padded tails cannot change the RNG.
            x, visibility = x.clone(), visibility.clone()
            for b, original in enumerate(original_indices.tolist()):
                length = int(lengths[b])
                generator = torch.Generator().manual_seed(seed + original * 1_000_003)
                cx, cv = corrupt(x[b:b + 1, :length], visibility[b:b + 1, :length],
                                 corruption, severity, generator, lengths[b:b + 1])
                x[b, :length], visibility[b, :length] = cx[0], cv[0]
        x, visibility, lengths = x.to(device), visibility.to(device), lengths.to(device)
        batch = model(x, visibility, frame_mask_from_lengths(visibility, lengths))
        descriptors = (descriptor_builder or (lambda output: _descriptors(output, include_reliable=True)))(batch)
        for name, value in descriptors.items():
            outputs.setdefault(name, []).append(value.cpu())
        output_labels.append(labels)
        output_indices.append(original_indices)
    return (
        {name: torch.cat(values).numpy() for name, values in outputs.items()},
        torch.cat(output_labels).numpy(),
        torch.cat(output_indices).numpy(),
    )


def retrieval_metrics(
    gallery_embeddings,
    gallery_labels,
    probe_embeddings,
    probe_labels,
    exclusion_mask: np.ndarray | None = None,
    *, descriptor=GLOBAL_DESCRIPTOR, part_weight=0.5,
):
    similarities = descriptor_similarity(probe_embeddings, gallery_embeddings, descriptor, part_weight)
    if exclusion_mask is None:
        exclusion_mask = np.zeros_like(similarities, dtype=bool)
    if exclusion_mask.shape != similarities.shape:
        raise ValueError("exclusion_mask must match the probe-by-gallery similarity matrix")
    if np.any((~exclusion_mask).sum(axis=1) == 0):
        raise ValueError("evaluation protocol excluded every gallery sample for a probe")
    similarities = similarities.copy()
    similarities[exclusion_mask] = -np.inf
    ranking = np.argsort(-similarities, axis=1)
    ranked_labels = gallery_labels[ranking]
    ranked_valid = ~np.take_along_axis(exclusion_mask, ranking, axis=1)
    matches = (ranked_labels == probe_labels[:, None]) & ranked_valid
    correct = matches[:, 0].astype(np.float32)
    metrics = {"rank1": float(correct.mean())}
    for k in (5, 10):
        metrics[f"rank{k}"] = float(matches[:, : min(k, matches.shape[1])].any(axis=1).mean())
    aps = []
    for row in matches:
        positions = np.flatnonzero(row)
        aps.append(float(np.mean(np.arange(1, len(positions) + 1) / (positions + 1))) if len(positions) else 0.0)
    metrics["map"] = float(np.mean(aps))
    genuine_mask = (probe_labels[:, None] == gallery_labels[None, :]) & ~exclusion_mask
    impostor_mask = (probe_labels[:, None] != gallery_labels[None, :]) & ~exclusion_mask
    genuine = similarities[genuine_mask]
    impostor = similarities[impostor_mask]
    if len(genuine) and len(impostor):
        # Same thresholds and the same FAR/FRR definitions as before, counted by
        # binary search instead of a Python loop. The loop rescanned both score
        # arrays 2001 times -- about 1.4e10 element comparisons per call, 11.4 s
        # on the CASIA-B probe-by-gallery matrix -- which dominated evaluation.
        genuine_sorted, impostor_sorted = np.sort(genuine), np.sort(impostor)
        thresholds = np.quantile(np.concatenate([genuine, impostor]), np.linspace(0, 1, 2001))
        false_accept = 1.0 - np.searchsorted(impostor_sorted, thresholds, side="left") / impostor_sorted.size
        false_reject = np.searchsorted(genuine_sorted, thresholds, side="left") / genuine_sorted.size
        eer_index = int(np.argmin(np.abs(false_accept - false_reject)))
        metrics["eer"] = float((false_accept[eer_index] + false_reject[eer_index]) / 2)
        for far in (1e-2, 1e-3):
            threshold = np.quantile(impostor, 1 - far, method="higher")
            accepted = np.searchsorted(genuine_sorted, threshold, side="left")
            metrics[f"tar_far_{far:.0e}"] = float(1.0 - accepted / genuine_sorted.size)
    else:
        metrics.update({"eer": float("nan"), "tar_far_1e-02": float("nan"), "tar_far_1e-03": float("nan")})
    return metrics, correct


def _single_view_gallery(
    gallery_x, gallery_y, gallery_views,
    probe_x, probe_y, probe_conditions, probe_views,
    descriptors, part_weight=0.5,
):
    """The official indoor-benchmark protocol, for comparability with published numbers.

    The repository's own protocol pools every gallery view into one gallery and
    excludes identical-view pairs. Published CASIA-B and OU-MVLP numbers come
    from a different task: for each condition, a probe-view x gallery-view
    matrix whose gallery is a *single* view, averaged with the identical-view
    diagonal excluded. See `single_view_gallery_evaluation` in FastPoseGait.
    Numbers from the two protocols are not interchangeable, so both are emitted
    and labelled rather than one standing in for the other.
    """
    views = sorted(set(np.unique(gallery_views)).union(np.unique(probe_views)))
    if len(views) < 2:
        raise ValueError("the single-view-gallery protocol needs at least two camera views")
    rows = []
    for descriptor in descriptors:
        for condition in np.unique(probe_conditions):
            accuracy = np.full((len(views), len(views)), np.nan)
            for i, probe_view in enumerate(views):
                selected = (probe_conditions == condition) & (probe_views == probe_view)
                if not selected.any():
                    continue
                for j, gallery_view in enumerate(views):
                    in_gallery = gallery_views == gallery_view
                    if not in_gallery.any():
                        continue
                    similarity = descriptor_similarity(probe_x[descriptor][selected], gallery_x[descriptor][in_gallery],
                                                       descriptor, part_weight)
                    predicted = gallery_y[in_gallery][np.argmax(similarity, axis=1)]
                    accuracy[i, j] = float((predicted == probe_y[selected]).mean())
            off_diagonal = accuracy.copy()
            np.fill_diagonal(off_diagonal, np.nan)
            if np.all(np.isnan(off_diagonal)):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_probe_view = np.nanmean(off_diagonal, axis=1)
            rows.append({
                "protocol": "single_view_gallery",
                "descriptor": descriptor,
                "corruption": f"official_condition:{condition}",
                "severity": 0.0,
                "delta_rank1": None,
                "rank1": float(np.nanmean(per_probe_view)),
                "rank5": float("nan"), "rank10": float("nan"), "map": float("nan"),
                "eer": float("nan"),
                "tar_far_1e-02": float("nan"), "tar_far_1e-03": float("nan"),
            })
            for probe_view, rank1 in zip(views, per_probe_view):
                if np.isnan(rank1):
                    continue
                rows.append({
                    "protocol": "single_view_gallery_per_view",
                    "descriptor": descriptor,
                    "corruption": f"official_condition:{condition}",
                    "probe_view": str(probe_view),
                    "severity": 0.0,
                    "delta_rank1": None,
                    "rank1": float(rank1),
                    "rank5": float("nan"), "rank10": float("nan"), "map": float("nan"),
                    "eer": float("nan"),
                    "tar_far_1e-02": float("nan"), "tar_far_1e-03": float("nan"),
                })
    return rows


def _evaluate(model, data, cfg, device, seed, realizations):
    stage = cfg["evaluation"].get("split", "test")
    if stage not in {"validation", "test"}:
        raise ValueError("evaluation.split must be validation or test")
    gallery_indices = data.indices(["val_gallery"] if stage == "validation" else ["gallery"])
    probe_indices = data.indices(["val_probe"] if stage == "validation" else ["probe", "test"])
    if not len(gallery_indices) or not len(probe_indices):
        raise ValueError("evaluation requires non-empty gallery and probe/test splits")
    options = _dataset_options(cfg, training=False)
    batch_size = int(cfg["evaluation"].get("batch_size", 32))
    part_weight = float(cfg["evaluation"].get("part_weight", .5))
    descriptors = list(cfg["evaluation"].get("descriptors", [GLOBAL_DESCRIPTOR]))
    descriptor_builder = _descriptor_builder(descriptors)
    gallery_x, gallery_y, _ = _embeddings(
        model, data, gallery_indices, device, dataset_options=options, batch_size=batch_size,
        descriptor_builder=descriptor_builder,
    )
    unknown = set(descriptors).difference(gallery_x)
    if unknown:
        raise ValueError(f"unknown retrieval descriptors: {sorted(unknown)}")
    gallery_views = data.views[gallery_indices]
    rows, correctness = [], {}
    clean_probe_x, clean_probe_y, _ = _embeddings(
        model, data, probe_indices, device, dataset_options=options, batch_size=batch_size,
        descriptor_builder=descriptor_builder,
    )
    probe_conditions = data.conditions[probe_indices]
    probe_views = data.views[probe_indices]
    exclude_identical_view = bool(cfg["dataset"].get("exclude_identical_view", False))

    def view_exclusions(selected_probe_views):
        if not exclude_identical_view:
            return None
        return selected_probe_views[:, None] == gallery_views[None, :]

    clean_rank1 = {}
    for descriptor in descriptors:
        metrics, _ = retrieval_metrics(
            gallery_x[descriptor],
            gallery_y,
            clean_probe_x[descriptor],
            clean_probe_y,
            view_exclusions(probe_views),
            descriptor=descriptor, part_weight=part_weight,
        )
        clean_rank1[descriptor] = metrics["rank1"]
    for field_name, values in (("condition", probe_conditions), ("view", probe_views)):
        for value in np.unique(values):
            selected = values == value
            for descriptor in descriptors:
                metrics, correct = retrieval_metrics(
                    gallery_x[descriptor],
                    gallery_y,
                    clean_probe_x[descriptor][selected],
                    clean_probe_y[selected],
                    view_exclusions(probe_views[selected]),
                    descriptor=descriptor, part_weight=part_weight,
                )
                rows.append({
                    "protocol": "clean_gallery",
                    "descriptor": descriptor,
                    "corruption": f"official_{field_name}:{value}",
                    "severity": 0.0,
                    "delta_rank1": None,
                    **metrics,
                })
                aligned = np.full(len(probe_indices), np.nan, dtype=np.float32)
                aligned[selected] = correct
                correctness[f"{descriptor}_official_{field_name}_{value}"] = aligned
    # Whether the benchmark has an official single-view-gallery protocol is a
    # property of the dataset, so the flag lives beside `exclude_identical_view`
    # in its config rather than in the shared evaluation block.
    if cfg["dataset"].get("single_view_gallery", False):
        rows.extend(
            _single_view_gallery(
                gallery_x, gallery_y, gallery_views,
                clean_probe_x, clean_probe_y, probe_conditions, probe_views,
                descriptors, part_weight,
            )
        )
    families = list(cfg["evaluation"]["corruption_families"])
    severities = [float(v) for v in cfg["evaluation"]["severities"]]
    protocols = list(cfg["evaluation"].get("gallery_protocols", ["clean"]))
    for family_index, family in enumerate(families):
        for severity in severities:
            for gallery_protocol in protocols:
                if gallery_protocol not in {"clean", "occluded"}:
                    raise ValueError(f"unknown gallery protocol: {gallery_protocol}")
                realization_metrics: dict[str, list] = {name: [] for name in descriptors}
                realization_correct: dict[str, list] = {name: [] for name in descriptors}
                count = 1 if severity == 0 else realizations
                for realization in range(count):
                    corruption_seed = int(cfg["evaluation"].get("corruption_seed", 1729)) + family_index * 10_000 + realization
                    if severity == 0:
                        probe_x, probe_y = clean_probe_x, clean_probe_y
                    else:
                        probe_x, probe_y, _ = _embeddings(
                            model, data, probe_indices, device, family, severity,
                            corruption_seed, options, batch_size, descriptor_builder,
                        )
                    if gallery_protocol == "occluded" and severity > 0:
                        current_gallery_x, current_gallery_y, _ = _embeddings(
                            model, data, gallery_indices, device, family, severity,
                            corruption_seed + 5_000, options, batch_size, descriptor_builder,
                        )
                    else:
                        current_gallery_x, current_gallery_y = gallery_x, gallery_y
                    for descriptor in descriptors:
                        metrics, correct = retrieval_metrics(
                            current_gallery_x[descriptor],
                            current_gallery_y,
                            probe_x[descriptor],
                            probe_y,
                            view_exclusions(probe_views),
                            descriptor=descriptor, part_weight=part_weight,
                        )
                        realization_metrics[descriptor].append(metrics)
                        realization_correct[descriptor].append(correct)
                protocol_name = f"{gallery_protocol}_gallery"
                for descriptor in descriptors:
                    averaged = {
                        key: float(np.mean([item[key] for item in realization_metrics[descriptor]]))
                        for key in realization_metrics[descriptor][0]
                    }
                    rows.append({
                        "protocol": protocol_name,
                        "descriptor": descriptor,
                        "corruption": family,
                        "severity": severity,
                        "delta_rank1": float(clean_rank1[descriptor] - averaged["rank1"]),
                        **averaged,
                    })
                    correctness[f"{descriptor}_{protocol_name}_{family}_{severity:g}"] = np.mean(
                        realization_correct[descriptor], axis=0
                    )
    return rows, correctness, probe_indices



@torch.no_grad()
def validation_score(model, data, cfg, device):
    val = cfg["validation"]
    descriptor = val.get("descriptor", "global")
    gallery_indices, probe_indices = data.indices(["val_gallery"]), data.indices(["val_probe"])
    if not len(gallery_indices) or not len(probe_indices):
        raise ValueError("checkpoint selection requires validation gallery and probe")
    options = _dataset_options(cfg, False)
    batch_size = int(cfg["evaluation"].get("batch_size", 32))
    part_weight = float(cfg["evaluation"].get("part_weight", .5))
    descriptor_builder = _descriptor_builder([descriptor])
    gallery, gy, _ = _embeddings(model, data, gallery_indices, device,
                                  dataset_options=options, batch_size=batch_size,
                                  descriptor_builder=descriptor_builder)
    gv, pv = data.views[gallery_indices], data.views[probe_indices]
    pc = data.conditions[probe_indices]
    exclusion = pv[:, None] == gv[None] if cfg["dataset"].get("exclude_identical_view", False) else None
    metrics = {}
    families = list(val.get("corruption_families", ["body_part", "random_joint"]))
    for number, family in enumerate([None] + families):
        name = "clean" if family is None else family
        probe, py, _ = _embeddings(model, data, probe_indices, device, family,
                                   0 if family is None else float(val.get("severity", .3)),
                                   int(val.get("corruption_seed", 2718)) + number * 10000,
                                   options, batch_size, descriptor_builder)
        if cfg["dataset"].get("single_view_gallery", False):
            rows = _single_view_gallery(gallery, gy, gv, probe, py, pc, pv, [descriptor], part_weight)
            aggregate_rows = [r for r in rows if r["protocol"] == "single_view_gallery"]
            metrics[name] = float(np.mean([r["rank1"] for r in aggregate_rows]))
            for row in aggregate_rows:
                metrics[name + "_" + row["corruption"].split(":")[-1]] = row["rank1"]
        else:
            scores = descriptor_similarity(probe[descriptor], gallery[descriptor], descriptor, part_weight)
            if exclusion is not None:
                if np.any((~exclusion).sum(1) == 0):
                    raise ValueError("validation excluded every gallery candidate")
                scores[exclusion] = -np.inf
            metrics[name] = float((gy[scores.argmax(1)] == py).mean())
    robust_weight = float(val.get("robust_weight", .25))
    if not 0 <= robust_weight <= 1:
        raise ValueError("validation.robust_weight must be in [0,1]")
    robust = float(np.mean([metrics[f] for f in families])) if families else metrics["clean"]
    metrics["score"] = (1 - robust_weight) * metrics["clean"] + robust_weight * robust
    return metrics


def run_experiment(project_root: Path, cfg: dict, seed: int, epochs: int | None, device_name: str, realizations: int | None):
    cfg = copy.deepcopy(cfg)
    seed_everything(seed)
    torch.use_deterministic_algorithms(bool(cfg["train"].get("deterministic", False)))
    if cfg["train"].get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    dataset_path = resolve_dataset_path(project_root, cfg["dataset"])
    data = PoseData.load(dataset_path)
    data, held_out = development_split(data, cfg)
    cfg.setdefault("validation", {})["held_out_identities"] = held_out
    stage = cfg["evaluation"].get("split", "test")
    if stage == "validation" and not held_out:
        raise ValueError("validation evaluation requires held-out validation identities")
    device = select_device(device_name)
    train_ids = np.unique(data.labels[data.split == "train"])
    if not len(train_ids):
        raise ValueError("no training identities")
    backbone = str(cfg["model"].get("backbone", "tcn"))
    model = encoder_from_config(cfg, data.x.shape[-1], data.x.shape[2], len(train_ids)).to(device)
    epochs = int(epochs if epochs is not None else cfg.get("epochs", cfg["dataset"].get("epochs", cfg["train"]["epochs"])))
    realizations = int(realizations if realizations is not None else cfg["evaluation"]["corruption_realizations"])
    if epochs < 1 or realizations < 1:
        raise ValueError("epochs and corruption realizations must be positive")
    cfg["epochs"] = cfg["train"]["epochs"] = epochs
    cfg["evaluation"]["corruption_realizations"] = realizations
    cfg["train_identities"] = train_ids.tolist()
    cfg["source_sha256"] = source_fingerprint(project_root)
    cfg["dataset_sha256"] = data.checksum()
    cfg["config_fingerprint"] = configuration_fingerprint(cfg)
    suffix = "" if backbone == "tcn" else f"_{backbone}"
    run_id = f"{cfg['dataset']['name']}{suffix}_{cfg['method_name']}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir = project_root / cfg.get("results_root", "results_v2") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit = git_commit(project_root)
    frozen_cfg = {**cfg, "seed": seed, "device": str(device), "dataset_path": str(dataset_path),
                  "run_id": run_id, "git_commit": commit}
    with (run_dir / "config.yaml").open("x") as handle:
        yaml.safe_dump(frozen_cfg, handle, sort_keys=False)
    validator = (lambda current: validation_score(current, data, cfg, device)) if held_out else None
    try:
        history, gpu_hours = _train(model, data, cfg, device, seed, epochs,
                                   output_dir=run_dir, checkpoint_config=frozen_cfg, validate=validator)
        selected_epoch = epochs
        if validator is not None:
            selected_epoch = torch.load(run_dir / "checkpoint-best.pt", map_location="cpu", weights_only=True)["epoch"]
        # Preserve the selected model BEFORE the potentially long evaluation.
        save_checkpoint(run_dir / "checkpoint.pt", model, frozen_cfg, selected_epoch)
        rows, correctness, probe_indices = _evaluate(model, data, cfg, device, seed, realizations)
    except BaseException as error:
        failure = {"status": "failed", "type": type(error).__name__, "message": str(error), "time_utc": datetime.now(timezone.utc).isoformat()}
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    np.savez_compressed(run_dir / "probe_outcomes.npz", probe_indices=probe_indices,
                        probe_labels=data.labels[probe_indices], **correctness)
    metrics = {"run_id": run_id, "git_commit": commit,
               "dataset": cfg["dataset"]["name"], "method": cfg["method_name"], "backbone": backbone,
               "experiment_id": cfg.get("experiment_id", "corrected_v2"),
               "config_fingerprint": cfg["config_fingerprint"], "source_sha256": cfg["source_sha256"],
               "evaluation_split": stage, "seed": seed, "epochs": epochs, "selected_epoch": selected_epoch,
               "device": str(device), "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "train_hours": gpu_hours, "dataset_sha256": cfg["dataset_sha256"],
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "platform": platform.platform(),
                               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
               "results": rows, "status": "complete"}
    with (run_dir / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
    return run_dir, metrics
