"""Checkpoint-only controls for part-aware retrieval descriptors.

The controls deliberately do not train or alter the saved backbone.  All
descriptor variants are derived from a single encoder forward per corruption
realization, preventing random-head and fusion sweeps from multiplying GPU
work.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import PoseData, development_split, resolve_dataset_path
from .experiment import (
    _dataset_options,
    _embeddings,
    _single_view_gallery,
    descriptor_similarity,
    git_commit,
    retrieval_metrics,
    select_device,
)
from .model import encoder_from_config
from .provenance import source_fingerprint


GLOBAL = "global"
RAW_RELIABLE = "raw_reliable"
RAW_UNIFORM = "raw_uniform"
FROZEN_RELIABLE = "frozen_head_reliable"


def _reset_module(module: torch.nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def fresh_part_heads(model, seeds: list[int], device: torch.device) -> dict[int, torch.nn.ModuleList]:
    """Create exact architectural copies of the saved part heads with fresh weights."""
    if not hasattr(model, "part_heads"):
        raise ValueError("descriptor controls require a backbone with independent part_heads")
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    result = {}
    for seed in seeds:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            heads = copy.deepcopy(model.part_heads)
            heads.apply(_reset_module)
        result[int(seed)] = heads.to(device).eval().requires_grad_(False)
    return result


def _structured(embedding: torch.Tensor, parts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if embedding.shape[-1] != parts.shape[-1]:
        raise ValueError(
            "raw part width must match the global embedding width for score fusion; "
            f"got {parts.shape[-1]} and {embedding.shape[-1]}"
        )
    vectors = torch.cat([embedding[:, None], parts], dim=1)
    all_weights = torch.cat([torch.ones_like(weights[:, :1]), weights], dim=1)
    return torch.cat([vectors, all_weights[..., None]], dim=-1)


def control_descriptor_builder(model, random_head_seeds: list[int], device: torch.device):
    """Return a batch builder for deterministic and random frozen-head controls."""
    random_heads = fresh_part_heads(model, random_head_seeds, device)

    def build(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = output["embedding"]
        reliability = output["reliability"]
        presence = (reliability > 0).to(reliability.dtype)
        descriptors = {
            GLOBAL: embedding,
            RAW_RELIABLE: _structured(embedding, output["raw_parts"], reliability),
            RAW_UNIFORM: _structured(embedding, output["raw_parts"], presence),
            FROZEN_RELIABLE: _structured(embedding, output["parts"], reliability),
        }
        features = output["part_features"]
        for seed, heads in random_heads.items():
            projected = []
            for part, head in enumerate(heads):
                vector = F.normalize(head(features[:, part]), dim=-1)
                projected.append(vector * presence[:, part, None])
            descriptors[f"random_head_{seed}_reliable"] = _structured(
                embedding, torch.stack(projected, dim=1), reliability
            )
        return descriptors

    return build


def _weights_for(descriptor: str, fusion_weights: list[float]) -> list[float]:
    return [0.0] if descriptor == GLOBAL else fusion_weights


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def outcome_key(descriptor: str, fusion_weight: float, protocol: str, corruption: str, severity: float) -> str:
    return _safe(
        f"{descriptor}_w{fusion_weight:g}_{protocol}_{corruption}_{severity:g}"
    )


@torch.no_grad()
def evaluate_descriptor_controls(
    model,
    data: PoseData,
    cfg: dict,
    device: torch.device,
    seed: int,
    realizations: int,
    random_head_seeds: list[int],
    fusion_weights: list[float],
):
    """Evaluate every P0 descriptor while sharing every encoder forward."""
    stage = cfg["evaluation"].get("split", "test")
    if stage not in {"validation", "test"}:
        raise ValueError("evaluation.split must be validation or test")
    gallery_indices = data.indices(["val_gallery"] if stage == "validation" else ["gallery"])
    probe_indices = data.indices(["val_probe"] if stage == "validation" else ["probe", "test"])
    if not len(gallery_indices) or not len(probe_indices):
        raise ValueError("evaluation requires non-empty gallery and probe/test splits")
    options = _dataset_options(cfg, training=False)
    batch_size = int(cfg["evaluation"].get("batch_size", 32))
    builder = control_descriptor_builder(model, random_head_seeds, device)

    def embeddings(indices, corruption=None, severity=0.0, corruption_seed=0):
        return _embeddings(
            model,
            data,
            indices,
            device,
            corruption,
            severity,
            corruption_seed,
            options,
            batch_size,
            descriptor_builder=builder,
        )

    gallery_x, gallery_y, _ = embeddings(gallery_indices)
    clean_probe_x, clean_probe_y, _ = embeddings(probe_indices)
    descriptors = list(gallery_x)
    gallery_views = data.views[gallery_indices]
    probe_conditions = data.conditions[probe_indices]
    probe_views = data.views[probe_indices]
    exclude_identical_view = bool(cfg["dataset"].get("exclude_identical_view", False))

    def exclusions(selected_probe_views):
        if not exclude_identical_view:
            return None
        return selected_probe_views[:, None] == gallery_views[None, :]

    rows, correctness = [], {}
    clean_rank1 = {}
    for descriptor in descriptors:
        for weight in _weights_for(descriptor, fusion_weights):
            metrics, _ = retrieval_metrics(
                gallery_x[descriptor], gallery_y, clean_probe_x[descriptor], clean_probe_y,
                exclusions(probe_views), descriptor=descriptor, part_weight=weight,
            )
            clean_rank1[(descriptor, weight)] = metrics["rank1"]

    for field_name, values in (("condition", probe_conditions), ("view", probe_views)):
        for value in np.unique(values):
            selected = values == value
            for descriptor in descriptors:
                for weight in _weights_for(descriptor, fusion_weights):
                    metrics, correct = retrieval_metrics(
                        gallery_x[descriptor], gallery_y,
                        clean_probe_x[descriptor][selected], clean_probe_y[selected],
                        exclusions(probe_views[selected]), descriptor=descriptor, part_weight=weight,
                    )
                    corruption = f"official_{field_name}:{value}"
                    rows.append({
                        "protocol": "clean_gallery", "descriptor": descriptor,
                        "fusion_weight": weight, "corruption": corruption,
                        "severity": 0.0, "delta_rank1": None, **metrics,
                    })
                    aligned = np.full(len(probe_indices), np.nan, dtype=np.float32)
                    aligned[selected] = correct
                    correctness[outcome_key(descriptor, weight, "clean_gallery", corruption, 0)] = aligned

    if cfg["dataset"].get("single_view_gallery", False):
        for descriptor in descriptors:
            for weight in _weights_for(descriptor, fusion_weights):
                official = _single_view_gallery(
                    {descriptor: gallery_x[descriptor]}, gallery_y, gallery_views,
                    {descriptor: clean_probe_x[descriptor]}, clean_probe_y,
                    probe_conditions, probe_views, [descriptor], weight,
                )
                for row in official:
                    row["fusion_weight"] = weight
                rows.extend(official)

    families = list(cfg["evaluation"]["corruption_families"])
    severities = [float(value) for value in cfg["evaluation"]["severities"]]
    protocols = list(cfg["evaluation"].get("gallery_protocols", ["clean"]))
    for family_index, family in enumerate(families):
        for severity in severities:
            for gallery_protocol in protocols:
                if gallery_protocol not in {"clean", "occluded"}:
                    raise ValueError(f"unknown gallery protocol: {gallery_protocol}")
                variants = [
                    (descriptor, weight)
                    for descriptor in descriptors
                    for weight in _weights_for(descriptor, fusion_weights)
                ]
                realization_metrics = {variant: [] for variant in variants}
                realization_correct = {variant: [] for variant in variants}
                count = 1 if severity == 0 else realizations
                for realization in range(count):
                    corruption_seed = (
                        int(cfg["evaluation"].get("corruption_seed", 1729))
                        + family_index * 10_000 + realization
                    )
                    if severity == 0:
                        probe_x, probe_y = clean_probe_x, clean_probe_y
                    else:
                        probe_x, probe_y, _ = embeddings(
                            probe_indices, family, severity, corruption_seed
                        )
                    if gallery_protocol == "occluded" and severity > 0:
                        current_gallery_x, current_gallery_y, _ = embeddings(
                            gallery_indices, family, severity, corruption_seed + 5_000
                        )
                    else:
                        current_gallery_x, current_gallery_y = gallery_x, gallery_y
                    for descriptor, weight in variants:
                        metrics, correct = retrieval_metrics(
                            current_gallery_x[descriptor], current_gallery_y,
                            probe_x[descriptor], probe_y, exclusions(probe_views),
                            descriptor=descriptor, part_weight=weight,
                        )
                        realization_metrics[(descriptor, weight)].append(metrics)
                        realization_correct[(descriptor, weight)].append(correct)
                protocol_name = f"{gallery_protocol}_gallery"
                for descriptor, weight in variants:
                    averaged = {
                        key: float(np.mean([item[key] for item in realization_metrics[(descriptor, weight)]]))
                        for key in realization_metrics[(descriptor, weight)][0]
                    }
                    rows.append({
                        "protocol": protocol_name, "descriptor": descriptor,
                        "fusion_weight": weight, "corruption": family, "severity": severity,
                        "delta_rank1": float(clean_rank1[(descriptor, weight)] - averaged["rank1"]),
                        **averaged,
                    })
                    correctness[outcome_key(descriptor, weight, protocol_name, family, severity)] = np.mean(
                        realization_correct[(descriptor, weight)], axis=0
                    )
    return rows, correctness, probe_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_descriptor_controls(
    project_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    device_name: str = "auto",
    random_head_seeds: list[int] | None = None,
    fusion_weights: list[float] | None = None,
    realizations: int | None = None,
):
    """Load one baseline checkpoint and write one immutable P0 result directory."""
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = copy.deepcopy(checkpoint["config"])
    if cfg.get("method_name") != "metric_baseline":
        raise ValueError("P0 controls must use a metric_baseline checkpoint")
    if cfg.get("evaluation", {}).get("split") != "validation":
        raise ValueError("P0 must run on the validation split; freeze the design before official test access")
    random_head_seeds = list(random_head_seeds or range(1001, 1011))
    fusion_weights = [float(value) for value in (fusion_weights or [0, .25, .5, .75, 1])]
    if len(set(random_head_seeds)) < 10:
        raise ValueError("P0 requires at least 10 distinct fresh random-head seeds")
    if any(not 0 <= value <= 1 for value in fusion_weights):
        raise ValueError("fusion weights must lie in [0,1]")
    if sorted(set(fusion_weights)) != sorted(fusion_weights):
        raise ValueError("fusion weights must be unique")

    data_path = resolve_dataset_path(project_root, cfg["dataset"])
    data = PoseData.load(data_path)
    data, held_out = development_split(data, cfg)
    expected_held_out = sorted(cfg.get("validation", {}).get("held_out_identities", []))
    if expected_held_out and held_out != expected_held_out:
        raise ValueError("checkpoint validation identities do not match the reconstructed split")
    train_ids = cfg.get("train_identities") or np.unique(data.labels[data.split == "train"]).tolist()
    device = select_device(device_name)
    model = encoder_from_config(cfg, data.x.shape[-1], data.x.shape[2], len(train_ids)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    control_realizations = int(
        realizations if realizations is not None else cfg["evaluation"]["corruption_realizations"]
    )
    if control_realizations < 1:
        raise ValueError("corruption realizations must be positive")

    source_run = checkpoint_path.parent.name
    run_id = f"{source_run}_descriptor_controls"
    output_root = output_root if output_root.is_absolute() else project_root / output_root
    run_dir = output_root / run_id
    if run_dir.exists():
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text())
            if existing.get("status") == "complete" and existing.get("source_checkpoint") == str(checkpoint_path):
                return run_dir, existing
        raise FileExistsError(
            f"incomplete P0 output already exists at {run_dir}; move it aside before retrying"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    control_config = {
        "experiment_id": "descriptor_controls_p0",
        "source_run": source_run,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset": cfg["dataset"],
        "method_name": cfg["method_name"],
        "model": cfg["model"],
        "pose": cfg.get("pose", {}),
        "train": cfg["train"],
        "validation": cfg["validation"],
        "evaluation": {**cfg["evaluation"], "corruption_realizations": control_realizations},
        "seed": int(cfg["seed"]),
        "random_head_seeds": random_head_seeds,
        "fusion_weights": fusion_weights,
        "held_out_identities": held_out,
    }
    with (run_dir / "config.yaml").open("x") as handle:
        yaml.safe_dump(control_config, handle, sort_keys=False)
    started = time.perf_counter()
    try:
        rows, correctness, probe_indices = evaluate_descriptor_controls(
            model, data, cfg, device, int(cfg["seed"]), control_realizations,
            random_head_seeds, fusion_weights,
        )
    except BaseException as error:
        failure = {
            "status": "failed", "type": type(error).__name__, "message": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    elapsed_hours = (time.perf_counter() - started) / 3600
    np.savez_compressed(
        run_dir / "probe_outcomes.npz", probe_indices=probe_indices,
        probe_labels=data.labels[probe_indices], **correctness,
    )
    metrics = {
        "run_id": run_id,
        "status": "complete",
        "experiment_id": "descriptor_controls_p0",
        "source_run": source_run,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": control_config["source_checkpoint_sha256"],
        "source_checkpoint_epoch": control_config["source_checkpoint_epoch"],
        "git_commit": git_commit(project_root),
        "source_sha256": source_fingerprint(project_root),
        "dataset": cfg["dataset"]["name"],
        "dataset_sha256": cfg.get("dataset_sha256", data.checksum()),
        "method": cfg["method_name"],
        "backbone": cfg["model"].get("backbone", "tcn"),
        "evaluation_split": cfg["evaluation"]["split"],
        "seed": int(cfg["seed"]),
        "random_head_seeds": random_head_seeds,
        "fusion_weights": fusion_weights,
        "evaluation_hours": elapsed_hours,
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "results": rows,
    }
    with (run_dir / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
    return run_dir, metrics
