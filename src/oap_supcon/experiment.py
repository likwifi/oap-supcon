from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .augment import corrupt, temporal_crop_view, training_views
from .data import IdentityBatchSampler, PoseData, PoseDataset, resolve_dataset_path
from .losses import part_contrastive, supervised_contrastive, temporal_contrastive
from .model import PoseEncoder


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


def load_configuration(project_root: Path, dataset: str, method: str) -> dict:
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
    return cfg


def _train(
    model: PoseEncoder,
    data: PoseData,
    cfg: dict,
    device: torch.device,
    seed: int,
    epochs: int,
):
    train_indices = data.indices(["train"])
    if not len(train_indices):
        raise ValueError("dataset has no train split")
    train_labels = data.labels[train_indices]
    label_values = sorted(np.unique(train_labels).tolist())
    label_map = {value: index for index, value in enumerate(label_values)}
    dataset = PoseDataset(data, train_indices)
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        sampler = IdentityBatchSampler(
            train_labels,
            int(train_cfg["identities_per_batch"]),
            int(train_cfg["samples_per_identity"]),
            seed + epoch,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(train_cfg["num_workers"]))
        model.train()
        totals = []
        component_totals = {"ce": [], "global": [], "part": [], "temporal": []}
        generator = torch.Generator(device=device.type).manual_seed(seed * 10000 + epoch)
        for x, visibility, labels, _ in loader:
            x, visibility = x.to(device), visibility.to(device)
            targets = torch.tensor([label_map[int(v)] for v in labels], device=device)
            maximum_severity = float(train_cfg["train_severity"])
            if train_cfg.get("severity_schedule") == "linear":
                start = float(train_cfg.get("train_severity_start", 0.1))
                progress = epoch / max(1, epochs - 1)
                train_severity = start + progress * (maximum_severity - start)
            else:
                train_severity = maximum_severity
            (xa, va), (xb, vb) = training_views(
                x, visibility, cfg["method"]["augmentation"], train_severity, generator
            )
            a, b = model(xa, va), model(xb, vb)
            repeated_labels = torch.cat([targets, targets])
            method = cfg["method"]
            loss = torch.zeros((), device=device)
            ce_weight = float(method.get("ce_weight", 0))
            if ce_weight:
                ce = (F.cross_entropy(a["logits"], targets) + F.cross_entropy(b["logits"], targets)) / 2
                loss = loss + ce_weight * ce
                component_totals["ce"].append(float(ce.detach().cpu()))
            global_weight = float(method.get("global_weight", 0))
            if global_weight:
                global_loss = supervised_contrastive(
                    torch.cat([a["projection"], b["projection"]]), repeated_labels, float(train_cfg["temperature"])
                )
                loss = loss + global_weight * global_loss
                component_totals["global"].append(float(global_loss.detach().cpu()))
            part_weight = float(method.get("part_weight", 0))
            if part_weight:
                part_loss = part_contrastive(
                    torch.cat([a["parts"], b["parts"]]),
                    torch.cat([a["reliability"], b["reliability"]]),
                    repeated_labels,
                    float(train_cfg.get("part_temperature", train_cfg["temperature"])),
                    bool(method.get("visibility_gating", True)),
                )
                loss = loss + part_weight * part_loss
                component_totals["part"].append(float(part_loss.detach().cpu()))
            temporal_weight = float(method.get("temporal_weight", 0))
            if temporal_weight:
                crop_min = float(train_cfg.get("temporal_crop_min", 0.5))
                crop_max = float(train_cfg.get("temporal_crop_max", 0.8))
                c1, v1 = temporal_crop_view(x, visibility, generator, crop_min, crop_max)
                c2, v2 = temporal_crop_view(x, visibility, generator, crop_min, crop_max)
                temp = temporal_contrastive(
                    model.project(c1, v1), model.project(c2, v2),
                    float(train_cfg.get("temporal_temperature", train_cfg["temperature"])),
                )
                loss = loss + temporal_weight * temp
                component_totals["temporal"].append(float(temp.detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        history.append({
            "epoch": epoch + 1,
            "train_severity": train_severity,
            "loss": float(np.mean(totals)),
            **{
                f"{name}_loss": float(np.mean(values)) if values else None
                for name, values in component_totals.items()
            },
        })
    return history, (time.perf_counter() - started) / 3600


@torch.no_grad()
def _embeddings(model, data: PoseData, indices: np.ndarray, device, corruption: str | None = None, severity: float = 0, seed: int = 0):
    loader = DataLoader(PoseDataset(data, indices), batch_size=128, shuffle=False)
    outputs, output_labels, output_indices = [], [], []
    model.eval()
    generator = torch.Generator(device=device.type).manual_seed(seed)
    for x, visibility, labels, original_indices in loader:
        x, visibility = x.to(device), visibility.to(device)
        if corruption and severity > 0:
            x, visibility = corrupt(x, visibility, corruption, severity, generator)
        outputs.append(model.embed(x, visibility).cpu())
        output_labels.append(labels)
        output_indices.append(original_indices)
    return torch.cat(outputs).numpy(), torch.cat(output_labels).numpy(), torch.cat(output_indices).numpy()


def retrieval_metrics(
    gallery_embeddings,
    gallery_labels,
    probe_embeddings,
    probe_labels,
    exclusion_mask: np.ndarray | None = None,
):
    similarities = probe_embeddings @ gallery_embeddings.T
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
        thresholds = np.quantile(np.concatenate([genuine, impostor]), np.linspace(0, 1, 2001))
        false_accept = np.array([(impostor >= threshold).mean() for threshold in thresholds])
        false_reject = np.array([(genuine < threshold).mean() for threshold in thresholds])
        eer_index = int(np.argmin(np.abs(false_accept - false_reject)))
        metrics["eer"] = float((false_accept[eer_index] + false_reject[eer_index]) / 2)
        for far in (1e-2, 1e-3):
            threshold = np.quantile(impostor, 1 - far, method="higher")
            metrics[f"tar_far_{far:.0e}"] = float((genuine >= threshold).mean())
    else:
        metrics.update({"eer": float("nan"), "tar_far_1e-02": float("nan"), "tar_far_1e-03": float("nan")})
    return metrics, correct


def _evaluate(model, data, cfg, device, seed, realizations):
    gallery_indices = data.indices(["gallery"])
    probe_indices = data.indices(["probe", "test"])
    if not len(gallery_indices) or not len(probe_indices):
        raise ValueError("evaluation requires non-empty gallery and probe/test splits")
    gallery_x, gallery_y, _ = _embeddings(model, data, gallery_indices, device)
    gallery_views = data.views[gallery_indices]
    rows, correctness = [], {}
    clean_probe_x, clean_probe_y, _ = _embeddings(model, data, probe_indices, device)
    probe_conditions = data.conditions[probe_indices]
    probe_views = data.views[probe_indices]
    exclude_identical_view = bool(cfg["dataset"].get("exclude_identical_view", False))

    def view_exclusions(selected_probe_views):
        if not exclude_identical_view:
            return None
        return selected_probe_views[:, None] == gallery_views[None, :]

    clean_metrics, _ = retrieval_metrics(
        gallery_x, gallery_y, clean_probe_x, clean_probe_y, view_exclusions(probe_views)
    )
    clean_rank1 = clean_metrics["rank1"]
    for field_name, values in (("condition", probe_conditions), ("view", probe_views)):
        for value in np.unique(values):
            selected = values == value
            metrics, correct = retrieval_metrics(
                gallery_x,
                gallery_y,
                clean_probe_x[selected],
                clean_probe_y[selected],
                view_exclusions(probe_views[selected]),
            )
            rows.append({
                "protocol": "clean_gallery",
                "corruption": f"official_{field_name}:{value}",
                "severity": 0.0,
                "delta_rank1": None,
                **metrics,
            })
            aligned = np.full(len(probe_indices), np.nan, dtype=np.float32)
            aligned[selected] = correct
            correctness[f"official_{field_name}_{value}"] = aligned
    families = list(cfg["evaluation"]["corruption_families"])
    severities = [float(v) for v in cfg["evaluation"]["severities"]]
    protocols = list(cfg["evaluation"].get("gallery_protocols", ["clean"]))
    for family_index, family in enumerate(families):
        for severity in severities:
            for gallery_protocol in protocols:
                if gallery_protocol not in {"clean", "occluded"}:
                    raise ValueError(f"unknown gallery protocol: {gallery_protocol}")
                realization_metrics, realization_correct = [], []
                count = 1 if severity == 0 else realizations
                for realization in range(count):
                    corruption_seed = seed * 100_000 + family_index * 10_000 + realization
                    probe_x, probe_y, _ = _embeddings(
                        model, data, probe_indices, device, family, severity, corruption_seed
                    )
                    if gallery_protocol == "occluded" and severity > 0:
                        current_gallery_x, current_gallery_y, _ = _embeddings(
                            model, data, gallery_indices, device, family, severity, corruption_seed + 5_000
                        )
                    else:
                        current_gallery_x, current_gallery_y = gallery_x, gallery_y
                    metrics, correct = retrieval_metrics(
                        current_gallery_x,
                        current_gallery_y,
                        probe_x,
                        probe_y,
                        view_exclusions(probe_views),
                    )
                    realization_metrics.append(metrics)
                    realization_correct.append(correct)
                averaged = {
                    key: float(np.mean([item[key] for item in realization_metrics]))
                    for key in realization_metrics[0]
                }
                protocol_name = f"{gallery_protocol}_gallery"
                row = {
                    "protocol": protocol_name,
                    "corruption": family,
                    "severity": severity,
                    "delta_rank1": float(clean_rank1 - averaged["rank1"]),
                    **averaged,
                }
                rows.append(row)
                correctness[f"{protocol_name}_{family}_{severity:g}"] = np.mean(
                    realization_correct, axis=0
                )
    return rows, correctness, probe_indices


def run_experiment(project_root: Path, cfg: dict, seed: int, epochs: int | None, device_name: str, realizations: int | None):
    seed_everything(seed)
    dataset_path = resolve_dataset_path(project_root, cfg["dataset"])
    data = PoseData.load(dataset_path)
    device = select_device(device_name)
    train_ids = np.unique(data.labels[data.split == "train"])
    if not len(train_ids):
        raise ValueError("no training identities")
    model_cfg = cfg["model"]
    model = PoseEncoder(
        data.x.shape[-1], data.x.shape[2], len(train_ids), int(model_cfg["hidden_dim"]),
        int(model_cfg["embedding_dim"]), float(model_cfg["dropout"])
    ).to(device)
    epochs = int(epochs if epochs is not None else cfg["dataset"].get("epochs", cfg["train"]["epochs"]))
    realizations = int(realizations if realizations is not None else cfg["evaluation"]["corruption_realizations"])
    run_id = f"{cfg['dataset']['name']}_{cfg['method_name']}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir = project_root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_cfg = {**cfg, "seed": seed, "epochs": epochs, "device": str(device), "dataset_path": str(dataset_path), "run_id": run_id}
    with (run_dir / "config.yaml").open("x") as handle:
        yaml.safe_dump(frozen_cfg, handle, sort_keys=False)
    try:
        history, gpu_hours = _train(model, data, cfg, device, seed, epochs)
        rows, correctness, probe_indices = _evaluate(model, data, cfg, device, seed, realizations)
    except BaseException as error:
        failure = {"status": "failed", "type": type(error).__name__, "message": str(error), "time_utc": datetime.now(timezone.utc).isoformat()}
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    torch.save({"model": model.state_dict(), "config": frozen_cfg}, run_dir / "checkpoint.pt")
    with (run_dir / "history.json").open("x") as handle:
        json.dump(history, handle, indent=2)
    np.savez_compressed(
        run_dir / "probe_outcomes.npz", probe_indices=probe_indices,
        probe_labels=data.labels[probe_indices], **correctness
    )
    metrics = {
        "run_id": run_id,
        "git_commit": git_commit(project_root),
        "dataset": cfg["dataset"]["name"],
        "method": cfg["method_name"],
        "seed": seed,
        "epochs": epochs,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_hours": gpu_hours,
        "dataset_sha256": data.checksum(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "results": rows,
        "status": "complete",
    }
    with (run_dir / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
    return run_dir, metrics
