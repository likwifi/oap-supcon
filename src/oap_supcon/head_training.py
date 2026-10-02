"""P1 frozen-backbone training for local descriptor heads only."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .augment import frame_mask_from_lengths, training_views
from .data import IdentityBatchSampler, PoseData, PoseDataset, development_split, resolve_dataset_path
from .experiment import (
    _evaluate,
    git_commit,
    seed_everything,
    select_device,
    validation_score,
)
from .losses import part_batch_hard_triplet
from .model import encoder_from_config
from .provenance import source_fingerprint
from .training import atomic_json, dataset_options, save_checkpoint


HEAD_MODES = ("distill", "triplet_variance")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_worker(_):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(torch.initial_seed() % (2**32))


def part_variances(parts: torch.Tensor, reliability: torch.Tensor, minimum: float) -> torch.Tensor:
    """Coordinate-wise variance per normalized part over observed samples."""
    values = []
    for part in range(parts.shape[1]):
        selected = parts[reliability[:, part] >= minimum, part]
        if len(selected) > 1:
            values.append(selected.float().var(0, unbiased=False).mean())
        else:
            values.append(parts.new_zeros((), dtype=torch.float32))
    return torch.stack(values)


def variance_floor_loss(variances: torch.Tensor, floor: float) -> torch.Tensor:
    if floor <= 0:
        raise ValueError("variance floor must be positive")
    # Normalizing by the floor makes a fully collapsed part contribute one,
    # rather than an ineffectual 1e-4-scale penalty.
    return torch.square(F.relu(floor - variances) / floor).mean()


def distillation_loss(
    parts: torch.Tensor,
    global_embedding: torch.Tensor,
    reliability: torch.Tensor,
    minimum: float,
) -> torch.Tensor:
    weights = reliability * (reliability >= minimum)
    cosine = (parts * global_embedding.detach()[:, None]).sum(-1)
    return ((1 - cosine) * weights).sum() / weights.sum().clamp_min(1e-8)


def _freeze_backbone(model) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "part_heads"):
        raise ValueError("P1 requires a backbone with independent part_heads")
    for parameter in model.part_heads.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.part_heads.train()
    return sum(parameter.numel() for parameter in model.part_heads.parameters())


def train_frozen_heads(
    model,
    data: PoseData,
    cfg: dict,
    device: torch.device,
    seed: int,
    mode: str,
    epochs: int,
    output_dir: Path,
    checkpoint_config: dict,
    *,
    learning_rate: float = 1e-3,
    variance_floor: float = 1e-4,
    variance_weight: float = .1,
):
    if mode not in HEAD_MODES:
        raise ValueError(f"unknown head mode {mode}; choose from {', '.join(HEAD_MODES)}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    trainable = _freeze_backbone(model)
    indices = data.indices(["train"])
    labels = data.labels[indices]
    dataset = PoseDataset(data, indices, seed=seed, **dataset_options(cfg, True))
    train = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.part_heads.parameters(), lr=learning_rate,
        weight_decay=float(train.get("weight_decay", 1e-4)),
    )
    warmup = min(2, max(0, epochs - 1))

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = bool(train.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaled = use_amp and amp_dtype == torch.float16
    scaler = (
        torch.amp.GradScaler("cuda", enabled=scaled)
        if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=scaled)
    )
    minimum = float(train.get("min_part_reliability", .05))
    best_score, best_epoch, history = -math.inf, None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.eval()
        model.part_heads.train()
        sampler = IdentityBatchSampler(
            labels, int(train["identities_per_batch"]), int(train["samples_per_identity"]),
            seed + epoch, views=data.views[indices] if train.get("diverse_views", False) else None,
        )
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=int(train.get("num_workers", 0)),
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed * 1000 + epoch),
            pin_memory=device.type == "cuda",
        )
        main_rng = torch.Generator(device=device.type).manual_seed(seed * 10_000 + epoch)
        maximum = float(train.get("train_severity", .5))
        start_severity = float(train.get("train_severity_start", .05))
        ramp = max(1, round(epochs * float(train.get("severity_ramp_fraction", .5))) - 1)
        severity = start_severity + min(1, epoch / ramp) * (maximum - start_severity)
        losses, objective_losses, variance_losses, variance_log, reliability_log = [], [], [], [], []
        current_lr = optimizer.param_groups[0]["lr"]
        for x, confidence, lengths, identity, sample_ids in loader:
            x, confidence, lengths = x.to(device), confidence.to(device), lengths.to(device)
            frame_mask = frame_mask_from_lengths(confidence, lengths)
            (xa, va), (xb, vb) = training_views(
                x, confidence, cfg["method"]["augmentation"], severity, main_rng, lengths
            )
            labels_twice = torch.cat([identity, identity]).to(device)
            ids_twice = sample_ids.to(device).repeat(2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(
                    torch.cat([xa, xb]), torch.cat([va, vb]), torch.cat([frame_mask, frame_mask])
                )
                if mode == "distill":
                    objective = distillation_loss(
                        output["parts"], output["embedding"], output["reliability"], minimum
                    )
                    variance_penalty = output["parts"].new_zeros(())
                else:
                    objective = part_batch_hard_triplet(
                        output["parts"], output["reliability"], labels_twice,
                        float(train.get("triplet_margin", .2)), ids_twice, minimum,
                    )
                    variances = part_variances(output["parts"], output["reliability"], minimum)
                    variance_penalty = variance_floor_loss(variances, variance_floor)
                loss = objective + (variance_weight * variance_penalty if mode == "triplet_variance" else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite head loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.part_heads.parameters(), float(train.get("gradient_clip", 5.0)),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            objective_losses.append(float(objective.detach()))
            variance_losses.append(float(variance_penalty.detach()))
            with torch.no_grad():
                variance_log.append(
                    part_variances(output["parts"], output["reliability"], minimum).cpu().numpy()
                )
                reliability_log.append(output["reliability"].mean(0).cpu().numpy())
        row = {
            "epoch": epoch + 1, "learning_rate": current_lr, "train_severity": severity,
            "loss": float(np.mean(losses)), "objective_loss": float(np.mean(objective_losses)),
            "variance_floor_loss": float(np.mean(variance_losses)),
            "part_embedding_variance": np.mean(variance_log, axis=0).tolist(),
            "part_reliability": np.mean(reliability_log, axis=0).tolist(),
        }
        interval = int(cfg.get("validation", {}).get("every_epochs", 5))
        if (epoch + 1) % max(1, interval) == 0 or epoch + 1 == epochs:
            metrics = validation_score(model, data, cfg, device)
            row["validation"] = metrics
            if metrics["score"] > best_score:
                best_score, best_epoch = float(metrics["score"]), epoch + 1
                save_checkpoint(
                    output_dir / "checkpoint-best.pt", model, checkpoint_config,
                    best_epoch, validation=metrics,
                )
        history.append(row)
        atomic_json(output_dir / "history.json", history)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            save_checkpoint(output_dir / "checkpoint-last.pt", model, checkpoint_config, epoch + 1)
        print(
            f"head epoch {epoch + 1}/{epochs}: mode={mode} loss={row['loss']:.4f} "
            f"variance={np.mean(row['part_embedding_variance']):.6g} lr={current_lr:.3g}"
            + (f" val={row['validation']['score']:.4f}" if "validation" in row else ""),
            flush=True,
        )
        scheduler.step()
    selected = torch.load(output_dir / "checkpoint-best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"])
    model.eval()
    return history, (time.perf_counter() - started) / 3600, int(best_epoch), trainable


def run_head_training(
    project_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    mode: str,
    device_name: str = "auto",
    epochs: int = 40,
    learning_rate: float = 1e-3,
    variance_floor: float = 1e-4,
    variance_weight: float = .1,
    realizations: int | None = None,
):
    checkpoint_path = checkpoint_path.resolve()
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = copy.deepcopy(source["config"])
    if cfg.get("method_name") != "metric_baseline":
        raise ValueError("P1 must start from a metric_baseline checkpoint")
    if cfg.get("evaluation", {}).get("split") != "validation":
        raise ValueError("P1 must use the development validation split")
    if mode not in HEAD_MODES:
        raise ValueError(f"unknown head mode {mode}; choose from {', '.join(HEAD_MODES)}")
    seed = int(cfg["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(bool(cfg["train"].get("deterministic", False)))
    if cfg["train"].get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    data = PoseData.load(resolve_dataset_path(project_root, cfg["dataset"]))
    data, held_out = development_split(data, cfg)
    expected = sorted(cfg.get("validation", {}).get("held_out_identities", []))
    if expected and held_out != expected:
        raise ValueError("checkpoint validation identities do not match the reconstructed split")
    train_ids = cfg.get("train_identities") or np.unique(data.labels[data.split == "train"]).tolist()
    device = select_device(device_name)
    model = encoder_from_config(cfg, data.x.shape[-1], data.x.shape[2], len(train_ids)).to(device)
    model.load_state_dict(source["model"])

    cfg["method_name"] = f"head_{mode}"
    cfg["method"] = {**cfg["method"], "head_mode": mode}
    cfg["validation"] = {**cfg["validation"], "descriptor": "reliable_parts"}
    cfg["evaluation"] = {**cfg["evaluation"], "descriptors": ["global", "reliable_parts"]}
    control_realizations = int(
        realizations if realizations is not None else cfg["evaluation"]["corruption_realizations"]
    )
    cfg["evaluation"]["corruption_realizations"] = control_realizations
    source_run = checkpoint_path.parent.name
    run_id = f"{source_run}_head_{mode}"
    output_root = output_root if output_root.is_absolute() else project_root / output_root
    run_dir = output_root / run_id
    if run_dir.exists():
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text())
            if existing.get("status") == "complete" and existing.get("source_checkpoint") == str(checkpoint_path):
                return run_dir, existing
        raise FileExistsError(f"incomplete P1 output already exists at {run_dir}; move it aside before retrying")
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_cfg = {
        **cfg,
        "experiment_id": "frozen_head_p1", "run_id": run_id,
        "seed": seed, "device": str(device), "dataset_path": str(resolve_dataset_path(project_root, cfg["dataset"])),
        "source_run": source_run, "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_epoch": int(source.get("epoch", -1)),
        "head_training": {
            "mode": mode, "epochs": int(epochs), "learning_rate": float(learning_rate),
            "variance_floor": float(variance_floor), "variance_weight": float(variance_weight),
            "backbone_frozen": True, "batch_norm_frozen": True,
        },
    }
    with (run_dir / "config.yaml").open("x") as handle:
        yaml.safe_dump(frozen_cfg, handle, sort_keys=False)
    try:
        history, train_hours, selected_epoch, trainable = train_frozen_heads(
            model, data, cfg, device, seed, mode, int(epochs), run_dir, frozen_cfg,
            learning_rate=learning_rate, variance_floor=variance_floor,
            variance_weight=variance_weight,
        )
        save_checkpoint(run_dir / "checkpoint.pt", model, frozen_cfg, selected_epoch)
        rows, correctness, probe_indices = _evaluate(
            model, data, cfg, device, seed, control_realizations
        )
    except BaseException as error:
        failure = {
            "status": "failed", "type": type(error).__name__, "message": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    np.savez_compressed(
        run_dir / "probe_outcomes.npz", probe_indices=probe_indices,
        probe_labels=data.labels[probe_indices], **correctness,
    )
    metrics = {
        "run_id": run_id, "status": "complete", "experiment_id": "frozen_head_p1",
        "source_run": source_run, "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": frozen_cfg["source_checkpoint_sha256"],
        "source_checkpoint_epoch": frozen_cfg["source_checkpoint_epoch"],
        "git_commit": git_commit(project_root), "source_sha256": source_fingerprint(project_root),
        "dataset": cfg["dataset"]["name"], "dataset_sha256": cfg.get("dataset_sha256", data.checksum()),
        "method": cfg["method_name"], "backbone": cfg["model"].get("backbone", "tcn"),
        "evaluation_split": cfg["evaluation"]["split"], "seed": seed,
        "epochs": int(epochs), "selected_epoch": selected_epoch,
        "trainable_parameters": trainable,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_hours": train_hours,
        "final_part_variance": float(np.mean(history[selected_epoch - 1]["part_embedding_variance"])),
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
