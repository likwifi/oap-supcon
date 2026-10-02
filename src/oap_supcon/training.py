"""Training engine: separate augmentation RNGs, validation and durable artifacts."""
from __future__ import annotations

import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import frame_mask_from_lengths, temporal_training_views, training_views
from .data import IdentityBatchSampler, PoseDataset
from .losses import (batch_hard_triplet, part_batch_hard_triplet, part_contrastive,
                     supervised_contrastive, temporal_contrastive)


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, model, config, epoch, **metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, **metadata}, temporary)
    os.replace(temporary, path)


def _seed_worker(_):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(torch.initial_seed() % (2**32))


def dataset_options(cfg, training):
    return {"weight_coordinates": bool(cfg.get("pose", {}).get("weight_coordinates", False)),
            "clip_length": cfg["train" if training else "evaluation"].get("clip_length"),
            "clip_mode": "random" if training else "center"}


def loss_gradient_cosines(terms, method, shared_features):
    """Measure objective agreement at the shared encoder representation.

    The diagnostic uses ``autograd.grad`` and therefore does not alter the
    optimizer gradients. Zero-gradient objectives are omitted rather than
    serialized as NaN.
    """
    gradients = {}
    for name, value in terms.items():
        weight = float(method.get(f"{name}_weight", 0))
        if not weight:
            continue
        gradient = torch.autograd.grad(
            weight * value, shared_features, retain_graph=True, allow_unused=True
        )[0]
        if gradient is None:
            continue
        flattened = gradient.detach().float().reshape(-1)
        norm = flattened.norm()
        if torch.isfinite(norm) and norm > 0:
            gradients[name] = flattened / norm
    return {
        f"{first}__{second}": float(torch.dot(gradients[first], gradients[second]))
        for first, second in combinations(sorted(gradients), 2)
    }


def train_model(model, data, cfg, device, seed, epochs, *, output_dir=None,
                checkpoint_config=None, validate=None):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    indices = data.indices(["train"])
    if not len(indices):
        raise ValueError("dataset has no train split")
    labels = data.labels[indices]
    label_map = {v: i for i, v in enumerate(sorted(np.unique(labels).tolist()))}
    train = cfg["train"]
    method = cfg["method"]
    dataset = PoseDataset(data, indices, seed=seed, **dataset_options(cfg, True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    schedule = train.get("scheduler", "constant")
    if schedule not in {"constant", "cosine"}:
        raise ValueError(f"unknown scheduler: {schedule}")
    warmup = min(int(train.get("warmup_epochs", 0)), max(0, epochs - 1))

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        floor = float(train.get("min_lr_ratio", 0.01))
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor) if schedule == "cosine" else None
    use_amp = bool(train.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaled = use_amp and amp_dtype == torch.float16
    scaler = (torch.amp.GradScaler("cuda", enabled=scaled)
              if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=scaled))
    history, best_score, best_epoch = [], -math.inf, None
    started = time.perf_counter()
    for epoch in range(epochs):
        sampler = IdentityBatchSampler(labels, int(train["identities_per_batch"]),
                                       int(train["samples_per_identity"]), seed + epoch,
                                       views=data.views[indices] if train.get("diverse_views", False) else None)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(train["num_workers"]),
                            worker_init_fn=_seed_worker,
                            generator=torch.Generator().manual_seed(seed * 1000 + epoch),
                            pin_memory=device.type == "cuda")
        model.train()
        main_rng = torch.Generator(device=device.type).manual_seed(seed * 10000 + epoch)
        temporal_rng = torch.Generator(device=device.type).manual_seed(seed * 10000 + epoch + 1_000_000)
        maximum = float(train["train_severity"])
        severity = maximum
        if train.get("severity_schedule") == "linear":
            start = float(train.get("train_severity_start", .1))
            ramp = max(1, round(epochs * float(train.get("severity_ramp_fraction", 1.0))) - 1)
            severity = start + min(1, epoch / ramp) * (maximum - start)
        totals, accuracies, visibility_log, gradient_norms = [], [], [], []
        part_reliability_log, part_presence_log, part_variance_log = [], [], []
        gradient_cosine_log = []
        skipped_steps = 0
        components = {name: [] for name in (
            "ce", "clean_ce", "global", "part", "temporal", "triplet", "part_triplet"
        )}
        current_lr = optimizer.param_groups[0]["lr"]
        for batch_index, (x, confidence, lengths, identity, sample_ids) in enumerate(loader):
            x, confidence, lengths = x.to(device), confidence.to(device), lengths.to(device)
            frame_mask = frame_mask_from_lengths(confidence, lengths)
            target = torch.tensor([label_map[int(v)] for v in identity], device=device)
            (xa, va), (xb, vb) = training_views(x, confidence, method["augmentation"], severity, main_rng, lengths)
            repeat_labels = torch.cat([target, target])
            clean_weight = float(method.get("clean_ce_weight", 0))
            xs, vs, fs = [xa, xb], [va, vb], [frame_mask, frame_mask]
            if clean_weight:
                xs.append(x); vs.append(confidence); fs.append(frame_mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(torch.cat(xs), torch.cat(vs), torch.cat(fs))
                count = len(x)
                paired = {key: value[:2 * count] for key, value in out.items() if key != "joint_features"}
                terms = {}
                if method.get("ce_weight", 0):
                    terms["ce"] = F.cross_entropy(paired["logits"], repeat_labels,
                                                  label_smoothing=float(train.get("label_smoothing", 0)))
                if clean_weight:
                    terms["clean_ce"] = F.cross_entropy(out["logits"][2 * count:], target,
                                                        label_smoothing=float(train.get("label_smoothing", 0)))
                if method.get("global_weight", 0):
                    terms["global"] = supervised_contrastive(paired["projection"], repeat_labels, float(train["temperature"]))
                if method.get("part_weight", 0):
                    terms["part"] = part_contrastive(
                        paired["parts"], paired["reliability"], repeat_labels,
                        float(train.get("part_temperature", train["temperature"])),
                        bool(method.get("visibility_gating", True)), float(train.get("min_part_reliability", 0)))
                ids = sample_ids.to(device).repeat(2)
                if method.get("triplet_weight", 0):
                    terms["triplet"] = batch_hard_triplet(paired["embedding"], repeat_labels,
                                                         float(train.get("triplet_margin", .2)), ids)
                if method.get("part_triplet_weight", 0):
                    terms["part_triplet"] = part_batch_hard_triplet(
                        paired["parts"], paired["reliability"], repeat_labels,
                        float(train.get("triplet_margin", .2)), ids,
                        float(train.get("min_part_reliability", 0)),
                    )
                if method.get("temporal_weight", 0):
                    first, second = temporal_training_views(
                        x, confidence, method["augmentation"], severity, temporal_rng, lengths,
                        float(train.get("temporal_crop_min", .5)), float(train.get("temporal_crop_max", .8)))
                    # Temporal auxiliary forwards must not overwrite the running
                    # BN statistics of the main-view distribution.
                    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
                    states = [m.training for m in bns]
                    for m in bns:
                        m.eval()
                    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                    try:
                        with torch.random.fork_rng(devices=cuda_devices):
                            torch.manual_seed(seed * 1_000_000 + epoch * len(loader) + batch_index)
                            projected = model.project(torch.cat([first[0], second[0]]), torch.cat([first[1], second[1]]))
                            terms["temporal"] = temporal_contrastive(projected[:count], projected[count:],
                                float(train.get("temporal_temperature", train["temperature"])), target)
                    finally:
                        for m, state in zip(bns, states):
                            m.train(state)
                if not terms:
                    raise ValueError("at least one loss weight must be nonzero")
                loss = sum(float(method[f"{name}_weight"]) * value for name, value in terms.items())
            diagnostic_interval = int(train.get("gradient_diagnostics_every_epochs", 0))
            if (diagnostic_interval > 0 and batch_index == 0
                    and (epoch + 1) % diagnostic_interval == 0):
                gradient_cosine_log.append(loss_gradient_cosines(
                    terms, method, out["joint_features"]
                ))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}, batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train.get("gradient_clip", 5.0)),
                                                 error_if_nonfinite=not scaled)
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(loss.detach()))
            if torch.isfinite(grad):
                gradient_norms.append(float(grad))
            else:
                # Float16 overflow is handled by GradScaler's skipped step and
                # reduced scale, not mistaken for a fatal model error.
                skipped_steps += 1
            accuracies.append(float((paired["logits"].argmax(1) == repeat_labels).float().mean()))
            visibility_log.append(float(paired["reliability"].detach().mean()))
            with torch.no_grad():
                reliability = paired["reliability"].detach().float()
                parts = paired["parts"].detach().float()
                present = reliability > float(train.get("min_part_reliability", 0))
                part_reliability_log.append(reliability.mean(0).cpu().numpy())
                part_presence_log.append(present.float().mean(0).cpu().numpy())
                variance = []
                for part in range(parts.shape[1]):
                    selected = parts[present[:, part], part]
                    variance.append(float(selected.var(0, unbiased=False).mean()) if len(selected) > 1 else 0.0)
                part_variance_log.append(variance)
            for name, value in terms.items():
                components[name].append(float(value.detach()))
        if not gradient_norms:
            raise FloatingPointError("every optimizer step in this epoch had non-finite gradients")
        row = {"epoch": epoch + 1, "train_severity": severity, "learning_rate": current_lr,
               "skipped_optimizer_steps": skipped_steps,
               "loss": float(np.mean(totals)), "train_accuracy": float(np.mean(accuracies)),
               "mean_part_reliability": float(np.mean(visibility_log)), "gradient_norm": float(np.mean(gradient_norms)),
               "part_reliability": np.mean(part_reliability_log, axis=0).tolist(),
               "part_presence_rate": np.mean(part_presence_log, axis=0).tolist(),
               "part_embedding_variance": np.mean(part_variance_log, axis=0).tolist(),
               **{f"{name}_loss": float(np.mean(values)) if values else None for name, values in components.items()}}
        if gradient_cosine_log:
            keys = sorted({key for values in gradient_cosine_log for key in values})
            row["loss_gradient_cosine"] = {
                key: float(np.mean([values[key] for values in gradient_cosine_log if key in values]))
                for key in keys
            }
        interval = int(cfg.get("validation", {}).get("every_epochs", 5))
        if validate is not None and ((epoch + 1) % max(1, interval) == 0 or epoch + 1 == epochs):
            metrics = validate(model)
            row["validation"] = metrics
            score = float(metrics["score"])
            if not math.isfinite(score):
                raise FloatingPointError("non-finite validation score")
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                if output_dir is not None:
                    save_checkpoint(output_dir / "checkpoint-best.pt", model, checkpoint_config,
                                    best_epoch, validation=metrics)
        history.append(row)
        if output_dir is not None:
            atomic_json(output_dir / "history.json", history)
            interval = max(1, int(train.get("checkpoint_every", 5)))
            if (epoch + 1) % interval == 0 or epoch + 1 == epochs:
                save_checkpoint(output_dir / "checkpoint-last.pt", model, checkpoint_config, epoch + 1)
        print(f"epoch {epoch + 1}/{epochs}: loss={row['loss']:.4f} accuracy={row['train_accuracy']:.3f} lr={current_lr:.3g}"
              + (f" val={row['validation']['score']:.4f}" if "validation" in row else ""), flush=True)
        if scheduler is not None:
            scheduler.step()
    if validate is not None and output_dir is not None:
        selected = torch.load(output_dir / "checkpoint-best.pt", map_location=device, weights_only=True)
        model.load_state_dict(selected["model"])
    return history, (time.perf_counter() - started) / 3600
