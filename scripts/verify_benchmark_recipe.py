#!/usr/bin/env python3
"""Real training-data integration check, NOT a benchmark or test-set evaluation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oap_supcon.augment import training_views, frame_mask_from_lengths
from oap_supcon.data import PoseData, PoseDataset, development_split, resolve_dataset_path
from oap_supcon.experiment import load_configuration, seed_everything
from oap_supcon.losses import batch_hard_triplet, supervised_contrastive, part_contrastive
from oap_supcon.model import encoder_from_config
from oap_supcon.provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="casia_b_pose")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("audits/benchmark_v2/integration.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_everything(11)
    root = Path(__file__).resolve().parents[1]
    cfg = load_configuration(root, args.dataset, "oap_v2", "benchmark_v2")
    data = PoseData.load(resolve_dataset_path(root, cfg["dataset"]))
    data, held = development_split(data, cfg)
    train_ids = np.unique(data.labels[data.split == "train"])
    selected_ids = np.random.default_rng(11).choice(train_ids, min(4, len(train_ids)), replace=False)
    indices = np.concatenate([np.flatnonzero((data.labels == identity) & (data.split == "train"))[:2]
                              for identity in selected_ids])
    dataset = PoseDataset(data, indices, weight_coordinates=False, clip_length=60, clip_mode="random", seed=11)
    samples = [dataset[i] for i in range(len(dataset))]
    device = torch.device(args.device)
    x = torch.stack([s[0] for s in samples]).to(device)
    confidence = torch.stack([s[1] for s in samples]).to(device)
    lengths = torch.tensor([s[2] for s in samples], device=device)
    mapping = {int(identity): i for i, identity in enumerate(train_ids)}
    labels = torch.tensor([mapping[s[3]] for s in samples], device=device)
    model = encoder_from_config(cfg, data.x.shape[-1], data.x.shape[2], len(train_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])
    gen = torch.Generator(device=device.type).manual_seed(11)
    losses, seconds = [], []
    for _ in range(args.steps):
        start = time.perf_counter()
        a, b = training_views(x, confidence, "mixed_partial", .3, gen, lengths)
        mask = frame_mask_from_lengths(confidence, lengths)
        out = model(torch.cat([a[0], b[0], x]), torch.cat([a[1], b[1], confidence]), torch.cat([mask] * 3))
        n = len(x); target = labels.repeat(2)
        loss = F.cross_entropy(out["logits"][:2*n], target, label_smoothing=.1)
        loss = loss + .5 * F.cross_entropy(out["logits"][2*n:], labels, label_smoothing=.1)
        loss = loss + batch_hard_triplet(out["embedding"][:2*n], target, sample_ids=torch.arange(n, device=device).repeat(2))
        loss = loss + .2 * supervised_contrastive(out["projection"][:2*n], target, .07)
        loss = loss + .2 * part_contrastive(out["parts"][:2*n], out["reliability"][:2*n], target, .1, min_reliability=.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite objective")
        optimizer.step()
        losses.append(float(loss.detach()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        print(f"integration step {len(losses)}: loss={losses[-1]:.4f}, {seconds[-1]:.2f}s", flush=True)
    result = {"purpose": "software integration only; not a benchmark result", "device": str(device),
              "source_sha256": source_fingerprint(root), "backbone": cfg["model"]["backbone"],
              "parameters": sum(p.numel() for p in model.parameters()), "training_identities": len(train_ids),
              "validation_identities": held, "sampled_training_ids": selected_ids.tolist(),
              "sampled_sequences": len(indices), "frames": 60, "losses": losses, "step_seconds": seconds,
              "finite_gradients": True, "test_sequences_used": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
