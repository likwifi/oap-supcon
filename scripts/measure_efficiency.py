#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oap_supcon.data import PoseData, PoseDataset, resolve_dataset_path
from oap_supcon.experiment import _dataset_options, select_device
from oap_supcon.model import encoder_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    data_path = resolve_dataset_path(Path(__file__).resolve().parents[1], cfg["dataset"])
    data = PoseData.load(data_path)
    train_ids = cfg.get("train_identities", np.unique(data.labels[data.split == "train"]).tolist())
    model_cfg = cfg["model"]
    # The checkpoint records its own backbone; hardcoding PoseEncoder here made
    # every stgcn/transformer checkpoint fail to load.
    model = encoder_from_config(cfg, data.x.shape[-1], data.x.shape[2], len(train_ids))
    model.load_state_dict(checkpoint["model"])
    device = select_device(args.device)
    model.to(device).eval()
    # Time the representation the run was actually evaluated on, not the padded
    # buffer: on CASIA-B those differ by a factor of three.
    x, visibility, length, _, _ = PoseDataset(data, np.array([0]), **_dataset_options(cfg, training=False))[0]
    x, visibility = x[:length], visibility[:length]
    descriptors = list(cfg.get("evaluation", {}).get("descriptors", ["global"]))
    unused = ("projector", "classifier") if any(d != "global" for d in descriptors) else ("projector", "part_projector", "part_heads", "classifier")
    x, visibility = x.unsqueeze(0).to(device), visibility.unsqueeze(0).to(device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x, visibility)
        synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model(x, visibility)
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    result = {
        "device": str(device), "batch_size": 1, "warmup": args.warmup, "iterations": args.iterations,
        "latency_ms_mean": float(np.mean(timings)), "latency_ms_std": float(np.std(timings)),
        "throughput_sequences_per_second": float(1000 / np.mean(timings)),
        "backbone": str(model_cfg.get("backbone", "tcn")),
        "sequence_frames": int(x.shape[1]),
        "timed_operation": "encoder forward (all heads), excluding gallery matching",
        "descriptors": descriptors,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        # The part projector is only dead weight at inference when the run does
        # not match on part descriptors.
        "inference_parameters": sum(
            p.numel() for name, p in model.named_parameters()
            if not name.startswith(unused)
        ),
    }
    output = args.checkpoint.parent / "efficiency.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
