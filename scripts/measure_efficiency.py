#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oap_supcon.data import PoseData, PoseDataset
from oap_supcon.experiment import select_device
from oap_supcon.model import PoseEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    data = PoseData.load(cfg["dataset_path"])
    train_ids = np.unique(data.labels[data.split == "train"])
    model_cfg = cfg["model"]
    model = PoseEncoder(data.x.shape[-1], data.x.shape[2], len(train_ids), int(model_cfg["hidden_dim"]), int(model_cfg["embedding_dim"]), float(model_cfg["dropout"]))
    model.load_state_dict(checkpoint["model"])
    device = select_device(args.device)
    model.to(device).eval()
    x, visibility, _, _ = PoseDataset(data, np.array([0]))[0]
    x, visibility = x.unsqueeze(0).to(device), visibility.unsqueeze(0).to(device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.no_grad():
        for _ in range(args.warmup):
            model.embed(x, visibility)
        synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model.embed(x, visibility)
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    result = {
        "device": str(device), "batch_size": 1, "warmup": args.warmup, "iterations": args.iterations,
        "latency_ms_mean": float(np.mean(timings)), "latency_ms_std": float(np.std(timings)),
        "throughput_sequences_per_second": float(1000 / np.mean(timings)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_parameters": sum(
            p.numel() for name, p in model.named_parameters()
            if not name.startswith(("projector", "part_projector", "classifier"))
        ),
    }
    output = args.checkpoint.parent / "efficiency.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
