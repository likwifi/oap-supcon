"""Prepare official FastPoseGait baselines on this project's exact data splits.

Raw coordinates are exported before OAP normalization or corruption. Official
model, optimizer, loss and transforms stay in the external framework. Preparing
files is separate from launching its CUDA/distributed training process.

Stages:
- ``validation``: the fixed development holdout; official test sequences are
  never exported.
- ``final``: the official training identities and the official test set. On
  CASIA-B this reproduces published scores; on Gait3D it scores the available
  pose subset, which published full-benchmark numbers do not cover.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil

import numpy as np
import yaml

from .data import PoseData, development_split, resolve_dataset_path
from .experiment import load_configuration

# Official template per (dataset, model); only these recipes are published.
TEMPLATES = {
    ("casia_b_pose", "gpgait"): "gpgait/gpgait.yaml",
    ("casia_b_pose", "gpgait++"): "gpgait++/gpgait++.yaml",
    ("casia_b_pose", "gaitgraph2"): "gaitgraph2/gaitgraph2.yaml",
    ("gait3d", "gpgait"): "gpgait/gpgait_Gait3D.yaml",
    ("gait3d", "gaitgraph2"): "gaitgraph2/gaitgraph2_Gait3D.yaml",
}
DEFAULT_MODELS = {"casia_b_pose": ("gpgait", "gpgait++"), "gait3d": ("gpgait", "gaitgraph2")}
# These models end in an identity classifier sized by the training identities.
# GaitGraph2's num_class is its contrastive embedding width and is left as published.
CLASSIFIER_MODELS = {"gpgait", "gpgait++"}
SPLIT_RECIPES = {"casia_b_pose": "benchmark_v3_control", "gait3d": "gait3d_v2_available"}
FRAMEWORK_NAMES = {"casia_b_pose": "CASIA-B", "gait3d": "Gait3D"}


def reference_entries(data, indices):
    if data.x.shape[2:] != (17, 2):
        raise ValueError("the official CASIA-B templates require 2D COCO-17 poses")
    pattern = re.compile(r"(\d{3})-(nm-0[1-6]|bg-0[1-2]|cl-0[1-2])-(\d{3})")
    valid_views = {f"{angle:03d}" for angle in range(0, 181, 18)}
    entries, seen = [], set()
    for index in indices:
        match = pattern.fullmatch(str(data.sequence_ids[index]))
        if match is None:
            raise ValueError(f"invalid CASIA-B sequence ID: {data.sequence_ids[index]}")
        identity, sequence, view = match.groups()
        key = (identity, sequence, view)
        if (int(identity) != int(data.labels[index]) or view != str(data.views[index])
                or sequence[:2].upper() != str(data.conditions[index]) or view not in valid_views):
            raise ValueError(f"sequence metadata disagree for {data.sequence_ids[index]}")
        if key in seen:
            raise ValueError(f"duplicate sequence: {data.sequence_ids[index]}")
        seen.add(key)
        length = int(data.frame_counts[index])
        # The official RescaleCenter divides by per-frame height. Catch unusable
        # clean input rather than silently introducing NaNs in the reference.
        y = data.x[index, :length, :, 1]
        if np.any(np.ptp(y, axis=1) <= 1e-6):
            raise ValueError(f"zero-height frame in {data.sequence_ids[index]}")
        # Directory levels follow the official CASIA-B pickle layout.
        entries.append((int(index), identity, sequence, view, view, length))
    return entries


def gait3d_entries(data, indices):
    """Official Gait3D layout: <identity>/<camid_videoid>/<seq>/<seq>.pkl."""
    if data.x.shape[2:] != (17, 2):
        raise ValueError("the official Gait3D pose templates require 2D COCO-17 poses")
    pattern = re.compile(r"(\d{4})-(camid\d+_videoid\d+)-(seq\d+)")
    entries, seen = [], set()
    for index in indices:
        sequence_id = str(data.sequence_ids[index])
        match = pattern.fullmatch(sequence_id)
        if match is None:
            raise ValueError(f"invalid Gait3D sequence ID: {sequence_id}")
        identity, camera_video, sequence = match.groups()
        if int(identity) != int(data.labels[index]) or not camera_video.startswith(f"{data.views[index]}_"):
            raise ValueError(f"sequence metadata disagree for {sequence_id}")
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence: {sequence_id}")
        seen.add(sequence_id)
        y = data.x[index, :int(data.frame_counts[index]), :, 1]
        if np.any(np.ptp(y, axis=1) <= 1e-6):
            raise ValueError(f"zero-height frame in {sequence_id}")
        entries.append((int(index), identity, camera_video, sequence, sequence, int(data.frame_counts[index])))
    return entries


def reference_configuration(template, destination, train_ids, gpu_count, name, *, classifier=True):
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    cfg = copy.deepcopy(template)
    if cfg["evaluator_cfg"]["restore_hint"] != cfg["trainer_cfg"]["total_iter"]:
        raise ValueError("the template must evaluate its final, predetermined iteration")
    cfg["data_cfg"].update(dataset_root=str(destination / "poses"),
                           dataset_partition=str(destination / "partition.json"))
    if classifier:
        cfg["model_cfg"]["num_class"] = len(train_ids)
    cfg["trainer_cfg"].update(with_test=False, save_name=name)
    cfg["evaluator_cfg"]["save_name"] = name
    sampler = cfg["evaluator_cfg"]["sampler"]
    # Variable-length ("all_ordered") inference places one sequence per GPU.
    if sampler.get("sample_type", "all_ordered") == "all_ordered":
        sampler["batch_size"] = gpu_count
    return cfg


def _stage_indices(data, dataset, stage):
    """Exported indices, training identities, evaluated identities and probe IDs."""
    if stage == "validation":
        indices = data.indices(["train", "val_gallery", "val_probe"])
        evaluated = np.unique(data.labels[data.indices(["val_gallery", "val_probe"])])
        probes = data.indices(["val_probe"])
    else:
        indices = data.indices(["train", "gallery", "probe"])
        evaluated = np.unique(data.labels[data.indices(["gallery", "probe"])])
        probes = data.indices(["probe"])
    train_ids = np.unique(data.labels[data.split == "train"])
    if dataset == "casia_b_pose" and stage == "final":
        if set(train_ids) != set(range(1, 75)) or set(evaluated) != set(range(75, 125)):
            raise ValueError("the CASIA-B reproduction needs the official 74/50 identity split")
    return indices, train_ids.tolist(), evaluated.tolist(), probes


def prepare_fastposegait(project_root, destination, models=None, gpu_count=1, dry_run=False,
                         dataset="casia_b_pose", stage="validation"):
    project_root, destination = Path(project_root).resolve(), Path(destination).resolve()
    if dataset not in DEFAULT_MODELS or stage not in {"validation", "final"}:
        raise ValueError("dataset must be casia_b_pose or gait3d, and stage validation or final")
    models = tuple(models or DEFAULT_MODELS[dataset])
    available = {model for (name, model) in TEMPLATES if name == dataset}
    if not models or len(set(models)) != len(models) or set(models) - available:
        raise ValueError(f"select distinct {dataset} models from {sorted(available)}")
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if destination.exists():
        raise FileExistsError(f"choose a new preparation directory: {destination}")
    cfg = load_configuration(project_root, dataset, "metric_baseline", SPLIT_RECIPES[dataset])
    source = resolve_dataset_path(project_root, cfg["dataset"])
    data = PoseData.load(source)
    source_checksum = data.checksum()
    if dataset == "casia_b_pose" and (
            set(data.labels[data.split == "train"]) != set(range(1, 75))
            or set(data.labels[np.isin(data.split, ["gallery", "probe"])]) != set(range(75, 125))):
        raise ValueError("prepare the canonical CASIA-B 74/50 identity split first")
    held = []
    if stage == "validation":
        data, held = development_split(data, cfg)
    indices, train_ids, evaluated, probes = _stage_indices(data, dataset, stage)
    entries = (reference_entries if dataset == "casia_b_pose" else gait3d_entries)(data, indices)
    official_test = int(np.isin(data.split[indices], ["gallery", "probe"]).sum())
    width = 3 if dataset == "casia_b_pose" else 4
    external = project_root / "external/FastPoseGait"
    configurations, template_hashes = {}, {}
    for model in models:
        content = (external / "configs" / TEMPLATES[(dataset, model)]).read_bytes()
        template_hashes[model] = hashlib.sha256(content).hexdigest()
        configurations[model] = reference_configuration(
            yaml.safe_load(content), destination, train_ids, gpu_count,
            f"oap_{dataset}_{stage}_" + model.replace("++", "pp"), classifier=model in CLASSIFIER_MODELS,
        )
    digest = hashlib.sha256()
    for path in sorted((external / "fastposegait").rglob("*.py")):
        digest.update(str(path.relative_to(external)).encode()); digest.update(path.read_bytes())
    partition = {"TRAIN_SET": [f"{identity:0{width}d}" for identity in train_ids],
                 "TEST_SET": [f"{identity:0{width}d}" for identity in evaluated]}
    if dataset == "gait3d":
        partition["PROBE_SET"] = sorted(str(data.sequence_ids[index]) for index in probes)
    differences = []
    if dataset == "gait3d":
        differences = [
            "frames are stacked in numeric frame-ID order; the official pretreatment sorts file "
            "names lexicographically, which differs only where frame IDs change digit count",
            "HRNet confidence above 1 was clipped to 1 by the project converter",
            "only the available-pose subset is exported; missing identities and probes are absent",
        ]
    manifest = {
        "purpose": ("published-score reproduction on the official CASIA-B protocol"
                    if dataset == "casia_b_pose" and stage == "final" else
                    "official framework on this project's identical data split"),
        "dataset": dataset, "framework_dataset": FRAMEWORK_NAMES[dataset], "stage": stage,
        "source_dataset": str(source), "pose_source": cfg["dataset"].get("pose_source"),
        "source_dataset_sha256": source_checksum, "development_dataset_sha256": data.checksum(),
        "coordinate_export": "raw x plus source confidence; no OAP normalization",
        "preprocessing_differences": differences,
        "training_identities": train_ids, "evaluated_identities": evaluated,
        "validation_identities": held, "official_test_sequences_exported": official_test,
        "test_sequences_exported": official_test, "probes": len(probes),
        "sequences": len(entries), "frames": sum(entry[-1] for entry in entries),
        "external_source_sha256": digest.hexdigest(), "template_sha256": template_hashes,
        "gpu_count": gpu_count, "destination": str(destination), "prepared": not dry_run,
        "configurations": {model: str(destination / f"{model}.yaml") for model in models},
        "run_from": str(destination),
        "run_note": ("Run main.py from this directory: the framework reads configs/default.yaml and, "
                     "for Gait3D, datasets/Gait3D/Gait3D.json (the probe list) relative to the working "
                     "directory. Both are written here."),
    }
    if dry_run:
        return manifest
    destination.mkdir(parents=True, exist_ok=False)
    for index, identity, level_two, level_three, filename, length in entries:
        path = destination / "poses" / identity / level_two / level_three / f"{filename}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.concatenate([data.x[index, :length], data.visibility[index, :length, :, None]], axis=-1)
        with path.open("xb") as handle:
            pickle.dump(raw, handle, protocol=4)
    # TEST_SET is the framework's required key. At the validation stage it holds
    # only the fixed validation identities, never the original test population.
    text = json.dumps(partition, indent=2) + "\n"
    (destination / "partition.json").write_text(text)
    framework = destination / "datasets" / FRAMEWORK_NAMES[dataset]
    framework.mkdir(parents=True)
    (framework / f"{FRAMEWORK_NAMES[dataset]}.json").write_text(text)
    (destination / "configs").mkdir()
    shutil.copy2(external / "configs/default.yaml", destination / "configs/default.yaml")
    for model, configuration in configurations.items():
        (destination / f"{model}.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
