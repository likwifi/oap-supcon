"""Fingerprint scientific inputs without treating repeated seeds as new recipes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def configuration_fingerprint(cfg):
    excluded = {"seed", "device", "dataset_path", "run_id", "git_commit", "results_root",
                "config_fingerprint", "source_sha256", "dataset_sha256", "parameters", "train_identities"}
    scientific = {k: v for k, v in cfg.items() if k not in excluded}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def source_fingerprint(root: Path):
    digest = hashlib.sha256()
    for path in sorted((root / "src/oap_supcon").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
