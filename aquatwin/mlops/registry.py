"""Minimal file-based model registry (MLflow-compatible concepts, zero infrastructure).

``artifacts/models/registry.json`` records, per model: version, created time,
metrics, feature list, and the SHA-256 of the dataset manifest it was trained
on -- enough for lineage, rollback and champion/challenger comparisons.  An
optional MLflow hook logs the same payload when ``MLFLOW_TRACKING_URI`` is set.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def file_hash(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"models": {}}

    def register(self, name: str, metrics: dict, data_manifest: str | Path, extra: dict | None = None) -> int:
        versions = self.data["models"].setdefault(name, [])
        version = len(versions) + 1
        entry = dict(version=version, created=time.strftime("%Y-%m-%dT%H:%M:%S"), metrics=metrics,
                     data_hash=file_hash(data_manifest), stage="production", **(extra or {}))
        for v in versions:
            v["stage"] = "archived"
        versions.append(entry)
        self.path.write_text(json.dumps(self.data, indent=2, default=float))
        self._mlflow(name, entry)
        return version

    def latest(self, name: str) -> dict | None:
        v = self.data["models"].get(name)
        return v[-1] if v else None

    @staticmethod
    def _mlflow(name: str, entry: dict) -> None:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:  # pragma: no cover - optional dependency
            import mlflow
            with mlflow.start_run(run_name=f"{name}-v{entry['version']}"):
                flat = {k: v for k, v in _flatten(entry["metrics"]).items() if isinstance(v, (int, float))}
                mlflow.log_metrics(flat)
                mlflow.log_param("data_hash", entry["data_hash"])
        except Exception:
            pass


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
