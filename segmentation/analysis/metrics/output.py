"""Atomic writer for UUID-addressed metric measurement bundles."""

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import uuid

from segmentation.analysis.metrics.catalog import (
    CatalogError,
    OBSERVATION_KEY,
    resolve_catalog_root,
    sha256_file,
)
from segmentation.analysis.metrics.provenance import (
    dataset_provenance,
    file_identity,
    package_versions,
    sanitize_error,
    source_provenance,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if type(value).__module__.split(".", 1)[0] in ("numpy", "torch"):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return _json_value(item())
            except (TypeError, ValueError, RuntimeError):
                return str(value)
    return str(value)


def _scalar(value):
    if type(value).__module__.split(".", 1)[0] in ("numpy", "torch"):
        item = getattr(value, "item", None)
        try:
            value = item() if callable(item) else None
        except (TypeError, ValueError, RuntimeError):
            value = None
    return value


class MetricOutput:
    """Write one immutable measurement under the configured catalog root."""

    def __init__(
        self,
        metric,
        checkpoint,
        data_type,
        *,
        model_name,
        training_phase,
        protocol,
        group_checkpoint=None,
        **parameters,
    ):
        if "output_dir" in parameters:
            raise TypeError("output_dir was removed; measurements are allocated under MEASUREMENT_CATALOG_ROOT")
        self.root = resolve_catalog_root()
        measurements = self.root / "measurements"
        while True:
            self.measurement_id = str(uuid.uuid4())
            self.path = measurements / self.measurement_id
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                continue
        self.metric = str(metric)
        self.data_type = str(data_type)
        protocol = _json_value(protocol)
        self.observation_columns = list(protocol.get("observation_columns", ()))
        if not self.observation_columns:
            raise CatalogError("Metric protocol must declare observation_columns")
        self.observation_table_columns = list(OBSERVATION_KEY) + self.observation_columns
        if len(set(self.observation_table_columns)) != len(self.observation_table_columns):
            raise CatalogError("Observation table column names must be unique")
        self._states = None
        self._observations = []
        self._observation_keys = set()
        self._observation_counts = {}
        self._active_sample = None
        self._terminal = False
        self._input_hash_cache = {}
        self._artifacts = {}
        inputs = {"prototype_checkpoint": file_identity(checkpoint, self._input_hash_cache)}
        if group_checkpoint is not None:
            inputs["group_checkpoint"] = file_identity(group_checkpoint, self._input_hash_cache)
        self.result = {
            "measurement_id": self.measurement_id,
            "metric": self.metric,
            "status": "running",
            "created_at_utc": _now(),
            "completed_at_utc": None,
            "model": {
                "label": model_name,
                "training_stage": training_phase,
                "dataset": data_type,
                "architecture": None,
                "inputs": inputs,
            },
            "dataset": {"name": data_type, "split": "val"},
            "protocol": protocol,
            "parameters": _json_value(parameters),
            "execution": {
                **source_provenance(Path(__file__).resolve().parents[3]),
                "command": list(sys.argv),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "packages": package_versions(),
                "host": socket.gethostname(),
                "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "population": None,
            "selected_prototype_ids": [],
            "scores": {},
            "artifacts": {},
        }
        self._write_result()

    def _ensure_running(self):
        if self._terminal:
            raise CatalogError("A terminal measurement is immutable")

    def _write_result(self):
        temporary = self.path / f".result.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(self.result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path / "result.json")
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_dataset(self, dataset):
        self._ensure_running()
        checkout = Path(__file__).resolve().parents[3]
        sample_ids, dataset_fields, parts_fields = dataset_provenance(
            dataset, self.data_type, checkout, self._input_hash_cache
        )
        self.result["dataset"].update(dataset_fields)
        self.result["protocol"].setdefault("parts", {}).update(parts_fields)
        self.set_candidates(sample_ids)
        self._write_result()

    def set_candidates(self, sample_ids):
        self._ensure_running()
        if self._states is not None:
            raise CatalogError("Candidate population has already been registered")
        candidates = [str(sample_id) for sample_id in sample_ids]
        if len(set(candidates)) != len(candidates):
            raise CatalogError("Candidate sample IDs must be unique")
        self._states = dict.fromkeys(candidates, "not_attempted")
        self.result["dataset"]["candidate_count"] = len(candidates)

    def update_metadata(self, **fields):
        self._ensure_running()
        for key, value in fields.items():
            self.result[key] = _json_value(value)
        self._write_result()

    def register_input(self, role, path):
        self._ensure_running()
        self.result["model"]["inputs"][role] = file_identity(path, self._input_hash_cache)
        self._write_result()

    def update_model_metadata(self, **fields):
        self._ensure_running()
        self.result["model"].update(_json_value(fields))
        self._write_result()

    def update_protocol(self, **fields):
        self._ensure_running()
        self.result["protocol"].update(_json_value(fields))
        self._write_result()

    def record_model_runtime(self, model, *, grouping_active, excluded_evaluation_class_ids):
        epsilon = _scalar(getattr(model, "epsilon", None))
        activation = getattr(model, "prototype_activation_function", None)
        self.update_protocol(
            model={
                "epsilon": float(epsilon) if isinstance(epsilon, (int, float)) else None,
                "prototype_activation_function": activation if isinstance(activation, str) else None,
            },
            input={
                "source": "cached validation image opened by get_img_path and converted to RGB",
                "resolution": "native cached image resolution; no metric resize before feature extraction",
                "normalization": {
                    "conversion": "PIL RGB uint8 to tensor in [0, 1]",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
            prototype_selection={
                "class_identity_rule": (
                    "select nonzero prototype_class_identity entries at model class index "
                    "evaluation_class_id - 1"
                ),
                "excluded_evaluation_class_ids": list(excluded_evaluation_class_ids),
                "grouping_active": bool(grouping_active),
                "group_nonzero_rule": (
                    "torch.nonzero(group_projection[class].weight.data.sum(dim=0)); "
                    "map selected positions through nonzero prototype_class_identity[:, class]"
                ),
                "group_intersection_rule": (
                    "retain model prototype IDs selected by the grouping checkpoint"
                    if grouping_active else "no grouping-checkpoint filter"
                ),
            },
        )

    def set_selected_prototype_ids(self, prototype_ids):
        self.update_metadata(selected_prototype_ids=list(dict.fromkeys(int(value) for value in prototype_ids)))

    def _require_candidate(self, sample_id):
        if self._states is None or sample_id not in self._states:
            raise CatalogError(f"Unknown candidate sample: {sample_id}")
        if self._states[sample_id] != "not_attempted":
            raise CatalogError(f"Sample has already been attempted: {sample_id}")

    def start_sample(self, sample_id):
        self._ensure_running()
        sample_id = str(sample_id)
        self._require_candidate(sample_id)
        if self._active_sample is not None:
            raise CatalogError(f"Sample {self._active_sample} is already being processed")
        self._active_sample = sample_id

    def mark_excluded(self, sample_id):
        """Exclude a candidate before processing (for example, a missing part annotation)."""
        self._ensure_running()
        sample_id = str(sample_id)
        self._require_candidate(sample_id)
        self._states[sample_id] = "excluded"

    def add_observation(self, **observation):
        self._ensure_running()
        missing = [key for key in self.observation_table_columns if key not in observation]
        if missing:
            raise CatalogError(f"Observation is missing columns: {', '.join(missing)}")
        sample_id = str(observation["sample_id"])
        if self._states is None or sample_id not in self._states:
            raise CatalogError(f"Observation is outside the candidate population: {sample_id}")
        key = tuple(str(observation[name]) for name in OBSERVATION_KEY)
        if key in self._observation_keys:
            raise CatalogError(f"Duplicate observation key: {key}")
        if self._active_sample != sample_id:
            raise CatalogError(f"Observation sample is not currently being processed: {sample_id}")
        normalized = {"sample_id": sample_id}
        for name in ("class_id", "prototype_id", "repetition"):
            try:
                normalized[name] = str(int(observation[name]))
            except (TypeError, ValueError) as error:
                raise CatalogError(f"Observation key {name} must be an integer") from error
        for name in self.observation_columns:
            value = observation[name]
            if value is None:
                normalized[name] = ""
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise CatalogError(f"Observation {name} must be zero, one, or unavailable") from error
            if math.isnan(number):
                normalized[name] = ""
            elif number in (0.0, 1.0):
                normalized[name] = str(int(number))
            else:
                raise CatalogError(f"Observation {name} must be zero, one, or unavailable")
        self._observation_keys.add(key)
        self._observations.append(normalized)
        self._observation_counts[sample_id] = self._observation_counts.get(sample_id, 0) + 1

    def complete_sample(self, sample_id):
        self._ensure_running()
        sample_id = str(sample_id)
        if self._active_sample != sample_id:
            raise CatalogError(f"Sample is not currently being processed: {sample_id}")
        self._states[sample_id] = "processed"
        self._active_sample = None

    def _write_observations(self):
        filename = "observations.csv"
        if filename in self._artifacts:
            return self._artifacts[filename]
        path = self.path / filename
        if path.exists():
            raise CatalogError(f"Refusing to overwrite an existing measurement artifact: {path}")
        temporary = self.path / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.observation_table_columns, extrasaction="raise")
                writer.writeheader()
                writer.writerows(self._observations)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest, size = sha256_file(path)
        self._artifacts[filename] = {"path": filename, "bytes": size, "sha256": digest}
        return self._artifacts[filename]

    def _finalize_files(self):
        """Write the observations and population counts; return False before candidates exist."""
        if self._states is None:
            return False
        if self._active_sample is not None:
            self._states[self._active_sample] = "failed"
            self._active_sample = None
        self.result["artifacts"] = {"observations": self._write_observations()}
        states = list(self._states.values())
        self.result["population"] = {
            "candidates": len(states),
            "contributing": len(self._observation_counts),
            "excluded": states.count("excluded"),
            "failed": states.count("failed"),
            "not_attempted": states.count("not_attempted"),
            "observations": len(self._observations),
        }
        return True

    def finish(self, value, *, numerator=None, denominator=None, reason=None):
        self._ensure_running()
        if self._states is None:
            raise CatalogError("Cannot complete a measurement before registering its candidate population")
        if self._active_sample is not None:
            raise CatalogError("Cannot complete a measurement while a sample is being processed")
        if "not_attempted" in self._states.values():
            raise CatalogError("A complete measurement cannot contain unattempted candidates")
        finite = value is not None and math.isfinite(float(value))
        value = float(value) if finite else None
        if not finite and not reason:
            reason = "score is undefined"
        if denominator == 0:
            reason = reason or "score denominator is zero"
            value = None
        self._finalize_files()
        self.result["scores"] = {
            self.metric: {
                "value": value,
                "unit": "fraction",
                "numerator": _json_value(numerator),
                "denominator": _json_value(denominator),
                **({"reason": str(reason)} if reason else {}),
            }
        }
        self.result["status"] = "complete"
        self.result["completed_at_utc"] = _now()
        self._write_result()
        self._terminal = True

    def fail(self, error):
        self._ensure_running()
        message = sanitize_error(error)
        self._finalize_files()
        self.result["status"] = "failed"
        self.result["completed_at_utc"] = _now()
        self.result["failure"] = {"type": type(error).__name__, "message": message}
        self.result["scores"] = {
            self.metric: {
                "value": None,
                "unit": "fraction",
                "numerator": None,
                "denominator": None,
                "reason": "measurement failed before a final score was available",
            }
        }
        self._write_result()
        self._terminal = True
