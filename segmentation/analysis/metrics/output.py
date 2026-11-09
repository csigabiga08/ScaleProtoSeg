"""Atomic writer for native, UUID-addressed metric measurement bundles."""

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
    SAMPLE_COLUMNS,
    SCHEMA_VERSION,
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
from segmentation.analysis.metrics.population import ordered_ids_sha256, population_counts


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


class MetricOutput:
    """Write one immutable native measurement under the configured catalog root."""

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
        self.observation_table_columns = list(OBSERVATION_KEY) + self.observation_columns
        if not self.observation_columns:
            raise CatalogError("Metric protocol must declare observation_columns")
        if len(set(self.observation_table_columns)) != len(self.observation_table_columns):
            raise CatalogError("Observation table column names must be unique")
        self._candidates = None
        self._sample_index = {}
        self._observations = []
        self._observation_keys = set()
        self._observation_counts = {}
        self._active_sample = None
        self._terminal = False
        self._input_hash_cache = {}
        self._artifact_cache = {}
        checkout = Path(__file__).resolve().parents[3]
        inputs = {
            "prototype_checkpoint": file_identity(checkpoint, self._input_hash_cache),
        }
        if group_checkpoint is not None:
            inputs["group_checkpoint"] = file_identity(group_checkpoint, self._input_hash_cache)
        self.result = {
            "schema_version": SCHEMA_VERSION,
            "measurement_id": self.measurement_id,
            "origin": "native",
            "metric": self.metric,
            "status": "running",
            "created_at_utc": _now(),
            "started_at_utc": _now(),
            "completed_at_utc": None,
            "model": {
                "label": model_name,
                "training_stage": training_phase,
                "dataset": data_type,
                "training_seed": None,
                "architecture": None,
                "inputs": inputs,
            },
            "dataset": {
                "name": data_type,
                "split": "val",
                "identity": "partial",
            },
            "protocol": _json_value(protocol),
            "parameters": _json_value(parameters),
            "execution": {
                **source_provenance(checkout),
                "command": list(sys.argv),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "packages": package_versions(),
                "host": socket.gethostname(),
                "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "population": None,
            "selected_prototype_ids": [],
            "contributing_prototype_ids": [],
            "scores": {},
            "observation_schema": {
                "version": 1,
                "columns": self.observation_table_columns,
                "key": list(OBSERVATION_KEY),
            },
            "artifacts": {},
            "provenance_quality": {
                "missing_fields": [
                    "model architecture",
                    "dataset manifest identity",
                    "part annotation mapping identity",
                    "part annotation provenance",
                    "model epsilon",
                    "model.prototype_activation_function",
                    "parts.mapping_identity",
                    "parts.class_dependent_part_definitions",
                    "parts.annotation_provenance",
                ],
                "warnings": ["Dataset identity is partial; no full dataset content hash was computed."],
            },
        }
        if self.result["execution"]["commit"] is None:
            self.result["provenance_quality"]["missing_fields"].append("source commit")
        for role, identity in inputs.items():
            if identity["status"] != "available":
                self.result["provenance_quality"]["missing_fields"].append(f"{role} SHA-256")
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
        sample_ids, dataset_fields, parts_fields, warnings, has_missing_parts = dataset_provenance(
            dataset, self.data_type, checkout, self._input_hash_cache
        )
        self.result["dataset"].update(dataset_fields)
        self.result["provenance_quality"]["warnings"] = warnings
        if has_missing_parts:
            self.result["provenance_quality"]["missing_fields"].append("part annotation coverage")
        self.result["protocol"].setdefault("parts", {}).update(parts_fields)
        mapping_metadata = parts_fields
        mapping_identity = mapping_metadata["mapping_identity"]
        missing_fields = self.result["provenance_quality"]["missing_fields"]
        if mapping_identity.get("status") == "available":
            if "part annotation mapping identity" in missing_fields:
                missing_fields.remove("part annotation mapping identity")
            if "parts.mapping_identity" in missing_fields:
                missing_fields.remove("parts.mapping_identity")
        elif "parts.mapping_identity" not in missing_fields:
            missing_fields.append("parts.mapping_identity")
        definitions = mapping_metadata["class_dependent_part_definitions"]
        if definitions is not None and "parts.class_dependent_part_definitions" in missing_fields:
            missing_fields.remove("parts.class_dependent_part_definitions")
        elif definitions is None and "parts.class_dependent_part_definitions" not in missing_fields:
            missing_fields.append("parts.class_dependent_part_definitions")
        if "part annotation provenance" not in missing_fields:
            missing_fields.append("part annotation provenance")
        if "parts.annotation_provenance" not in missing_fields:
            missing_fields.append("parts.annotation_provenance")
        self.set_candidates(sample_ids)
        self._write_result()

    def set_candidates(self, sample_ids):
        self._ensure_running()
        if self._candidates is not None:
            raise CatalogError("Candidate population has already been registered")
        candidates = [str(sample_id) for sample_id in sample_ids]
        if len(set(candidates)) != len(candidates):
            raise CatalogError("Candidate sample IDs must be unique")
        self._candidates = [
            {
                "sample_index": index,
                "sample_id": sample_id,
                "outcome": "not_attempted",
                "reason": "",
                "observation_count": 0,
            }
            for index, sample_id in enumerate(candidates)
        ]
        self._sample_index = {sample_id: index for index, sample_id in enumerate(candidates)}
        self.result["dataset"]["candidate_count"] = len(candidates)
        self.result["dataset"]["candidate_order_sha256"] = ordered_ids_sha256(candidates)

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
        if fields.get("architecture") is not None:
            missing = self.result["provenance_quality"]["missing_fields"]
            if "model architecture" in missing:
                missing.remove("model architecture")
        self._write_result()

    def update_protocol(self, **fields):
        self._ensure_running()
        self.result["protocol"].update(_json_value(fields))
        self._write_result()

    def record_model_runtime(self, model, *, grouping_active, excluded_evaluation_class_ids):
        epsilon = getattr(model, "epsilon", None)
        if type(epsilon).__module__.split(".", 1)[0] in ("numpy", "torch"):
            item = getattr(epsilon, "item", None)
            try:
                epsilon = item() if callable(item) else None
            except (TypeError, ValueError, RuntimeError):
                epsilon = None
        epsilon = float(epsilon) if isinstance(epsilon, (int, float)) else None
        activation = getattr(model, "prototype_activation_function", None)
        activation = activation if isinstance(activation, str) else None
        missing_fields = self.result["provenance_quality"]["missing_fields"]
        if epsilon is None and "model epsilon" not in missing_fields:
            missing_fields.append("model epsilon")
        elif epsilon is not None and "model epsilon" in missing_fields:
            missing_fields.remove("model epsilon")
        if activation is None and "model.prototype_activation_function" not in missing_fields:
            missing_fields.append("model.prototype_activation_function")
        elif activation is not None and "model.prototype_activation_function" in missing_fields:
            missing_fields.remove("model.prototype_activation_function")
        self.update_protocol(
            model={
                "epsilon": epsilon,
                "prototype_activation_function": activation,
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

    def start_sample(self, sample_id):
        self._ensure_running()
        sample_id = str(sample_id)
        if self._candidates is None or sample_id not in self._sample_index:
            raise CatalogError(f"Unknown candidate sample: {sample_id}")
        if self._active_sample is not None:
            raise CatalogError(f"Sample {self._active_sample} is already being processed")
        row = self._candidates[self._sample_index[sample_id]]
        if row["outcome"] != "not_attempted":
            raise CatalogError(f"Sample has already been attempted: {sample_id}")
        self._active_sample = sample_id

    def mark_sample(self, sample_id, outcome, reason=None):
        self._ensure_running()
        sample_id = str(sample_id)
        if outcome != "excluded":
            raise CatalogError("Only candidates excluded before processing may be marked directly")
        if self._candidates is None or sample_id not in self._sample_index:
            raise CatalogError(f"Unknown candidate sample: {sample_id}")
        row = self._candidates[self._sample_index[sample_id]]
        if row["outcome"] != "not_attempted":
            raise CatalogError(f"Sample has already been attempted: {sample_id}")
        row.update(outcome=outcome, reason=str(reason or ""), observation_count=0)

    def add_observation(self, **observation):
        self._ensure_running()
        missing = [key for key in self.observation_table_columns if key not in observation]
        if missing:
            raise CatalogError(f"Observation is missing columns: {', '.join(missing)}")
        sample_id = str(observation["sample_id"])
        if self._candidates is None or sample_id not in self._sample_index:
            raise CatalogError(f"Observation is outside the candidate population: {sample_id}")
        key = tuple(str(observation[name]) for name in OBSERVATION_KEY)
        if key in self._observation_keys:
            raise CatalogError(f"Duplicate observation key: {key}")
        if self._active_sample != sample_id:
            raise CatalogError(f"Observation sample is not currently being processed: {sample_id}")
        normalized = {}
        normalized["sample_id"] = sample_id
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

    def complete_sample(self, sample_id, reason=None):
        self._ensure_running()
        sample_id = str(sample_id)
        if self._active_sample != sample_id:
            raise CatalogError(f"Sample is not currently being processed: {sample_id}")
        row = self._candidates[self._sample_index[sample_id]]
        count = self._observation_counts.get(sample_id, 0)
        row.update(
            outcome="contributed" if count else "processed_no_observations",
            reason=str(reason or ""),
            observation_count=count,
        )
        self._active_sample = None

    def _write_csv(self, filename, columns, rows):
        path = self.path / filename
        if filename in self._artifact_cache:
            return self._artifact_cache[filename]
        if path.exists():
            raise CatalogError(f"Refusing to overwrite an existing measurement artifact: {path}")
        temporary = self.path / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest, size = sha256_file(path)
        artifact = {
            "path": filename,
            "bytes": size,
            "sha256": digest,
            "schema_version": 1,
            "columns": list(columns),
        }
        self._artifact_cache[filename] = artifact
        return artifact

    def _finalize_files(self):
        if self._candidates is None:
            return False
        if self._active_sample is not None:
            row = self._candidates[self._sample_index[self._active_sample]]
            row.update(
                outcome="failed",
                reason="processing interrupted",
                observation_count=self._observation_counts.get(self._active_sample, 0),
            )
            self._active_sample = None
        observation_artifact = self._write_csv(
            "observations.csv", self.observation_table_columns, self._observations
        )
        sample_artifact = self._write_csv("samples.csv", SAMPLE_COLUMNS, self._candidates)
        self.result["artifacts"] = {
            "observations": observation_artifact,
            "samples": sample_artifact,
        }
        self.result["population"] = population_counts(self._candidates, len(self._observations))
        contributing = list(dict.fromkeys(int(row["prototype_id"]) for row in self._observations))
        contributing.sort()
        self.result["contributing_prototype_ids"] = contributing
        contributing_classes = sorted({int(row["class_id"]) for row in self._observations})
        self.result["contributing_class_ids"] = contributing_classes
        self.result["population"]["contributing_prototypes"] = len(contributing)
        self.result["population"]["contributing_classes"] = len(contributing_classes)
        self.result["population"]["selected_prototypes"] = len(self.result["selected_prototype_ids"])
        processed_ids = [
            row["sample_id"] for row in self._candidates
            if row["outcome"] in ("processed_no_observations", "contributed")
        ]
        self.result["dataset"]["processed_order_sha256"] = ordered_ids_sha256(processed_ids)
        contributing_ids = [
            row["sample_id"] for row in self._candidates if row["outcome"] == "contributed"
        ]
        self.result["dataset"]["contributing_order_sha256"] = ordered_ids_sha256(contributing_ids)
        return True

    def _publish_terminal(self, status):
        self.result["status"] = status
        self.result["completed_at_utc"] = _now()
        self._write_result()
        self._terminal = True

    def finish(self, value, *, numerator=None, denominator=None, reason=None):
        self._ensure_running()
        if self._candidates is None:
            raise CatalogError("Cannot complete a measurement before registering its candidate population")
        if self._active_sample is not None:
            raise CatalogError("Cannot complete a measurement while a sample is being processed")
        if any(sample["outcome"] == "not_attempted" for sample in self._candidates):
            raise CatalogError("A complete measurement cannot contain unattempted candidates")
        if any(sample["outcome"] == "failed" for sample in self._candidates):
            raise CatalogError("A measurement with failed samples must be published as failed")
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
        self._publish_terminal("complete")

    def fail(self, error):
        self._ensure_running()
        message = sanitize_error(error)
        if self._active_sample is not None and self._candidates is not None:
            row = self._candidates[self._sample_index[self._active_sample]]
            row.update(
                outcome="failed",
                reason=message,
                observation_count=self._observation_counts.get(self._active_sample, 0),
            )
            self._active_sample = None
        has_population = self._finalize_files()
        self.result["status"] = "failed"
        self.result["completed_at_utc"] = _now()
        self.result["failure"] = {"type": type(error).__name__, "message": message}
        if not self.result["scores"]:
            self.result["scores"] = {
                self.metric: {
                    "value": None,
                    "unit": "fraction",
                    "numerator": None,
                    "denominator": None,
                    "reason": "measurement failed before a final score was available",
                }
            }
        if not has_population:
            self.result["population"] = None
        self._write_result()
        self._terminal = True
