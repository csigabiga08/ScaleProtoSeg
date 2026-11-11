"""Atomic writer for UUID-addressed metric measurement bundles."""

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

from segmentation.catalog import (
    CatalogError,
    OBSERVATION_KEY,
    resolve_catalog_root,
    sha256_file,
)
from segmentation.catalog.schema import (
    Artifact,
    DatasetMetadata,
    ExecutionMetadata,
    Failure,
    MeasurementOutput,
    ModelMetadata,
    Population,
    Score,
)


def _file_identity(path):
    absolute = Path(path).expanduser().resolve()
    digest, size = sha256_file(absolute)
    return {"path": str(absolute), "bytes": size, "sha256": digest}


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


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
        self._candidate_count = None
        self._observations = []
        self._observation_keys = set()
        self._observation_counts = {}
        self._terminal = False
        self._artifacts = {}
        inputs = {"prototype_checkpoint": _file_identity(checkpoint)}
        if group_checkpoint is not None:
            inputs["group_checkpoint"] = _file_identity(group_checkpoint)
        self.result = MeasurementOutput(
            measurement_id=self.measurement_id,
            metric=self.metric,
            status="running",
            created_at_utc=_now(),
            completed_at_utc=None,
            model=ModelMetadata(
                label=model_name,
                training_stage=training_phase,
                dataset=self.data_type,
                architecture=None,
                inputs={role: Artifact(**identity) for role, identity in inputs.items()},
            ),
            dataset=DatasetMetadata(name=self.data_type, split="val"),
            protocol=protocol,
            parameters=_json_value(parameters),
            execution=ExecutionMetadata(commit=_git_commit(), command=list(sys.argv)),
            population=None,
            selected_prototype_ids=[],
            scores={},
            artifacts={},
        )
        self._write_result()

    def _ensure_running(self):
        if self._terminal:
            raise CatalogError("A terminal measurement is immutable")

    def _write_result(self):
        temporary = self.path / f".result.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(self.result.model_dump(mode="json", exclude_unset=True), stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path / "result.json")
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_dataset(self, dataset):
        self._ensure_running()
        self.set_candidates(dataset.img_ids)
        self._write_result()

    def set_candidates(self, sample_ids):
        self._ensure_running()
        self._candidate_count = len(set(str(sample_id) for sample_id in sample_ids))
        self.result.dataset.candidate_count = self._candidate_count

    def update_metadata(self, **fields):
        self._ensure_running()
        for key, value in fields.items():
            setattr(self.result, key, _json_value(value))
        self._write_result()

    def register_input(self, role, path):
        self._ensure_running()
        self.result.model.inputs[role] = Artifact(**_file_identity(path))
        self._write_result()

    def update_model_metadata(self, **fields):
        self._ensure_running()
        for key, value in _json_value(fields).items():
            setattr(self.result.model, key, value)
        self._write_result()

    def set_selected_prototype_ids(self, prototype_ids):
        self.update_metadata(selected_prototype_ids=list(dict.fromkeys(int(value) for value in prototype_ids)))

    def add_observation(self, **observation):
        self._ensure_running()
        missing = [key for key in self.observation_table_columns if key not in observation]
        if missing:
            raise CatalogError(f"Observation is missing columns: {', '.join(missing)}")
        sample_id = str(observation["sample_id"])
        key = tuple(str(observation[name]) for name in OBSERVATION_KEY)
        if key in self._observation_keys:
            raise CatalogError(f"Duplicate observation key: {key}")
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
        self.result.artifacts = {"observations": Artifact(**self._write_observations())}
        self.result.population = Population(
            candidates=self._candidate_count,
            contributing=len(self._observation_counts),
            observations=len(self._observations),
        )

    def finish(self, value, *, numerator=None, denominator=None, reason=None):
        self._ensure_running()
        finite = value is not None and math.isfinite(float(value))
        value = float(value) if finite else None
        if not finite and not reason:
            reason = "score is undefined"
        if denominator == 0:
            reason = reason or "score denominator is zero"
            value = None
        self._finalize_files()
        self.result.scores = {
            self.metric: Score(
                value=value,
                unit="fraction",
                numerator=_json_value(numerator),
                denominator=_json_value(denominator),
                **({"reason": str(reason)} if reason else {}),
            )
        }
        self.result.status = "complete"
        self.result.completed_at_utc = _now()
        self._write_result()
        self._terminal = True

    def fail(self, error):
        self._ensure_running()
        message = str(error)[:500]
        self._finalize_files()
        self.result.status = "failed"
        self.result.completed_at_utc = _now()
        self.result.failure = Failure(type=type(error).__name__, message=message)
        self.result.scores = {
            self.metric: Score(
                value=None,
                unit="fraction",
                numerator=None,
                denominator=None,
                reason="measurement failed before a final score was available",
            )
        }
        self._write_result()
        self._terminal = True
