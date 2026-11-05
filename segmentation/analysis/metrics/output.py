"""Atomic writer for native, UUID-addressed metric measurement bundles."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
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


_PACKAGES = (
    "torch", "torchvision", "numpy", "pandas", "Pillow", "opencv-python",
    "scikit-image", "scipy", "tqdm",
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|credential|private[_-]?key)\b\s*[:=]"
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|credential|private[_-]?key)\b\s*[:=]\s*)"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s,;]+)'
)


def _sanitize_error(error):
    message = f"{type(error).__name__}: {error}"
    message = _SENSITIVE_VALUE.sub(r"\1[redacted]", message)
    return "".join(character for character in message if character.isprintable())[:500]


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
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _run_git(checkout, *arguments):
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_provenance(checkout):
    if _run_git(checkout, "rev-parse", "--show-toplevel") is None:
        return {
            "commit": None,
            "tracked_diff": None,
            "relevant_untracked_source": [],
            "submodules": [],
            "missing": ["git source identity"],
        }
    diff = _run_git(checkout, "diff", "--binary", "HEAD")
    untracked = _run_git(checkout, "ls-files", "--others", "--exclude-standard") or ""
    source_files = []
    omitted_source = []
    for relative in untracked.splitlines():
        path = Path(relative)
        if path.suffix not in (".py", ".sh") or path.parts[0] not in ("segmentation", "scripts"):
            continue
        source = checkout / path
        if source.is_symlink():
            omitted_source.append(str(path))
            continue
        try:
            source.resolve().relative_to(checkout.resolve())
            contents = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            omitted_source.append(str(path))
            continue
        if _SENSITIVE_ASSIGNMENT.search(contents):
            omitted_source.append(str(path))
            continue
        source_files.append({
            "path": str(path),
            "sha256": hashlib.sha256(contents.encode()).hexdigest(),
            "content": contents,
        })
    submodules = _run_git(checkout, "submodule", "status") or ""
    tracked_diff = None
    if diff:
        tracked_diff = {
            "sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "content": None if _SENSITIVE_ASSIGNMENT.search(diff) else diff,
        }
        if tracked_diff["content"] is None:
            tracked_diff["content_omitted_for_sensitive_content"] = True
    return {
        "commit": _run_git(checkout, "rev-parse", "HEAD"),
        "tracked_diff": tracked_diff,
        "relevant_untracked_source": source_files,
        "sensitive_looking_untracked_source_omitted": omitted_source,
        "submodules": [line.strip() for line in submodules.splitlines() if line.strip()],
        "missing": [],
    }


def _file_identity(path, cache):
    absolute = str(Path(path).expanduser().resolve())
    if absolute not in cache:
        source = Path(absolute)
        if source.is_file():
            try:
                digest, size = sha256_file(source)
                cache[absolute] = {
                    "path": absolute,
                    "bytes": size,
                    "sha256": digest,
                    "status": "available",
                }
            except OSError as error:
                cache[absolute] = {
                    "path": absolute,
                    "status": "unreadable",
                    "error": str(error)[:200],
                }
        else:
            cache[absolute] = {"path": absolute, "status": "missing"}
    return dict(cache[absolute])


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
            "prototype_checkpoint": _file_identity(checkpoint, self._input_hash_cache),
        }
        if group_checkpoint is not None:
            inputs["group_checkpoint"] = _file_identity(group_checkpoint, self._input_hash_cache)
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
                "seed": parameters.get("seed"),
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
                **_source_provenance(checkout),
                "command": list(sys.argv),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "packages": self._package_versions(),
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

    @staticmethod
    def _package_versions():
        versions = {}
        for package in _PACKAGES:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return versions

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
        sample_ids = list(dataset.img_ids)
        annotations = Path(dataset.annotations_dir).expanduser().resolve()
        part_dir = annotations.parent.parent / "annotations_PIDS" / annotations.name
        part_available = [sample_id for sample_id in sample_ids if (part_dir / f"{sample_id}.npy").is_file()]
        self.result["dataset"].update({
            "split": "val",
            "processed_root": str(annotations.parent.parent),
            "annotations_directory": str(annotations),
            "part_annotations_directory": str(part_dir),
            "semantic_id_annotations_directory": str(
                annotations.parent.parent / "annotations_SIDS" / annotations.name
            ),
            "candidate_count": len(sample_ids),
            "part_annotation_count": len(part_available),
            "part_annotation_coverage": len(part_available) / len(sample_ids) if sample_ids else None,
            "semantic_id_annotation_count": sum(
                (annotations.parent.parent / "annotations_SIDS" / annotations.name / f"{sample_id}.npy").is_file()
                for sample_id in sample_ids
            ),
            "identity": "partial",
        })
        self.result["provenance_quality"]["warnings"] = [
            "Dataset identity is partial; image paths are not content fingerprints.",
            f"{len(sample_ids) - len(part_available)} candidates lack part annotations."
            if len(sample_ids) != len(part_available) else "Every candidate has a part annotation.",
        ]
        if len(part_available) != len(sample_ids):
            self.result["provenance_quality"]["missing_fields"].append("part annotation coverage")
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
        self.result["dataset"]["candidate_order_sha256"] = hashlib.sha256(
            json.dumps(candidates, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    def update_metadata(self, **fields):
        self._ensure_running()
        for key, value in fields.items():
            self.result[key] = _json_value(value)
        self._write_result()

    def register_input(self, role, path):
        self._ensure_running()
        self.result["model"]["inputs"][role] = _file_identity(path, self._input_hash_cache)
        self._write_result()

    def update_model_metadata(self, **fields):
        self._ensure_running()
        self.result["model"].update(_json_value(fields))
        if fields.get("architecture") is not None:
            missing = self.result["provenance_quality"]["missing_fields"]
            if "model architecture" in missing:
                missing.remove("model architecture")
        self._write_result()

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

    def _population(self):
        outcomes = [sample["outcome"] for sample in self._candidates or ()]
        return {
            "candidates": len(outcomes),
            "attempted": sum(value in ("processed_no_observations", "contributed", "failed") for value in outcomes),
            "processed": sum(value in ("processed_no_observations", "contributed") for value in outcomes),
            "contributing": outcomes.count("contributed"),
            "failed": outcomes.count("failed"),
            "excluded": outcomes.count("excluded"),
            "not_attempted": outcomes.count("not_attempted"),
            "processed_no_observations": outcomes.count("processed_no_observations"),
            "observations": len(self._observations),
        }

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
        self.result["population"] = self._population()
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
        self.result["dataset"]["processed_order_sha256"] = hashlib.sha256(
            json.dumps(processed_ids, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        contributing_ids = [
            row["sample_id"] for row in self._candidates if row["outcome"] == "contributed"
        ]
        self.result["dataset"]["contributing_order_sha256"] = hashlib.sha256(
            json.dumps(contributing_ids, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
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
        message = _sanitize_error(error)
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
