"""Reusable image-support counts derived from finalized consistency observations."""

import csv
import hashlib
import json
import math
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from segmentation.catalog import (
    CatalogError,
    load_observations,
    read_result,
    sha256_file,
    validate_result,
)
from segmentation.catalog.schema import Artifact

CSV_NAME = "part-summary-v1.csv"
METADATA_NAME = "part-summary-v1.meta.json"
COUNT_COLUMNS = ("class_id", "prototype_id", "part_id", "observations", "hits")
SUPPORT_LIMITS = (1, 2, 5, 10, 20, 30)
NonnegativeInt = Annotated[int, Field(ge=0)]


class SummaryModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class PrototypeClass(SummaryModel):
    class_id: int
    prototype_id: int


class PartCount(PrototypeClass):
    part_id: NonnegativeInt
    observations: NonnegativeInt
    hits: NonnegativeInt

    @model_validator(mode="after")
    def valid_counts(self):
        if self.hits > self.observations:
            raise ValueError("Part hits cannot exceed observations")
        return self


class PartSummaryMetadata(SummaryModel):
    schema_version: Literal[1]
    calculation: Literal["binary-part-image-counts-v1"]
    measurement_id: str
    source_observations: Artifact
    context_sha256: str
    generated_at_utc: str
    threshold: Optional[Annotated[float, Field(ge=0, le=1)]]
    part_ids: list[int]
    analysis_part_ids: list[int]
    prototypes: list[PrototypeClass]
    selected_prototype_ids: Optional[list[int]]
    observation_rows: NonnegativeInt
    csv: Artifact


class PartSummary(SummaryModel):
    metadata: PartSummaryMetadata
    counts: list[PartCount]

    @model_validator(mode="after")
    def complete_coordinates(self):
        metadata = self.metadata
        parts = metadata.part_ids
        prototypes = [(item.class_id, item.prototype_id) for item in metadata.prototypes]
        if parts != sorted(set(parts)) or any(part < 0 for part in parts):
            raise ValueError("Summary part IDs must be unique, nonnegative and sorted")
        if len({pid for _, pid in prototypes}) != len(prototypes):
            raise ValueError("A prototype must belong to exactly one class")
        if not set(metadata.analysis_part_ids) <= set(parts) or any(part <= 0 for part in metadata.analysis_part_ids):
            raise ValueError("Analysis part IDs must be recorded positive part IDs")
        expected = {(cid, pid, part) for cid, pid in prototypes for part in parts}
        actual = [(row.class_id, row.prototype_id, row.part_id) for row in self.counts]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Summary must contain every prototype/part coordinate exactly once")
        if any(row.observations > metadata.observation_rows for row in self.counts):
            raise ValueError("Part support exceeds the number of source observations")
        return self


def summary_context_sha256(record):
    """Fingerprint interpretation and population metadata, not mutable model descriptions."""
    context = {
        key: record.get(key)
        for key in ("measurement_id", "metric", "status", "dataset", "protocol", "parameters", "selected_prototype_ids")
    }
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _require_consistency(record):
    if record["metric"] != "consistency" or record["status"] != "complete":
        raise CatalogError("Part-support summaries require a complete consistency measurement")


def _summary_paths(bundle):
    bundle = Path(bundle).resolve()
    derived = bundle / "derived"
    if bundle not in derived.resolve().parents:
        raise CatalogError("Derived directory must remain inside its measurement bundle")
    paths = (derived / CSV_NAME, derived / METADATA_NAME)
    if any(bundle not in path.resolve().parents for path in paths):
        raise CatalogError("Derived artifacts must remain inside their measurement bundle")
    return paths


def part_summary_signature(bundle):
    """Cheap cache invalidation for independently published derived files."""
    return tuple(
        (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None for path in _summary_paths(bundle)
    )


def _integer(value):
    if isinstance(value, bool) or str(int(value)) != str(value):
        raise ValueError(f"Expected an integer ID or count, got {value!r}")
    return int(value)


def _build_counts(record, rows):
    columns = [
        column
        for column in record["protocol"].get("observation_columns", [])
        if column.startswith("part_") and column[5:].isdigit()
    ]
    part_ids = sorted({_integer(column[5:]) for column in columns})
    if not part_ids or len(part_ids) != len(columns):
        raise CatalogError("Measurement must declare unique part-activation columns")
    counts, classes, seen = {}, {}, set()
    for row in rows:
        cid, pid = _integer(row["class_id"]), _integer(row["prototype_id"])
        if pid in classes and classes[pid] != cid:
            raise CatalogError("A prototype must belong to exactly one semantic class")
        classes[pid] = cid
        key = (str(row["sample_id"]), cid, pid)
        if key in seen:
            raise CatalogError("Repeated sample/class/prototype rows cannot count as independent image support")
        seen.add(key)
        for part in part_ids:
            pair = counts.setdefault((cid, pid, part), [0, 0])
            value = row[f"part_{part}"]
            if value is None or value == "" or value == "nan":
                continue
            number = float(value)
            if math.isnan(number):
                continue
            if number not in (0, 1):
                raise CatalogError("Part-support summaries require binary or missing part observations")
            pair[0] += 1
            pair[1] += int(number)
    result = [
        PartCount(class_id=cid, prototype_id=pid, part_id=part, observations=n, hits=h)
        for (cid, pid, part), (n, h) in sorted(counts.items())
    ]
    prototypes = [PrototypeClass(class_id=cid, prototype_id=pid) for pid, cid in sorted(classes.items())]
    return part_ids, prototypes, result


@contextmanager
def _atomic_text(path):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_part_summary(bundle, record, rows):
    """Publish counts first and metadata last; never modify the original bundle evidence."""
    _require_consistency(record)
    try:
        part_ids, prototypes, counts = _build_counts(record, rows)
        valid = record["protocol"].get("parts", {}).get("valid_part_ids", part_ids)
        threshold = record["protocol"].get("consistency", {}).get("threshold", record["parameters"].get("threshold"))
        csv_path, metadata_path = _summary_paths(bundle)
        metadata = PartSummaryMetadata(
            schema_version=1,
            calculation="binary-part-image-counts-v1",
            measurement_id=record["measurement_id"],
            source_observations=Artifact(**record["artifacts"]["observations"]),
            context_sha256=summary_context_sha256(record),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            threshold=float(threshold) if threshold is not None else None,
            part_ids=part_ids,
            analysis_part_ids=[part for part in part_ids if part > 0 and part in valid],
            prototypes=prototypes,
            selected_prototype_ids=record.get("selected_prototype_ids"),
            observation_rows=len(rows),
            csv=Artifact(path=CSV_NAME, bytes=0, sha256="0" * 64),
        )
        summary = PartSummary(metadata=metadata, counts=counts)
        csv_path.parent.mkdir(exist_ok=True)
        with _atomic_text(csv_path) as stream:
            writer = csv.DictWriter(stream, fieldnames=COUNT_COLUMNS)
            writer.writeheader()
            writer.writerows(row.model_dump() for row in counts)
        digest, size = sha256_file(csv_path)
        metadata.csv = Artifact(path=CSV_NAME, bytes=size, sha256=digest)
        with _atomic_text(metadata_path) as stream:
            json.dump(metadata.model_dump(mode="json"), stream, indent=2, allow_nan=False)
            stream.write("\n")
        return summary
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise CatalogError(f"Cannot write part summary: {error}") from error


def load_part_summary(bundle, record=None):
    """Verify derived integrity and advertised source identity without rereading observations."""
    csv_path, metadata_path = _summary_paths(bundle)
    if not metadata_path.exists():
        return None
    record = record if record is not None else validate_result(read_result(Path(bundle) / "result.json"))
    _require_consistency(record)
    try:
        metadata = PartSummaryMetadata.model_validate(read_result(metadata_path))
        if (
            metadata.measurement_id != record["measurement_id"]
            or metadata.context_sha256 != summary_context_sha256(record)
            or metadata.source_observations.model_dump() != record["artifacts"]["observations"]
        ):
            raise CatalogError("Part summary is stale: source observations or protocol changed")
        if metadata.csv.path != CSV_NAME or sha256_file(csv_path) != (metadata.csv.sha256, metadata.csv.bytes):
            raise CatalogError("Part summary CSV hash or byte size mismatch")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(COUNT_COLUMNS):
                raise CatalogError("Part summary CSV columns disagree with its schema")
            counts = [PartCount(**{key: _integer(value) for key, value in row.items()}) for row in reader]
        return PartSummary(metadata=metadata, counts=counts)
    except (ValidationError, KeyError, TypeError, ValueError, OSError, csv.Error) as error:
        raise CatalogError(f"Invalid part summary: {error}") from error


def backfill_part_summary(bundle):
    """Create missing/stale summaries; return 'current' without changing valid summaries."""
    record = validate_result(read_result(Path(bundle) / "result.json"))
    _require_consistency(record)
    try:
        if load_part_summary(bundle, record) is not None:
            return "current"
    except CatalogError:
        pass
    record, rows = load_observations(bundle)
    write_part_summary(bundle, record, rows)
    return "created"


def _prototype_support(summary, threshold):
    threshold = summary.metadata.threshold if threshold is None else threshold
    if threshold is None or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A consistency threshold between 0 and 1 is required")
    prototypes = {
        item.prototype_id: {
            "class_id": item.class_id,
            "prototype_id": item.prototype_id,
            "maximum_support": 0,
            "qualifying_support": 0,
        }
        for item in summary.metadata.prototypes
    }
    parts = set(summary.metadata.analysis_part_ids)
    for row in summary.counts:
        if row.part_id not in parts:
            continue
        item = prototypes[row.prototype_id]
        item["maximum_support"] = max(item["maximum_support"], row.observations)
        if row.observations and row.hits / row.observations > threshold:
            item["qualifying_support"] = max(item["qualifying_support"], row.observations)
    return list(prototypes.values()), threshold


def _support_statistics(population, minimum_support, threshold):
    if type(minimum_support) is not int or minimum_support < 1:
        raise ValueError("Minimum support must be a positive integer")
    numerator = sum(item["qualifying_support"] >= minimum_support for item in population)
    denominator = len(population)
    supported = sum(item["maximum_support"] >= minimum_support for item in population)
    return {
        "minimum_support": minimum_support,
        "threshold": threshold,
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "rare_only": sum(0 < item["qualifying_support"] < minimum_support for item in population),
        "insufficient_evidence": denominator - supported,
        "evidence_coverage": supported / denominator if denominator else None,
        "prototypes": population,
    }


def support_statistics(summary, minimum_support=1, threshold=None):
    """Evaluate any support limit from counts, keeping all observed prototypes in the denominator."""
    population, threshold = _prototype_support(summary, threshold)
    return _support_statistics(population, minimum_support, threshold)


def support_curve(summary, limits=SUPPORT_LIMITS, threshold=None):
    """Reuse the qualifying-support vector across every requested limit."""
    population, threshold = _prototype_support(summary, threshold)
    return [
        {key: value for key, value in _support_statistics(population, limit, threshold).items() if key != "prototypes"}
        for limit in limits
    ]
