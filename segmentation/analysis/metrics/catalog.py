"""Validation and integrity helpers for native metric measurement bundles."""

import csv
import hashlib
import json
import os
from pathlib import Path
import re

from dotenv import dotenv_values
from pydantic import ValidationError

from segmentation.analysis.metrics.schema import Artifact, validate_metadata
from segmentation.analysis.metrics.population import ordered_ids_sha256, population_counts


SCHEMA_VERSION = 1
SAMPLE_COLUMNS = ("sample_index", "sample_id", "outcome", "reason", "observation_count")
SAMPLE_OUTCOMES = {
    "not_attempted",
    "excluded",
    "processed_no_observations",
    "contributed",
    "failed",
}
OBSERVATION_KEY = ("sample_id", "class_id", "prototype_id", "repetition")


class CatalogError(RuntimeError):
    """Raised when a catalog root or measurement bundle violates its contract."""


def resolve_catalog_root(checkout=None):
    """Resolve the required catalog root from the environment or this checkout's .env."""
    checkout = Path(checkout) if checkout is not None else Path(__file__).resolve().parents[3]
    configured_file_value = dotenv_values(checkout / ".env").get("MEASUREMENT_CATALOG_ROOT")
    if configured_file_value is not None:
        os.environ.setdefault("MEASUREMENT_CATALOG_ROOT", configured_file_value)
    configured = os.environ.get("MEASUREMENT_CATALOG_ROOT")
    if not configured:
        raise CatalogError("Set MEASUREMENT_CATALOG_ROOT to an existing writable catalog root")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise CatalogError(f"MEASUREMENT_CATALOG_ROOT must already exist: {root}")
    if not os.access(root, os.W_OK | os.X_OK):
        raise CatalogError(f"MEASUREMENT_CATALOG_ROOT is not writable: {root}")
    measurements = root / "measurements"
    try:
        measurements.mkdir(exist_ok=True)
    except OSError as error:
        raise CatalogError(f"Cannot create catalog measurements directory under {root}: {error}") from error
    if not measurements.is_dir() or not os.access(measurements, os.W_OK | os.X_OK):
        raise CatalogError(f"Catalog measurements directory is not writable: {measurements}")
    return root


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def read_result(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        return json.loads(Path(path).read_text(), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CatalogError(f"Invalid result JSON at {path}: {error}") from error


def artifact_path(bundle, artifact, role):
    """Resolve an existing artifact without allowing escape from its bundle."""
    try:
        Artifact.model_validate(artifact)
    except ValidationError as error:
        raise CatalogError(f"Invalid {role} artifact metadata: {error.errors(include_input=False)}") from error
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"Artifact path must be relative to its measurement: {role}")
    path = (Path(bundle) / relative).resolve()
    if Path(bundle).resolve() not in path.parents or not path.is_file():
        raise CatalogError(f"Missing or out-of-bundle artifact: {role}")
    return path


def ensure_external_output(path, root=None):
    """Reject symlinks and derived output inside canonical measurement storage."""
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise CatalogError(f"Derived export destination cannot be a symlink: {requested}")
    target = requested.resolve()
    root = Path(root) if root is not None else resolve_catalog_root()
    canonical = (root / "measurements").resolve()
    if target == canonical or canonical in target.parents:
        raise CatalogError(f"Derived exports must be outside catalog measurements: {target}")
    return target


def create_fresh_output_dir(path, root=None):
    output_dir = ensure_external_output(path, root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as error:
        raise CatalogError(
            f"Derived output directory must not already exist; choose a fresh path: {output_dir}"
        ) from error
    return output_dir


def write_derived_csv(path, rows, columns):
    """Write a caller-defined CSV table to a new derived artifact."""
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _read_csv(path, expected_columns=None, *, include_columns=False):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            if not columns:
                raise CatalogError(f"CSV table has no header: {path}")
            if expected_columns is not None and tuple(columns) != tuple(expected_columns):
                raise CatalogError(f"Unexpected CSV schema in {path}")
            rows = list(reader)
            return (columns, rows) if include_columns else rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CatalogError(f"Cannot read catalog CSV {path}: {error}") from error


def _integer(value, field):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise CatalogError(f"Invalid {field} in measurement record") from error
    if str(parsed) != str(value):
        raise CatalogError(f"Invalid {field} in measurement record")
    return parsed


def validate_result(result):
    """Validate the versioned metadata envelope without importing model dependencies."""
    if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
        version = result.get("schema_version") if isinstance(result, dict) else None
        raise CatalogError(f"Unsupported measurement schema version: {version}")
    try:
        validate_metadata(result)
    except ValidationError as error:
        details = []
        for item in error.errors(include_url=False, include_input=False):
            path = ".".join(map(str, item["loc"]))
            label = "checkpoint identity " if path.startswith("model.inputs.") else (
                "part mapping identity " if "parts.mapping_identity" in path else ""
            )
            details.append(f"{label}{path}: {item['type']}: {item['msg']}")
        raise CatalogError(f"Invalid measurement metadata: {'; '.join(details)}") from error
    return result


def _legacy_table_columns(result, role, artifact):
    columns = artifact.get("columns")
    if role == "observations":
        observation_schema = result.get("observation_schema", {})
        if observation_schema.get("version") not in (None, 1):
            raise CatalogError("Unsupported legacy observation schema version")
        schema_columns = observation_schema.get("columns")
        if columns is not None and schema_columns is not None and columns != schema_columns:
            raise CatalogError("Legacy observation artifact schema disagrees with result metadata")
        if columns is None:
            columns = schema_columns
    if columns is None:
        return None
    if not isinstance(columns, list) or not columns or any(not isinstance(value, str) for value in columns):
        raise CatalogError(f"Legacy {role} artifact has no declared table columns")
    if artifact.get("columns") is not None and artifact["columns"] != columns:
        raise CatalogError(f"Legacy {role} artifact schema disagrees with result metadata")
    return columns


def _load_legacy_measurement(bundle, result, *, verify_hashes):
    """Validate available legacy evidence without claiming native population completeness."""
    bundle = bundle.resolve()
    artifacts = result["artifacts"]
    paths = {}
    for role, artifact in artifacts.items():
        path = artifact_path(bundle, artifact, role)
        size = path.stat().st_size
        if artifact.get("bytes") is not None:
            if artifact["bytes"] != size:
                raise CatalogError(f"Legacy artifact byte size mismatch: {path}")
        digest = artifact.get("sha256")
        if digest is not None:
            if verify_hashes and sha256_file(path)[0] != digest:
                raise CatalogError(f"Legacy artifact hash mismatch: {path}")
        if role in ("observations", "samples") and artifact.get("schema_version") not in (None, 1):
            raise CatalogError(f"Unsupported legacy table schema version for {role}")
        paths[role] = path

    tables = {}
    observation_rows = []
    if "observations" in paths:
        artifact = artifacts["observations"]
        columns = _legacy_table_columns(result, "observations", artifact)
        parsed_columns, observation_rows = _read_csv(
            paths["observations"], columns, include_columns=True
        )
        columns = parsed_columns if columns is None else columns
        if not isinstance(columns, list) or not columns:
            raise CatalogError("Legacy observations artifact has no declared table columns")
        if artifact.get("columns") is not None and artifact["columns"] != columns:
            raise CatalogError("Legacy observations artifact schema disagrees with result metadata")
        schema_columns = result.get("observation_schema", {}).get("columns")
        if schema_columns is not None and schema_columns != columns:
            raise CatalogError("Legacy observation artifact schema disagrees with result metadata")
        if any(field not in columns for field in OBSERVATION_KEY):
            raise CatalogError("Legacy observations must use canonical sample/class/prototype/repetition keys")
        tables["observations"] = observation_rows
        keys = set()
        observation_counts = {}
        for row in observation_rows:
            for field in ("class_id", "prototype_id", "repetition"):
                _integer(row[field], field)
            key = tuple(row[name] for name in OBSERVATION_KEY)
            if key in keys:
                raise CatalogError(f"Duplicate legacy observation key: {key}")
            keys.add(key)
            observation_counts[row["sample_id"]] = observation_counts.get(row["sample_id"], 0) + 1
            for column, value in row.items():
                if (
                    re.fullmatch(r"(?:noisy_)?part_\d+", column)
                    and value.strip().lower() not in ("", "0", "1", "nan", "null", "none")
                ):
                    raise CatalogError(f"Invalid legacy presence value {value!r} in {column}")
    else:
        observation_counts = {}

    sample_rows = None
    if "samples" in paths:
        artifact = artifacts["samples"]
        columns = _legacy_table_columns(result, "samples", artifact)
        parsed_columns, sample_rows = _read_csv(
            paths["samples"], columns, include_columns=True
        )
        columns = parsed_columns if columns is None else columns
        if not isinstance(columns, list) or not columns:
            raise CatalogError("Legacy samples artifact has no declared table columns")
        if artifact.get("columns") is not None and artifact["columns"] != columns:
            raise CatalogError("Legacy samples artifact schema disagrees with result metadata")
        if tuple(columns) != SAMPLE_COLUMNS:
            raise CatalogError("Legacy sample tables must use the canonical partial-samples columns")
        tables["samples"] = sample_rows
        seen_ids = set()
        previous_index = -1
        for row in sample_rows:
            index = _integer(row["sample_index"], "sample_index")
            if index < 0 or index <= previous_index:
                raise CatalogError("Legacy partial sample indices must be unique and ordered")
            previous_index = index
            if row["sample_id"] in seen_ids:
                raise CatalogError(f"Duplicate legacy sample ID: {row['sample_id']}")
            seen_ids.add(row["sample_id"])
            if row["outcome"] not in SAMPLE_OUTCOMES:
                raise CatalogError(f"Invalid legacy sample outcome: {row['outcome']}")
            if row["outcome"] in ("excluded", "failed") and not row["reason"]:
                raise CatalogError(f"Legacy sample outcome {row['outcome']} requires a reason")
            count = _integer(row["observation_count"], "observation_count")
            if count < 0:
                raise CatalogError("Legacy sample observation counts cannot be negative")
            if "observations" in paths:
                if count != observation_counts.get(row["sample_id"], 0):
                    raise CatalogError(f"Legacy observation count mismatch for sample {row['sample_id']}")
                if row["outcome"] == "contributed" and count == 0:
                    raise CatalogError("A legacy contributing sample must have observations")
                if row["outcome"] == "processed_no_observations" and count:
                    raise CatalogError("A legacy sample without observations has observation rows")

    coverage = result.get("dataset", {}).get("identity_coverage", {})
    membership_complete = (
        isinstance(coverage, dict) and coverage.get("candidate_membership") == "complete"
    )
    if membership_complete:
        if sample_rows is None:
            raise CatalogError("Legacy record claims complete candidate membership without samples.csv")
        sample_indices = [_integer(row["sample_index"], "sample_index") for row in sample_rows]
        if sample_indices != list(range(len(sample_rows))):
            raise CatalogError("Complete legacy sample membership must use consecutive sample indices")
        candidate_count = result.get("dataset", {}).get("candidate_count")
        if candidate_count is not None and candidate_count != len(sample_rows):
            raise CatalogError("Legacy candidate count disagrees with its complete sample table")
        if set(observation_counts) - {row["sample_id"] for row in sample_rows}:
            raise CatalogError("Legacy observations contain samples outside complete candidate membership")
    return result, tables


def _validate_observation_rows(rows, observation_columns):
    keys = set()
    counts = {}
    for row in rows:
        for field in ("class_id", "prototype_id", "repetition"):
            parsed = _integer(row[field], field)
            if str(parsed) != row[field]:
                raise CatalogError(f"Invalid {field} in observation table")
        key = tuple(row[name] for name in OBSERVATION_KEY)
        if key in keys:
            raise CatalogError(f"Duplicate observation key: {key}")
        keys.add(key)
        counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
        for column in observation_columns[4:]:
            value = row[column]
            if value not in ("", "0", "1"):
                raise CatalogError(f"Invalid presence value {value!r} in {column}")
    return counts


def _load_measurement(bundle, *, verify_hashes=True):
    """Load a measurement and validate finalized artifacts and population counts."""
    bundle = Path(bundle)
    result = validate_result(read_result(bundle / "result.json"))
    if result["status"] not in ("complete", "failed"):
        return result, {}
    if result.get("origin") == "legacy":
        return _load_legacy_measurement(bundle, result, verify_hashes=verify_hashes)

    artifacts = result["artifacts"]
    if result["status"] == "complete" and set(artifacts) != {"observations", "samples"}:
        raise CatalogError("A completed native measurement must declare observations.csv and samples.csv")
    if result["status"] == "complete" and {
        path.name for path in bundle.iterdir()
    } != {"result.json", "observations.csv", "samples.csv"}:
        raise CatalogError("A completed native measurement must contain exactly three canonical files")
    expected_artifact_paths = {
        "observations": "observations.csv",
        "samples": "samples.csv",
    }
    paths = {}
    for role, artifact in artifacts.items():
        if role in expected_artifact_paths and artifact.get("path") != expected_artifact_paths[role]:
            raise CatalogError(f"Unexpected canonical artifact path for {role}")
        if artifact.get("schema_version") != 1:
            raise CatalogError(f"Unsupported artifact schema version for {role}")
        path = artifact_path(bundle, artifact, role)
        size = path.stat().st_size
        if size != artifact.get("bytes"):
            raise CatalogError(f"Artifact byte size mismatch: {path}")
        if verify_hashes and sha256_file(path)[0] != artifact.get("sha256"):
            raise CatalogError(f"Artifact hash mismatch: {path}")
        paths[role] = path

    if "observations" not in artifacts or "samples" not in artifacts:
        if result["status"] == "complete":
            raise CatalogError("Completed measurement is missing observation or sample artifacts")
        tables = {}
        if "observations" in artifacts:
            observation_schema = result.get("observation_schema", {})
            observation_columns = observation_schema.get("columns")
            if observation_schema.get("version") != 1 or not isinstance(observation_columns, list):
                raise CatalogError("Measurement is missing its observation table schema")
            if artifacts["observations"].get("columns") != observation_columns:
                raise CatalogError("Observation artifact schema disagrees with result.json")
            rows = _read_csv(paths["observations"], observation_columns)
            _validate_observation_rows(rows, observation_columns)
            tables["observations"] = rows
        return result, tables

    observation_schema = result.get("observation_schema", {})
    observation_columns = observation_schema.get("columns")
    if observation_schema.get("version") != 1 or not isinstance(observation_columns, list):
        raise CatalogError("Measurement is missing its observation table schema")
    if artifacts["observations"].get("columns") != observation_columns:
        raise CatalogError("Observation artifact schema disagrees with result.json")
    if artifacts["samples"].get("columns") != list(SAMPLE_COLUMNS):
        raise CatalogError("Sample artifact schema disagrees with result.json")
    rows = _read_csv(paths["observations"], observation_columns)
    sample_rows = _read_csv(paths["samples"], SAMPLE_COLUMNS)
    counts = _validate_observation_rows(rows, observation_columns)

    outcomes = []
    seen_samples = set()
    for index, row in enumerate(sample_rows):
        if _integer(row["sample_index"], "sample_index") != index:
            raise CatalogError("Sample table indices must be consecutive and ordered")
        sample_id = row["sample_id"]
        if sample_id in seen_samples:
            raise CatalogError(f"Duplicate candidate sample ID: {sample_id}")
        seen_samples.add(sample_id)
        if row["outcome"] not in SAMPLE_OUTCOMES:
            raise CatalogError(f"Invalid sample outcome: {row['outcome']}")
        if row["outcome"] in ("excluded", "failed") and not row["reason"]:
            raise CatalogError(f"Sample outcome {row['outcome']} requires a reason")
        count = _integer(row["observation_count"], "observation_count")
        if count != counts.get(sample_id, 0):
            raise CatalogError(f"Observation count mismatch for sample {sample_id}")
        if row["outcome"] == "contributed" and count == 0:
            raise CatalogError("A contributing sample must have observations")
        if row["outcome"] == "processed_no_observations" and count:
            raise CatalogError("A sample without observations has observation rows")
        outcomes.append(row["outcome"])
    if set(counts) - seen_samples:
        raise CatalogError("Observation table contains samples outside the candidate population")

    population = result.get("population")
    if not isinstance(population, dict):
        raise CatalogError("Finalized measurement is missing population counts")
    expected = population_counts(sample_rows, len(rows))
    for key, value in expected.items():
        if population.get(key) != value:
            raise CatalogError(f"Population count mismatch for {key}")
    if result["status"] == "complete" and (expected["not_attempted"] or expected["failed"]):
        raise CatalogError("Completed measurement contains unattempted or failed samples")
    dataset = result.get("dataset", {})
    candidate_ids = [row["sample_id"] for row in sample_rows]
    processed_ids = [
        row["sample_id"] for row in sample_rows
        if row["outcome"] in ("processed_no_observations", "contributed")
    ]
    contributing_ids = [row["sample_id"] for row in sample_rows if row["outcome"] == "contributed"]
    for field, values in (
        ("candidate_order_sha256", candidate_ids),
        ("processed_order_sha256", processed_ids),
        ("contributing_order_sha256", contributing_ids),
    ):
        if dataset.get(field) != ordered_ids_sha256(values):
            raise CatalogError(f"Dataset {field} disagrees with samples.csv")
    contributing_prototypes = sorted({int(row["prototype_id"]) for row in rows})
    contributing_classes = sorted({int(row["class_id"]) for row in rows})
    if result.get("contributing_prototype_ids") != contributing_prototypes:
        raise CatalogError("Contributing prototype IDs disagree with observations.csv")
    if result.get("contributing_class_ids") != contributing_classes:
        raise CatalogError("Contributing class IDs disagree with observations.csv")
    if population.get("contributing_prototypes") != len(contributing_prototypes):
        raise CatalogError("Contributing prototype count disagrees with observations.csv")
    if population.get("contributing_classes") != len(contributing_classes):
        raise CatalogError("Contributing class count disagrees with observations.csv")
    selected_prototypes = result.get("selected_prototype_ids")
    if not isinstance(selected_prototypes, list) or len(set(selected_prototypes)) != len(selected_prototypes):
        raise CatalogError("Selected prototype IDs must be a unique list")
    if population.get("selected_prototypes") != len(selected_prototypes):
        raise CatalogError("Selected prototype count disagrees with result.json")
    return result, {"observations": rows, "samples": sample_rows}


def load_measurement(bundle, *, verify_hashes=True):
    """Load a measurement and validate finalized artifacts and population counts."""
    result, _tables = _load_measurement(bundle, verify_hashes=verify_hashes)
    return result


def load_observations(bundle, *, verify_hashes=True):
    """Load validated finalized observations together with their result metadata."""
    result, tables = _load_measurement(bundle, verify_hashes=verify_hashes)
    if result["status"] == "running":
        raise CatalogError("Running measurements cannot be summarized or plotted")
    if "observations" not in result.get("artifacts", {}) or "observations" not in tables:
        raise CatalogError("Measurement has no finalized observation evidence")
    return result, tables["observations"]


def iter_measurements(root):
    """Yield catalog metadata records without loading model code or reading table payloads."""
    measurements = Path(root) / "measurements"
    if not measurements.is_dir():
        return
    for bundle in sorted(measurements.iterdir(), key=lambda path: path.name):
        if bundle.is_dir() and (bundle / "result.json").is_file():
            yield validate_result(read_result(bundle / "result.json"))
