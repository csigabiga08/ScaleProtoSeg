"""Validation and integrity helpers for native metric measurement bundles."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re


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


def _read_dotenv(path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?MEASUREMENT_CATALOG_ROOT\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split("#", 1)[0].strip()
        os.environ.setdefault("MEASUREMENT_CATALOG_ROOT", value)


def resolve_catalog_root(checkout=None):
    """Resolve the required catalog root from the environment or this checkout's .env."""
    checkout = Path(checkout) if checkout is not None else Path(__file__).resolve().parents[3]
    _read_dotenv(checkout / ".env")
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


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        return json.loads(Path(path).read_text(), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CatalogError(f"Invalid result JSON at {path}: {error}") from error


def _read_csv(path, expected_columns):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise CatalogError(f"Unexpected CSV schema in {path}")
            return list(reader)
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
    for key in ("measurement_id", "metric", "status", "artifacts", "scores"):
        if key not in result:
            raise CatalogError(f"Measurement result is missing {key}")
    if result["status"] not in ("running", "complete", "failed"):
        raise CatalogError(f"Invalid measurement status: {result['status']}")
    if not isinstance(result["artifacts"], dict) or not isinstance(result["scores"], dict):
        raise CatalogError("Measurement artifacts and scores must be objects")
    if result["status"] in ("complete", "failed"):
        if result.get("origin") != "native":
            raise CatalogError("A terminal measurement must declare native origin")
        for section in ("model", "dataset", "protocol", "execution", "provenance_quality"):
            if not isinstance(result.get(section), dict):
                raise CatalogError(f"Terminal measurement is missing native {section} metadata")
        model = result["model"]
        if any(key not in model for key in ("label", "training_stage", "dataset", "training_seed", "inputs")):
            raise CatalogError("Terminal measurement is missing native model fields")
        if "seed" in model:
            raise CatalogError("Metric parameter seeds must not be stored as model seed")
        if not isinstance(model["inputs"], dict) or not model["inputs"]:
            raise CatalogError("Terminal measurement is missing model inputs")
        dataset = result["dataset"]
        if any(key not in dataset for key in ("name", "split", "identity")):
            raise CatalogError("Terminal measurement is missing native dataset fields")
        if not isinstance(dataset["identity"], str):
            raise CatalogError("Terminal measurement is missing dataset identity")
        protocol = result["protocol"]
        if (
            protocol.get("version") != 1
            or protocol.get("metric") != result["metric"]
            or not isinstance(protocol.get("observation_columns"), list)
            or not protocol["observation_columns"]
        ):
            raise CatalogError("Terminal measurement is missing its native protocol")
        execution = result["execution"]
        execution_fields = (
            "commit", "tracked_diff", "relevant_untracked_source", "submodules", "missing",
            "command", "python", "python_executable", "packages", "host", "scheduler_job_id",
        )
        if any(field not in execution for field in execution_fields):
            raise CatalogError("Terminal measurement is missing native execution provenance")
        if not isinstance(execution["packages"], dict):
            raise CatalogError("Terminal measurement is missing execution package provenance")
        provenance_quality = result["provenance_quality"]
        if (
            not isinstance(provenance_quality.get("missing_fields"), list)
            or not isinstance(provenance_quality.get("warnings"), list)
        ):
            raise CatalogError("Terminal measurement is missing provenance quality metadata")
        score = result["scores"].get(result["metric"])
        if not isinstance(score, dict) or score.get("unit") != "fraction":
            raise CatalogError("A terminal measurement must contain its fractional metric score")
        if not {"value", "numerator", "denominator"} <= set(score):
            raise CatalogError("Metric score is missing value, numerator, or denominator")
        if score["value"] is None:
            if not score.get("reason"):
                raise CatalogError("An undefined metric score must include a reason")
        elif not isinstance(score["value"], (int, float)) or not math.isfinite(score["value"]):
            raise CatalogError("Metric score value must be finite or null")
    return result


def load_measurement(bundle, *, verify_hashes=True):
    """Load a measurement and validate finalized artifacts and population counts."""
    bundle = Path(bundle)
    result = validate_result(_read_json(bundle / "result.json"))
    if result["status"] not in ("complete", "failed"):
        return result

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
    for role, artifact in artifacts.items():
        if role in expected_artifact_paths and artifact.get("path") != expected_artifact_paths[role]:
            raise CatalogError(f"Unexpected canonical artifact path for {role}")
        if artifact.get("schema_version") != 1:
            raise CatalogError(f"Unsupported artifact schema version for {role}")
        relative = Path(artifact.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise CatalogError(f"Artifact path must be relative to its measurement: {role}")
        path = bundle / relative
        if not path.is_file():
            raise CatalogError(f"Missing measurement artifact: {path}")
        size = path.stat().st_size
        if size != artifact.get("bytes"):
            raise CatalogError(f"Artifact byte size mismatch: {path}")
        if verify_hashes and sha256_file(path)[0] != artifact.get("sha256"):
            raise CatalogError(f"Artifact hash mismatch: {path}")

    if "observations" not in artifacts or "samples" not in artifacts:
        if result["status"] == "complete":
            raise CatalogError("Completed measurement is missing observation or sample artifacts")
        return result

    observation_schema = result.get("observation_schema", {})
    observation_columns = observation_schema.get("columns")
    if observation_schema.get("version") != 1 or not isinstance(observation_columns, list):
        raise CatalogError("Measurement is missing its observation table schema")
    if artifacts["observations"].get("columns") != observation_columns:
        raise CatalogError("Observation artifact schema disagrees with result.json")
    if artifacts["samples"].get("columns") != list(SAMPLE_COLUMNS):
        raise CatalogError("Sample artifact schema disagrees with result.json")
    rows = _read_csv(bundle / artifacts["observations"]["path"], observation_columns)
    sample_rows = _read_csv(bundle / artifacts["samples"]["path"], SAMPLE_COLUMNS)
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
    expected = {
        "candidates": len(sample_rows),
        "attempted": sum(value in ("processed_no_observations", "contributed", "failed") for value in outcomes),
        "processed": sum(value in ("processed_no_observations", "contributed") for value in outcomes),
        "contributing": outcomes.count("contributed"),
        "failed": outcomes.count("failed"),
        "excluded": outcomes.count("excluded"),
        "not_attempted": outcomes.count("not_attempted"),
        "processed_no_observations": outcomes.count("processed_no_observations"),
        "observations": len(rows),
    }
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
        digest = hashlib.sha256(
            json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        if dataset.get(field) != digest:
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
    return result


def iter_measurements(root):
    """Yield catalog metadata records without loading model code or reading table payloads."""
    measurements = Path(root) / "measurements"
    if not measurements.is_dir():
        return
    for bundle in sorted(measurements.iterdir(), key=lambda path: path.name):
        if bundle.is_dir() and (bundle / "result.json").is_file():
            yield validate_result(_read_json(bundle / "result.json"))
