"""Locate, validate, and read metric measurement bundles."""

import csv
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from segmentation.catalog.schema import Measurement


OBSERVATION_KEY = ("sample_id", "class_id", "prototype_id", "repetition")


class CatalogError(RuntimeError):
    """Raised when a catalog root or measurement bundle violates its contract."""


def resolve_catalog_root(checkout=None):
    """Resolve the required catalog root from the environment or this checkout's .env."""
    checkout = Path(checkout) if checkout is not None else Path(__file__).resolve().parents[2]
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


def validate_result(result):
    """Validate the metadata only; callers keep the original dictionary unchanged."""
    try:
        Measurement.model_validate(result)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors(include_url=False, include_input=False)
        )
        raise CatalogError(f"Invalid measurement metadata: {details}") from error
    return result


def _read_observations(path, columns):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise CatalogError(f"Observation columns disagree with the protocol in {path}")
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CatalogError(f"Cannot read observations {path}: {error}") from error


def _load(bundle, verify_hashes):
    bundle = Path(bundle)
    result = validate_result(read_result(bundle / "result.json"))
    artifact = result["artifacts"].get("observations")
    if result["status"] == "running" or artifact is None:
        return result, None
    relative = Path(artifact["path"])
    path = (bundle / relative).resolve()
    if relative.is_absolute() or bundle.resolve() not in path.parents or not path.is_file():
        raise CatalogError(f"Missing or out-of-bundle observations artifact: {bundle}")
    if path.stat().st_size != artifact["bytes"]:
        raise CatalogError(f"Artifact byte size mismatch: {path}")
    if verify_hashes and sha256_file(path)[0] != artifact["sha256"]:
        raise CatalogError(f"Artifact hash mismatch: {path}")
    columns = list(OBSERVATION_KEY) + list(result["protocol"].get("observation_columns", []))
    rows = _read_observations(path, columns)
    population = result.get("population") or {}
    if population.get("observations", len(rows)) != len(rows):
        raise CatalogError("Observation count disagrees with observations.csv")
    contributing = len({row["sample_id"] for row in rows})
    if population.get("contributing", contributing) != contributing:
        raise CatalogError("Contributing sample count disagrees with observations.csv")
    return result, rows


def load_measurement(bundle, *, verify_hashes=True):
    """Load validated metadata, checking the observations artifact when one is finalized."""
    return _load(bundle, verify_hashes)[0]


def load_observations(bundle, *, verify_hashes=True):
    """Load validated metadata together with its finalized observation rows."""
    result, rows = _load(bundle, verify_hashes)
    if result["status"] == "running":
        raise CatalogError("Running measurements have no finalized observation evidence")
    if rows is None:
        raise CatalogError("Measurement has no finalized observation evidence")
    return result, rows


def iter_measurements(root):
    """Yield catalog metadata records without reading observation tables."""
    measurements = Path(root) / "measurements"
    if not measurements.is_dir():
        return
    for bundle in sorted(measurements.iterdir(), key=lambda path: path.name):
        if bundle.is_dir() and (bundle / "result.json").is_file():
            yield validate_result(read_result(bundle / "result.json"))
