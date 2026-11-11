"""Reviewed, dry-run-first linking of historical measurements to registered models."""

import copy
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path

from segmentation.catalog import CatalogError, iter_measurements, load_measurement, validate_result
from segmentation.catalog.cli import find_measurement
from segmentation.catalog.models import find_model, model_snapshot


def migration_plan(root):
    measurements = {}
    for record in iter_measurements(root):
        if record.get("model", {}).get("registration_id"):
            continue
        description = (record.get("legacy_metadata") or {}).get("description") or {}
        measurements[record["measurement_id"]] = {
            "model": None,
            "grouping_model": None,
            "confirm_historical_identity": False,
            "historical_model_label": record["model"].get("label"),
            "historical_checkpoint": description.get("checkpoint"),
            "historical_group_checkpoint": description.get("group_checkpoint"),
            "historical_parameters": description.get("parameters", record["parameters"]),
        }
    return {"measurements": measurements}


def _linked_record(record, entry, root, cache):
    def resolve(reference):
        if reference not in cache:
            cache[reference] = find_model(reference, root, verify=True)
        return cache[reference]

    main = resolve(entry.get("model"))
    group = resolve(entry["grouping_model"]) if entry.get("grouping_model") else None
    old_inputs = record.get("model", {}).get("inputs", {})
    description = (record.get("legacy_metadata") or {}).get("description") or {}
    if main["dataset"] != (record.get("dataset", {}).get("name") or description.get("dataset")):
        raise CatalogError(f"Dataset does not match registered model: {record['measurement_id']}")
    if group and group["dataset"] != main["dataset"]:
        raise CatalogError("Main and grouping models must use the same dataset")
    historical_group = old_inputs.get("group_checkpoint") or {}
    if (historical_group.get("sha256") or description.get("group_checkpoint")) and group is None:
        raise CatalogError(f"Historical grouping model must be supplied: {record['measurement_id']}")
    for role, registration in (("prototype_checkpoint", main), ("group_checkpoint", group)):
        if registration is None:
            continue
        old_hash = (old_inputs.get(role) or {}).get("sha256")
        if old_hash:
            if old_hash != registration["checkpoint"]["sha256"]:
                raise CatalogError(f"Historical checkpoint hash mismatch: {record['measurement_id']}/{role}")
        elif entry.get("confirm_historical_identity") is not True:
            raise CatalogError(f"Explicit historical-identity confirmation required: {record['measurement_id']}")
    result = copy.deepcopy(record)
    result["model"] = model_snapshot(main)
    result["grouping_model"] = model_snapshot(group) if group else None
    if group:
        result["model"]["inputs"]["group_checkpoint"] = group["checkpoint"]
    for parameter in ("quantile_scope", "merge_left_right_parts"):
        value = description.get("parameters", {}).get(parameter)
        if value is not None:
            current = result["parameters"].get(parameter)
            if current is not None and current != value:
                raise CatalogError(f"Conflicting historical parameter: {record['measurement_id']}/{parameter}")
            result["parameters"][parameter] = value
            if parameter == "quantile_scope":
                result["protocol"]["quantile_scope"] = value
            else:
                result["protocol"].setdefault("parts", {})[parameter] = value
    result["registration_migration"] = {
        "original_model": record["model"],
        "historical_identity_confirmed_by_user": entry.get("confirm_historical_identity") is True,
    }
    return validate_result(result)


def migrate(root, manifest, *, apply=False):
    root = Path(root).resolve()
    entries = manifest.get("measurements")
    if not isinstance(entries, dict) or not entries:
        raise CatalogError("Manifest must contain a nonempty measurements mapping")
    cache, changes = {}, []
    for identifier, entry in entries.items():
        if not isinstance(entry, dict):
            raise CatalogError(f"Invalid manifest entry: {identifier}")
        bundle, record = find_measurement(root, identifier)
        if bundle.is_symlink() or bundle.resolve().parent != root / "measurements":
            raise CatalogError(f"Out-of-catalog bundle: {identifier}")
        if record["status"] == "running" or record.get("model", {}).get("registration_id"):
            raise CatalogError(f"Measurement is running or already linked: {identifier}")
        load_measurement(bundle)
        original = (bundle / "result.json").read_bytes()
        changes.append((bundle, original, _linked_record(record, entry, root, cache)))
    if not apply:
        return {"status": "dry_run", "measurements": len(changes)}
    backup = root / "backups" / f"registration-{uuid.uuid4()}"
    with (root / ".registration-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for bundle, original, _ in changes:
            if (bundle / "result.json").read_bytes() != original:
                raise CatalogError(f"Measurement changed during migration: {bundle.name}")
        backup.mkdir(parents=True)
        for bundle, original, _ in changes:
            with (backup / f"{bundle.name}.json").open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
        receipt = {"status": "applying", "backup": str(backup), "applied": []}

        def save_receipt():
            temporary = backup / ".receipt.json.tmp"
            with temporary.open("w") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(backup / "receipt.json")

        save_receipt()
        try:
            for bundle, original, result in changes:
                path = bundle / "result.json"
                if path.read_bytes() != original:
                    raise CatalogError(f"Measurement changed during migration: {bundle.name}")
                temporary = bundle / f".registration-{uuid.uuid4()}.json"
                try:
                    with temporary.open("x") as stream:
                        json.dump(result, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                receipt["applied"].append(
                    {
                        "measurement_id": bundle.name,
                        "original_sha256": hashlib.sha256(original).hexdigest(),
                    }
                )
                save_receipt()
            receipt["status"] = "complete"
        except BaseException as error:
            receipt.update(status="failed", error=str(error))
            raise
        finally:
            save_receipt()
    return receipt
