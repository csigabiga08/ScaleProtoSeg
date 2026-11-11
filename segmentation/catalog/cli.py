"""Metadata-only query and validation commands for the metric catalog."""

import argparse
import json
from pathlib import Path
import sys

from segmentation.catalog import (
    CatalogError,
    load_measurement,
    read_result,
    resolve_catalog_root,
    validate_result,
)


def configured_catalog_root(checkout=None):
    """Resolve the environment/.env setting without exposing a CLI root override."""
    return resolve_catalog_root(checkout=checkout)


def _load_record(bundle):
    return validate_result(read_result(bundle / "result.json"))


def _get_path(value, path):
    current = value
    for component in path.split("."):
        if isinstance(current, dict):
            current = current.get(component)
        else:
            return None
    return current


def _quantile(record):
    protocol = record.get("protocol", {})
    if isinstance(protocol, dict) and "quantile" in protocol:
        return protocol["quantile"]
    return record.get("parameters", {}).get("quantile")


def _checkpoint_terms(record):
    model = record.get("model", {})
    parts = [model.get("label"), model.get("inputs", {})]
    return " ".join(str(value) for value in parts if value is not None).lower()


def _matches(record, *, metric=None, dataset=None, quantile=None, statuses=None, checkpoint=None, variants=()):
    if metric and record.get("metric") != metric:
        return False
    if dataset and _get_path(record, "dataset.name") != dataset:
        return False
    if statuses and record.get("status") not in statuses:
        return False
    if quantile is not None:
        current = _quantile(record)
        try:
            if current is None or float(current) != float(quantile):
                return False
        except (TypeError, ValueError):
            return False
    if checkpoint and checkpoint.lower() not in _checkpoint_terms(record):
        return False
    for expression in variants:
        if "=" not in expression:
            raise CatalogError(f"Variant filter must be dotted.path=value: {expression!r}")
        key, expected = expression.split("=", 1)
        actual = _get_path(record, key)
        if actual is None:
            actual = _get_path(record, "parameters." + key)
        if str(actual).lower() != expected.lower():
            return False
    return True


def find_records(
    root=None,
    *,
    all_status=False,
    statuses=None,
    metric=None,
    dataset=None,
    quantile=None,
    checkpoint=None,
    variants=(),
):
    """Read and filter only result.json files; do not open table/checkpoint payloads."""
    root = Path(root) if root is not None else configured_catalog_root()
    measurements = root / "measurements"
    if not measurements.is_dir():
        return []
    selected_statuses = set(statuses or ())
    if not all_status and selected_statuses - {"complete"}:
        raise CatalogError("Selecting running or failed records requires --all-status")
    if not all_status and not selected_statuses:
        selected_statuses.add("complete")
    records = []
    for bundle in sorted(measurements.iterdir(), key=lambda path: path.name):
        if not bundle.is_dir() or not (bundle / "result.json").is_file():
            continue
        try:
            record = _load_record(bundle)
        except CatalogError as error:
            print(f"warning: skipping {bundle.name}: {error}", file=sys.stderr)
            continue
        if _matches(
            record,
            metric=metric,
            dataset=dataset,
            quantile=quantile,
            statuses=selected_statuses,
            checkpoint=checkpoint,
            variants=variants,
        ):
            records.append(record)
    return records


def find_measurement(root, measurement_id):
    root = Path(root)
    measurement_id = str(measurement_id)
    if measurement_id in ("", ".", "..") or Path(measurement_id).name != measurement_id:
        raise CatalogError(f"Invalid measurement ID: {measurement_id!r}")
    bundle = root / "measurements" / measurement_id
    if not bundle.is_dir():
        matches = (
            [
                path
                for path in (root / "measurements").iterdir()
                if path.is_dir() and read_result(path / "result.json").get("measurement_id") == measurement_id
            ]
            if (root / "measurements").is_dir()
            else []
        )
        if len(matches) != 1:
            raise CatalogError(f"Measurement ID not found or ambiguous: {measurement_id}")
        bundle = matches[0]
    record = _load_record(bundle)
    return bundle, record


def validate_measurement(bundle):
    """Validate metadata, artifact integrity and observation counts; return the verdict."""
    try:
        return {"measurement_id": load_measurement(bundle)["measurement_id"], "status": "valid"}
    except CatalogError as error:
        return {"measurement_id": Path(bundle).name, "status": "invalid", "error": str(error)}


def _filter_arguments(parser):
    parser.add_argument("--all-status", action="store_true", help="Include running, failed, and complete records.")
    parser.add_argument("--status", action="append", choices=("running", "complete", "failed"))
    parser.add_argument("--metric")
    parser.add_argument("--dataset")
    parser.add_argument("--quantile")
    parser.add_argument("--checkpoint", help="Substring of model label, checkpoint path, or SHA-256.")
    parser.add_argument("--variant", action="append", default=[], metavar="DOTTED.PATH=VALUE")


def _command_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List metadata (default: complete measurements only).")
    _filter_arguments(listing)
    validate = commands.add_parser("validate", help="Validate selected metadata and canonical artifacts.")
    validate.add_argument("measurement_ids", nargs="*")
    validate.add_argument("--all-status", action="store_true")
    commands.add_parser("architectures", help="Print source-defined architecture JSON schemas.")
    commands.add_parser("list-models", help="List registered checkpoint metadata.")
    register = commands.add_parser("register-model", help="Register one checkpoint; reject duplicate names or hashes.")
    register.add_argument("--name", required=True)
    register.add_argument("--architecture", required=True)
    register.add_argument("--schema-version", type=int, default=1)
    register.add_argument("--config", type=Path, required=True, help="Architecture configuration JSON file.")
    register.add_argument("--dataset", required=True)
    register.add_argument("--seed", type=int)
    register.add_argument("--tags", default="{}", help="JSON object of universal string tags.")
    register.add_argument("--checkpoint", required=True)
    relocate = commands.add_parser("relocate-model", help="Update a location only if checkpoint contents match.")
    relocate.add_argument("model")
    relocate.add_argument("checkpoint")
    commands.add_parser("migration-plan", help="Print a manifest requiring reviewed model references.")
    migration = commands.add_parser("migrate", help="Validate a reviewed manifest; writes require --apply.")
    migration.add_argument("manifest", type=Path)
    migration.add_argument("--apply", action="store_true", help="Back up original metadata before rewriting it.")
    return parser


def _filters(args):
    return {
        "all_status": args.all_status,
        "statuses": args.status,
        "metric": args.metric,
        "dataset": args.dataset,
        "quantile": args.quantile,
        "checkpoint": args.checkpoint,
        "variants": args.variant,
    }


def main(argv=None):
    parser = _command_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "architectures":
            from segmentation.catalog.architectures import ARCHITECTURES

            print(
                json.dumps(
                    {
                        f"{name}/{version}": schema.model_json_schema()
                        for (name, version), schema in ARCHITECTURES.items()
                    },
                    indent=2,
                )
            )
            return 0
        root = configured_catalog_root()
        if args.command in ("list-models", "register-model", "relocate-model"):
            from segmentation.catalog.models import list_models, register_model, relocate_model

            if args.command == "list-models":
                result = list_models(root)
            elif args.command == "register-model":
                result = register_model(
                    root=root,
                    name=args.name,
                    architecture=args.architecture,
                    schema_version=args.schema_version,
                    config=read_result(args.config),
                    dataset=args.dataset,
                    seed=args.seed,
                    tags=json.loads(args.tags),
                    checkpoint=args.checkpoint,
                )
            else:
                result = relocate_model(args.model, args.checkpoint, root)
            print(json.dumps(result, indent=2))
            return 0
        if args.command in ("migration-plan", "migrate"):
            from segmentation.catalog.migrate import migrate, migration_plan

            result = (
                migration_plan(root)
                if args.command == "migration-plan"
                else migrate(
                    root,
                    read_result(args.manifest),
                    apply=args.apply,
                )
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "list":
            records = find_records(root, **_filters(args))
            print(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "validate":
            selected = (
                [find_measurement(root, measurement_id)[0] for measurement_id in args.measurement_ids]
                if args.measurement_ids
                else [
                    root / "measurements" / record["measurement_id"]
                    for record in find_records(root, all_status=args.all_status)
                ]
            )
            diagnostics = [validate_measurement(bundle) for bundle in selected]
            print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
            return 0 if all(item["status"] == "valid" for item in diagnostics) else 1
    except (CatalogError, OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
