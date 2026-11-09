"""Metadata-only query and explicit derived-export commands for the metric catalog."""

import argparse
import csv
import json
from pathlib import Path
import sys

from segmentation.analysis.metrics.catalog import (
    CatalogError,
    create_fresh_output_dir,
    ensure_external_output,
    load_measurement,
    load_observations,
    read_result,
    resolve_catalog_root,
    validate_result,
    write_derived_csv,
)
from segmentation.analysis.metrics.summaries import SummaryError, summarize_observations


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
    return " ".join(
        str(value)
        for value in parts
        if value is not None
    ).lower()


def _matches(record, *, metric=None, dataset=None, quantile=None, statuses=None,
             checkpoint=None, variants=()):
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


def find_records(root=None, *, all_status=False, statuses=None, metric=None, dataset=None, quantile=None,
                 checkpoint=None, variants=()):
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
            record, metric=metric, dataset=dataset, quantile=quantile,
            statuses=selected_statuses, checkpoint=checkpoint, variants=variants,
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
        matches = [
            path for path in (root / "measurements").iterdir()
            if path.is_dir() and read_result(path / "result.json").get("measurement_id") == measurement_id
        ] if (root / "measurements").is_dir() else []
        if len(matches) != 1:
            raise CatalogError(f"Measurement ID not found or ambiguous: {measurement_id}")
        bundle = matches[0]
    record = _load_record(bundle)
    return bundle, record


def _flatten(value, prefix=""):
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, name))
    elif isinstance(value, list):
        flattened[prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        flattened[prefix] = value
    return flattened


def export_records(records, output, root=None):
    output = ensure_external_output(output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    flattened = [_flatten(record) for record in records]
    keys = sorted({key for row in flattened for key in row})
    try:
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flattened)
    except FileExistsError as error:
        raise CatalogError(f"Derived export file already exists: {output}") from error
    return output


def validate_measurement(bundle):
    """Validate metadata, artifact integrity and observation counts; return the verdict."""
    try:
        return {"measurement_id": load_measurement(bundle)["measurement_id"], "status": "valid"}
    except CatalogError as error:
        return {"measurement_id": Path(bundle).name, "status": "invalid", "error": str(error)}


def export_summary(result, summary, output_dir, root=None):
    output_dir = create_fresh_output_dir(output_dir, root)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tables = {
        "classes.csv": summary.get("classes", []),
        "prototypes.csv": summary.get("prototypes", []),
        "parts.csv": summary.get("parts", []),
        "support.csv": summary.get("support", []),
        "stability_rows.csv": summary.get("stability_rows", []),
    }
    for name, rows in tables.items():
        if not rows:
            continue
        columns = list(rows[0])
        write_derived_csv(output_dir / name, rows, columns)
    return output_dir


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
    for name in ("list", "filter"):
        listing = commands.add_parser(name, help="List metadata (default: complete measurements only).")
        _filter_arguments(listing)
    export = commands.add_parser("export", help="Export filtered metadata to a flat comparison CSV.")
    _filter_arguments(export)
    export.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate", help="Validate selected metadata and canonical artifacts.")
    validate.add_argument("measurement_ids", nargs="*")
    validate.add_argument("--all-status", action="store_true")
    summarize = commands.add_parser("summarize", help="Recompute one measurement's metric from observations.")
    summarize.add_argument("measurement_id")
    summarize.add_argument("--output-dir", type=Path)
    summarize.add_argument(
        "--all-status",
        action="store_true",
        help="Allow summaries of failed records with finalized observations.",
    )
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
        root = configured_catalog_root()
        if args.command in ("list", "filter", "export"):
            records = find_records(root, **_filters(args))
            if args.command == "export":
                print(export_records(records, args.output, root))
            else:
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
        if args.command == "summarize":
            bundle, _discovered_result = find_measurement(root, args.measurement_id)
            result, rows = load_observations(bundle)
            status = result.get("status")
            if status == "failed" and not args.all_status:
                raise CatalogError(
                    "Summaries default to complete measurements; pass --all-status "
                    "to inspect a failed record"
                )
            summary = summarize_observations(result, rows)
            if args.output_dir:
                print(export_summary(result, summary, args.output_dir, root))
            else:
                print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
    except (CatalogError, SummaryError, OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
