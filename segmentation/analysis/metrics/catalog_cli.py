"""Metadata-only query and explicit derived-export commands for the metric catalog."""

import argparse
import csv
import json
from pathlib import Path
import sys

from segmentation.analysis.metrics.catalog import (
    CatalogError,
    load_measurement,
    resolve_catalog_root,
    validate_result,
)
from segmentation.analysis.metrics.summaries import SummaryError, summarize_observations
from segmentation.analysis.metrics.visualizations import render_measurements


def configured_catalog_root(checkout=None):
    """Resolve the environment/.env setting without exposing a CLI root override."""
    return resolve_catalog_root(checkout=checkout)


def _read_metadata(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CatalogError(f"Cannot read measurement metadata {path}: {error}") from error


def _load_record(bundle):
    return validate_result(_read_metadata(bundle / "result.json"))


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


def _matches(record, *, metric=None, dataset=None, quantile=None, origin=None, statuses=None,
             checkpoint=None, provenance_gap=None, variants=()):
    if metric and record.get("metric") != metric:
        return False
    if dataset:
        dataset_name = _get_path(record, "dataset.name") or _get_path(record, "dataset.dataset")
        if dataset_name != dataset:
            return False
    if origin and record.get("origin") != origin:
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
    if provenance_gap:
        quality = record.get("provenance_quality", {})
        gaps = quality.get("missing_fields", []) + quality.get("warnings", [])
        if not any(provenance_gap.lower() in str(gap).lower() for gap in gaps):
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
                 origin=None, checkpoint=None, provenance_gap=None, variants=()):
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
            record, metric=metric, dataset=dataset, quantile=quantile, origin=origin,
            statuses=selected_statuses, checkpoint=checkpoint, provenance_gap=provenance_gap,
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
        matches = [
            path for path in (root / "measurements").iterdir()
            if path.is_dir() and _read_metadata(path / "result.json").get("measurement_id") == measurement_id
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


def ensure_external_output(path, root=None):
    """Reject any derived output inside canonical measurement storage."""
    target = Path(path).expanduser().resolve()
    root = Path(root) if root is not None else configured_catalog_root()
    canonical = (root / "measurements").resolve()
    if target == canonical or canonical in target.parents:
        raise CatalogError(f"Derived exports must be outside catalog measurements: {target}")
    return target


def export_records(records, output, root=None):
    output = ensure_external_output(output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    flattened = [_flatten(record) for record in records]
    keys = sorted({key for row in flattened for key in row} | {"metadata_json"})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for record, row in zip(records, flattened):
            row["metadata_json"] = json.dumps(record, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    return output


def _artifact_path(bundle, result, role):
    artifact = result.get("artifacts", {}).get(role)
    if not isinstance(artifact, dict) or not artifact.get("path"):
        raise CatalogError(f"Measurement has no {role} artifact")
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"Artifact path must be relative to its measurement: {role}")
    path = (bundle / relative).resolve()
    if bundle.resolve() not in path.parents:
        raise CatalogError(f"Artifact path escapes measurement: {role}")
    if not path.is_file():
        raise CatalogError(f"Missing measurement artifact: {path}")
    return path


def read_observations(bundle, result):
    """Read observation rows only for explicit aggregation/plot operations."""
    bundle = Path(bundle)
    result = load_measurement(bundle)
    path = _artifact_path(bundle, result, "observations")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise CatalogError(f"Observation table has no header: {path}")
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CatalogError(f"Cannot read observations from {path}: {error}") from error


def validate_measurement(bundle):
    """Validate native bundles fully and report legacy partial-evidence limitations."""
    bundle = Path(bundle)
    result = load_measurement(bundle)
    if result.get("origin") == "legacy":
        needs = list(result.get("provenance_quality", {}).get("missing_fields", []))
        artifacts = result.get("artifacts", {})
        for role, artifact in artifacts.items():
            if artifact.get("bytes") is None or artifact.get("sha256") is None:
                needs.append(f"artifact integrity metadata is incomplete: {role}")
        coverage = result.get("dataset", {}).get("identity_coverage", {})
        membership_complete = (
            isinstance(coverage, dict) and coverage.get("candidate_membership") == "complete"
        )
        if "samples" not in artifacts or not membership_complete:
            needs.append("candidate membership/order is unknown")
        if any("candidate membership/order" in str(item).lower() for item in needs):
            needs.append("candidate membership/order is unknown")
        needs = list(dict.fromkeys(needs))
        return {
            "measurement_id": result["measurement_id"],
            "status": "needs-attention" if needs else "valid",
            "origin": "legacy",
            "needs": needs,
            "diagnostics": [
                "Legacy metadata is intentionally not promoted to native completeness.",
                "Partial sample rows are validated but are not reconciled as a complete candidate population.",
            ],
        }
    return {
        "measurement_id": result["measurement_id"],
        "status": "valid",
        "origin": result.get("origin", "native"),
        "needs": [],
        "diagnostics": [],
    }


def export_summary(result, summary, output_dir, root=None):
    output_dir = ensure_external_output(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
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
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                })
    return output_dir


def _filter_arguments(parser):
    parser.add_argument("--all-status", action="store_true", help="Include running, failed, and complete records.")
    parser.add_argument("--status", action="append", choices=("running", "complete", "failed"))
    parser.add_argument("--metric")
    parser.add_argument("--dataset")
    parser.add_argument("--quantile")
    parser.add_argument("--origin", choices=("native", "legacy"))
    parser.add_argument("--checkpoint", help="Substring of model label, checkpoint path, or SHA-256.")
    parser.add_argument("--provenance-gap", help="Substring of a missing provenance field or warning.")
    parser.add_argument("--variant", action="append", default=[], metavar="DOTTED.PATH=VALUE")


def _record_id_map(root, measurement_ids):
    bundles = []
    for measurement_id in measurement_ids:
        bundle, result = find_measurement(root, measurement_id)
        bundles.append((bundle, result))
    if len({record.get("measurement_id") for _, record in bundles}) != len(bundles):
        raise CatalogError("Measurement IDs must be unique; attempts are never implicitly pooled")
    return bundles


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
    plot = commands.add_parser("plot", help="Plot explicitly selected consistency measurements.")
    plot.add_argument("measurement_ids", nargs="+")
    plot.add_argument("--output-dir", required=True, type=Path)
    return parser


def _filters(args):
    return {
        "all_status": args.all_status,
        "statuses": args.status,
        "metric": args.metric,
        "dataset": args.dataset,
        "quantile": args.quantile,
        "origin": args.origin,
        "checkpoint": args.checkpoint,
        "provenance_gap": args.provenance_gap,
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
            bundle, result = find_measurement(root, args.measurement_id)
            status = result.get("status")
            if status == "running":
                raise CatalogError("Running measurements cannot be summarized")
            if status == "failed" and not args.all_status:
                raise CatalogError(
                    "Summaries default to complete measurements; pass --all-status "
                    "to inspect a failed record"
                )
            rows = read_observations(bundle, result)
            summary = summarize_observations(result, rows)
            if args.output_dir:
                print(export_summary(result, summary, args.output_dir, root))
            else:
                print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "plot":
            bundles = _record_id_map(root, args.measurement_ids)
            if any(record.get("metric") != "consistency" for _, record in bundles):
                raise SummaryError("The visualization command accepts consistency measurements only")
            summaries = [
                summarize_observations(record, read_observations(bundle, record))
                for bundle, record in bundles
            ]
            rendered = render_measurements(
                [record for _, record in bundles], summaries, args.output_dir, root=root
            )
            print(json.dumps({
                "output_dir": str(rendered["output_dir"]),
                "support_comparable": rendered["support_comparable"],
            }))
            return 0
    except (CatalogError, SummaryError, OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
