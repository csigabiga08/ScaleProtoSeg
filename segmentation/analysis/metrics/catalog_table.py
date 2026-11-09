"""Read-only tabular views of catalog metadata for inspection, comparison and aggregation."""

import json
import math
import os
from pathlib import Path

from dotenv import dotenv_values
import pandas as pd

from segmentation.analysis.metrics.catalog import CatalogError, read_result, validate_result


MISSING = "∅"
IDENTITY_INPUTS = (
    "metric",
    "dataset.name",
    "model.label",
    "model.training_stage",
    "model.inputs.group_checkpoint.path",
)
SCORE_COLUMNS = ("score", "numerator", "denominator")
# Large lookup tables that are identical per dataset and only clutter field pickers.
EXCLUDED_PREFIXES = (
    "artifacts.",
    "protocol.classes.",
    "protocol.parts.left_right_part_groups.",
    "legacy_metadata.description.left_right_part_groups.",
)


def read_catalog_root(checkout=None):
    """Resolve MEASUREMENT_CATALOG_ROOT like the writers do, but require read access only."""
    checkout = Path(checkout) if checkout is not None else Path(__file__).resolve().parents[3]
    configured = os.environ.get("MEASUREMENT_CATALOG_ROOT") or dotenv_values(checkout / ".env").get(
        "MEASUREMENT_CATALOG_ROOT"
    )
    if not configured:
        raise CatalogError("Set MEASUREMENT_CATALOG_ROOT to an existing catalog root")
    root = Path(configured).expanduser().resolve()
    if not (root / "measurements").is_dir():
        raise CatalogError(f"No measurements directory under catalog root: {root}")
    return root


def catalog_signature(root):
    """Cheap change detector: bundle names with their result.json modification times."""
    signature = []
    for bundle in sorted((Path(root) / "measurements").iterdir()):
        result = bundle / "result.json"
        if result.is_file():
            signature.append((bundle.name, result.stat().st_mtime_ns))
    return tuple(signature)


def load_records(root):
    """Return (valid records, [(bundle name, error)]) without opening observation tables."""
    records, errors = [], []
    for bundle in sorted((Path(root) / "measurements").iterdir()):
        if not bundle.is_dir() or not (bundle / "result.json").is_file():
            continue
        try:
            records.append(validate_result(read_result(bundle / "result.json")))
        except CatalogError as error:
            errors.append((bundle.name, str(error)))
    return records, errors


def flatten_record(record, prefix=""):
    """Flatten nested metadata into dotted paths; lists stay whole as one value."""
    flat = {}
    for key, value in record.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_record(value, path + "."))
        elif not any(f"{path}.".startswith(excluded) for excluded in EXCLUDED_PREFIXES):
            flat[path] = value
    return flat


def display_value(value):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _number(value):
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def records_frame(records):
    """One row per measurement: numeric score columns plus every flattened field as text."""
    rows = []
    for record in records:
        row = {path: display_value(value) for path, value in flatten_record(record).items()}
        score = (record.get("scores") or {}).get(record.get("metric")) or {}
        row["score"] = _number(score.get("value"))
        row["numerator"] = _number(score.get("numerator"))
        row["denominator"] = _number(score.get("denominator"))
        row["score_reason"] = score.get("reason") or ""
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in IDENTITY_INPUTS + ("measurement_id", "status"):
        if column not in frame:
            frame[column] = MISSING
    for column in SCORE_COLUMNS:
        if column not in frame:
            frame[column] = math.nan
    for column in frame.columns:
        if column not in SCORE_COLUMNS:
            frame[column] = frame[column].fillna(MISSING).astype(object)
    return frame.set_index("measurement_id", drop=False)


def input_columns(frame):
    """Run inputs: model/dataset identity followed by every recorded parameter."""
    parameters = sorted(column for column in frame.columns if column.startswith("parameters."))
    return list(IDENTITY_INPUTS) + parameters


def varying_columns(frame, columns):
    return [column for column in columns if frame[column].nunique(dropna=False) > 1]


def filter_frame(frame, selections):
    """Keep rows whose column value is in the selected values; empty selections keep all."""
    mask = pd.Series(True, index=frame.index)
    for column, values in selections.items():
        if values:
            mask &= frame[column].isin(list(values))
    return frame[mask]


def differing_fields(frame, measurement_ids, columns):
    """Fields (rows) that differ across the chosen measurements (columns), with scores on top."""
    subset = frame.loc[list(measurement_ids)]
    differing = [column for column in columns if subset[column].astype(str).nunique() > 1]
    table = subset[list(SCORE_COLUMNS) + differing].T
    table.columns = [str(measurement_id)[:8] for measurement_id in subset.index]
    return table


def aggregate(frame, by):
    """Score statistics per group; metric is always a group key so metrics never mix.

    ``mean`` averages run scores; ``pooled`` is sum(numerator) / sum(denominator) and is
    undefined when any run in the group lacks either count.
    """
    by = list(dict.fromkeys(["metric", *by]))
    work = frame[by + list(SCORE_COLUMNS)].copy()
    work["_incomplete"] = work["numerator"].isna() | work["denominator"].isna()
    grouped = work.groupby(by, sort=True)
    result = grouped["score"].agg(["count", "mean", "std", "min", "max"])
    result.insert(0, "runs", grouped.size())
    numerator = grouped["numerator"].sum()
    denominator = grouped["denominator"].sum()
    defined = ~grouped["_incomplete"].any() & (denominator > 0)
    result["pooled"] = (numerator / denominator).where(defined)
    return result.rename(columns={"count": "scored"}).reset_index()


def pivot_means(frame, rows, column):
    """Mean run score per (rows, column value), plus the unweighted mean across the column."""
    rows = list(dict.fromkeys(["metric", *[row for row in rows if row != column]]))
    table = aggregate(frame, rows + [column]).pivot_table(
        index=rows, columns=column, values="mean", aggfunc="first"
    )
    table.columns = [str(value) for value in table.columns]
    table[f"mean over {column}"] = table.mean(axis=1, skipna=False)
    return table.reset_index()
