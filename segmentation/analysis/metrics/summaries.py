"""Recompute metric summaries from catalog observation rows."""

import math
import re
from collections import defaultdict


class SummaryError(ValueError):
    """Raised when observations cannot be interpreted by a known metric protocol."""


_PART_COLUMN = re.compile(r"^part_(\d+)$")
_NOISY_PART_COLUMN = re.compile(r"^noisy_part_(\d+)$")
_CONSISTENCY_AGGREGATIONS = {"unweighted mean over class/prototype rows"}
_STABILITY_AGGREGATIONS = {
    "unweighted mean of exact all-part equality per paired sample/class/prototype row",
}
_CONSISTENCY_COMPARISONS = {
    "mean part presence > threshold",
    "strictly greater than threshold",
}
_STABILITY_MISSING_RULES = {
    "zero versus zero for parts without annotated centroids",
    "nan compares unequal, including nan versus nan",
}


def _protocol_info(result):
    metric = result.get("metric")
    if metric not in ("consistency", "stability"):
        raise SummaryError(f"Unsupported metric for summary: {metric!r}")
    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise SummaryError("Measurement has no protocol metadata")
    section = protocol.get(metric)
    if not isinstance(section, dict):
        raise SummaryError(f"Measurement has no {metric} protocol section")
    aggregation = section.get("aggregation")
    expected = _CONSISTENCY_AGGREGATIONS if metric == "consistency" else _STABILITY_AGGREGATIONS
    if aggregation not in expected:
        raise SummaryError(f"Unsupported {metric} aggregation protocol: {aggregation!r}")
    if metric == "consistency":
        comparison = section.get("strict_comparison")
        if comparison not in _CONSISTENCY_COMPARISONS:
            raise SummaryError(f"Unsupported consistency threshold comparison: {comparison!r}")
        threshold = section.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise SummaryError("Consistency protocol must record a finite threshold")
        return metric, protocol, section, float(threshold)

    missing_rule = section.get("missing_value_comparison")
    if missing_rule not in _STABILITY_MISSING_RULES:
        raise SummaryError(f"Unsupported stability missing-value comparison: {missing_rule!r}")
    return metric, protocol, section, None


def _columns(protocol, rows, pattern):
    declared = protocol.get("observation_columns", [])
    names = set(declared)
    for row in rows:
        names.update(row)
    found = {}
    for name in names:
        match = pattern.fullmatch(str(name))
        if match:
            found[int(match.group(1))] = str(name)
    return found


def _reject_unrecorded_part_columns(protocol, rows):
    declared = set(protocol.get("observation_columns", []))
    for row in rows:
        for column in row:
            is_part = (
                _PART_COLUMN.fullmatch(str(column))
                or _NOISY_PART_COLUMN.fullmatch(str(column))
            )
            if is_part and column not in declared:
                raise SummaryError(f"Observation contains an unrecorded part column: {column}")


def _presence(value, column, row_index):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "null", "none"):
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError) as error:
        raise SummaryError(f"Invalid presence value in row {row_index}, {column}: {value!r}") from error
    if not math.isfinite(parsed):
        return None
    if parsed not in (0.0, 1.0):
        raise SummaryError(f"Presence values must be zero or one in row {row_index}, {column}")
    return int(parsed)


def _row_key(row, row_index):
    required = ("sample_id", "class_id", "prototype_id", "repetition")
    missing = [field for field in required if field not in row]
    if missing:
        raise SummaryError(f"Observation row {row_index} is missing keys: {', '.join(missing)}")
    try:
        return (
            str(row["sample_id"]),
            int(row["class_id"]),
            int(row["prototype_id"]),
            int(row["repetition"]),
        )
    except (TypeError, ValueError) as error:
        raise SummaryError(f"Invalid observation key in row {row_index}") from error


def _class_label(result, class_id):
    labels = result.get("protocol", {}).get("classes", {}).get("evaluation_class_id_to_label", {})
    return labels.get(str(class_id), labels.get(class_id))


def _saved_score(result):
    score = result.get("scores", {}).get(result.get("metric"))
    if not isinstance(score, dict):
        return None
    value = score.get("value")
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize_observations(result, observations):
    """Derive exact means, flags, support, and score without pooling incompatible rows."""
    metric, protocol, metric_protocol, threshold = _protocol_info(result)
    rows = list(observations)
    _reject_unrecorded_part_columns(protocol, rows)
    keys = set()
    clean_columns = _columns(protocol, rows, _PART_COLUMN)
    if metric == "consistency":
        valid = protocol.get("parts", {}).get("valid_part_ids")
        if not isinstance(valid, list) or not valid:
            raise SummaryError("Consistency protocol has no valid part IDs")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in valid):
            raise SummaryError("Consistency protocol has invalid valid_part_ids")
        valid_ids = {int(value) for value in valid}
        if len(valid_ids) != len(valid):
            raise SummaryError("Consistency protocol has duplicate valid part IDs")
        if not valid_ids <= set(clean_columns):
            raise SummaryError("Consistency protocol does not declare every valid part column")
        clean_columns = {part_id: name for part_id, name in clean_columns.items() if part_id in valid_ids}
        if not clean_columns:
            raise SummaryError("No consistency part columns are available")
        grouped = defaultdict(list)
        support = defaultdict(lambda: defaultdict(set))
        for index, row in enumerate(rows):
            key = _row_key(row, index)
            if key in keys:
                raise SummaryError(f"Duplicate observation key: {key}")
            keys.add(key)
            for part_id in clean_columns:
                support[key[1], part_id].setdefault(key[2], set())
            values = {}
            for part_id, column in clean_columns.items():
                value = _presence(row.get(column), column, index)
                values[part_id] = value
                if value is not None:
                    support[key[1], part_id][key[2]].add(key[0])
            grouped[key[1], key[2]].append((key, values))

        prototypes = []
        parts = []
        for (class_id, prototype_id), observations_for_proto in sorted(grouped.items()):
            per_part = {}
            for part_id in sorted(clean_columns):
                available = [
                    values[part_id]
                    for _, values in observations_for_proto
                    if values[part_id] is not None
                ]
                mean = sum(available) / len(available) if available else None
                per_part[part_id] = (mean, len(available))
                parts.append({
                    "class_id": class_id,
                    "class_label": _class_label(result, class_id),
                    "prototype_id": prototype_id,
                    "part_id": part_id,
                    "mean_presence": mean,
                    "available_observations": len(available),
                })
            best = max((value for value, _ in per_part.values() if value is not None), default=None)
            consistent = bool(best is not None and best > threshold)
            prototypes.append({
                "class_id": class_id,
                "class_label": _class_label(result, class_id),
                "prototype_id": prototype_id,
                "max_part_frequency": best,
                "consistent": consistent,
            })
        numerator = sum(row["consistent"] for row in prototypes)
        denominator = len(prototypes)
        support_rows = _support_rows(support)
        class_values = _class_summaries(prototypes, result)
        summary = {
            "measurement_id": result.get("measurement_id"),
            "metric": metric,
            "status": result.get("status"),
            "aggregation": metric_protocol["aggregation"],
            "quantile": protocol.get("quantile"),
            "threshold": threshold,
            "score": _score(numerator, denominator),
            "classes": class_values,
            "prototypes": prototypes,
            "parts": parts,
            "support": support_rows,
        }
    else:
        noisy_columns = _columns(protocol, rows, _NOISY_PART_COLUMN)
        if not clean_columns or set(clean_columns) != set(noisy_columns):
            raise SummaryError("Stability requires paired clean and noisy part columns")
        grouped = defaultdict(list)
        support = defaultdict(lambda: defaultdict(set))
        stability_rows = []
        for index, row in enumerate(rows):
            key = _row_key(row, index)
            if key in keys:
                raise SummaryError(f"Duplicate observation key: {key}")
            keys.add(key)
            for part_id in clean_columns:
                support[key[1], part_id].setdefault(key[2], set())
            equalities = []
            for part_id, column in sorted(clean_columns.items()):
                clean = _presence(row.get(column), column, index)
                noisy_name = noisy_columns[part_id]
                noisy = _presence(row.get(noisy_name), noisy_name, index)
                equalities.append(None if clean is None or noisy is None else clean == noisy)
                if clean is not None:
                    support[key[1], part_id][key[2]].add(key[0])
            stable = all(value is True for value in equalities)
            item = {
                "sample_id": key[0],
                "class_id": key[1],
                "class_label": _class_label(result, key[1]),
                "prototype_id": key[2],
                "repetition": key[3],
                "part_equalities": equalities,
                "stable": stable,
            }
            stability_rows.append(item)
            grouped[key[1], key[2]].append(item)

        prototypes = []
        for (class_id, prototype_id), items in sorted(grouped.items()):
            stable_count = sum(row["stable"] for row in items)
            prototypes.append({
                "class_id": class_id,
                "class_label": _class_label(result, class_id),
                "prototype_id": prototype_id,
                "stable_observations": stable_count,
                "observations": len(items),
                "value": stable_count / len(items) if items else None,
            })
        numerator = sum(row["stable"] for row in stability_rows)
        denominator = len(stability_rows)
        summary = {
            "measurement_id": result.get("measurement_id"),
            "metric": metric,
            "status": result.get("status"),
            "aggregation": metric_protocol["aggregation"],
            "quantile": protocol.get("quantile"),
            "score": _score(numerator, denominator),
            "classes": _class_summaries(stability_rows, result),
            "prototypes": prototypes,
            "parts": [],
            "support": _support_rows(support),
            "stability_rows": stability_rows,
        }
    saved = _saved_score(result)
    value = summary["score"]["value"]
    summary["recorded_score"] = saved
    summary["score_matches_record"] = (
        None if saved is None or value is None else abs(saved - value) <= 1e-10
    )
    summary["support_sample_sets"] = {
        f"{item['class_id']}:{item['part_id']}": item["sample_ids"] for item in summary["support"]
    }
    return summary


def _score(numerator, denominator):
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "unit": "fraction",
    }


def _class_summaries(rows, result):
    groups = defaultdict(list)
    for row in rows:
        groups[row["class_id"]].append(row)
    summaries = []
    for class_id, items in sorted(groups.items()):
        if "consistent" in items[0]:
            numerator = sum(item["consistent"] for item in items)
        else:
            numerator = sum(item["stable"] for item in items)
        denominator = len(items)
        summaries.append({
            "class_id": class_id,
            "class_label": _class_label(result, class_id),
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else None,
        })
    return summaries


def _support_rows(support):
    result = []
    for (class_id, part_id), per_prototype in sorted(support.items()):
        sets = list(per_prototype.values())
        union = set().union(*sets) if sets else set()
        result.append({
            "class_id": class_id,
            "part_id": part_id,
            "usable_images": len(union),
            "sample_ids": sorted(union),
            "prototype_support_matches": (
                bool(sets) and bool(sets[0]) and all(values == sets[0] for values in sets)
            ),
        })
    return result
