"""Compare Revised mean-part profiles with image/part event overlap in frozen cohorts."""

import argparse
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from statistics import mean

from segmentation.analysis.metrics.part_events import compare_events, event_profiles
from segmentation.analysis.support_sensitivity import (
    QUANTILES,
    STUDIES,
    aggregate,
    write_csv,
)
from segmentation.catalog import load_observations, sha256_file
from segmentation.catalog.table import read_catalog_root

SUBSETS = {"unrestricted": (1, 2), "supported": (10, 5)}
HIGH_RHO = 0.9
HIGH_JACCARD = 0.9
LOW_JACCARD = 0.5


def summarize(pairs):
    eligible = [row for row in pairs if row["eligible"]]
    rho = [row for row in eligible if row["rho"] is not None]
    events = [row for row in eligible if row["jaccard"] is not None]
    joint = [row for row in rho if row["jaccard"] is not None]
    high = [row for row in joint if row["rho"] >= HIGH_RHO]
    low = [row for row in high if row["jaccard"] < LOW_JACCARD]
    high_overlap = [row for row in high if row["jaccard"] >= HIGH_JACCARD]
    result = {
        "pairs": len(pairs),
        "eligible_pairs": len(eligible),
        "insufficient_parts": len(pairs) - len(eligible),
        "undefined_rho": len(eligible) - len(rho),
        "undefined_jaccard": len(eligible) - len(events),
        "joint_defined": len(joint),
        "high_rho": len(high),
        "high_rho_low_jaccard": len(low),
        "high_rho_high_jaccard": len(high_overlap),
        "eligible_fraction": len(eligible) / len(pairs) if pairs else None,
        "high_rho_fraction": len(high) / len(joint) if joint else None,
        "high_rho_low_jaccard_fraction": len(low) / len(joint) if joint else None,
        "low_jaccard_given_high_rho": len(low) / len(high) if high else None,
        "high_jaccard_given_high_rho": len(high_overlap) / len(high) if high else None,
    }
    for key, values in (
        ("rho", [row["rho"] for row in rho]),
        ("profile_mad", [row["profile_mad"] for row in eligible]),
        ("jaccard", [row["jaccard"] for row in events]),
        ("high_rho_jaccard", [row["jaccard"] for row in high]),
        ("availability_dice", [row["availability_dice"] for row in eligible]),
        (
            "absolute_matched_rho_difference",
            [abs(row["rho"] - row["matched_rho"]) for row in rho if row["matched_rho"] is not None],
        ),
    ):
        result[f"{key}_mean"] = mean(values) if values else None
    return result


def analyze_measurement(record, rows):
    profiles = event_profiles(record, rows)
    by_class = defaultdict(list)
    for profile in profiles.values():
        by_class[profile.class_id].append(profile)
    labels = record["protocol"]["classes"]["evaluation_class_id_to_label"]
    pairs, classes, measurements = [], [], []
    for subset, (support, parts) in SUBSETS.items():
        subset_pairs = []
        for cid, values in sorted(by_class.items()):
            class_pairs = []
            for left, right in combinations(sorted(values, key=lambda value: value.prototype_id), 2):
                row = {
                    "subset": subset,
                    "class_label": labels[str(cid)],
                    "left_prototype": left.prototype_id,
                    "right_prototype": right.prototype_id,
                    "both_consistent": left.consistent and right.consistent,
                    **compare_events(left, right, minimum_support=support, minimum_parts=parts),
                }
                class_pairs.append(row)
            subset_pairs.extend(class_pairs)
            for population in ("all", "both_consistent"):
                selected = [row for row in class_pairs if population == "all" or row["both_consistent"]]
                classes.append(
                    {
                        "subset": subset,
                        "population": population,
                        "class_label": labels[str(cid)],
                        "prototypes": len(values) if population == "all" else sum(p.consistent for p in values),
                        **summarize(selected),
                    }
                )
        pairs.extend(subset_pairs)
        for population in ("all", "both_consistent"):
            selected = [row for row in subset_pairs if population == "all" or row["both_consistent"]]
            measurements.append(
                {
                    "subset": subset,
                    "population": population,
                    "prototypes": (
                        len(profiles) if population == "all" else sum(p.consistent for p in profiles.values())
                    ),
                    **summarize(selected),
                }
            )
    score = record["scores"]["consistency"]
    if (sum(p.consistent for p in profiles.values()), len(profiles)) != (score["numerator"], score["denominator"]):
        raise ValueError("Event profiles do not reconstruct recorded consistency")
    return pairs, classes, measurements


def frozen_cohort(manifest, dataset):
    """Reuse exact Revised measurements, not a new selection from a changing index."""
    if manifest["analysis"] != f"{dataset}-support-sensitivity-v1":
        raise ValueError("Expected this dataset's frozen support-sensitivity manifest")
    cells = {}
    for item in manifest["measurements"]:
        if item["setting"] != "R":
            continue
        key = item["model"], item["quantile"]
        if key in cells:
            raise ValueError("Duplicate Revised measurement")
        if item["architecture"] != STUDIES[dataset].cohort[item["model"]][0]:
            raise ValueError("Architecture differs from the declared cohort")
        cells[key] = item
    if cells.keys() != set(product(STUDIES[dataset].cohort, QUANTILES)):
        raise ValueError("Incomplete or unexpected Revised cohort")
    return cells


def run_analysis(root, source, output, dataset):
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing analysis: {output}")
    source_hash = sha256_file(source)[0]
    manifest = json.loads(source.read_text())
    cells = frozen_cohort(manifest, dataset)
    pairs, classes, measurements, sources = [], [], [], []
    for (model, quantile), item in sorted(cells.items()):
        bundle = root / "measurements" / item["measurement_id"]
        if sha256_file(bundle / "result.json")[0] != item["result_sha256"]:
            raise ValueError("Source measurement differs from frozen support cohort")
        record, rows = load_observations(bundle)
        identity = {
            "architecture": item["architecture"],
            "model": model,
            "quantile": quantile,
            "measurement_id": item["measurement_id"],
        }
        a, b, c = analyze_measurement(record, rows)
        for target, values in ((pairs, a), (classes, b), (measurements, c)):
            target.extend({**identity, **row} for row in values)
        if sha256_file(bundle / "result.json")[0] != item["result_sha256"]:
            raise ValueError("Source measurement changed during analysis")
        sources.append(item)
        print(f"Validated {dataset} {model}, q={quantile}", flush=True)
    metrics = tuple(
        key
        for key in measurements[0]
        if key not in ("architecture", "model", "quantile", "measurement_id", "subset", "population")
    )
    fields = ("architecture", "subset", "population")
    seeds, summary = aggregate(measurements, fields, metrics)
    quantile_summary = []
    for q in QUANTILES:
        _, values = aggregate(
            [row for row in measurements if row["quantile"] == q], (*fields, "quantile"), metrics, quantiles=(q,)
        )
        quantile_summary.extend(values)
    examples = []
    for architecture in sorted({row["architecture"] for row in pairs}):
        candidates = [
            row
            for row in pairs
            if row["architecture"] == architecture
            and row["subset"] == "supported"
            and row["eligible"]
            and row["rho"] is not None
            and row["rho"] >= HIGH_RHO
            and row["jaccard"] is not None
            and row["jaccard"] < LOW_JACCARD
        ]
        if candidates:
            examples.append(max(candidates, key=lambda row: row["rho"] - row["jaccard"]))
    if sha256_file(source)[0] != source_hash:
        raise ValueError("Frozen cohort manifest changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (
        ("pairs.csv", pairs),
        ("classes.csv", classes),
        ("measurements.csv", measurements),
        ("seeds.csv", seeds),
        ("summary.csv", summary),
        ("quantile-summary.csv", quantile_summary),
        ("examples.csv", examples),
    ):
        if values:
            write_csv(output / name, values)
    report = {
        "analysis": f"{dataset}-profile-event-overlap-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "source_manifest": {"path": str(source), "sha256": source_hash},
        "source_sha256": {
            str(path): sha256_file(path)[0]
            for path in (Path(__file__), Path(__file__).parent / "metrics" / "part_events.py")
        },
        "subsets": {
            key: {"minimum_joint_image_support": support, "minimum_shared_parts": parts}
            for key, (support, parts) in SUBSETS.items()
        },
        "cutoffs": {"high_rho": HIGH_RHO, "high_jaccard": HIGH_JACCARD, "low_jaccard_strict": LOW_JACCARD},
        "matching": (
            "Within class only. Pearson and MAD use each prototype's own image means on the same "
            "jointly supported parts used for Jaccard. Event overlap intersects observed image/part "
            "cells; missing values are never negatives. Matched-image Pearson is a sensitivity check."
        ),
        "aggregation": (
            "Pool unordered within-class pairs within each measurement; average three quantiles within "
            "run, then unweighted mean and sample SD across three runs. No pooling across runs. "
            "Undefined fractions stay undefined; incomplete quantile/run groups are not averaged."
        ),
        "example_selection": (
            "One illustrative supported high-rho/low-Jaccard pair per architecture, maximizing rho-J "
            "over all runs and quantiles; ties use first model/quantile/class/prototype in sorted order."
        ),
        "seed_design": manifest["seed_design"],
        "scope_provenance": manifest["scope_provenance"],
        "limitations": [
            "Exploratory follow-up, not a preregistered or held-out study.",
            "Pair-weighted within-run summaries; class sizes and prototype budgets differ.",
            "Pairs, parts and quantiles are not independent training replicates.",
            "Three-run sample SD is not a confidence interval; no image bootstrap or significance test.",
            "Support and similarity cutoffs are descriptive, not calibrated reliability thresholds.",
            "High profile correlation or event overlap does not establish dispensability.",
            "Low event overlap alone does not establish semantic specialization.",
            "Cityscapes five-part subset excludes person and rider by taxonomy.",
        ],
        "measurements": sources,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset", choices=STUDIES, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing results are never overwritten."
    )
    args = parser.parse_args()
    summary = run_analysis(
        (args.root or read_catalog_root()).resolve(),
        args.cohort_manifest.resolve(),
        args.output.resolve(),
        args.dataset,
    )
    for row in summary:
        if row["population"] == "all":
            print(
                f"{row['architecture']} {row['subset']}: "
                f"eligible={row['eligible_pairs_mean']:.1f}, joint-defined={row['joint_defined_mean']:.1f}, "
                f"high-rho/low-J={row['high_rho_low_jaccard_fraction_mean']}"
            )


if __name__ == "__main__":
    main()
