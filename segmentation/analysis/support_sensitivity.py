"""Reproduce the six-seed Pascal consistency support-sensitivity study."""

import argparse
import csv
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from segmentation.catalog import iter_measurements, load_observations, sha256_file
from segmentation.catalog.part_summary import (
    SUPPORT_LIMITS,
    load_part_summary,
    support_statistics,
)
from segmentation.catalog.table import read_catalog_root

COHORT = {
    "pascalvoc": "ProtoSeg",
    "pascalvoc_seed-42": "ProtoSeg",
    "pascalvoc_seed-43": "ProtoSeg",
    "scaleproto_pascal": "ScaleProtoSeg",
    "scaleproto_pascal_seed_191095": "ScaleProtoSeg",
    "scaleproto_pascal_seed_310792": "ScaleProtoSeg",
}
QUANTILES = (0.7, 0.8, 0.9)
SETTINGS = {
    (False, False, False): "O",
    (True, False, False): "F",
    (True, True, False): "R",
    (True, True, True): "M",
}
FLAGS = ("require_semantic_match", "exclude_background_centroid", "merge_left_right_parts")
TRANSITIONS = (("O", "F"), ("F", "R"), ("R", "M"))
THRESHOLD = 0.8
METRICS = (
    "value",
    "loss",
    "rare_only_fraction",
    "evidence_coverage",
    "numerator",
    "denominator",
)


def freeze_cohort(records):
    """Reject ambiguous, incomplete, duplicate, or incompatible Pascal cells."""
    cells = {}
    identities, datasets, classes, image_orders = defaultdict(set), set(), set(), set()
    for record in records:
        name = record["model"]["name"]
        if (
            name not in COHORT
            or record["dataset"]["name"] != "pascal"
            or record["metric"] != "consistency"
            or record["status"] != "complete"
        ):
            continue
        params, protocol = record["parameters"], record["protocol"]
        flags = tuple(params[key] for key in FLAGS)
        if flags not in SETTINGS or any(type(flag) is not bool for flag in flags):
            raise ValueError(f"Unexpected annotation policy: {record['measurement_id']}")
        if (
            params["quantile"] not in QUANTILES
            or params["threshold"] != THRESHOLD
            or protocol["quantile"] != params["quantile"]
            or protocol["consistency"]["threshold"] != THRESHOLD
            or params.get("quantile_scope") != "class"
            or protocol.get("quantile_scope", "class") != "class"
            or params["official_parts_only"]
            or record["model"]["architecture"] != COHORT[name]
        ):
            raise ValueError(f"Incompatible protocol: {record['measurement_id']}")
        if protocol["centroids"]["exclude_background_centroid"] != flags[1]:
            raise ValueError("Centroid policy disagrees with parameters")
        for flag, value in zip(FLAGS, flags):
            declared = protocol["parts"].get(flag, protocol["prototype_selection"].get(flag))
            if declared is not None and declared != value:
                raise ValueError(f"Conflicting {flag}")
        description = record["legacy_metadata"]["description"]
        historical = description["parameters"]
        for flag, value in zip(FLAGS, flags):
            if historical.get(flag, False) != value:
                raise ValueError(f"Historical annotation policy disagrees: {flag}")
        scope = historical.get("quantile_scope")
        command = record["execution"].get("command") or []
        if "--quantile-scope" in command:
            scope = command[command.index("--quantile-scope") + 1]
        if scope not in (None, "class"):
            raise ValueError("Historical activation scope disagrees")
        key = (name, params["quantile"], SETTINGS[flags])
        if key in cells:
            raise ValueError(f"Duplicate cohort cell: {key}")
        cells[key] = record
        grouping = record.get("grouping_model")
        identities[name].add(
            (
                record["model"]["inputs"]["prototype_checkpoint"]["sha256"],
                grouping["inputs"]["prototype_checkpoint"]["sha256"] if grouping else None,
            )
        )
        datasets.add(json.dumps(record["dataset"], sort_keys=True))
        classes.add(json.dumps(protocol["classes"], sort_keys=True))
        image_orders.add((description["validation_images"], description["image_order_sha256"]))
    expected = set(product(COHORT, QUANTILES, SETTINGS.values()))
    if set(cells) != expected:
        raise ValueError(f"Incomplete cohort; missing cells: {sorted(expected - cells.keys())}")
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError("Checkpoint or grouping identity changes across settings")
    if len({next(iter(values))[0] for values in identities.values()}) != len(COHORT):
        raise ValueError("Seed replicates must have distinct trained checkpoints")
    if len(datasets) != 1 or len(classes) != 1 or len(image_orders) != 1:
        raise ValueError("Dataset, class identities, or candidate image order differ")
    return cells


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (row["sample_id"], int(row["class_id"]), int(row["prototype_id"]))
        if key in indexed:
            raise ValueError("Repeated image/class/prototype observation")
        indexed[key] = row
    return indexed


def evidence_from_rows(rows, parts, population=()):
    """Independent raw-row reconstruction, also used after matching image rows."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for cid, pid in population:
        counts[cid, pid]
    for row in rows:
        values = counts[int(row["class_id"]), int(row["prototype_id"])]
        for part in parts:
            value = row[f"part_{part}"]
            if value is None or value == "" or str(value).lower() == "nan":
                continue
            number = float(value)
            if number not in (0, 1):
                raise ValueError("Expected a binary part hit")
            values[part][0] += 1
            values[part][1] += int(number)
    return {
        key: {
            "maximum_support": max((n for n, _ in values.values()), default=0),
            "qualifying_support": max((n for n, hits in values.values() if hits / n > THRESHOLD), default=0),
        }
        for key, values in counts.items()
    }


def statistics(evidence, minimum_support):
    denominator = len(evidence)
    if not denominator:
        raise ValueError("Empty prototype population")
    baseline = sum(item["qualifying_support"] > 0 for item in evidence.values())
    numerator = sum(item["qualifying_support"] >= minimum_support for item in evidence.values())
    return {
        "value": numerator / denominator,
        "loss": (baseline - numerator) / denominator,
        "rare_only_fraction": (baseline - numerator) / baseline if baseline else None,
        "evidence_coverage": sum(item["maximum_support"] >= minimum_support for item in evidence.values())
        / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def paired_statistics(left, right, minimum_support):
    common = left.keys() & right.keys()
    before = {key for key in common if left[key]["qualifying_support"] >= minimum_support}
    after = {key for key in common if right[key]["qualifying_support"] >= minimum_support}
    a, b = statistics(left, minimum_support), statistics(right, minimum_support)
    return {
        "left_value": a["value"],
        "right_value": b["value"],
        "delta": b["value"] - a["value"],
        "left_denominator": len(left),
        "right_denominator": len(right),
        "common_prototypes": len(common),
        "left_only_prototypes": len(left.keys() - common),
        "right_only_prototypes": len(right.keys() - common),
        "retained_consistent": len(before & after),
        "lost_consistency": len(before - after),
        "gained_consistency": len(after - before),
        "retained_inconsistent": len(common - (before | after)),
        "left_evidence_coverage": a["evidence_coverage"],
        "right_evidence_coverage": b["evidence_coverage"],
    }


def aggregate(rows, group_fields, metrics, quantiles=QUANTILES):
    """Average quantiles within seed, then summarize three seed-level estimates."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields) + (row["model"],)].append(row)
    seeds = []
    for key, values in sorted(grouped.items()):
        if sorted(row["quantile"] for row in values) != sorted(quantiles):
            raise ValueError("Each seed must contain each declared quantile exactly once")
        item = dict(zip(group_fields, key[:-1]))
        item["model"] = key[-1]
        for metric in metrics:
            observed = [row[metric] for row in values]
            item[metric] = mean(observed) if all(value is not None for value in observed) else None
        seeds.append(item)
    groups = defaultdict(list)
    for seed in seeds:
        groups[tuple(seed[field] for field in group_fields)].append(seed)
    summaries = []
    for key, values in sorted(groups.items()):
        if len(values) != 3:
            raise ValueError("Expected three independent training seeds per architecture")
        item = dict(zip(group_fields, key))
        item["seeds"] = len(values)
        for metric in metrics:
            observed = [row[metric] for row in values]
            complete = all(value is not None for value in observed)
            item[f"{metric}_mean"] = mean(observed) if complete else None
            item[f"{metric}_sd"] = stdev(observed) if complete else None
        summaries.append(item)
    return seeds, summaries


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output, seeds, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    colors = dict(zip(SETTINGS.values(), ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(6.1, 2.65), sharey=True, layout="constrained")
        for architecture, axis in zip(("ProtoSeg", "ScaleProtoSeg"), axes):
            for setting, color in colors.items():
                for model in COHORT:
                    points = sorted(
                        (
                            row
                            for row in seeds
                            if row["model"] == model and COHORT[model] == architecture and row["setting"] == setting
                        ),
                        key=lambda row: row["minimum_support"],
                    )
                    axis.plot(
                        [row["minimum_support"] for row in points],
                        [100 * row["value"] for row in points],
                        color=color,
                        alpha=0.3,
                        linewidth=0.7,
                    )
                points = sorted(
                    (row for row in summaries if row["architecture"] == architecture and row["setting"] == setting),
                    key=lambda row: row["minimum_support"],
                )
                axis.plot(
                    [row["minimum_support"] for row in points],
                    [100 * row["value_mean"] for row in points],
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.6,
                    label=setting,
                )
            axis.set(title=architecture, xlabel="Minimum supporting images $m$", xscale="log", ylim=(0, 100))
            axis.set_xticks(SUPPORT_LIMITS)
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.minorticks_off()
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Support-aware consistency (%)")
        axes[1].legend(ncol=4, loc="upper right", fontsize=7)
        figure.savefig(output / "support-curves.pdf")
        figure.savefig(output / "support-curves.png", dpi=180)
        plt.close(figure)


def run_analysis(root, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing analysis: {output}")
    cells = freeze_cohort(iter_measurements(root))
    curves, prototype_rows, pairs, manifest = [], [], [], []
    for name, quantile in product(COHORT, QUANTILES):
        variants = {}
        for setting in SETTINGS.values():
            record = cells[name, quantile, setting]
            bundle = root / "measurements" / record["measurement_id"]
            record_hash = sha256_file(bundle / "result.json")[0]
            loaded, rows = load_observations(bundle)
            if loaded != record:
                raise ValueError("Measurement metadata changed during analysis")
            indexed = index_rows(rows)
            summary = load_part_summary(bundle, record)
            if summary is None:
                raise ValueError(f"Missing support summary: {record['measurement_id']}")
            evidence = evidence_from_rows(rows, summary.metadata.analysis_part_ids)
            stored = {
                (item["class_id"], item["prototype_id"]): {
                    key: item[key] for key in ("maximum_support", "qualifying_support")
                }
                for item in support_statistics(summary)["prototypes"]
            }
            if evidence != stored:
                raise ValueError("Raw-row support reconstruction disagrees with persisted counts")
            baseline = statistics(evidence, 1)
            score = record["scores"]["consistency"]
            if (baseline["numerator"], baseline["denominator"]) != (score["numerator"], score["denominator"]):
                raise ValueError("Reconstructed consistency disagrees with recorded score")
            if not math.isclose(baseline["value"], score["value"], abs_tol=1e-12):
                raise ValueError("Reconstructed consistency value disagrees with recorded score")
            identity = {
                "architecture": COHORT[name],
                "model": name,
                "quantile": quantile,
                "setting": setting,
                "measurement_id": record["measurement_id"],
            }
            for limit in SUPPORT_LIMITS:
                curves.append(
                    {
                        **identity,
                        "minimum_support": limit,
                        **statistics(evidence, limit),
                        "images": len({key[0] for key in indexed}),
                        "observation_rows": len(rows),
                    }
                )
            for (cid, pid), values in sorted(evidence.items()):
                prototype_rows.append({**identity, "class_id": cid, "prototype_id": pid, **values})
            description = record["legacy_metadata"]["description"]
            manifest.append(
                {
                    **identity,
                    "result_sha256": record_hash,
                    "model_identity": record["model"],
                    "grouping_identity": record.get("grouping_model"),
                    "parameters": record["parameters"],
                    "protocol": record["protocol"],
                    "observation_artifact": record["artifacts"]["observations"],
                    "selected_prototype_ids": record.get("selected_prototype_ids"),
                    "population": record["population"],
                    "execution": record["execution"],
                    "legacy_description": description,
                    "legacy_source_identity": record["legacy_metadata"]["primary_source_relative_identity"],
                    "explicit_description_scope": description["parameters"].get("quantile_scope"),
                }
            )
            variants[setting] = (evidence, indexed, summary.metadata.analysis_part_ids)
            if sha256_file(bundle / "result.json")[0] != record_hash:
                raise ValueError("Measurement metadata changed during analysis")
        for left_setting, right_setting in TRANSITIONS:
            left, left_rows, left_parts = variants[left_setting]
            right, right_rows, right_parts = variants[right_setting]
            common = left.keys() & right.keys()
            common_rows = left_rows.keys() & right_rows.keys()
            populations = {
                "native": (left, right),
                "common_prototypes": ({key: left[key] for key in common}, {key: right[key] for key in common}),
                "common_rows": (
                    evidence_from_rows((left_rows[key] for key in sorted(common_rows)), left_parts, common),
                    evidence_from_rows((right_rows[key] for key in sorted(common_rows)), right_parts, common),
                ),
            }
            for population, (before, after) in populations.items():
                for limit in SUPPORT_LIMITS:
                    pairs.append(
                        {
                            "architecture": COHORT[name],
                            "model": name,
                            "quantile": quantile,
                            "transition": left_setting + right_setting,
                            "population": population,
                            "minimum_support": limit,
                            **paired_statistics(before, after, limit),
                            "left_rows": len(left_rows),
                            "right_rows": len(right_rows),
                            "common_rows": len(common_rows),
                        }
                    )
        print(f"Validated {name}, q={quantile}", flush=True)
    seeds, summaries = aggregate(curves, ("architecture", "setting", "minimum_support"), METRICS)
    quantile_summaries = []
    for quantile in QUANTILES:
        _, group = aggregate(
            [row for row in curves if row["quantile"] == quantile],
            ("architecture", "setting", "minimum_support", "quantile"),
            METRICS,
            quantiles=(quantile,),
        )
        quantile_summaries.extend(group)
    pair_metrics = tuple(
        key
        for key in pairs[0]
        if key not in ("architecture", "model", "quantile", "transition", "population", "minimum_support")
    )
    paired_seeds, paired_summaries = aggregate(
        pairs, ("architecture", "transition", "population", "minimum_support"), pair_metrics
    )
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("measurements.csv", curves),
        ("prototypes.csv", prototype_rows),
        ("seed-curves.csv", seeds),
        ("summary.csv", summaries),
        ("quantile-summary.csv", quantile_summaries),
        ("paired-measurements.csv", pairs),
        ("paired-seeds.csv", paired_seeds),
        ("paired-summary.csv", paired_summaries),
    ):
        write_csv(output / filename, rows)
    report = {
        "analysis": "pascal-support-sensitivity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "analysis_source_sha256": sha256_file(Path(__file__))[0],
        "threshold": THRESHOLD,
        "support_limits": SUPPORT_LIMITS,
        "aggregation": "unweighted quantile mean within seed, then mean and sample SD across three seeds",
        "seed_design": "User confirmed only training random seed differs within architecture.",
        "scope_provenance": (
            "All records declare class scope. Where the original description and command omit scope, "
            "this agrees with the historical class-masked implementation (034e327) and later default "
            "quantile_scope='class' (c3837e3). Exact historical source revisions were not recorded."
        ),
        "matching": (
            "Class maps are verified identical. Common-prototype comparisons retain native image eligibility; "
            "common-row comparisons intersect image/class/prototype keys, retaining every common prototype "
            "even if it has no common eligible rows. Each setting retains its own part policy."
        ),
        "limitations": [
            "Three seeds per model; descriptive sample SD is not a confidence interval.",
            "Quantiles are sensitivity settings, not independent training replicates.",
            "Support limits are descriptive, not calibrated evidence or quality thresholds.",
            "Sequential annotation ablations do not identify independent causal contributions.",
            "Common-row matching does not align differing taxonomies or turn missing parts into negatives.",
            "The original exploratory case was already inspected; this is not a preregistered study.",
        ],
        "measurements": manifest,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot_curves(output, seeds, summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing outputs are never overwritten."
    )
    args = parser.parse_args()
    root = args.root or read_catalog_root()
    summaries = run_analysis(root.resolve(), args.output.resolve())
    for row in summaries:
        if row["minimum_support"] in (1, 10, 30):
            print(
                f"{row['architecture']} {row['setting']} m={row['minimum_support']}: "
                f"{100 * row['value_mean']:.2f} +/- {100 * row['value_sd']:.2f}%"
            )


if __name__ == "__main__":
    main()
