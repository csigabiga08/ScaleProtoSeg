"""Reproduce the Pascal and Cityscapes consistency support-sensitivity studies."""

import argparse
import csv
import json
import math
import platform
from collections import defaultdict
from dataclasses import dataclass
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
    "pascalvoc_seed-42_run02": "ProtoSeg",
    "pascalvoc_seed-43": "ProtoSeg",
    "scaleproto_pascal": "ScaleProtoSeg",
    "scaleproto_pascal_seed_191095": "ScaleProtoSeg",
    "scaleproto_pascal_seed_310792": "ScaleProtoSeg",
}
CITYSCAPES_COHORT = {
    "cityscapes_kld_imnet_4_16v2": ("ProtoSeg", "d64131b1-1777-495f-910f-e11115f2aee8", None),
    "cityscapes_kld_imnet_4_16v2_seed42_run01": ("ProtoSeg", "4080c4d9-bf19-4021-a15d-b00e2bc67b3e", None),
    "cityscapes_kld_imnet_4_16v2_seed-43": ("ProtoSeg", "0870ebca-b219-4bfd-a9fd-29b7242bf57d", None),
    "scaleproto_cityscapes": (
        "ScaleProtoSeg",
        "68a7815b-f857-4c4a-832c-7e7c5eda8c9d",
        "ddd803b5-b59f-4825-91dc-19697eea82a5",
    ),
    "scaleproto_cityscapes_seed_191095": (
        "ScaleProtoSeg",
        "e1185bb2-fc50-4bcf-8098-0cd9c125cabd",
        "407cdbe8-79fc-4635-9552-9515993a3878",
    ),
    "scaleproto_cityscapes_seed_310792": (
        "ScaleProtoSeg",
        "b9024bdd-bfc2-42e0-9fcb-0989b0bb13ea",
        "f34014d5-62ad-44cf-89ec-95b9e5715ce6",
    ),
}
# The scheduled native jobs use this class-masked evaluator, which did not serialize scope.
IMPLICIT_CLASS_COMMITS = {"0592b110e25b96fe46692e5f8d321e5ec728e6ce"}
QUANTILES = (0.7, 0.8, 0.9)
SETTINGS = {
    (False, False, False): "O",
    (True, False, False): "F",
    (True, True, False): "R",
    (True, True, True): "M",
}
FLAGS = ("require_semantic_match", "exclude_background_centroid", "merge_left_right_parts")
ANNOTATION_FLAGS = ("official_parts_only", *FLAGS)
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


@dataclass(frozen=True)
class Study:
    dataset: str
    cohort: dict
    settings: dict
    transitions: tuple
    seed_design: str
    scope_provenance: str
    limitations: tuple
    candidate_count: int = None
    legacy_only: bool = False
    include_image_counts: bool = False
    plot: bool = False


STUDIES = {
    "pascal": Study(
        "pascal",
        {name: (architecture, None, None) for name, architecture in COHORT.items()},
        {(False, *flags): setting for flags, setting in SETTINGS.items()},
        TRANSITIONS,
        "Matched seed family; ProtoSeg seed42 uses run02, not the resumed pascalvoc_seed-42 checkpoint.",
        "All records declare class scope. Where the original description and command omit scope, "
        "this agrees with the historical class-masked implementation (034e327) and later default "
        "quantile_scope='class' (c3837e3). Exact historical source revisions were not recorded.",
        (
            "Sequential annotation ablations do not identify independent causal contributions.",
            "The original exploratory case was already inspected; this is not a preregistered study.",
        ),
        legacy_only=True,
        plot=True,
    ),
    "cityscapes": Study(
        "cityscapes",
        CITYSCAPES_COHORT,
        {(False, False, False, False): "O", (True, False, True, False): "R"},
        (("O", "R"),),
        "Preselected 4_16v2 and grouped ScaleProtoSeg registrations; verify training provenance.",
        "Imported class scope as recorded; implicit native scope accepted only for "
        f"verified class-masked evaluator commits: {sorted(IMPLICIT_CLASS_COMMITS)}.",
        ("O/R is a combined revision, not an attribution to either change.",),
        candidate_count=500,
        include_image_counts=True,
    ),
}


def legacy_description(record):
    return (record.get("legacy_metadata") or {}).get("description") or {}


def select_cohort(records, study):
    """Return the unique eligible measurements and missing cells in a study's grid."""
    cells = {}
    identities = defaultdict(set)
    datasets, legacy_image_orders, class_maps = set(), set(), set()
    for record in records:
        model, dataset = record["model"], record["dataset"]
        name = model["name"]
        if (
            name not in study.cohort
            or dataset["name"] != study.dataset
            or dataset["split"] != "val"
            or record["metric"] != "consistency"
            or record["status"] != "complete"
        ):
            continue
        params, protocol = record["parameters"], record["protocol"]
        q = params["quantile"]
        flags = tuple(params.get(field, False) for field in ANNOTATION_FLAGS)
        if not study.legacy_only and (q not in QUANTILES or flags not in study.settings):
            continue
        description = legacy_description(record)
        historical = description.get("parameters", {})
        command = record["execution"].get("command") or []
        command_scope = command[command.index("--quantile-scope") + 1] if "--quantile-scope" in command else None
        scopes = {
            scope
            for scope in (
                params.get("quantile_scope"),
                protocol.get("quantile_scope"),
                historical.get("quantile_scope"),
                command_scope,
            )
            if scope is not None
        }
        if scopes == {"image"} and not study.legacy_only:
            continue
        if scopes != {"class"} and (scopes or record["execution"].get("commit") not in IMPLICIT_CLASS_COMMITS):
            raise ValueError(f"Unverified or conflicting quantile scope: {record['measurement_id']}")
        architecture, model_id, grouping_id = study.cohort[name]
        grouping = record.get("grouping_model") or {}
        if (
            model["architecture"] != architecture
            or (
                model_id is not None
                and (model.get("registration_id"), grouping.get("registration_id")) != (model_id, grouping_id)
            )
            or (study.candidate_count is not None and dataset.get("candidate_count") != study.candidate_count)
            or q not in QUANTILES
            or flags not in study.settings
            or params["threshold"] != THRESHOLD
            or protocol["quantile"] != q
            or protocol["consistency"]["threshold"] != THRESHOLD
            or (
                study.legacy_only
                and (params.get("quantile_scope") != "class" or "image_order_sha256" not in description)
            )
            or any(type(flag) is not bool for flag in flags)
        ):
            raise ValueError(f"Incompatible {study.dataset} measurement: {record['measurement_id']}")
        for field, flag in zip(ANNOTATION_FLAGS, flags):
            declarations = [
                protocol["parts"].get(field),
                protocol.get("prototype_selection", {}).get(field),
                historical.get(field, False) if study.legacy_only else historical.get(field),
            ]
            if field == "exclude_background_centroid":
                declarations.append(protocol.get("centroids", {}).get(field))
            if any(value is not None and value != flag for value in declarations):
                raise ValueError(f"Conflicting {field}: {record['measurement_id']}")
        key = name, q, study.settings[flags]
        if key in cells:
            raise ValueError(
                f"Duplicate {study.dataset} cell {key}: {cells[key]['measurement_id']}, {record['measurement_id']}"
            )
        cells[key] = record
        identities[name].add(
            (
                model["inputs"]["prototype_checkpoint"]["sha256"],
                grouping["inputs"]["prototype_checkpoint"]["sha256"] if grouping else None,
            )
        )
        datasets.add(json.dumps(dataset, sort_keys=True))
        if study.legacy_only:
            class_maps.add(json.dumps(protocol["classes"], sort_keys=True))
        if description.get("image_order_sha256"):
            legacy_image_orders.add((description.get("validation_images"), description["image_order_sha256"]))
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError("Checkpoint or grouping identity changes across settings")
    if len({next(iter(values))[0] for values in identities.values()}) != len(identities):
        raise ValueError("Cohort must contain distinct trained checkpoints")
    if max(len(datasets), len(legacy_image_orders), len(class_maps)) > 1:
        raise ValueError("Dataset, imported candidate image order, or required class maps differ")
    expected = set(product(study.cohort, QUANTILES, study.settings.values()))
    return cells, sorted(expected - cells.keys())


def canonical_variant(variant, labels, canonical_ids):
    """Resolve measurement-local class IDs before matching observations."""
    evidence, rows, parts = variant
    by_id = {int(key): canonical_ids[label] for key, label in labels.items()}
    if len(by_id) != len(labels) or len(set(by_id.values())) != len(labels):
        raise ValueError("Ambiguous class mapping")
    return (
        {(by_id[cid], pid): item for (cid, pid), item in evidence.items()},
        {(sid, by_id[cid], pid): {**row, "class_id": by_id[cid]} for (sid, cid, pid), row in rows.items()},
        parts,
    )


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


def compare_variants(left_variant, right_variant, identity, include_image_counts):
    left, left_rows, left_parts = left_variant
    right, right_rows, right_parts = right_variant
    common = left.keys() & right.keys()
    common_rows = sorted(left_rows.keys() & right_rows.keys())
    if not common_rows:
        raise ValueError(f"No matched observations: {identity}")
    counts = {"left_rows": len(left_rows), "right_rows": len(right_rows), "common_rows": len(common_rows)}
    if include_image_counts:
        left_images = {key[0] for key in left_rows}
        right_images = {key[0] for key in right_rows}
        counts.update(
            left_images=len(left_images), right_images=len(right_images), common_images=len(left_images & right_images)
        )
    populations = {
        "native": (left, right),
        "common_prototypes": ({key: left[key] for key in common}, {key: right[key] for key in common}),
        "common_rows": (
            evidence_from_rows((left_rows[key] for key in common_rows), left_parts, common),
            evidence_from_rows((right_rows[key] for key in common_rows), right_parts, common),
        ),
    }
    return [
        {
            **identity,
            "population": population,
            "minimum_support": limit,
            **paired_statistics(before, after, limit),
            **counts,
        }
        for population, (before, after) in populations.items()
        for limit in SUPPORT_LIMITS
    ]


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


def plot_curves(output, seeds, summaries, study):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    colors = dict(zip(study.settings.values(), ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(6.1, 2.65), sharey=True, layout="constrained")
        for architecture, axis in zip(("ProtoSeg", "ScaleProtoSeg"), axes):
            for setting, color in colors.items():
                for model, (model_architecture, _, _) in study.cohort.items():
                    points = sorted(
                        (
                            row
                            for row in seeds
                            if row["model"] == model
                            and model_architecture == architecture
                            and row["setting"] == setting
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


def run_analysis(root, output, dataset="pascal"):
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing analysis: {output}")
    study = STUDIES[dataset]
    cells, missing = select_cohort(iter_measurements(root), study)
    if missing:
        raise ValueError(f"Incomplete {dataset} cohort; missing cells: {missing}")
    curves, prototype_rows, pairs, manifest = [], [], [], []
    for name, quantile in product(study.cohort, QUANTILES):
        architecture = study.cohort[name][0]
        variants = {}
        for setting in study.settings.values():
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
                "architecture": architecture,
                "model": name,
                "quantile": quantile,
                "setting": setting,
                "measurement_id": record["measurement_id"],
            }
            images = len({key[0] for key in indexed})
            for limit in SUPPORT_LIMITS:
                curves.append(
                    {
                        **identity,
                        "minimum_support": limit,
                        **statistics(evidence, limit),
                        "images": images,
                        "observation_rows": len(rows),
                    }
                )
            for (cid, pid), values in sorted(evidence.items()):
                prototype_rows.append({**identity, "class_id": cid, "prototype_id": pid, **values})
            legacy = record.get("legacy_metadata") or {}
            description = legacy_description(record)
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
                    "legacy_source_identity": legacy.get("primary_source_relative_identity"),
                    "explicit_description_scope": description.get("parameters", {}).get("quantile_scope"),
                }
            )
            variants[setting] = (evidence, indexed, summary.metadata.analysis_part_ids)
            if sha256_file(bundle / "result.json")[0] != record_hash:
                raise ValueError("Measurement metadata changed during analysis")
        # Preserve native IDs in source exports; normalize only paired comparisons.
        labels = {
            setting: cells[name, quantile, setting]["protocol"]["classes"]["evaluation_class_id_to_label"]
            for setting in variants
        }
        all_labels = sorted({label for mapping in labels.values() for label in mapping.values()})
        canonical_ids = {label: i for i, label in enumerate(all_labels)}
        matched = {
            setting: canonical_variant(variant, labels[setting], canonical_ids) for setting, variant in variants.items()
        }
        for left_setting, right_setting in study.transitions:
            pairs.extend(
                compare_variants(
                    matched[left_setting],
                    matched[right_setting],
                    {
                        "architecture": architecture,
                        "model": name,
                        "quantile": quantile,
                        "transition": left_setting + right_setting,
                    },
                    study.include_image_counts,
                )
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
        "analysis": f"{dataset}-support-sensitivity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "analysis_source_sha256": sha256_file(Path(__file__))[0],
        "threshold": THRESHOLD,
        "support_limits": SUPPORT_LIMITS,
        "aggregation": "unweighted quantile mean within seed, then mean and sample SD across three seeds",
        "seed_design": study.seed_design,
        "scope_provenance": study.scope_provenance,
        "matching": (
            "Class labels normalize measurement-local class IDs. Common-prototype comparisons retain native image eligibility; "
            "common-row comparisons intersect image/class/prototype keys, retaining every common prototype "
            "even if it has no common eligible rows. Each setting retains its own part policy."
        ),
        "limitations": [
            "Three seeds per model; descriptive sample SD is not a confidence interval.",
            "Quantiles are sensitivity settings, not independent training replicates.",
            "Support limits are descriptive, not calibrated evidence or quality thresholds.",
            "Common-row matching does not align differing taxonomies or turn missing parts into negatives.",
            *study.limitations,
        ],
        "measurements": manifest,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if study.plot:
        plot_curves(output, seeds, summaries, study)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset", choices=STUDIES, default="pascal")
    parser.add_argument("--check-cohort", action="store_true", help="List missing metadata cells without scoring.")
    parser.add_argument("--output", type=Path, help="New directory; existing outputs are never overwritten.")
    args = parser.parse_args()
    root = args.root or read_catalog_root()
    if args.check_cohort:
        if args.output:
            parser.error("--check-cohort does not take --output")
        cells, missing = select_cohort(iter_measurements(root), STUDIES[args.dataset])
        print(f"{args.dataset.capitalize()} cohort: {len(cells)}/{len(cells) + len(missing)} completed cells.")
        for name, q, setting in missing:
            print(f"Missing: {name}, q={q}, {setting}")
        return int(bool(missing))
    if args.output is None:
        parser.error("--output is required for analysis")
    summaries = run_analysis(root.resolve(), args.output.resolve(), args.dataset)
    for row in summaries:
        if row["minimum_support"] in (1, 10, 30):
            print(
                f"{row['architecture']} {row['setting']} m={row['minimum_support']}: "
                f"{100 * row['value_mean']:.2f} +/- {100 * row['value_sd']:.2f}%"
            )


if __name__ == "__main__":
    raise SystemExit(main())
