"""Reconcile article reference tables without substituting incomplete seed cohorts."""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean, median, stdev

from segmentation.analysis.support_sensitivity import (
    ANNOTATION_FLAGS,
    IMPLICIT_CLASS_COMMITS,
    QUANTILES,
    STUDIES,
    aggregate,
    evidence_from_rows,
    legacy_description,
    statistics,
    write_csv,
)
from segmentation.catalog import iter_measurements, load_observations, sha256_file
from segmentation.catalog.table import read_catalog_root

REPLACED_MODEL = "pascalvoc_seed-42"
MATCHED_MODEL = "pascalvoc_seed-42_run02"


def article_cohorts():
    return {
        dataset: {
            MATCHED_MODEL if name == REPLACED_MODEL else name: architecture
            for name, (architecture, _, _) in study.cohort.items()
        }
        for dataset, study in STUDIES.items()
    }


def gin_config(path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in values:
                raise ValueError(f"Repeated config assignment: {path}")
            values[key.strip()] = value.strip()
    seed = int(values.pop("train.random_seed"))
    values.pop("Trainer.gpus", None)
    return seed, values


def class_scope(record):
    historical = legacy_description(record).get("parameters", {})
    command = record["execution"].get("command") or []
    command_scope = command[command.index("--quantile-scope") + 1] if "--quantile-scope" in command else None
    declarations = {
        value
        for value in (
            record["parameters"].get("quantile_scope"),
            record["protocol"].get("quantile_scope"),
            historical.get("quantile_scope"),
            command_scope,
        )
        if value is not None
    }
    if declarations == {"image"}:
        return False
    if declarations != {"class"} and (declarations or record["execution"].get("commit") not in IMPLICIT_CLASS_COMMITS):
        raise ValueError(f"Unverified quantile scope: {record['measurement_id']}")
    return True


def select_cells(records, identities):
    cells = {}
    for record in records:
        dataset, model = record["dataset"]["name"], record["model"]["name"]
        if (dataset, model) not in identities or record["status"] != "complete":
            continue
        params = record["parameters"]
        flags = tuple(params.get(field, False) for field in ANNOTATION_FLAGS)
        setting = STUDIES[dataset].settings.get(flags)
        if setting not in ("O", "R", "M") or params["quantile"] not in QUANTILES:
            continue
        if not class_scope(record):
            continue
        model_hash, group_hash = identities[dataset, model]["identity"]
        grouping = record.get("grouping_model")
        observed_identity = (
            record["model"]["inputs"]["prototype_checkpoint"]["sha256"],
            grouping["inputs"]["prototype_checkpoint"]["sha256"] if grouping else None,
        )
        if observed_identity != (model_hash, group_hash) or record["dataset"]["split"] != "val":
            raise ValueError(f"Unexpected checkpoint, grouping or split: {record['measurement_id']}")
        if record["model"]["architecture"] != identities[dataset, model]["architecture"]:
            raise ValueError("Unexpected architecture")
        protocol = record["protocol"]
        if protocol["quantile"] != params["quantile"]:
            raise ValueError("Conflicting quantiles")
        for field, value in zip(ANNOTATION_FLAGS, flags):
            declarations = (
                protocol["parts"].get(field),
                protocol.get("centroids", {}).get(field),
                protocol.get("prototype_selection", {}).get(field),
                legacy_description(record).get("parameters", {}).get(field),
            )
            if type(value) is not bool or any(item is not None and item != value for item in declarations):
                raise ValueError(f"Conflicting annotation policy: {record['measurement_id']}")
        metric = record["metric"]
        if metric == "consistency":
            if params["threshold"] != 0.8 or protocol[metric]["threshold"] != 0.8:
                raise ValueError("Unexpected consistency threshold")
        elif metric == "stability":
            if (
                params.get("std_dev") != 0.2
                or params.get("seed") != 0
                or protocol[metric]["standard_deviation"] != 0.2
                or protocol[metric]["seed"] != 0
            ):
                raise ValueError("Unexpected noise protocol")
        else:
            continue
        key = dataset, model, setting, metric, params["quantile"]
        if key in cells:
            raise ValueError(f"Duplicate reference-table cell: {key}")
        cells[key] = record
    return cells


def reconstruct(record, rows):
    keys = [(row["sample_id"], row["class_id"], row["prototype_id"], row["repetition"]) for row in rows]
    if len(set(keys)) != len(keys) or any(row["repetition"] != "0" for row in rows):
        raise ValueError("Repeated observations or unsupported repetitions")
    if record["metric"] == "consistency":
        parts = record["protocol"]["parts"]["valid_part_ids"]
        result = statistics(evidence_from_rows(rows, parts), 1)
        return result["numerator"], result["denominator"]
    parts = [column for column in record["protocol"]["observation_columns"] if column.startswith("part_")]
    if not parts or {f"noisy_{column}" for column in parts} != {
        column for column in record["protocol"]["observation_columns"] if column.startswith("noisy_part_")
    }:
        raise ValueError("Unpaired stability columns")
    stable = 0
    for row in rows:
        clean, noisy = [float(row[column]) for column in parts], [float(row[f"noisy_{column}"]) for column in parts]
        if any(value not in (0, 1) for value in clean + noisy):
            raise ValueError("Stability requires binary, nonmissing observations")
        stable += clean == noisy
    return stable, len(rows)


def check_score(record, numerator, denominator):
    score = record["scores"][record["metric"]]
    if (
        not denominator
        or (numerator, denominator) != (score["numerator"], score["denominator"])
        or not math.isclose(numerator / denominator, score["value"], rel_tol=0, abs_tol=1e-10)
    ):
        raise ValueError(f"Score reconstruction failed: {record['measurement_id']}")


def activation_statistics(rows):
    ids, values = set(), []
    for row in rows:
        pid = int(row["proto_id"])
        if pid in ids:
            raise ValueError("Repeated activation prototype")
        ids.add(pid)
        inside_pixels, outside_pixels = int(row["inside_pixels"]), int(row["outside_pixels"])
        inside, outside = float(row["inside_mean"]), float(row["outside_mean"])
        ratio = row["imprecise_activation"]
        defined = inside_pixels > 0 and outside_pixels > 0 and math.isfinite(inside) and inside != 0
        if not defined:
            if ratio and math.isfinite(float(ratio)):
                raise ValueError("Activation ratio should be undefined")
            continue
        number = float(ratio)
        if not math.isfinite(number) or not math.isclose(number, outside / inside, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Activation ratio reconstruction failed")
        values.append(number)
    return {
        "prototypes": len(values),
        "mean": mean(values) if values else None,
        "prototype_sd": stdev(values) if len(values) > 1 else None,
        "median": median(values) if values else None,
    }


def run_analysis(root, manifests, checkpoint_root, output):
    cohorts = article_cohorts()
    identities, frozen = {}, {}
    for dataset, manifest_path in manifests.items():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["measurements"]:
            name, model = item["model"], item["model_identity"]
            if name not in cohorts[dataset]:
                continue
            grouping = item.get("grouping_identity")
            identities[dataset, name] = {
                "architecture": cohorts[dataset][name],
                "identity": (
                    model["inputs"]["prototype_checkpoint"]["sha256"],
                    grouping["inputs"]["prototype_checkpoint"]["sha256"] if grouping else None,
                ),
                "checkpoint": model["inputs"]["prototype_checkpoint"]["path"],
                "seed": model.get("seed"),
            }
            frozen[item["measurement_id"]] = item["result_sha256"]
    matched_checkpoint = checkpoint_root / "protoseg/checkpoints" / MATCHED_MODEL / "checkpoints/push_best.pth"
    identities["pascal", MATCHED_MODEL] = {
        "architecture": "ProtoSeg",
        "identity": (sha256_file(matched_checkpoint)[0], None),
        "checkpoint": str(matched_checkpoint),
        "seed": 42,
    }
    config_evidence = []
    for dataset, cohort in cohorts.items():
        configs, seeds = [], []
        for name, architecture in cohort.items():
            identity = identities[dataset, name]
            if sha256_file(identity["checkpoint"])[0] != identity["identity"][0]:
                raise ValueError(f"Checkpoint changed: {name}")
            if architecture != "ProtoSeg":
                continue
            config = Path(identity["checkpoint"]).parents[1] / "config.gin"
            seed, values = gin_config(config)
            configs.append(values)
            seeds.append(seed)
            identity["seed"] = seed
            config_evidence.append({"model": name, "seed": seed, "path": str(config), "sha256": sha256_file(config)[0]})
        if len(set(seeds)) != 3 or any(config != configs[0] for config in configs[1:]):
            raise ValueError(f"Expected matching ProtoSeg configs and three distinct seeds: {dataset}")
    cells = select_cells(iter_measurements(root), identities)
    measurements, sources, missing = [], [], []
    for dataset, cohort in cohorts.items():
        settings = ("O", "R", "M") if dataset == "pascal" else ("O", "R")
        for name, setting, metric, q in product(cohort, settings, ("consistency", "stability"), QUANTILES):
            key = dataset, name, setting, metric, q
            record = cells.get(key)
            row = {
                "dataset": dataset,
                "architecture": cohort[name],
                "model": name,
                "setting": setting,
                "metric": metric,
                "quantile": q,
                "value": None,
                "numerator": None,
                "denominator": None,
                "images": None,
                "prototypes": None,
                "measurement_id": None,
            }
            if record is None:
                missing.append(dict(row))
            else:
                bundle = root / "measurements" / record["measurement_id"]
                result_hash = sha256_file(bundle / "result.json")[0]
                if metric == "consistency" and frozen.get(record["measurement_id"]) != result_hash:
                    raise ValueError("Consistency source differs from the frozen support study")
                loaded, observations = load_observations(bundle)
                numerator, denominator = reconstruct(loaded, observations)
                check_score(loaded, numerator, denominator)
                row.update(
                    value=numerator / denominator,
                    numerator=numerator,
                    denominator=denominator,
                    images=len({item["sample_id"] for item in observations}),
                    prototypes=len({(item["class_id"], item["prototype_id"]) for item in observations}),
                    measurement_id=record["measurement_id"],
                )
                sources.append(
                    {
                        "measurement_id": record["measurement_id"],
                        "result_sha256": result_hash,
                        "observations": loaded["artifacts"]["observations"],
                    }
                )
            measurements.append(row)
    seeds, summary = aggregate(measurements, ("dataset", "architecture", "setting", "metric"), ("value",))
    activations, activation_sources = [], []
    for (dataset, name), identity in identities.items():
        checkpoint = Path(identity["checkpoint"])
        directory = checkpoint.parents[1] / "metrics-imprecise-activations"
        stem = f"imprecise_activations_checkpoint_{checkpoint.stem}"
        description, observations = directory / f"description_{stem}.json", directory / f"{stem}.csv"
        row = dict(dataset=dataset, architecture=identity["architecture"], model=name, seed=identity["seed"])
        if not description.is_file() or not observations.is_file():
            row.update(prototypes=None, mean=None, prototype_sd=None, median=None)
        else:
            metadata = json.loads(description.read_text())
            if (
                metadata["status"] != "complete"
                or Path(metadata["checkpoint"]).resolve() != checkpoint.resolve()
                or metadata["dataset"] != dataset
                or metadata["split"] != "val"
                or metadata["activation"] != "log_similarity"
                or metadata["aggregation"] != "pooled_validation_pixels"
                or metadata["interpolation"] != "nearest"
            ):
                raise ValueError(f"Incompatible activation measurement: {description}")
            with observations.open(newline="") as stream:
                row.update(activation_statistics(list(csv.DictReader(stream))))
            if row["prototypes"] != metadata["prototypes_with_defined_ratio"] or not math.isclose(
                row["mean"], metadata["score"], rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError("Activation summary disagrees with metadata")
            activation_sources.append(
                {
                    "model": name,
                    "description": str(description),
                    "description_sha256": sha256_file(description)[0],
                    "observations": str(observations),
                    "observations_sha256": sha256_file(observations)[0],
                    "image_order_sha256": metadata["image_order_sha256"],
                    "evaluated_images": metadata["evaluated_images"],
                }
            )
        activations.append(row)
    for dataset in cohorts:
        orders = {item["image_order_sha256"] for item in activation_sources if item["model"] in cohorts[dataset]}
        if len(orders) != 1:
            raise ValueError("Activation candidate image order differs within dataset")
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("measurements.csv", measurements),
        ("seeds.csv", seeds),
        ("summary.csv", summary),
        ("imprecise-activation.csv", activations),
    ):
        write_csv(output / filename, rows)
    manifest = {
        "analysis": "article-reference-tables-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": sha256_file(Path(__file__))[0],
        "cohorts": [
            {"dataset": dataset, "model": model, **identity} for (dataset, model), identity in identities.items()
        ],
        "excluded_model": REPLACED_MODEL,
        "config_evidence": config_evidence,
        "source_manifests": {str(path): sha256_file(path)[0] for path in manifests.values()},
        "measurements": sources,
        "missing_cells": missing,
        "activation_sources": activation_sources,
        "limitations": [
            "Missing seed measurements suppress the entire three-seed aggregate; no two-seed substitution.",
            "The prior Pascal ProtoSeg support/overlap aggregates include the excluded resumed seed42 checkpoint.",
            "ProtoSeg configs match apart from training seed and GPU declaration; checkpoint hashes are verified.",
            "ScaleProtoSeg uses the existing released seed family and grouping identities; full training configs unavailable.",
            "Historical evaluator revisions are incomplete; scope provenance follows the frozen support studies.",
            "Activation metadata identify checkpoint paths but do not record historical checkpoint hashes.",
            "Activation SD is across prototypes; reference-table SD is across quantile-averaged seeds.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"verified_measurements": len(sources), "missing_cells": len(missing), "summary": summary}, indent=2)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=read_catalog_root())
    parser.add_argument("--pascal-manifest", type=Path, required=True)
    parser.add_argument("--cityscapes-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory must not already exist")
    run_analysis(
        args.root,
        {"pascal": args.pascal_manifest, "cityscapes": args.cityscapes_manifest},
        args.checkpoint_root,
        args.output,
    )


if __name__ == "__main__":
    main()
