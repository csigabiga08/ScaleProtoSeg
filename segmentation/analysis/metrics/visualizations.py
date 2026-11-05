"""Render catalog consistency summaries without the legacy evaluator or path layout."""

import csv
import json
from pathlib import Path

from segmentation.analysis.metrics.catalog import CatalogError, resolve_catalog_root


class VisualizationError(ValueError):
    """Raised when selected measurements cannot be compared honestly."""


def _known_checkpoint(record):
    identity = record.get("model", {}).get("inputs", {}).get("prototype_checkpoint")
    if not isinstance(identity, dict) or identity.get("status") != "available" or not identity.get("sha256"):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has unknown checkpoint identity; comparability is unverified"
        )
    return identity["sha256"]


def _protocol_without_quantile(record):
    protocol = record.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("quantile") is None:
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has no recorded protocol quantile"
        )
    required = ("activation", "centroids", "parts", "classes", "prototype_selection", "consistency")
    if any(not isinstance(protocol.get(field), dict) or not protocol[field] for field in required):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has incomplete protocol identity; comparability is unverified"
        )
    mapping = protocol["parts"].get("mapping_identity")
    if not isinstance(mapping, dict) or mapping.get("status") != "available" or not mapping.get("sha256"):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has unknown part-mapping identity"
        )
    normalized = json.loads(json.dumps(protocol, sort_keys=True))
    normalized.pop("quantile", None)
    return normalized


def _compatibility_signature(record):
    dataset = record.get("dataset", {})
    dataset_identity = dataset.get("identity")
    if not isinstance(dataset_identity, str) or dataset_identity.lower() in ("", "unknown", "partial", "none"):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has unknown dataset identity; comparability is unverified"
        )
    if not dataset.get("name") or not dataset.get("split"):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has incomplete dataset identity"
        )
    inputs = record.get("model", {}).get("inputs", {})
    group = inputs.get("group_checkpoint")
    if group is not None and (group.get("status") != "available" or not group.get("sha256")):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has unknown group-checkpoint identity"
        )
    parameters = record.get("parameters", {})
    if not isinstance(parameters, dict):
        raise VisualizationError(
            f"Measurement {record.get('measurement_id')} has invalid expanded parameters"
        )
    parameters = dict(parameters)
    parameters.pop("quantile", None)
    parameters.pop("activation_quantile", None)
    return (
        _known_checkpoint(record),
        None if group is None else group["sha256"],
        dataset.get("name"),
        dataset.get("split"),
        dataset_identity,
        _protocol_without_quantile(record),
        parameters,
    )


def _ensure_external(path, root):
    target = Path(path).expanduser().resolve()
    canonical = (Path(root) / "measurements").resolve()
    if target == canonical or canonical in target.parents:
        raise CatalogError(f"Derived exports must be outside catalog measurements: {target}")
    return target


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _class_label(record, class_id):
    labels = record.get("protocol", {}).get("classes", {}).get("evaluation_class_id_to_label", {})
    return labels.get(str(class_id), labels.get(class_id, str(class_id)))


def _part_label(record, class_id, part_id):
    class_label = str(_class_label(record, class_id))
    definitions = record.get("protocol", {}).get("parts", {}).get("class_dependent_part_definitions")
    if isinstance(definitions, dict):
        definition = next(
            (value for name, value in definitions.items() if str(name).casefold() == class_label.casefold()),
            None,
        )
        if isinstance(definition, dict):
            names = definition.get("part_id_to_name")
            if isinstance(names, dict):
                name = names.get(str(part_id), names.get(part_id))
                if name:
                    return str(name)
            ids = definition.get("part_ids")
            if isinstance(ids, dict):
                for name, identifier in ids.items():
                    if identifier == part_id:
                        return str(name)
    return f"Part {part_id}"


def _tables(records, summaries):
    measurements = []
    heatmap_rows = []
    class_rows = []
    prototype_rows = []
    support_rows = []
    support_sets = []
    for record, summary in zip(records, summaries):
        measurement_id = record["measurement_id"]
        quantile = summary.get("quantile")
        measurements.append({
            "measurement_id": measurement_id,
            "quantile": quantile,
            "score": summary.get("score", {}).get("value"),
        })
        for row in summary.get("parts", []):
            heatmap_rows.append({
                "measurement_id": measurement_id,
                "quantile": quantile,
                "class_id": row["class_id"],
                "class_name": row.get("class_label") or _class_label(record, row["class_id"]),
                "prototype_id": row["prototype_id"],
                "part_id": row["part_id"],
                "part_name": _part_label(record, row["class_id"], row["part_id"]),
                "mean_presence": row["mean_presence"],
                "available_observations": row["available_observations"],
                "above_threshold": (
                    row["mean_presence"] is not None
                    and row["mean_presence"] > summary["threshold"]
                ),
            })
        for row in summary.get("classes", []):
            class_rows.append({
                "measurement_id": measurement_id,
                "quantile": quantile,
                "class_id": row["class_id"],
                "class_name": row.get("class_label") or _class_label(record, row["class_id"]),
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "consistency_fraction": row["value"],
            })
        for row in summary.get("prototypes", []):
            prototype_rows.append({
                "measurement_id": measurement_id,
                "quantile": quantile,
                "class_id": row["class_id"],
                "class_name": row.get("class_label") or _class_label(record, row["class_id"]),
                "prototype_id": row["prototype_id"],
                "max_part_frequency": row.get("max_part_frequency"),
                "consistent": row.get("consistent"),
            })
        current_support = {}
        for row in summary.get("support", []):
            key = f"{row['class_id']}:{row['part_id']}"
            current_support[key] = tuple(row["sample_ids"])
            support_rows.append({
                "measurement_id": measurement_id,
                "quantile": quantile,
                "class_id": row["class_id"],
                "class_name": _class_label(record, row["class_id"]),
                "part_id": row["part_id"],
                "part_name": _part_label(record, row["class_id"], row["part_id"]),
                "usable_images": row["usable_images"],
                "prototype_support_matches": row["prototype_support_matches"],
            })
        support_sets.append((current_support, all(
            row["prototype_support_matches"] for row in summary.get("support", [])
        )))
    support_comparable = bool(support_sets) and all(consistent for _, consistent in support_sets)
    if len(support_sets) > 1:
        baseline = support_sets[0][0]
        support_comparable = support_comparable and all(support == baseline for support, _ in support_sets[1:])
    return {
        "measurements": measurements,
        "heatmaps": heatmap_rows,
        "classes": class_rows,
        "prototypes": prototype_rows,
        "support": support_rows,
    }, support_comparable


def render_measurements(records, summaries, output_dir, root=None):
    """Save four independent views, exact-value tables, and a combined four-page PDF."""
    records = list(records)
    summaries = list(summaries)
    if not records or len(records) != len(summaries):
        raise VisualizationError("Select at least one measurement and provide its matching summary")
    if len({record.get("measurement_id") for record in records}) != len(records):
        raise VisualizationError("Repeated attempts must be selected by unique measurement ID")
    if any(record.get("metric") != "consistency" for record in records):
        raise VisualizationError("Consistency plots require consistency measurements")
    if any(record.get("status") != "complete" for record in records):
        raise VisualizationError("Only complete measurements can be plotted")
    if len(records) > 1:
        signatures = [_compatibility_signature(record) for record in records]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise VisualizationError(
                "Selected measurements differ in checkpoint, dataset, or non-quantile protocol fields"
            )
    quantiles = [summary.get("quantile") for summary in summaries]
    if any(value is None for value in quantiles):
        raise VisualizationError("Every selected measurement must have a protocol quantile")

    root = Path(root) if root is not None else resolve_catalog_root()
    output_dir = _ensure_external(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, support_comparable = _tables(records, summaries)
    support_rows = tables["support"]
    if not support_comparable:
        for row in support_rows:
            row["support_view"] = "separate per selected quantile/attempt; support sets differ"
    else:
        for row in support_rows:
            row["support_view"] = "matching support sets across selected measurements"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    figures = []
    colors = plt.get_cmap("tab10")
    heatmap_rows = tables["heatmaps"]
    class_ids = sorted({row["class_id"] for row in heatmap_rows})
    measurements = tables["measurements"]
    fig, axes = plt.subplots(
        max(1, len(class_ids)), max(1, len(records)),
        squeeze=False, figsize=(5 * max(1, len(records)), 2.4 * max(1, len(class_ids)) + 1.5),
    )
    for class_index, class_id in enumerate(class_ids or [None]):
        for measurement_index, record in enumerate(records):
            ax = axes[class_index][measurement_index]
            local = [
                row for row in heatmap_rows
                if row["class_id"] == class_id and row["measurement_id"] == record["measurement_id"]
            ]
            protos = sorted({row["prototype_id"] for row in local})
            local_parts = sorted({row["part_id"] for row in local})
            matrix = np.full((len(protos), len(local_parts)), np.nan, dtype=float)
            for row in local:
                matrix[protos.index(row["prototype_id"]), local_parts.index(row["part_id"])] = (
                    np.nan if row["mean_presence"] is None else row["mean_presence"]
                )
            if matrix.size:
                image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues", aspect="auto")
                for i, prototype_id in enumerate(protos):
                    for j, part_id in enumerate(local_parts):
                        value = matrix[i, j]
                        if np.isfinite(value):
                            ax.text(j, i, f"{value:.6g}", ha="center", va="center", fontsize=7)
                fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
            part_names = {
                row["part_id"]: row["part_name"] for row in local
            }
            ax.set_xticks(range(len(local_parts)), [part_names.get(value, f"Part {value}") for value in local_parts],
                          rotation=25, ha="right")
            ax.set_yticks(range(len(protos)), [f"P{value}" for value in protos])
            ax.set_title(
                f"{_class_label(record, class_id) if class_id is not None else 'No observations'}"
                f" · q={measurements[measurement_index]['quantile']} · {record['measurement_id'][:8]}"
            )
    fig.suptitle("Prototype-part consistency (exact recorded means)", fontsize=14)
    fig.tight_layout()
    figures.append(("01_prototype_part_heatmaps", fig))

    class_rows = tables["classes"]
    fig, ax = plt.subplots(figsize=(max(7, len(class_rows) * 1.1), 5))
    classes = sorted({row["class_id"] for row in class_rows})
    width = 0.8 / max(1, len(records))
    x = np.arange(len(classes))
    for index, record in enumerate(records):
        values = []
        for class_id in classes:
            row = next(
                (
                    item
                    for item in class_rows
                    if item["measurement_id"] == record["measurement_id"]
                    and item["class_id"] == class_id
                ),
                None,
            )
            values.append(np.nan if row is None else row["consistency_fraction"])
        ax.bar(
            x + index * width,
            values,
            width=width,
            label=f"q={quantiles[index]} · {record['measurement_id'][:8]}",
        )
    ax.set_xticks(
        x + width * (len(records) - 1) / 2,
        [_class_label(records[0], value) for value in classes],
        rotation=25,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Consistent class/prototype rows")
    ax.legend(frameon=False)
    ax.set_title("Consistency by class")
    fig.tight_layout()
    figures.append(("02_consistency_by_class", fig))

    fig, ax = plt.subplots(figsize=(8, 5))
    by_prototype = {}
    for row in tables["prototypes"]:
        if row["max_part_frequency"] is not None:
            by_prototype.setdefault((row["class_id"], row["prototype_id"]), []).append(row)
    has_repeat_quantiles = len(set(quantiles)) != len(quantiles)
    for index, ((class_id, prototype_id), rows) in enumerate(sorted(by_prototype.items())):
        points = sorted(rows, key=lambda row: (row["quantile"], row["measurement_id"]))
        xs = [row["quantile"] for row in points]
        ys = [row["max_part_frequency"] for row in points]
        label = f"{_class_label(records[0], class_id)} P{prototype_id}"
        if len(points) > 1 and not has_repeat_quantiles:
            ax.plot(xs, ys, marker="o", color=colors(index % 10), label=label)
        else:
            ax.scatter(xs, ys, color=colors(index % 10), label=label)
    for summary in summaries:
        threshold = summary.get("threshold")
        if threshold is not None:
            ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1)
            break
    ax.set_xlabel("Activation quantile")
    ax.set_ylabel("Maximum part mean presence")
    ax.set_ylim(0, 1.02)
    ax.set_title("Prototype scores across quantiles (selected attempts remain separate)")
    if len(by_prototype) <= 16:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    figures.append(("03_prototype_scores_across_quantiles", fig))

    fig, ax = plt.subplots(figsize=(max(7, len(support_rows) * 0.3), 5))
    support_labels = [
        f"{row['class_name']} / {row['part_name']}\nq={row['quantile']} · {row['measurement_id'][:8]}"
        for row in support_rows
    ]
    ax.bar(np.arange(len(support_rows)), [row["usable_images"] for row in support_rows],
           color=[colors(index % 10) for index in range(len(support_rows))])
    ax.set_xticks(np.arange(len(support_rows)), support_labels, rotation=40, ha="right")
    ax.set_ylabel("Unique images with an observed part value")
    ax.set_title(
        "Usable-image support" if support_comparable
        else "Usable-image support shown separately (sample sets differ or are incomplete)"
    )
    fig.tight_layout()
    figures.append(("04_usable_images", fig))

    table_fields = {
        "01_heatmap_values.csv": (tables["heatmaps"], [
            "measurement_id", "quantile", "class_id", "class_name", "prototype_id", "part_id",
            "part_name", "mean_presence", "available_observations", "above_threshold",
        ]),
        "02_class_consistency.csv": (tables["classes"], [
            "measurement_id", "quantile", "class_id", "class_name", "numerator", "denominator",
            "consistency_fraction",
        ]),
        "03_prototype_scores.csv": (tables["prototypes"], [
            "measurement_id", "quantile", "class_id", "class_name", "prototype_id",
            "max_part_frequency", "consistent",
        ]),
        "04_usable_images.csv": (tables["support"], [
            "measurement_id", "quantile", "class_id", "class_name", "part_id", "usable_images",
            "part_name", "prototype_support_matches", "support_view",
        ]),
    }
    for filename, (rows, fields) in table_fields.items():
        _write_csv(output_dir / filename, rows, fields)
    with PdfPages(output_dir / "all_four_visualizations.pdf") as combined:
        for name, figure in figures:
            figure.savefig(output_dir / f"{name}.png", dpi=150, facecolor="white")
            figure.savefig(output_dir / f"{name}.pdf", facecolor="white")
            combined.savefig(figure, facecolor="white")
            plt.close(figure)
    (output_dir / "visualization_metadata.json").write_text(
        json.dumps({
            "measurement_ids": [record["measurement_id"] for record in records],
            "quantiles": quantiles,
            "support_comparable": support_comparable,
            "support_policy": (
                "support sample sets matched across selected measurements"
                if support_comparable else "separate support values retained per selected measurement"
            ),
            "visualization_methods": [
                "prototype-part heatmap", "per-class consistency bars",
                "prototype maximum-part trajectories", "usable-image support bars",
            ],
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "support_comparable": support_comparable,
        "tables": tables,
    }
