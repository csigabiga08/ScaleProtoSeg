"""CPU-only summaries and correlations of recorded prototype part activations."""

import numpy as np
import pandas as pd

from segmentation.data.pascal_parts import PASCAL_PART_LABELS


def mean_part_activations(observations, part_columns):
    """Average each class/prototype's parts, counting only nonmissing values per part."""
    frame = pd.DataFrame(observations, columns=["class_id", "prototype_id", *part_columns])
    frame[part_columns] = frame[part_columns].replace("", np.nan).astype(float)
    if np.isinf(frame[part_columns].to_numpy()).any():
        raise ValueError("Part activations must be finite or missing")
    return frame.groupby(["class_id", "prototype_id"])[part_columns].mean()


def consistent_prototypes(activations, threshold):
    """Match consistency scoring: at least one mean part activation strictly above threshold."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Consistency threshold must be between 0 and 1")
    return activations.gt(threshold).any(axis=1)


def _semantic_part_ids(record, class_id):
    labels = record["protocol"].get("classes", {}).get("evaluation_class_id_to_label", {})
    label = labels.get(str(class_id))
    dataset = record["dataset"].get("name")
    if dataset == "pascal":
        if label == "__background__":
            return set()
        if label not in PASCAL_PART_LABELS:
            raise ValueError(f"No Pascal part taxonomy for evaluation class {class_id}: {label}")
        ids = set(range(1, len(PASCAL_PART_LABELS[label]) + 1))
        parts = record["protocol"].get("parts", {})
        if parts.get("merge_left_right_parts"):
            if "left_right_part_groups" not in parts:
                raise ValueError("Missing recorded left/right part groups")
            for group in parts["left_right_part_groups"].get(label, []):
                ids.difference_update(set(group["source_part_ids"]) - {group["canonical_part_id"]})
        return ids
    if dataset == "cityscapes":
        if label is None:
            raise ValueError(f"Missing Cityscapes label for evaluation class {class_id}")
        if label in ("person", "rider"):
            return set(range(1, 5))
        if label in ("car", "truck", "bus"):
            return set(range(1, 6))
        return set()
    raise ValueError(f"No semantic part taxonomy for dataset: {dataset}")


def prototype_part_correlation(record, rows, *, consistent_only=False, semantic_parts_only=False, threshold=None):
    """Return mean vectors, Pearson correlations and shared-part counts.

    CSV part IDs are class-local. Coordinates are (evaluation class, part ID),
    so prototypes from different classes never share parts. Missing coordinates
    stay missing, including after semantic-class filtering.
    """
    columns = [
        column
        for column in record["protocol"].get("observation_columns", [])
        if column.startswith("part_") and column[5:].isdigit() and int(column[5:]) > 0
    ]
    columns.sort(key=lambda column: int(column[5:]))
    if not columns:
        raise ValueError("Measurement has no part-activation columns")
    observations = pd.DataFrame(rows, columns=["class_id", "prototype_id", *columns])
    for key in ("class_id", "prototype_id"):
        observations[key] = pd.to_numeric(observations[key], errors="raise")
        if not np.isfinite(observations[key]).all() or (observations[key] % 1 != 0).any():
            raise ValueError(f"{key} must contain integer IDs")
        observations[key] = observations[key].astype(int)
    means = mean_part_activations(observations, columns)
    if means.index.get_level_values("prototype_id").duplicated().any():
        raise ValueError("A prototype must belong to exactly one semantic class")
    if consistent_only:
        if threshold is None:
            threshold = record["protocol"].get("consistency", {}).get("threshold")
        if threshold is None:
            raise ValueError("A consistency threshold is required to filter prototypes")
        means = means.loc[consistent_prototypes(means, threshold)]

    prototype_ids = pd.Index(means.index.get_level_values("prototype_id"), name="prototype_id")
    coordinates = pd.MultiIndex.from_tuples(
        [(class_id, column) for class_id in means.index.get_level_values("class_id").unique() for column in columns],
        names=["class_id", "part"],
    )
    activations = pd.DataFrame(np.nan, index=prototype_ids, columns=coordinates)
    for (class_id, prototype_id), vector in means.iterrows():
        allowed = _semantic_part_ids(record, class_id) if semantic_parts_only else None
        for column, value in vector.items():
            if allowed is None or int(column[5:]) in allowed:
                activations.loc[prototype_id, (class_id, column)] = value
    present = activations.notna().astype("int64")
    shared = pd.DataFrame(present.to_numpy() @ present.to_numpy().T, index=prototype_ids, columns=prototype_ids)
    correlation = activations.T.corr(method="pearson", min_periods=2)
    return activations, correlation, shared
