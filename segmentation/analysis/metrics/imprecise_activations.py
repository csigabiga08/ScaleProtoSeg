"""Measure activation outside each prototype's ground-truth semantic class."""

import math
import os
from pathlib import Path
from typing import Optional

import argh
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.metrics.output import MetricOutput
from segmentation.checkpoints import load_model_checkpoint
from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, CITYSCAPES_CATEGORIES, PASCAL_CATEGORIES
from segmentation.data.dataset import PatchClassificationDataset
from settings import log


def _class_pixel_counts(labels, activation_shape, num_classes):
    """Count target pixels of each class covered by each native activation cell."""
    cells = np.arange(math.prod(activation_shape), dtype=np.float64).reshape(activation_shape)
    height, width = labels.shape
    # Match the nearest-neighbor resize used by the part metrics, without
    # materializing a full-resolution activation map for every prototype.
    pixel_cells = cv2.resize(cells, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    cell_count = cells.size
    offsets = labels.astype(np.int64) * cell_count + pixel_cells
    counts = np.bincount(offsets.ravel(), minlength=(num_classes + 1) * cell_count)
    return counts.reshape(num_classes + 1, cell_count)[1:]  # Converted label 0 is void.


def _activation_sums(activations, class_counts, prototype_classes):
    """Sum nearest-neighbor upsampled activations inside and outside each assigned class."""
    flattened = np.asarray(activations, dtype=np.float64).reshape(len(prototype_classes), -1)
    valid_counts = class_counts.sum(axis=0)
    sums = np.zeros((len(prototype_classes), 2), dtype=np.float64)
    for class_index in np.unique(prototype_classes):
        prototype_ids = np.flatnonzero(prototype_classes == class_index)
        inside_counts = class_counts[class_index]
        weights = np.column_stack((inside_counts, valid_counts - inside_counts))
        sums[prototype_ids] = flattened[prototype_ids] @ weights
    return sums


@argh.arg("data_type", choices=("cityscapes", "pascal"))
@torch.inference_mode()
def run_imprecise_activations(
    model_name: str,
    training_phase: str,
    data_type: str,
    *,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Compute the outside/inside mean activation ratio for each prototype.

    Args:
        model_name: Name of the evaluated model run.
        training_phase: Phase used to resolve the default <phase>_last.pth checkpoint.
        data_type: Semantic validation dataset, either cityscapes or pascal.
        checkpoint: Explicit path to a trained ProtoSeg or ScaleProto model.
        output_dir: Override the metrics-imprecise-activations result directory.
    """
    if data_type == "cityscapes":
        class_names = {
            target - 1: CITYSCAPES_CATEGORIES[source]
            for source, target in CITYSCAPES_19_EVAL_CATEGORIES.items()
            if target > 0
        }
    elif data_type == "pascal":
        class_names = dict(enumerate(PASCAL_CATEGORIES[1:]))
    else:
        raise ValueError("data_type must be 'cityscapes' or 'pascal'")

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser()
    else:
        model_path = Path(os.environ["RESULTS_DIR"]) / model_name
        if training_phase == "pruned":
            checkpoint_path = model_path / "pruned/checkpoints/push_last.pth"
        else:
            checkpoint_path = model_path / "checkpoints" / f"{training_phase}_last.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading model from {checkpoint_path} on {device}")
    model = load_model_checkpoint(checkpoint_path).to(device).eval()
    assignments = model.prototype_class_identity.detach().cpu().numpy() != 0
    num_classes = len(class_names)
    if assignments.ndim != 2 or assignments.shape[1] != num_classes or assignments.shape[0] == 0:
        raise ValueError(f"Expected a nonempty prototype assignment matrix with {num_classes} {data_type} classes")
    if not np.all(assignments.sum(axis=1) == 1):
        raise ValueError("Each prototype must be assigned to exactly one semantic class")
    prototype_classes = assignments.argmax(axis=1)
    num_prototypes = len(prototype_classes)

    dataset = PatchClassificationDataset(
        data_type=data_type,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_margin_size=0,
        only_19_from_cityscapes=(data_type == "cityscapes"),
        split_key="val",
        is_eval=True,
        push_prototypes=True,
    )
    output = MetricOutput(
        "imprecise_activations", checkpoint_path, data_type, model_name=model_name,
        training_phase=training_phase, output_dir=output_dir,
    )
    output.description.update(
        activation="log_similarity",
        aggregation="pooled_validation_pixels",
        interpolation="nearest",
    )
    output.record_dataset(dataset)
    log(f"Results directory: {output.path}")

    sums = np.zeros((num_prototypes, 2), dtype=np.float64)
    class_pixel_totals = np.zeros(num_classes, dtype=np.int64)
    evaluated_images = 0
    for image_id in tqdm(dataset.img_ids, desc="Computing imprecise activations"):
        labels = np.load(Path(dataset.annotations_dir) / f"{image_id}.npy")
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        labels = dataset.convert_targets(labels)
        with Image.open(dataset.get_img_path(image_id)) as image:
            if labels.shape != (image.height, image.width):
                raise ValueError(f"Image and semantic annotation sizes differ for {image_id}")
            if not np.any(labels):
                continue
            image_tensor = to_normalized_tensor(image.convert("RGB")).unsqueeze(0).to(device)

        distances = model.prototype_distances(image_tensor)[0]
        activations = torch.log((distances + 1) / (distances + model.epsilon)).cpu().numpy()
        if activations.ndim != 3 or activations.shape[0] != num_prototypes or not np.isfinite(activations).all():
            raise ValueError(f"Invalid prototype activation maps for {image_id}")
        class_counts = _class_pixel_counts(labels, activations.shape[-2:], num_classes)
        class_pixel_totals += class_counts.sum(axis=1)
        sums += _activation_sums(activations, class_counts, prototype_classes)
        evaluated_images += 1

    inside_pixels = class_pixel_totals[prototype_classes]
    outside_pixels = class_pixel_totals.sum() - inside_pixels
    counts = np.column_stack((inside_pixels, outside_pixels))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    inside, outside = means.T
    ratios = np.divide(
        outside, inside, out=np.full_like(inside, np.nan),
        where=np.isfinite(inside) & np.isfinite(outside) & (inside != 0),
    )
    results = pd.DataFrame({
        "proto_id": np.arange(num_prototypes),
        "class_index": prototype_classes,
        "class": [class_names[index] for index in prototype_classes],
        "inside_pixels": inside_pixels,
        "outside_pixels": outside_pixels,
        "inside_mean": inside,
        "outside_mean": outside,
        "imprecise_activation": ratios,
    })
    result_path = output.file("imprecise_activations", "csv")
    results.to_csv(result_path, index=False, na_rep="NaN")
    defined = np.isfinite(ratios)
    mean_ratio = float(ratios[defined].mean()) if defined.any() else float("nan")
    output.description["prototypes_with_defined_ratio"] = int(defined.sum())
    output.description["valid_pixels"] = int(class_pixel_totals.sum())
    output.finish(mean_ratio, evaluated_images, num_prototypes)
    log(f"Mean imprecise activation: {mean_ratio} ({defined.sum()}/{num_prototypes} prototypes)")
    log(f"Per-prototype results: {result_path}")


if __name__ == "__main__":
    argh.dispatch_command(run_imprecise_activations)
