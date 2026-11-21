"""Measure activation outside each prototype's ground-truth semantic class."""

import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.metrics.helpers import EvaluationContext
from segmentation.analysis.metrics.pipeline import MetricPlugin, run_metric
from segmentation.analysis.metrics.protocol import ImpreciseActivationsRunInput


def _class_pixel_counts(labels, activation_shape, num_classes):
    """Count target pixels of each class covered by each native activation cell."""
    cells = np.arange(math.prod(activation_shape), dtype=np.float64).reshape(activation_shape)
    height, width = labels.shape
    pixel_cells = cv2.resize(cells, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    cell_count = cells.size
    offsets = labels.astype(np.int64) * cell_count + pixel_cells
    counts = np.bincount(offsets.ravel(), minlength=(num_classes + 1) * cell_count)
    return counts.reshape(num_classes + 1, cell_count)[1:]


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


def _image_activation_scores(sums, counts):
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    inside, outside = means.T
    return np.divide(
        outside,
        inside + outside,
        out=np.full_like(inside, np.nan),
        where=np.isfinite(inside) & np.isfinite(outside) & ((inside + outside) != 0),
    )


def _mean_per_prototype(image_scores):
    image_scores = np.asarray(image_scores, dtype=np.float64)
    if image_scores.ndim != 2:
        raise ValueError("Image scores must be a two-dimensional array")
    defined = np.isfinite(image_scores)
    counts = defined.sum(axis=0)
    sums = np.where(defined, image_scores, 0).sum(axis=0)
    return np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)


class ImpreciseActivationsMetric(MetricPlugin):
    name = "imprecise_activations"
    empty_reason = "no prototypes with a defined imprecise activation score"

    def __init__(self):
        self.image_scores = []
        self.selected_count = 0

    def model_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_protocol(self, run_input, context):
        return run_input.to_protocol(
            semantic_id_to_evaluation_id=context.semantic_ids,
            evaluation_class_id_to_label={str(key + 1): value for key, value in context.class_labels.items()},
        )

    @torch.inference_mode()
    def evaluate(
        self,
        run_input,
        context,
        model,
        dataset,
        output,
        prototype_ids,
        selected_ids,
    ):
        self.selected_count = len(selected_ids)
        assignments = model.prototype_class_identity.detach().cpu().numpy() != 0
        num_classes = len(context.class_labels)
        if assignments.ndim != 2 or assignments.shape[1] != num_classes or assignments.shape[0] == 0:
            raise ValueError(f"Expected a nonempty prototype assignment matrix with {num_classes} classes")
        if not np.all(assignments.sum(axis=1) == 1):
            raise ValueError("Each prototype must be assigned to exactly one semantic class")
        if not selected_ids:
            return
        prototype_classes = assignments[selected_ids].argmax(axis=1)
        num_prototypes = len(selected_ids)
        device = self.model_device()

        for image_id in tqdm(dataset.img_ids, desc="Computing imprecise activations"):
            labels = np.load(Path(dataset.annotations_dir) / f"{image_id}.npy")
            if labels.ndim == 3:
                labels = labels[:, :, 0]
            if dataset.convert_targets is not None:
                labels = dataset.convert_targets(labels)
            labels = np.asarray(labels)
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Semantic annotation contains unmapped labels for {image_id}")
            if labels.ndim != 2 or np.any(labels < 0) or np.any(labels > num_classes):
                raise ValueError(f"Invalid semantic labels for {image_id}")
            if not np.any(labels):
                continue

            with Image.open(dataset.get_img_path(image_id)) as source_image:
                image = source_image.convert("RGB")
                if labels.shape != (image.height, image.width):
                    raise ValueError(f"Image and semantic annotation sizes differ for {image_id}")
                image_tensor = to_normalized_tensor(image).unsqueeze(0).to(device)

            distances = model.prototype_distances(image_tensor)[0]
            activations = torch.log((distances + 1) / (distances + model.epsilon)).cpu().numpy()
            if (
                activations.ndim != 3
                or activations.shape[0] != assignments.shape[0]
                or not np.isfinite(activations).all()
                or np.any(activations < 0)
            ):
                raise ValueError(f"Invalid prototype activation maps for {image_id}")

            activations = activations[selected_ids]
            class_counts = _class_pixel_counts(labels, activations.shape[-2:], num_classes)
            image_sums = _activation_sums(activations, class_counts, prototype_classes)
            image_class_counts = class_counts.sum(axis=1)
            valid_pixels = int(image_class_counts.sum())
            image_counts = np.column_stack(
                (image_class_counts[prototype_classes], valid_pixels - image_class_counts[prototype_classes])
            )
            scores = _image_activation_scores(image_sums, image_counts)
            self.image_scores.append(scores)
            for output_index, (prototype_id, class_index) in enumerate(zip(selected_ids, prototype_classes)):
                output.add_observation(
                    sample_id=image_id,
                    class_id=int(class_index) + 1,
                    prototype_id=prototype_id,
                    repetition=0,
                    inside_activation_sum=image_sums[output_index, 0],
                    inside_pixel_count=image_counts[output_index, 0],
                    outside_activation_sum=image_sums[output_index, 1],
                    outside_pixel_count=image_counts[output_index, 1],
                    imprecise_activation=float(scores[output_index]) if np.isfinite(scores[output_index]) else None,
                )

    def summarize(self, run_input, context):
        image_score_matrix = (
            np.asarray(self.image_scores, dtype=np.float64)
            if self.image_scores
            else np.empty((0, self.selected_count), dtype=np.float64)
        )
        prototype_scores = _mean_per_prototype(image_score_matrix)
        defined = np.isfinite(prototype_scores)
        mean_score = float(prototype_scores[defined].mean()) if defined.any() else float("nan")
        return mean_score, float(prototype_scores[defined].sum()), int(defined.sum())


def run_imprecise_activations(run_input: ImpreciseActivationsRunInput):
    return run_metric(run_input, ImpreciseActivationsMetric())


def _run_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Imprecise activations configuration must resolve to a mapping")
    values.pop("hydra", None)
    run_imprecise_activations(ImpreciseActivationsRunInput.model_validate(values))


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parent / "../../../config/hydra/metrics").resolve()),
    config_name="imprecise_activations",
)
def main(config: DictConfig):
    _run_config(config)


if __name__ == "__main__":
    main()
