"""Shared dataset preparation and part-presence computation for interpretability metrics."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from segmentation.analysis.equivariance import quantile_map
from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    MAX_PARTS_CITY,
    MAX_PARTS_PASCAL,
    PASCAL_CATEGORIES,
    PASCAL_FILTER_CLASS,
    PASCAL_ID_MAPPING,
)
from segmentation.data.cityscapes_parts import filter_official_parts
from segmentation.data.pascal_parts import merge_left_right_presence


@dataclass(frozen=True)
class EvaluationContext:
    data_type: str
    semantic_ids: dict
    class_labels: dict
    max_parts: int
    excluded_classes: tuple

    @classmethod
    def for_dataset(cls, data_type):
        if data_type == "cityscapes":
            mapping = CITYSCAPES_19_EVAL_CATEGORIES
            labels = {value - 1: CITYSCAPES_CATEGORIES[key] for key, value in mapping.items() if value > 0}
            return cls(data_type, mapping, labels, MAX_PARTS_CITY, ())
        if data_type == "pascal":
            mapping = PASCAL_ID_MAPPING
            labels = {
                value - 1: PASCAL_CATEGORIES[key + 1]
                for key, value in mapping.items()
                if value > 0 and key < len(PASCAL_CATEGORIES) - 1
            }
            return cls(data_type, mapping, labels, MAX_PARTS_PASCAL, tuple(PASCAL_FILTER_CLASS))
        raise NotImplementedError(f"Data type {data_type} not supported")

    @property
    def dataset_options(self):
        return dict(
            data_type=self.data_type, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            image_margin_size=0, window_size=(513, 513), only_19_from_cityscapes=self.data_type == "cityscapes",
            scales=(0.5, 1.5), split_key="val", is_eval=True, push_prototypes=True,
        )


def load_part_annotations(dataset, img_id, run_input):
    classes = Path(dataset.annotations_dir)
    parts_path = classes.parent.with_name(classes.parent.name + "_PIDS") / classes.name / f"{img_id}.npy"
    if not parts_path.is_file():
        return None
    parts = np.load(parts_path)
    semantic = np.load(classes / f"{img_id}.npy")
    if run_input.official_parts_only or run_input.require_semantic_match:
        sids = np.load(classes.parent.parent / "annotations_SIDS" / classes.name / f"{img_id}.npy")
        if run_input.official_parts_only:
            parts = filter_official_parts(sids, parts)
        if run_input.require_semantic_match:
            if sids.shape != semantic.shape or parts.shape != semantic.shape:
                raise ValueError(f"Pascal semantic, part and SID annotations must have matching shapes: {img_id}")
            valid = (sids == semantic) & (sids >= 1) & (sids <= 20) & (parts > 0)
            parts = np.where(valid, parts, 0)
    return dataset.convert_targets(semantic), parts


def part_intersections(
    img_tensor, img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile,
    filter_proto_ids, filter_class_ids, *, max_parts, missing_parts_nan,
    exclude_background_centroid=False, merge_left_right_parts=False,
):
    conv_features = ppnet.conv_features(img_tensor)
    _, distances = ppnet.forward_from_conv_features(conv_features)
    proto_dist = distances[0].permute(1, 2, 0).detach().cpu().numpy()
    width, height = img.size
    part_presence, rows = {}, []
    for class_id in np.unique(cls_ann):
        if class_id == 0 or class_id in filter_class_ids:
            continue
        proto_ids = ppnet.prototype_class_identity[:, class_id - 1].flatten().nonzero()
        if filter_proto_ids is not None:
            proto_ids = [p for p in proto_ids if p in filter_proto_ids]
        class_mask = cls_ann == class_id
        part_mask = part_ann * class_mask
        part_centroids = {}
        for part_id in np.unique(part_mask):
            if part_id <= 0:
                continue
            binary = (part_mask == part_id).astype(np.uint8)
            _, _, _, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            if exclude_background_centroid:
                centroids = centroids[1:]
            part_centroids[part_id] = np.round(centroids).astype(int)
        if not part_centroids:
            continue
        for prototype in proto_ids:
            presence = [np.nan if missing_parts_nan else 0] * (max_parts + 1)
            distance = proto_dist[:, :, prototype]
            activation = np.log((distance + 1) / (distance + ppnet.epsilon))
            activation = cv2.resize(activation, (width, height), interpolation=cv2.INTER_NEAREST)
            activation = quantile_map(
                torch.from_numpy(activation[None, :, :]), quantile, class_mask=torch.from_numpy(class_mask),
            ).numpy()
            for part_id, centroids in part_centroids.items():
                presence[part_id] = 0
                for x, y in centroids:
                    presence[part_id] += activation[y, x, 0]
            hits = [np.nan if np.isnan(value) else int(value != 0) for value in presence]
            if merge_left_right_parts:
                hits = merge_left_right_presence(hits, cls2name[class_id - 1])
                if not missing_parts_nan:
                    hits = [0 if np.isnan(hit) else hit for hit in hits]
            part_presence[prototype.item()] = hits
            rows.append(hits + [prototype.item(), cls2name[class_id - 1], img_id])
    return part_presence, rows
