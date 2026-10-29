"""Script to evaluate models for consistency"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import argh
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.equivariance import quantile_map
from segmentation.checkpoints import load_model_checkpoint
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
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model import PPNet
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import log


def proto_filter(ppnet: PPNetMultiScale) -> List[int]:
    """Callable to filter prototypes when not used in grouping mechanism."""
    tot_list_proto_ids = []
    for class_id in range(len(ppnet.group_projection)):
        proto_ids = torch.nonzero(ppnet.group_projection[class_id].weight.data.sum(dim=0))
        proto_ids = proto_ids.flatten().tolist()
        class_proto_ids = ppnet.prototype_class_identity[:, class_id].flatten().nonzero()
        class_proto_ids = class_proto_ids.flatten().tolist()
        filter_class_proto_ids = [class_proto_ids[proto_id] for proto_id in proto_ids]
        tot_list_proto_ids.extend(filter_class_proto_ids)
    return tot_list_proto_ids


def run_consistency(
    model_name: str,
    training_phase: str,
    data_type: str,
    *,
    quantile: float = 0.8,
    threshold: float = 0.8,
    group_name: Optional[str] = None,
    official_parts_only: bool = False,
    require_semantic_match: bool = False,
    exclude_background_centroid: bool = False,
):
    """Wrapper to compute the consistency across validation sets

    Args:
        model_name (str): Name of the model for testing.
        training_phase (str): Training phase for loading the model.
        data_type (str): Data type for testing: Cityscapes & Pascal.
        quantile (float, optional): Quantile to threshold the prototype activations. Defaults to 0.8.
        threshold (float, optional): Threshold for the consistency metric. Defaults to 0.8.
        group_name (str, optional): Name of the model when grouping is used. Defaults to None.
        official_parts_only (bool): Keep only official Cityscapes semantic/part pairs.
            Defaults to False. Filtered results go in an official_parts subdirectory.
        require_semantic_match (bool): For Pascal, keep positive parts only where
            panoptic SIDs match the original VOC foreground class IDs, as in ReProSeg.
            Defaults to False (legacy behavior). Results go in a semantic_match subdirectory.
        exclude_background_centroid (bool): Check only foreground part centroids.
            Defaults to False (legacy behavior). Results go in a foreground_centroids subdirectory.

    Raises:
        NotImplementedError: Data type not supported.
    """

    if official_parts_only and data_type != "cityscapes":
        raise ValueError("official_parts_only is supported only for cityscapes")
    if require_semantic_match and data_type != "pascal":
        raise ValueError("require_semantic_match is supported only for pascal")

    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
        # checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    output_path = os.path.join(model_path, f"metrics/{training_phase}")
    if official_parts_only:
        output_path = os.path.join(output_path, "official_parts")
    if require_semantic_match:
        output_path = os.path.join(output_path, "semantic_match")
    if exclude_background_centroid:
        output_path = os.path.join(output_path, "foreground_centroids")
    os.makedirs(output_path, exist_ok=True)

    global MAX_PARTS

    if data_type == "cityscapes":

        ID_MAPPING = CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = CITYSCAPES_CATEGORIES
        MAX_PARTS = MAX_PARTS_CITY
        FILTER_CLASS = []
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k] for i, k in pred2name.items()}

    elif data_type == "pascal":
        ID_MAPPING = PASCAL_ID_MAPPING
        CATEGORIES = PASCAL_CATEGORIES
        MAX_PARTS = MAX_PARTS_PASCAL
        FILTER_CLASS = PASCAL_FILTER_CLASS
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k + 1] for i, k in pred2name.items() if k < len(CATEGORIES) - 1}

    else:
        raise NotImplementedError(f"Data type {data_type} not supported")

    log(f"Loading model from {checkpoint_path}")
    log(f"Part labels: {'official Cityscapes parts only' if official_parts_only else 'all decoded part IDs'}")
    if data_type == "pascal":
        log(f"Pascal semantic agreement: {'required' if require_semantic_match else 'not required (legacy)'}")
    log(f"Part centroids: {'foreground only' if exclude_background_centroid else 'including background (legacy)'}")
    ppnet = load_model_checkpoint(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    if group_name is not None:
        group_path = os.path.join(os.environ["RESULTS_DIR"], group_name)
        checkpoint_path_group = os.path.join(group_path, f"checkpoints/th-0.05-nopush-group_last.pth")
        ppnet_group = load_model_checkpoint(checkpoint_path_group)
        ppnet_group.eval()
        proto_ids = proto_filter(ppnet_group)
    else:
        proto_ids = None

    val_dataset = PatchClassificationDataset(  # TODO: Adapt to new dataset
        data_type=data_type,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_margin_size=0,
        window_size=(513, 513),
        only_19_from_cityscapes=(data_type == "cityscapes"),
        scales=(0.5, 1.5),
        split_key="val",
        is_eval=True,
        push_prototypes=True,
    )

    i = 0
    tot_presence = []
    for sample_id in tqdm(
        range(len(val_dataset)), desc="Computing Consistency Score Validation", total=len(val_dataset)
    ):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        path_cls = Path(val_dataset.annotations_dir)
        path_part = str(path_cls.parent) + f"_PIDS/{path_cls.name}"

        # Attention for PASCAL only overlap
        if not os.path.exists(os.path.join(path_part, img_id + ".npy")):
            continue

        part_ann = np.load(os.path.join(path_part, img_id + ".npy"))
        if official_parts_only:
            # Filter in memory using native SIDs; keep the cache usable by both modes.
            sids_path = path_cls.parent.parent / "annotations_SIDS" / path_cls.name / f"{img_id}.npy"
            part_ann = filter_official_parts(np.load(sids_path), part_ann)
        cls_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))
        if require_semantic_match:
            sids_path = path_cls.parent.parent / "annotations_SIDS" / path_cls.name / f"{img_id}.npy"
            sids = np.load(sids_path)
            if sids.shape != cls_ann.shape or part_ann.shape != cls_ann.shape:
                raise ValueError(f"Pascal semantic, part and SID annotations must have matching shapes: {img_id}")
            # Compare native VOC IDs before convert_targets shifts 0..20 to 1..21.
            valid = (sids == cls_ann) & (sids >= 1) & (sids <= 20) & (part_ann > 0)
            part_ann = np.where(valid, part_ann, 0)
        cls_ann = val_dataset.convert_targets(cls_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        _, part_presence_list = part_intersect(
            img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, FILTER_CLASS,
            exclude_background_centroid=exclude_background_centroid,
        )

        tot_presence.extend(part_presence_list)
        i += 1

    part_columns = [f"part_{i}" for i in range(MAX_PARTS + 1)]
    df_presence = pd.DataFrame(tot_presence, columns=part_columns + ["proto_id", "class", "img_id"])
    df_presence.to_csv(os.path.join(output_path, f"part_presence_th_{threshold}_qt_{quantile}.csv"), index=False)

    df_mean = df_presence.groupby(["class", "proto_id"])[part_columns].mean().reset_index()
    df_flag = df_mean.copy()
    df_flag[[f"part_{i}" for i in range(MAX_PARTS + 1)]] = df_flag[[f"part_{i}" for i in range(MAX_PARTS + 1)]].apply(
        lambda x: (x > threshold).astype(int), axis=1
    )
    df_flag["is_consistent"] = df_flag[[f"part_{i}" for i in range(MAX_PARTS + 1)]].max(axis=1)
    df_mean = df_mean.merge(df_flag[["proto_id", "is_consistent"]], on="proto_id")
    df_mean.to_csv(os.path.join(output_path, f"part_presence_mean_th_{threshold}_qt_{quantile}.csv"), index=False)
    consistency_score = df_mean.is_consistent.mean()
    log(f"Consistency score: {consistency_score}")

    with open(os.path.join(output_path, f"consistency_score_th_{threshold}_qt_{quantile}.txt"), "w") as f:
        f.write(f"{consistency_score}")


@torch.no_grad()
def part_intersect(
    img: Image.Image,
    cls_ann: torch.Tensor,
    part_ann: torch.Tensor,
    ppnet: Union[PPNetMultiScale, PPNet],
    cls2name: Dict[int, str],
    img_id: int,
    quantile: float,
    filter_proto_ids: Optional[List[int]] = None,
    filter_class_ids: List[int] = [],
    *,
    exclude_background_centroid: bool = False,
) -> Tuple[Dict[int, List[int]], List[List[Any]]]:
    """Compute intersection between part labels and prototype activations."""

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    conv_features = ppnet.conv_features(img_tensor)
    _, distances = ppnet.forward_from_conv_features(conv_features)
    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    part_presence = {}
    part_presence_list = []

    for class_id in np.unique(cls_ann):

        if class_id == 0:
            continue

        if class_id in filter_class_ids:
            continue

        proto_ids = ppnet.prototype_class_identity[:, class_id - 1].flatten().nonzero()
        if filter_proto_ids is not None:
            proto_ids = [p for p in proto_ids if p in filter_proto_ids]
        y_mask = cls_ann == class_id

        part_centroids = {}
        part_mask = part_ann * y_mask

        for part_id in np.unique(part_mask):
            if part_id <= 0:
                continue
            bin_part_mask = (part_mask == part_id).astype(np.uint8)
            _, _, _, centroids = cv2.connectedComponentsWithStats(bin_part_mask, 8, cv2.CV_32S)
            # OpenCV returns the background (label 0) before the actual part components.
            if exclude_background_centroid:
                centroids = centroids[1:]
            centroids = np.round(centroids).astype(int)
            part_centroids[part_id] = centroids

        if len(part_centroids) == 0:
            continue

        else:
            for p in proto_ids:
                presence_list = [np.nan] * (MAX_PARTS + 1)
                proto_dist_img_j = proto_dist_[:, :, p]
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))

                upsampled_act_img_j = cv2.resize(
                    proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_NEAREST
                )
                upsampled_act_img_j = quantile_map(
                    torch.from_numpy(upsampled_act_img_j[None, :, :]),
                    quantile,
                    class_mask=torch.from_numpy(y_mask),
                ).numpy().astype(int)

                for part_id, centroids in part_centroids.items():
                    presence_list[part_id] = 0
                    for centroid in centroids:
                        x, y = centroid[0], centroid[1]
                        presence_list[part_id] += upsampled_act_img_j[y, x, 0]
                part_presence[p.item()] = [np.nan if np.isnan(val) else 0 if val == 0 else 1 for val in presence_list]
                part_presence_list.append(part_presence[p.item()] + [p.item(), cls2name[class_id - 1], img_id])

    return part_presence, part_presence_list


if __name__ == "__main__":
    argh.dispatch_command(run_consistency)
