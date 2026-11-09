"""Script to evaluate models for stability"""

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
from segmentation.analysis.metrics.output import MetricOutput
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
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.data.cityscapes_parts import filter_official_parts
from segmentation.data.pascal_parts import left_right_part_groups, merge_left_right_presence
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


@argh.arg("--quantile-scope", choices=("class", "image"))
def run_stability(
    model_name: str,
    training_phase: str,
    data_type: str,
    quantile: float = 0.8,
    std_dev: float = 0.2,
    group_name: str = None,
    checkpoint_path: str = None,
    *,
    output_dir: Optional[str] = None,
    group_checkpoint: Optional[str] = None,
    official_parts_only: bool = False,
    require_semantic_match: bool = False,
    seed: int = 0,
    exclude_background_centroid: bool = False,
    merge_left_right_parts: bool = False,
    quantile_scope: str = "class",
):
    """Wrapper to compute the stability across validation sets

    Args:
        model_name (str): Name of the model for testing.
        training_phase (str):  Training phase for loading the model.
        data_type (str): Data type for testing: Cityscapes & Pascal.
        quantile (float, optional): Quantile to threshold the prototype activations. Defaults to 0.8.
        quantile_scope (str): "class" (default) restricts the quantile and active
            area to the semantic class; "image" uses the whole image without masking.
        std_dev (float, optional): Standard deviation of the noise injected. Defaults to 0.2.
        group_name (str, optional): Name of the model when grouping is used. Defaults to None.
        output_dir (str, optional): Exact output directory, overriding the default
            metrics-allparts/filteredparts-foreground/background folder in the model
            directory. Parameters and a run description are saved per filename.
        group_checkpoint (str, optional): Explicit prototype-filter checkpoint,
            overriding group_name.
        official_parts_only (bool): Keep only documented Cityscapes semantic/part pairs.
        require_semantic_match (bool): For Pascal, require part/VOC semantic agreement.
        seed (int): Random noise seed. Defaults to 0.
        merge_left_right_parts (bool): Pascal only: OR left/right centroid hits
            separately for clean and noisy images before comparing part vectors.

    Raises:
        NotImplementedError: Data type not supported.
    """

    if quantile_scope not in ("class", "image"):
        raise ValueError("quantile_scope must be 'class' or 'image'")
    if official_parts_only and data_type != "cityscapes":
        raise ValueError("official_parts_only is supported only for cityscapes")
    if require_semantic_match and data_type != "pascal":
        raise ValueError("require_semantic_match is supported only for pascal")

    if merge_left_right_parts and data_type != "pascal":
        raise ValueError("merge_left_right_parts is supported only for pascal")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)
    elif training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
        # checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    global MAX_PARTS

    if data_type == "cityscapes":  # TODO: Potentially adapt to new mapping

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
    log(f"Activation quantile scope: {quantile_scope}")
    log(f"Part labels: {'official Cityscapes parts only' if official_parts_only else 'all decoded part IDs'}")
    if data_type == "pascal":
        log(f"Pascal semantic agreement: {'required' if require_semantic_match else 'not required (legacy)'}")
        log(f"Left/right parts: {'merged with per-image OR' if merge_left_right_parts else 'separate (legacy)'}")
    log(f"Part centroids: {'foreground only' if exclude_background_centroid else 'including background (legacy)'}")
    ppnet = load_model_checkpoint(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    checkpoint_path_group = None
    if group_checkpoint is not None or group_name is not None:
        if group_checkpoint is not None:
            checkpoint_path_group = os.path.expanduser(group_checkpoint)
        else:
            group_path = os.path.join(os.environ["RESULTS_DIR"], group_name)
            checkpoint_path_group = os.path.join(group_path, "checkpoints/th-0.05-nopush-group_last.pth")
        log(f"Loading prototype filter from {checkpoint_path_group}")
        ppnet_group = load_model_checkpoint(checkpoint_path_group)
        ppnet_group.eval()
        proto_ids = proto_filter(ppnet_group)
    else:
        proto_ids = None

    output = MetricOutput(
        "stability", checkpoint_path, data_type, model_name=model_name,
        training_phase=training_phase, quantile=quantile, std_dev=std_dev, seed=seed,
        group_checkpoint=checkpoint_path_group, official_parts_only=official_parts_only,
        require_semantic_match=require_semantic_match,
        exclude_background_centroid=exclude_background_centroid, output_dir=output_dir,
        merge_left_right_parts=merge_left_right_parts,
        quantile_scope=quantile_scope,
        part_groups=left_right_part_groups() if merge_left_right_parts else None,
    )
    output_path = str(output.path)
    log(f"Results directory: {output_path}")

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

    output.record_dataset(val_dataset)
    i = 0
    tot_presence = []
    tot_presence_noise = []

    for sample_id in tqdm(range(len(val_dataset)), desc="Computing Stability Score Validation", total=len(val_dataset)):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        path_cls = Path(val_dataset.annotations_dir)
        path_part = str(path_cls.parent) + f"_PIDS/{path_cls.name}"

        # Attention for PASCAL only overlap
        if not os.path.exists(os.path.join(path_part, img_id + ".npy")):
            continue

        part_ann = np.load(os.path.join(path_part, img_id + ".npy"))
        if official_parts_only:
            sids_path = path_cls.parent.parent / "annotations_SIDS" / path_cls.name / f"{img_id}.npy"
            part_ann = filter_official_parts(np.load(sids_path), part_ann)
        cls_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))
        if require_semantic_match:
            sids_path = path_cls.parent.parent / "annotations_SIDS" / path_cls.name / f"{img_id}.npy"
            sids = np.load(sids_path)
            if sids.shape != cls_ann.shape or part_ann.shape != cls_ann.shape:
                raise ValueError(f"Pascal semantic, part and SID annotations must have matching shapes: {img_id}")
            valid = (sids == cls_ann) & (sids >= 1) & (sids <= 20) & (part_ann > 0)
            part_ann = np.where(valid, part_ann, 0)
        cls_ann = val_dataset.convert_targets(cls_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        _, part_presence_list = part_intersect(
            img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids,
            exclude_background_centroid=exclude_background_centroid,
            merge_left_right_parts=merge_left_right_parts,
            quantile_scope=quantile_scope,
        )
        _, noise_part_presence_list = part_intersect(
            img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, std_dev,
            exclude_background_centroid=exclude_background_centroid,
            merge_left_right_parts=merge_left_right_parts,
            quantile_scope=quantile_scope,
        )

        tot_presence.extend(part_presence_list)
        tot_presence_noise.extend(noise_part_presence_list)
        i += 1

    df_presence = pd.DataFrame(
        tot_presence, columns=[f"part_{i}" for i in range(MAX_PARTS + 1)] + ["proto_id", "class", "img_id"]
    )
    df_presence_noise = pd.DataFrame(
        tot_presence_noise, columns=[f"part_stab_{i}" for i in range(MAX_PARTS + 1)] + ["proto_id", "class", "img_id"]
    )

    df_tot = df_presence.merge(df_presence_noise, on=["proto_id", "class", "img_id"])
    for i in range(MAX_PARTS + 1):
        df_tot[f"eq_part_{i}"] = (df_tot[f"part_{i}"] == df_tot[f"part_stab_{i}"]).astype(int)
    df_tot["sum_eq"] = df_tot[[f"eq_part_{i}" for i in range(MAX_PARTS + 1)]].sum(axis=1)
    df_tot["is_stable"] = (df_tot["sum_eq"] == MAX_PARTS + 1).astype(int)
    df_tot.to_csv(output.file("tot_presence_stab", "csv"), index=False)
    stab_score = df_tot.is_stable.mean()
    log(f"Stability score: {stab_score}")

    with output.file("stability_score", "txt").open("w") as f:
        f.write(f"{stab_score}")
    output.finish(stab_score, i, len(df_tot))


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
    noise: Optional[float] = None,
    filter_class_ids: List[int] = [],
    *,
    exclude_background_centroid: bool = False,
    merge_left_right_parts: bool = False,
    quantile_scope: str = "class",
) -> Tuple[Dict[int, List[int]], List[List[Any]]]:
    """Compute intersection between part labels and prototype activations."""

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()

    if noise is not None:
        noise = noise * torch.randn_like(img_tensor)
        img_tensor = img_tensor + noise
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
                presence_list = [0] * (MAX_PARTS + 1)  # Can keep 0 here instead of np.nan
                proto_dist_img_j = proto_dist_[:, :, p]
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))

                upsampled_act_img_j = cv2.resize(
                    proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_NEAREST
                )
                upsampled_act_img_j = quantile_map(
                    torch.from_numpy(upsampled_act_img_j[None, :, :]),
                    quantile,
                    class_mask=torch.from_numpy(y_mask),
                    scope=quantile_scope,
                ).numpy()

                for part_id, centroids in part_centroids.items():
                    for centroid in centroids:
                        x, y = centroid[0], centroid[1]
                        presence_list[part_id] += upsampled_act_img_j[y, x, 0]
                hits = [0 if val == 0 else 1 for val in presence_list]
                if merge_left_right_parts:
                    # Keep the stability vector binary: redundant right-hand columns
                    # are zero in both passes, so they cannot introduce NaN != NaN.
                    hits = [
                        0 if np.isnan(hit) else hit
                        for hit in merge_left_right_presence(hits, cls2name[class_id - 1])
                    ]
                part_presence[p.item()] = hits
                part_presence_list.append(part_presence[p.item()] + [p.item(), cls2name[class_id - 1], img_id])

    return part_presence, part_presence_list


if __name__ == "__main__":
    argh.dispatch_command(run_stability)
