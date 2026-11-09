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
    PASCAL_ID_MAPPING,
)
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.data.cityscapes_parts import filter_official_parts
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


def _selected_prototypes(model, allowed_ids):
    identity = getattr(model, "prototype_class_identity", None)
    if not isinstance(identity, torch.Tensor) or identity.ndim != 2:
        return []
    identity = identity.detach().cpu().numpy()
    allowed = None if allowed_ids is None else set(int(value) for value in allowed_ids)
    return [
        int(proto_id)
        for class_index in range(identity.shape[1])
        for proto_id in np.flatnonzero(identity[:, class_index])
        if allowed is None or int(proto_id) in allowed
    ]


def _protocol(data_type, class_map, semantic_map, max_parts, *, quantile, std_dev, seed,
              official_parts_only, require_semantic_match, exclude_background_centroid):
    return {
        "metric": "stability",
        "quantile": quantile,
        "parts": {
            "valid_part_ids": list(range(1, max_parts + 1)),
            "official_parts_only": official_parts_only,
            "require_semantic_match": require_semantic_match,
            "exclude_background_centroid": exclude_background_centroid,
        },
        "classes": {
            "semantic_id_to_evaluation_id": semantic_map,
            "evaluation_class_id_to_label": {str(key + 1): value for key, value in class_map.items()},
            "background_evaluation_class_id": 1 if data_type == "pascal" else None,
        },
        "stability": {
            "standard_deviation": std_dev,
            "seed": seed,
            "aggregation": "unweighted mean of exact all-part equality per paired sample/class/prototype row",
        },
        "observation_columns": (
            [f"part_{part_id}" for part_id in range(max_parts + 1)]
            + [f"noisy_part_{part_id}" for part_id in range(max_parts + 1)]
        ),
    }


def _paired_rows(part_rows, noisy_rows, max_parts):
    columns = [f"part_{part_id}" for part_id in range(max_parts + 1)] + ["proto_id", "class", "img_id"]
    noisy_columns = [
        f"noisy_part_{part_id}" for part_id in range(max_parts + 1)
    ] + ["proto_id", "class", "img_id"]
    clean = pd.DataFrame(part_rows, columns=columns)
    noisy = pd.DataFrame(noisy_rows, columns=noisy_columns)
    key = ["proto_id", "class", "img_id"]
    if clean.duplicated(key).any() or noisy.duplicated(key).any():
        raise ValueError("Stability observations must have unique sample/class/prototype pairing keys")
    if set(map(tuple, clean[key].values.tolist())) != set(map(tuple, noisy[key].values.tolist())):
        raise ValueError("Clean and noisy stability observations have different pairing keys")
    return clean.merge(noisy, on=key, validate="one_to_one")


def _add_observations(output, paired, class_map, max_parts):
    label_to_id = {label: model_class_index + 1 for model_class_index, label in class_map.items()}
    for _, row in paired.iterrows():
        observation = {
            "sample_id": row["img_id"],
            "class_id": label_to_id[row["class"]],
            "prototype_id": int(row["proto_id"]),
            "repetition": 0,
        }
        observation.update({
            f"part_{part_id}": row[f"part_{part_id}"]
            for part_id in range(max_parts + 1)
        })
        observation.update({
            f"noisy_part_{part_id}": row[f"noisy_part_{part_id}"]
            for part_id in range(max_parts + 1)
        })
        output.add_observation(**observation)


def run_stability(
    model_name: str,
    training_phase: str,
    data_type: str,
    quantile: float = 0.8,
    std_dev: float = 0.2,
    group_name: str = None,
    checkpoint_path: str = None,
    *,
    group_checkpoint: Optional[str] = None,
    official_parts_only: bool = False,
    require_semantic_match: bool = False,
    seed: int = 0,
    exclude_background_centroid: bool = False,
):
    """Wrapper to compute the stability across validation sets

    Args:
        model_name (str): Name of the model for testing.
        training_phase (str):  Training phase for loading the model.
        data_type (str): Data type for testing: Cityscapes & Pascal.
        quantile (float, optional): Quantile to threshold the prototype activations. Defaults to 0.8.
        std_dev (float, optional): Standard deviation of the noise injected. Defaults to 0.2.
        group_name (str, optional): Name of the model when grouping is used. Defaults to None.
        group_checkpoint (str, optional): Explicit prototype-filter checkpoint,
            overriding group_name.
        official_parts_only (bool): Keep only documented Cityscapes semantic/part pairs.
        require_semantic_match (bool): For Pascal, require part/VOC semantic agreement.
        seed (int): Random noise seed. Defaults to 0.

    Raises:
        NotImplementedError: Data type not supported.
    """

    if official_parts_only and data_type != "cityscapes":
        raise ValueError("official_parts_only is supported only for cityscapes")
    if require_semantic_match and data_type != "pascal":
        raise ValueError("require_semantic_match is supported only for pascal")

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
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k] for i, k in pred2name.items()}

    elif data_type == "pascal":
        ID_MAPPING = PASCAL_ID_MAPPING
        CATEGORIES = PASCAL_CATEGORIES
        MAX_PARTS = MAX_PARTS_PASCAL
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k + 1] for i, k in pred2name.items() if k < len(CATEGORIES) - 1}

    else:
        raise NotImplementedError(f"Data type {data_type} not supported")

    checkpoint_path_group = None
    if group_checkpoint is not None or group_name is not None:
        if group_checkpoint is not None:
            checkpoint_path_group = os.path.expanduser(group_checkpoint)
        else:
            group_path = os.path.join(os.environ["RESULTS_DIR"], group_name)
            checkpoint_path_group = os.path.join(group_path, "checkpoints/th-0.05-nopush-group_last.pth")

    protocol = _protocol(
        data_type, cls2name, ID_MAPPING, MAX_PARTS, quantile=quantile, std_dev=std_dev, seed=seed,
        official_parts_only=official_parts_only, require_semantic_match=require_semantic_match,
        exclude_background_centroid=exclude_background_centroid,
    )
    output = MetricOutput(
        "stability", checkpoint_path, data_type, model_name=model_name,
        training_phase=training_phase, protocol=protocol, quantile=quantile, std_dev=std_dev, seed=seed,
        group_checkpoint=checkpoint_path_group,
        official_parts_only=official_parts_only,
        require_semantic_match=require_semantic_match,
        exclude_background_centroid=exclude_background_centroid,
    )
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        log(f"Loading model from {checkpoint_path}")
        log(f"Part labels: {'official Cityscapes parts only' if official_parts_only else 'all decoded part IDs'}")
        if data_type == "pascal":
            log(f"Pascal semantic agreement: {'required' if require_semantic_match else 'not required (legacy)'}")
        log(f"Part centroids: {'foreground only' if exclude_background_centroid else 'including background (legacy)'}")
        ppnet = load_model_checkpoint(checkpoint_path)
        ppnet = ppnet.cuda()
        ppnet.eval()

        if checkpoint_path_group is not None:
            log(f"Loading prototype filter from {checkpoint_path_group}")
            if checkpoint_path_group == checkpoint_path:
                ppnet_group = ppnet
            else:
                ppnet_group = load_model_checkpoint(checkpoint_path_group)
            ppnet_group.eval()
            proto_ids = proto_filter(ppnet_group)
            if checkpoint_path_group == checkpoint_path:
                output.register_input("group_checkpoint", checkpoint_path)
        else:
            proto_ids = None
        output.set_selected_prototype_ids(_selected_prototypes(ppnet, proto_ids))
        output.update_model_metadata(architecture=type(ppnet).__name__)

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
        tot_presence = []
        tot_presence_noise = []

        for sample_index in tqdm(
            range(len(val_dataset)), desc="Computing Stability Score Validation", total=len(val_dataset)
        ):

            img_id = val_dataset.img_ids[sample_index]
            img_path = val_dataset.get_img_path(img_id)

            path_cls = Path(val_dataset.annotations_dir)
            path_part = str(path_cls.parent) + f"_PIDS/{path_cls.name}"

            # Pascal parts cover only a subset of the validation images.
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
            )
            _, noise_part_presence_list = part_intersect(
                img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, std_dev,
                exclude_background_centroid=exclude_background_centroid,
            )

            paired = _paired_rows(part_presence_list, noise_part_presence_list, MAX_PARTS)
            _add_observations(output, paired, cls2name, MAX_PARTS)
            tot_presence.extend(part_presence_list)
            tot_presence_noise.extend(noise_part_presence_list)

        df_tot = _paired_rows(tot_presence, tot_presence_noise, MAX_PARTS)
        for part_id in range(MAX_PARTS + 1):
            df_tot[f"eq_part_{part_id}"] = (
                df_tot[f"part_{part_id}"] == df_tot[f"noisy_part_{part_id}"]
            ).astype(int)
        df_tot["sum_eq"] = df_tot[[f"eq_part_{part_id}" for part_id in range(MAX_PARTS + 1)]].sum(axis=1)
        df_tot["is_stable"] = (df_tot["sum_eq"] == MAX_PARTS + 1).astype(int)
        stab_score = df_tot.is_stable.mean()
        numerator = float(df_tot.is_stable.sum()) if len(df_tot) else 0
        denominator = int(len(df_tot))
        log(f"Stability score: {stab_score}")

        output.finish(
            stab_score,
            numerator=numerator,
            denominator=denominator,
            reason="no eligible paired class/prototype observations" if not denominator else None,
        )
    except Exception as error:
        output.fail(error)
        raise


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
                ).numpy()

                for part_id, centroids in part_centroids.items():
                    for centroid in centroids:
                        x, y = centroid[0], centroid[1]
                        presence_list[part_id] += upsampled_act_img_j[y, x, 0]
                part_presence[p.item()] = [0 if val == 0 else 1 for val in presence_list]
                part_presence_list.append(part_presence[p.item()] + [p.item(), cls2name[class_id - 1], img_id])

    return part_presence, part_presence_list


if __name__ == "__main__":
    argh.dispatch_command(run_stability)
