"""Script to evaluate models for stability"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.equivariance import quantile_map
from segmentation.analysis.metrics.helpers import proto_filter, selected_prototypes
from segmentation.catalog.output import MetricOutput
from segmentation.analysis.metrics.protocol import StabilityRunInput
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
from segmentation.data.pascal_parts import left_right_part_groups, merge_left_right_presence
from segmentation.model.model import PPNet
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import log


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


def run_stability(run_input: StabilityRunInput):
    """Wrapper to compute the stability across validation sets

    Args:
        run_input: Validated model, dataset, checkpoint and metric options.

    Raises:
        NotImplementedError: Data type not supported.
    """

    model_name = run_input.model_name
    training_phase = run_input.training_phase
    data_type = run_input.data_type
    quantile = run_input.quantile
    std_dev = run_input.std_dev
    checkpoint_path = run_input.resolved_checkpoint_path
    official_parts_only = run_input.official_parts_only
    require_semantic_match = run_input.require_semantic_match
    seed = run_input.seed
    exclude_background_centroid = run_input.exclude_background_centroid
    merge_left_right_parts = run_input.merge_left_right_parts

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

    checkpoint_path_group = run_input.resolved_group_checkpoint_path

    protocol = run_input.to_protocol(
        semantic_id_to_evaluation_id=ID_MAPPING,
        evaluation_class_id_to_label={
            str(key + 1): value for key, value in cls2name.items()
        },
        valid_part_ids=range(1, MAX_PARTS + 1),
        observation_columns=(
            [f"part_{part_id}" for part_id in range(MAX_PARTS + 1)]
            + [f"noisy_part_{part_id}" for part_id in range(MAX_PARTS + 1)]
        ),
        left_right_part_groups=left_right_part_groups() if merge_left_right_parts else None,
    )
    output = MetricOutput(
        "stability", checkpoint_path, data_type, model_name=model_name,
        training_phase=training_phase, protocol=protocol, quantile=quantile, std_dev=std_dev, seed=seed,
        group_checkpoint=checkpoint_path_group,
        official_parts_only=official_parts_only,
        require_semantic_match=require_semantic_match,
        exclude_background_centroid=exclude_background_centroid,
        merge_left_right_parts=merge_left_right_parts,
    )
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        log(f"Loading model from {checkpoint_path}")
        log(f"Part labels: {'official Cityscapes parts only' if official_parts_only else 'all decoded part IDs'}")
        if data_type == "pascal":
            log(f"Pascal semantic agreement: {'required' if require_semantic_match else 'not required (legacy)'}")
            log(f"Left/right parts: {'merged with per-image OR' if merge_left_right_parts else 'separate'}")
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
        output.set_selected_prototype_ids(selected_prototypes(ppnet, proto_ids))
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
                merge_left_right_parts=merge_left_right_parts,
            )
            _, noise_part_presence_list = part_intersect(
                img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, std_dev,
                exclude_background_centroid=exclude_background_centroid,
                merge_left_right_parts=merge_left_right_parts,
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
    merge_left_right_parts: bool = False,
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


def _run_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Stability configuration must resolve to a mapping")
    values.pop("hydra", None)
    run_input = StabilityRunInput.model_validate(values)
    return run_stability(run_input)


@hydra.main(
    version_base=None,
    config_path=str(
        (Path(__file__).resolve().parent / "../../../config/hydra/metrics").resolve()
    ),
    config_name="stability",
)
def main(config: DictConfig):
    _run_config(config)


if __name__ == "__main__":
    main()
