"""Script to evaluate models for consistency"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.metrics.common import EvaluationContext, load_part_annotations, part_intersections
from segmentation.analysis.metrics.helpers import proto_filter, selected_prototypes
from segmentation.catalog.output import MetricOutput
from segmentation.analysis.metrics.protocol import ConsistencyRunInput
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
from segmentation.data.pascal_parts import left_right_part_groups
from segmentation.model.model import PPNet
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import log


def _add_observations(output, rows, class_map, max_parts):
    label_to_id = {label: model_class_index + 1 for model_class_index, label in class_map.items()}
    for row in rows:
        class_name = row[max_parts + 2]
        observation = {
            "sample_id": row[max_parts + 3],
            "class_id": label_to_id[class_name],
            "prototype_id": int(row[max_parts + 1]),
            "repetition": 0,
        }
        observation.update({
            f"part_{part_id}": row[part_id] for part_id in range(max_parts + 1)
        })
        output.add_observation(**observation)


def run_consistency(run_input: ConsistencyRunInput):
    """Wrapper to compute the consistency across validation sets

    Args:
        run_input: Validated model, dataset, checkpoint and metric options.

    Raises:
        NotImplementedError: Data type not supported.
    """

    model_name = run_input.model_name
    training_phase = run_input.training_phase
    data_type = run_input.data_type
    quantile = run_input.quantile
    threshold = run_input.threshold
    checkpoint = run_input.checkpoint
    official_parts_only = run_input.official_parts_only
    require_semantic_match = run_input.require_semantic_match
    exclude_background_centroid = run_input.exclude_background_centroid
    merge_left_right_parts = run_input.merge_left_right_parts
    checkpoint_path = run_input.resolved_checkpoint_path

    global MAX_PARTS

    context = EvaluationContext.for_dataset(data_type)
    ID_MAPPING = context.semantic_ids
    cls2name = context.class_labels
    MAX_PARTS = context.max_parts
    FILTER_CLASS = list(context.excluded_classes)

    checkpoint_path_group = run_input.resolved_group_checkpoint_path

    protocol = run_input.to_protocol(
        semantic_id_to_evaluation_id=ID_MAPPING,
        evaluation_class_id_to_label={
            str(key + 1): value for key, value in cls2name.items()
        },
        valid_part_ids=range(1, MAX_PARTS + 1),
        observation_columns=[f"part_{part_id}" for part_id in range(MAX_PARTS + 1)],
        excluded_evaluation_class_ids=(
            PASCAL_FILTER_CLASS if data_type == "pascal" else []
        ),
        left_right_part_groups=left_right_part_groups() if merge_left_right_parts else None,
    )
    output = MetricOutput(
        "consistency", checkpoint_path, data_type, model_name=model_name,
        training_phase=training_phase, protocol=protocol, quantile=quantile, threshold=threshold,
        group_checkpoint=checkpoint_path_group,
        official_parts_only=official_parts_only,
        require_semantic_match=require_semantic_match,
        exclude_background_centroid=exclude_background_centroid,
        merge_left_right_parts=merge_left_right_parts,
    )
    try:
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
        elif checkpoint is not None and hasattr(ppnet, "group_projection"):
            checkpoint_path_group = checkpoint_path
            proto_ids = proto_filter(ppnet)
            output.register_input("group_checkpoint", checkpoint_path)
        else:
            proto_ids = None
        selected_ids = selected_prototypes(ppnet, proto_ids, FILTER_CLASS)
        output.set_selected_prototype_ids(selected_ids)
        output.update_model_metadata(architecture=type(ppnet).__name__)

        val_dataset = PatchClassificationDataset(**context.dataset_options)

        output.record_dataset(val_dataset)
        tot_presence = []
        for sample_index in tqdm(
            range(len(val_dataset)), desc="Computing Consistency Score Validation", total=len(val_dataset)
        ):

            img_id = val_dataset.img_ids[sample_index]
            img_path = val_dataset.get_img_path(img_id)

            annotations = load_part_annotations(val_dataset, img_id, run_input)
            if annotations is None:
                continue
            cls_ann, part_ann = annotations

            with open(img_path, "rb") as f:
                img = Image.open(f).convert("RGB")

            _, part_presence_list = part_intersect(
                img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, FILTER_CLASS,
                exclude_background_centroid=exclude_background_centroid,
                merge_left_right_parts=merge_left_right_parts,
            )

            tot_presence.extend(part_presence_list)
            _add_observations(output, part_presence_list, cls2name, MAX_PARTS)

        part_columns = [f"part_{i}" for i in range(MAX_PARTS + 1)]
        df_presence = pd.DataFrame(tot_presence, columns=part_columns + ["proto_id", "class", "img_id"])

        if df_presence.empty:
            consistency_score = float("nan")
            numerator = 0
            denominator = 0
        else:
            df_mean = df_presence.groupby(["class", "proto_id"])[part_columns].mean().reset_index()
            df_flag = df_mean.copy()
            df_flag[part_columns] = df_flag[part_columns].apply(
                lambda column: (column > threshold).astype(int), axis=1
            )
            df_flag["is_consistent"] = df_flag[part_columns].max(axis=1)
            df_mean = df_mean.merge(df_flag[["proto_id", "is_consistent"]], on="proto_id")
            consistency_score = df_mean.is_consistent.mean()
            numerator = float(df_mean.is_consistent.sum())
            denominator = int(len(df_mean))
        log(f"Consistency score: {consistency_score}")

        output.finish(
            consistency_score,
            numerator=numerator,
            denominator=denominator,
            reason="no eligible class/prototype observations" if not denominator else None,
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
    filter_class_ids: List[int] = [],
    *,
    exclude_background_centroid: bool = False,
    merge_left_right_parts: bool = False,
) -> Tuple[Dict[int, List[int]], List[List[Any]]]:
    """Check original part-component centroids against prototype activations.

    Optional left/right pooling combines binary hits after computing centroids for
    the original labels. Masks are never merged and centroids are never recentered.
    """

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    return part_intersections(
        img_tensor, img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile,
        filter_proto_ids, filter_class_ids, max_parts=MAX_PARTS, missing_parts_nan=True,
        exclude_background_centroid=exclude_background_centroid, merge_left_right_parts=merge_left_right_parts,
    )


def _run_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Consistency configuration must resolve to a mapping")
    values.pop("hydra", None)
    run_input = ConsistencyRunInput.model_validate(values)
    return run_consistency(run_input)


@hydra.main(
    version_base=None,
    config_path=str(
        (Path(__file__).resolve().parent / "../../../config/hydra/metrics").resolve()
    ),
    config_name="consistency",
)
def main(config: DictConfig):
    _run_config(config)


if __name__ == "__main__":
    main()
