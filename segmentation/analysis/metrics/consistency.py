"""Consistency observations and scoring for the shared metric pipeline."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from segmentation.analysis.metrics.part_activations import (
    consistent_prototypes,
    mean_part_activations,
)
from segmentation.analysis.metrics.pipeline import PartMetricPlugin, run_metric
from segmentation.analysis.metrics.protocol import ConsistencyRunInput


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
        observation.update({f"part_{part_id}": row[part_id] for part_id in range(max_parts + 1)})
        output.add_observation(**observation)


class ConsistencyMetric(PartMetricPlugin):
    name = "consistency"
    missing_parts_nan = True
    empty_reason = "no eligible class/prototype observations"

    def __init__(self):
        self.rows = []

    def excluded_classes(self, context):
        return context.excluded_classes

    def protocol_options(self, context):
        return {"excluded_evaluation_class_ids": list(self.excluded_classes(context))}

    def implicit_grouping(self, run_input, model):
        return hasattr(model, "group_projection")

    def observe(self, output, variants, context):
        (rows,) = variants
        self.rows.extend(rows)
        _add_observations(output, rows, context.class_labels, context.max_parts)

    def summarize(self, run_input, context):
        part_columns = [f"part_{i}" for i in range(context.max_parts + 1)]
        df_presence = pd.DataFrame(self.rows, columns=part_columns + ["proto_id", "class", "img_id"])
        if df_presence.empty:
            return float("nan"), 0, 0
        df_mean = mean_part_activations(
            df_presence.rename(columns={"class": "class_id", "proto_id": "prototype_id"}), part_columns
        )
        flags = consistent_prototypes(df_mean, run_input.threshold)
        return flags.mean(), float(flags.sum()), int(len(flags))


def run_consistency(run_input: ConsistencyRunInput):
    return run_metric(run_input, ConsistencyMetric())


def _run_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Consistency configuration must resolve to a mapping")
    values.pop("hydra", None)
    return run_consistency(ConsistencyRunInput.model_validate(values))


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parent / "../../../config/hydra/metrics").resolve()),
    config_name="consistency",
)
def main(config: DictConfig):
    _run_config(config)


if __name__ == "__main__":
    main()
