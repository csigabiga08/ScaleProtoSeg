"""Stability observations and scoring for the shared metric pipeline."""

from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from segmentation.analysis.metrics.pipeline import MetricPlugin, run_metric
from segmentation.analysis.metrics.protocol import StabilityRunInput


def _paired_rows(part_rows, noisy_rows, max_parts):
    columns = [f"part_{part_id}" for part_id in range(max_parts + 1)] + ["proto_id", "class", "img_id"]
    noisy_columns = [f"noisy_part_{part_id}" for part_id in range(max_parts + 1)] + ["proto_id", "class", "img_id"]
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
        observation.update({f"part_{part_id}": row[f"part_{part_id}"] for part_id in range(max_parts + 1)})
        observation.update({f"noisy_part_{part_id}": row[f"noisy_part_{part_id}"] for part_id in range(max_parts + 1)})
        output.add_observation(**observation)


class StabilityMetric(MetricPlugin):
    name = "stability"
    missing_parts_nan = False
    part_prefixes = ("", "noisy_")
    empty_reason = "no eligible paired class/prototype observations"

    def __init__(self):
        self.clean_rows = []
        self.noisy_rows = []

    def prepare(self, run_input):
        torch.manual_seed(run_input.seed)
        torch.cuda.manual_seed(run_input.seed)
        return (None, run_input.std_dev)

    def observe(self, output, variants, context):
        clean, noisy = variants
        paired = _paired_rows(clean, noisy, context.max_parts)
        _add_observations(output, paired, context.class_labels, context.max_parts)
        self.clean_rows.extend(clean)
        self.noisy_rows.extend(noisy)

    def summarize(self, run_input, context):
        df_tot = _paired_rows(self.clean_rows, self.noisy_rows, context.max_parts)
        for part_id in range(context.max_parts + 1):
            df_tot[f"eq_part_{part_id}"] = (df_tot[f"part_{part_id}"] == df_tot[f"noisy_part_{part_id}"]).astype(int)
        df_tot["sum_eq"] = df_tot[[f"eq_part_{part_id}" for part_id in range(context.max_parts + 1)]].sum(axis=1)
        df_tot["is_stable"] = (df_tot["sum_eq"] == context.max_parts + 1).astype(int)
        numerator = float(df_tot.is_stable.sum()) if len(df_tot) else 0
        return df_tot.is_stable.mean(), numerator, int(len(df_tot))


def run_stability(run_input: StabilityRunInput):
    return run_metric(run_input, StabilityMetric())


def _run_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Stability configuration must resolve to a mapping")
    values.pop("hydra", None)
    return run_stability(StabilityRunInput.model_validate(values))


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parent / "../../../config/hydra/metrics").resolve()),
    config_name="stability",
)
def main(config: DictConfig):
    _run_config(config)


if __name__ == "__main__":
    main()
