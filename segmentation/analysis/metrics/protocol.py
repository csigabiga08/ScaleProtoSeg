"""Typed run inputs and serializers for metric protocols."""

from typing import Dict, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, model_validator


class MetricRunInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    model_name: str
    training_phase: str
    data_type: Literal["cityscapes", "pascal"]
    quantile: float = 0.8
    group_name: Optional[str] = None
    group_checkpoint: Optional[str] = None
    official_parts_only: bool = False
    require_semantic_match: bool = False
    exclude_background_centroid: bool = False

    @model_validator(mode="after")
    def validate_dataset_flags(self):
        if self.official_parts_only and self.data_type != "cityscapes":
            raise ValueError("official_parts_only is supported only for cityscapes")
        if self.require_semantic_match and self.data_type != "pascal":
            raise ValueError("require_semantic_match is supported only for pascal")
        return self

    def _protocol_parts(
        self,
        valid_part_ids: Sequence[int],
    ) -> Dict[str, object]:
        return {
            "valid_part_ids": list(valid_part_ids),
            "official_parts_only": self.official_parts_only,
            "require_semantic_match": self.require_semantic_match,
            "exclude_background_centroid": self.exclude_background_centroid,
        }

    def _protocol_classes(
        self,
        semantic_id_to_evaluation_id: Mapping[int, int],
        evaluation_class_id_to_label: Mapping[str, str],
    ) -> Dict[str, object]:
        return {
            "semantic_id_to_evaluation_id": dict(semantic_id_to_evaluation_id),
            "evaluation_class_id_to_label": dict(evaluation_class_id_to_label),
            "background_evaluation_class_id": 1 if self.data_type == "pascal" else None,
        }


class ConsistencyRunInput(MetricRunInput):
    checkpoint: Optional[str] = None
    threshold: float = 0.8

    def to_protocol(
        self,
        *,
        semantic_id_to_evaluation_id: Mapping[int, int],
        evaluation_class_id_to_label: Mapping[str, str],
        valid_part_ids: Sequence[int],
        observation_columns: Sequence[str],
        excluded_evaluation_class_ids: Sequence[int],
    ) -> Dict[str, object]:
        classes = self._protocol_classes(
            semantic_id_to_evaluation_id,
            evaluation_class_id_to_label,
        )
        classes["excluded_evaluation_class_ids"] = list(
            excluded_evaluation_class_ids
        )
        return {
            "metric": "consistency",
            "quantile": self.quantile,
            "parts": self._protocol_parts(valid_part_ids),
            "classes": classes,
            "consistency": {
                "threshold": self.threshold,
                "aggregation": "unweighted mean over class/prototype rows",
            },
            "observation_columns": list(observation_columns),
        }


class StabilityRunInput(MetricRunInput):
    checkpoint_path: Optional[str] = None
    std_dev: float = 0.2
    seed: int = 0

    def to_protocol(
        self,
        *,
        semantic_id_to_evaluation_id: Mapping[int, int],
        evaluation_class_id_to_label: Mapping[str, str],
        valid_part_ids: Sequence[int],
        observation_columns: Sequence[str],
    ) -> Dict[str, object]:
        return {
            "metric": "stability",
            "quantile": self.quantile,
            "parts": self._protocol_parts(valid_part_ids),
            "classes": self._protocol_classes(
                semantic_id_to_evaluation_id,
                evaluation_class_id_to_label,
            ),
            "stability": {
                "standard_deviation": self.std_dev,
                "seed": self.seed,
                "aggregation": (
                    "unweighted mean of exact all-part equality per paired "
                    "sample/class/prototype row"
                ),
            },
            "observation_columns": list(observation_columns),
        }
