"""Typed run inputs and serializers for metric protocols."""

from functools import cached_property
from typing import Dict, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricRunInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    model: str = Field(min_length=1)
    grouping_model: Optional[str] = None
    data_type: Literal["cityscapes", "pascal"]
    quantile: float = Field(default=0.8, ge=0, le=1)
    official_parts_only: bool = False
    require_semantic_match: bool = False
    exclude_background_centroid: bool = False
    merge_left_right_parts: bool = False

    @model_validator(mode="after")
    def validate_dataset_flags(self):
        if self.official_parts_only and self.data_type != "cityscapes":
            raise ValueError("official_parts_only is supported only for cityscapes")
        if self.require_semantic_match and self.data_type != "pascal":
            raise ValueError("require_semantic_match is supported only for pascal")
        if self.merge_left_right_parts and self.data_type != "pascal":
            raise ValueError("merge_left_right_parts is supported only for pascal")
        return self

    @cached_property
    def registration(self):
        from segmentation.catalog import CatalogError
        from segmentation.catalog.models import find_model

        record = find_model(self.model, verify=True)
        if record["dataset"] != self.data_type:
            raise CatalogError("Evaluation dataset must match the registered model dataset")
        return record

    @cached_property
    def group_registration(self):
        if self.grouping_model is None:
            return None
        from segmentation.catalog import CatalogError
        from segmentation.catalog.models import find_model

        record = (
            self.registration
            if self.grouping_model
            in (
                self.registration["model_id"],
                self.registration["name"],
            )
            else find_model(self.grouping_model, verify=True)
        )
        if record["dataset"] != self.data_type:
            raise CatalogError("Grouping model dataset must match the evaluation dataset")
        return record

    @property
    def model_name(self):
        return self.registration["name"]

    @property
    def training_phase(self):
        return self.registration["config"]["stage"]

    @property
    def resolved_checkpoint_path(self):
        return self.registration["checkpoint"]["path"]

    @property
    def resolved_group_checkpoint_path(self) -> Optional[str]:
        return self.group_registration["checkpoint"]["path"] if self.group_registration else None

    def _protocol_parts(
        self,
        valid_part_ids: Sequence[int],
        left_right_part_groups: Optional[Mapping[str, Sequence[Mapping[str, object]]]],
    ) -> Dict[str, object]:
        parts = {
            "valid_part_ids": list(valid_part_ids),
            "official_parts_only": self.official_parts_only,
            "require_semantic_match": self.require_semantic_match,
            "exclude_background_centroid": self.exclude_background_centroid,
            "merge_left_right_parts": self.merge_left_right_parts,
        }
        if self.merge_left_right_parts:
            if left_right_part_groups is None:
                raise ValueError("Merged Pascal parts require their canonical group definitions")
            parts["left_right_part_groups"] = {
                name: [dict(group) for group in groups] for name, groups in left_right_part_groups.items()
            }
        return parts

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
    threshold: float = Field(default=0.8, ge=0, le=1)

    def to_protocol(
        self,
        *,
        semantic_id_to_evaluation_id: Mapping[int, int],
        evaluation_class_id_to_label: Mapping[str, str],
        valid_part_ids: Sequence[int],
        observation_columns: Sequence[str],
        excluded_evaluation_class_ids: Sequence[int],
        left_right_part_groups: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
    ) -> Dict[str, object]:
        classes = self._protocol_classes(
            semantic_id_to_evaluation_id,
            evaluation_class_id_to_label,
        )
        classes["excluded_evaluation_class_ids"] = list(excluded_evaluation_class_ids)
        return {
            "metric": "consistency",
            "quantile": self.quantile,
            "parts": self._protocol_parts(valid_part_ids, left_right_part_groups),
            "classes": classes,
            "consistency": {
                "threshold": self.threshold,
                "aggregation": "unweighted mean over class/prototype rows",
            },
            "observation_columns": list(observation_columns),
        }


class StabilityRunInput(MetricRunInput):
    std_dev: float = Field(default=0.2, ge=0)
    seed: int = 0

    def to_protocol(
        self,
        *,
        semantic_id_to_evaluation_id: Mapping[int, int],
        evaluation_class_id_to_label: Mapping[str, str],
        valid_part_ids: Sequence[int],
        observation_columns: Sequence[str],
        left_right_part_groups: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
    ) -> Dict[str, object]:
        return {
            "metric": "stability",
            "quantile": self.quantile,
            "parts": self._protocol_parts(valid_part_ids, left_right_part_groups),
            "classes": self._protocol_classes(
                semantic_id_to_evaluation_id,
                evaluation_class_id_to_label,
            ),
            "stability": {
                "standard_deviation": self.std_dev,
                "seed": self.seed,
                "aggregation": ("unweighted mean of exact all-part equality per paired " "sample/class/prototype row"),
            },
            "observation_columns": list(observation_columns),
        }
