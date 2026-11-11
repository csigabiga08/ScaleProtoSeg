"""Versioned, dependency-light architecture configurations registered in source."""

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from segmentation.catalog.schema import Text


class ProtoSegConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    stage: Text
    backbone: Optional[Text] = None
    prototype_shape: Optional[list[Annotated[int, Field(gt=0)]]] = Field(default=None, min_length=4, max_length=4)


class ScaleProtoSegConfig(ProtoSegConfig):
    grouped: bool
    num_scales: Optional[Annotated[int, Field(gt=0)]] = None
    num_groups: Optional[Annotated[int, Field(gt=0)]] = None


ARCHITECTURES = {
    ("ProtoSeg", 1): ProtoSegConfig,
    ("ScaleProtoSeg", 1): ScaleProtoSegConfig,
}


def validate_config(architecture, version, config):
    schema = ARCHITECTURES.get((architecture, version))
    if schema is None:
        raise ValueError(f"Unknown architecture/schema version: {architecture}/{version}")
    return schema.model_validate(config).model_dump(exclude_none=True)
