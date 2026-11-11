"""Minimal metadata schema: what was measured, with which parameters, and the result."""

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False, validate_assignment=True)


class Artifact(Metadata):
    path: Text
    bytes: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Score(Metadata):
    value: Optional[FiniteFloat]
    unit: Literal["fraction"]
    numerator: Any
    denominator: Any
    reason: Optional[Text] = None

    @model_validator(mode="after")
    def undefined(self):
        if self.value is None and not self.reason:
            raise ValueError("undefined metric score requires a reason")
        return self


class ModelMetadata(Metadata):
    label: Text
    training_stage: Text
    dataset: Text
    architecture: Optional[Text]
    inputs: dict[str, Artifact]


class DatasetMetadata(Metadata):
    name: Text
    split: Text
    candidate_count: Optional[Annotated[int, Field(ge=0)]] = None


class ExecutionMetadata(Metadata):
    commit: Optional[Text]
    command: list[str]


class Population(Metadata):
    candidates: Optional[Annotated[int, Field(ge=0)]]
    contributing: Annotated[int, Field(ge=0)]
    observations: Annotated[int, Field(ge=0)]


class Failure(Metadata):
    type: Text
    message: str


class Measurement(Metadata):
    measurement_id: Text
    metric: Text
    status: Literal["running", "complete", "failed"]
    model: dict
    dataset: dict
    protocol: dict
    parameters: dict
    scores: dict[str, Score]
    artifacts: dict[str, Artifact]
    population: Optional[dict] = None

    @model_validator(mode="after")
    def complete(self):
        if self.status == "complete" and (
            self.metric not in self.scores or "observations" not in self.artifacts
        ):
            raise ValueError("a complete measurement needs its score and observations artifact")
        return self


class MeasurementOutput(Measurement):
    """Typed native record emitted by MetricOutput."""

    model: ModelMetadata
    dataset: DatasetMetadata
    execution: ExecutionMetadata
    population: Optional[Population] = None
    selected_prototype_ids: list[int]
    failure: Optional[Failure] = None
