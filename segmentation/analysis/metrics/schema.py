"""Minimal metadata schema: what was measured, with which parameters, and the result."""

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)


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
