"""Shared metadata schemas; filesystem integrity and scientific policy live elsewhere."""

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, model_validator,
)


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
ByteCount = Annotated[int, Field(ge=0)]
Vector = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
PartIds = Annotated[list[PositiveInt], Field(min_length=1)]
Definitions = Annotated[dict, Field(min_length=1)]


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)


class FileIdentity(Metadata):
    status: Literal["available", "unknown", "invalid", "missing", "unreadable"]
    path: Optional[Text] = None
    bytes: Optional[ByteCount] = None
    sha256: Optional[Digest] = None
    reason: Optional[Text] = None
    error: Optional[Text] = None

    @model_validator(mode="after")
    def evidence(self):
        if self.status == "available":
            if any(value is None for value in (self.path, self.bytes, self.sha256)):
                raise ValueError("incomplete file identity")
        elif not (self.reason or self.error):
            raise ValueError("unknown file identity requires a reason")
        return self


class CheckpointIdentity(FileIdentity):
    status: Literal["available", "unknown", "missing", "unreadable"]
    path: Text
    bytes: Optional[ByteCount]
    sha256: Optional[Digest]

    @model_validator(mode="after")
    def unavailable(self):
        if self.status != "available" and (self.bytes is not None or self.sha256 is not None):
            raise ValueError("unavailable checkpoint identity must have null bytes and SHA-256")
        return self


class Artifact(Metadata):
    path: Text
    bytes: Optional[ByteCount] = None
    sha256: Optional[Digest] = None
    schema_version: Optional[Literal[1]] = None
    columns: Optional[list[str]] = None


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


class Quality(Metadata):
    missing_fields: list[str]
    warnings: list[str]


class Activation(Metadata):
    prototype_distance_activation: Optional[Text]
    prototype_activation_function: Optional[Text]
    normalization_mean: Optional[Vector]
    normalization_std: Optional[Vector]
    resize_interpolation: Optional[Text]
    quantile_support: Optional[Text]
    quantile_interpolation: Optional[Text]
    active_comparison: Optional[Text]
    quantile_class_mask: Optional[Text]


class Centroids(Metadata):
    component_rule: Text
    background_component_centroid_included: bool
    exclude_background_centroid: bool


class Classes(Metadata):
    id_convention: Annotated[dict[str, Text], Field(min_length=1)]
    evaluation_class_id_to_label: Annotated[dict[str, Text], Field(min_length=1)]
    ignored_or_void_evaluation_class_id: int
    background_evaluation_class_id: Optional[int]
    excluded_evaluation_class_ids: list[int]


class Selection(Metadata):
    class_identity_rule: Text
    excluded_evaluation_class_ids: list[int]
    grouping_active: bool
    group_nonzero_rule: Text
    group_intersection_rule: Text


class ModelProtocol(Metadata):
    epsilon: Optional[Annotated[FiniteFloat, Field(gt=0)]]
    prototype_activation_function: Optional[Text]


class Normalization(Metadata):
    conversion: Optional[Text]
    mean: Optional[Vector]
    std: Optional[Vector]


class Input(Metadata):
    source: Text
    resolution: Text
    normalization: Normalization


class Consistency(Metadata):
    threshold: FiniteFloat
    strict_comparison: Optional[Text]
    aggregation: Optional[Text]


class NativeConsistency(Consistency):
    prototype_class_consistent: Optional[Text]


class Stability(Metadata):
    standard_deviation: FiniteFloat
    seed: Optional[int]
    noise_distribution: Optional[Text]
    noise_magnitude: FiniteFloat
    noise_injection_space: Optional[Text]
    repetitions: Optional[int]
    aggregation: Optional[Text]
    missing_value_comparison: Optional[Text]


class KnownIdentity(Metadata):
    status: Literal["available"]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]


class ComparableParts(Metadata):
    valid_part_ids: PartIds
    mapping_identity: KnownIdentity
    class_dependent_part_definitions: Optional[Definitions]


class NativeParts(ComparableParts):
    maximum_part_id: PositiveInt
    mapping_identity: FileIdentity
    annotation_provenance: Text

    @model_validator(mode="after")
    def part_ids(self):
        if (
            len(set(self.valid_part_ids)) != len(self.valid_part_ids)
            or max(self.valid_part_ids) != self.maximum_part_id
        ):
            raise ValueError("invalid protocol valid_part_ids")
        if self.mapping_identity.status not in ("available", "unknown", "invalid"):
            raise ValueError("invalid part mapping identity status")
        return self


class ProtocolHeader(Metadata):
    version: Literal[1]
    metric: str
    observation_columns: Annotated[list[str], Field(min_length=1)]


class ProtocolSections(Metadata):
    activation: Activation
    centroids: Centroids
    classes: Classes
    prototype_selection: Selection
    model: ModelProtocol
    input: Input


class ConsistencyProtocol(ProtocolHeader, ProtocolSections):
    metric: Literal["consistency"]
    parts: NativeParts
    consistency: NativeConsistency


class StabilityProtocol(ProtocolHeader, ProtocolSections):
    metric: Literal["stability"]
    parts: NativeParts
    stability: Stability


class ComparableProtocol(ProtocolSections):
    parts: ComparableParts
    consistency: Consistency


def unknown_fields(value, prefix="", *, comparison=False):
    """Find unknown declared protocol values, ignoring unrelated extension metadata."""
    if isinstance(value, FileIdentity):
        if value.status != "available":
            yield prefix
    elif isinstance(value, Metadata):
        for field in type(value).model_fields:
            if field == "background_evaluation_class_id":
                continue
            yield from unknown_fields(
                getattr(value, field), f"{prefix}.{field}".lstrip("."), comparison=comparison,
            )
    elif value is None or (
        isinstance(value, str) and (value.strip().casefold() if comparison else value) in (
            {"unknown", "unavailable", "partial", "null"} if comparison else {"unknown"}
        )
    ):
        yield prefix
    elif comparison and isinstance(value, dict):
        for field, child in value.items():
            yield from unknown_fields(child, f"{prefix}.{field}", comparison=True)


class Envelope(Metadata):
    schema_version: Literal[1]
    measurement_id: Text
    metric: Text
    status: Literal["running", "complete", "failed"]
    artifacts: dict[str, dict]
    scores: dict[str, dict]


class Historical(Envelope):
    origin: Literal["legacy"]
    provenance_quality: Quality


class Model(Metadata):
    label: Any
    training_stage: Any
    dataset: Any
    training_seed: Any
    inputs: dict[str, dict]

    @model_validator(mode="after")
    def no_metric_seed(self):
        if "seed" in self.model_extra:
            raise ValueError("Metric parameter seeds must not be stored as model seed")
        return self


class Dataset(Metadata):
    name: Any
    split: Any
    identity: str


class Execution(Metadata):
    commit: Any
    tracked_diff: Any
    relevant_untracked_source: Any
    submodules: Any
    missing: Any
    command: Any
    python: Any
    python_executable: Any
    packages: dict
    host: Any
    scheduler_job_id: Any


class NativeTerminal(Envelope):
    origin: Literal["native"]
    model: Model
    dataset: Dataset
    protocol: ProtocolHeader
    execution: Execution
    provenance_quality: Quality

    @model_validator(mode="after")
    def metric_identity(self):
        if self.protocol.metric != self.metric:
            raise ValueError("native protocol metric disagrees with measurement")
        Score.model_validate(self.scores.get(self.metric))
        return self


class CompleteModel(Model):
    inputs: Annotated[dict[str, CheckpointIdentity], Field(min_length=1)]


class NativeComplete(NativeTerminal):
    model: CompleteModel
    protocol: Annotated[Union[ConsistencyProtocol, StabilityProtocol], Field(discriminator="metric")]

    @model_validator(mode="after")
    def provenance(self):
        gaps = self.provenance_quality.missing_fields
        if "prototype_checkpoint" not in self.model.inputs:
            raise ValueError("missing prototype checkpoint identity")
        for role, identity in self.model.inputs.items():
            if identity.status != "available" and f"{role} SHA-256" not in gaps:
                raise ValueError(f"unjustified unknown checkpoint identity: {role}")
        convention = self.protocol.classes.id_convention
        if not {
            "semantic_mapping_input_ids", "semantic_mapping_output_ids", "model_class_index",
        } <= convention.keys():
            raise ValueError("missing native protocol class ID convention")
        for path in unknown_fields(self.protocol):
            if path not in gaps:
                raise ValueError(f"unjustified unknown protocol value: {path}")
        return self


def validate_metadata(result):
    """Validate metadata only; callers retain the original dictionary unchanged."""
    envelope = Envelope.model_validate(result)
    if envelope.status != "running":
        if result.get("origin") == "legacy":
            Historical.model_validate(result)
        else:
            model = NativeComplete if envelope.status == "complete" else NativeTerminal
            model.model_validate(result)
