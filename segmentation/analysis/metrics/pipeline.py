"""Shared evaluation lifecycle with metric-specific observation and scoring hooks."""

from abc import ABC, abstractmethod

from PIL import Image
from tqdm import tqdm

from segmentation.analysis.metrics.helpers import (
    EvaluationContext,
    load_part_annotations,
    part_intersect,
    proto_filter,
    selected_prototypes,
)
from segmentation.analysis.metrics.protocol import MetricRunInput
from segmentation.catalog.output import MetricOutput
from segmentation.checkpoints import load_model_checkpoint
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.data.pascal_parts import left_right_part_groups
from settings import log


class MetricPlugin(ABC):
    name: str
    missing_parts_nan: bool
    empty_reason: str
    part_prefixes = ("",)

    def prepare(self, run_input):
        """Initialize metric state and return the ordered noise levels per image."""
        return (None,)

    def excluded_classes(self, context):
        return ()

    def protocol_options(self, context):
        return {}

    @abstractmethod
    def observe(self, output, variants, context):
        """Record one sample's ordered clean/noisy observation rows."""

    @abstractmethod
    def summarize(self, run_input, context):
        """Return the score, numerator and denominator for the whole run."""


def run_metric(run_input: MetricRunInput, metric: MetricPlugin):
    context = EvaluationContext.for_dataset(run_input.data_type)
    excluded_classes = metric.excluded_classes(context)
    observation_columns = [
        f"{prefix}part_{part_id}" for prefix in metric.part_prefixes for part_id in range(context.max_parts + 1)
    ]
    protocol = run_input.to_protocol(
        semantic_id_to_evaluation_id=context.semantic_ids,
        evaluation_class_id_to_label={str(key + 1): value for key, value in context.class_labels.items()},
        valid_part_ids=range(1, context.max_parts + 1),
        observation_columns=observation_columns,
        left_right_part_groups=left_right_part_groups() if run_input.merge_left_right_parts else None,
        **metric.protocol_options(context),
    )
    output = MetricOutput(
        metric.name,
        run_input.resolved_checkpoint_path,
        run_input.data_type,
        model_name=run_input.model_name,
        training_phase=run_input.training_phase,
        protocol=protocol,
        group_checkpoint=run_input.resolved_group_checkpoint_path,
        model_ref=run_input.model,
        grouping_model_ref=run_input.grouping_model,
        **run_input.model_dump(exclude={"model", "grouping_model", "data_type"}),
    )
    try:
        noise_levels = metric.prepare(run_input)
        log(f"Loading model from {run_input.resolved_checkpoint_path}")
        log(
            f"Part labels: {'official Cityscapes parts only' if run_input.official_parts_only else 'all decoded part IDs'}"
        )
        if run_input.data_type == "pascal":
            log(
                f"Pascal semantic agreement: {'required' if run_input.require_semantic_match else 'not required (legacy)'}"
            )
            log(f"Left/right parts: {'merged with per-image OR' if run_input.merge_left_right_parts else 'separate'}")
        log(
            f"Part centroids: {'foreground only' if run_input.exclude_background_centroid else 'including background (legacy)'}"
        )
        model = load_model_checkpoint(run_input.resolved_checkpoint_path).cuda()
        model.eval()
        prototype_ids = None
        if run_input.resolved_group_checkpoint_path is not None:
            log(f"Loading prototype filter from {run_input.resolved_group_checkpoint_path}")
            if run_input.resolved_group_checkpoint_path == run_input.resolved_checkpoint_path:
                grouping_model = model
                output.register_input("group_checkpoint", run_input.resolved_checkpoint_path)
            else:
                grouping_model = load_model_checkpoint(run_input.resolved_group_checkpoint_path)
            if not hasattr(grouping_model, "group_projection"):
                raise ValueError("The registered grouping model has no group projection")
            grouping_model.eval()
            prototype_ids = proto_filter(grouping_model)
        output.set_selected_prototype_ids(selected_prototypes(model, prototype_ids, excluded_classes))
        output.update_model_metadata(runtime_class=type(model).__name__)

        dataset = PatchClassificationDataset(**context.dataset_options)
        output.record_dataset(dataset)
        for sample_index in tqdm(
            range(len(dataset)),
            desc=f"Computing {metric.name.title()} Score Validation",
            total=len(dataset),
        ):
            img_id = dataset.img_ids[sample_index]
            annotations = load_part_annotations(dataset, img_id, run_input)
            if annotations is None:
                continue
            semantic, parts = annotations
            with open(dataset.get_img_path(img_id), "rb") as stream:
                image = Image.open(stream).convert("RGB")
            variants = []
            for noise in noise_levels:
                _, rows = part_intersect(
                    image,
                    semantic,
                    parts,
                    model,
                    context.class_labels,
                    img_id,
                    run_input.quantile,
                    prototype_ids,
                    excluded_classes,
                    max_parts=context.max_parts,
                    missing_parts_nan=metric.missing_parts_nan,
                    noise=noise,
                    exclude_background_centroid=run_input.exclude_background_centroid,
                    merge_left_right_parts=run_input.merge_left_right_parts,
                    quantile_scope=run_input.quantile_scope,
                )
                variants.append(rows)
            metric.observe(output, variants, context)

        score, numerator, denominator = metric.summarize(run_input, context)
        log(f"{metric.name.title()} score: {score}")
        output.finish(
            score,
            numerator=numerator,
            denominator=denominator,
            reason=metric.empty_reason if not denominator else None,
        )
    except Exception as error:
        output.fail(error)
        raise
