"""Registered checkpoint fixtures for existing CPU evaluation tests."""

import os
from pathlib import Path

from segmentation.catalog import sha256_file
from segmentation.catalog.models import list_models, register_model


def registered_input(input_type, model_name="model", training_phase="push", data_type="pascal", **overrides):
    root = Path(os.environ.get("MEASUREMENT_CATALOG_ROOT", "/nonexistent-fixture-catalog"))
    override = overrides.pop("checkpoint", None) or overrides.pop("checkpoint_path", None)
    checkpoint = (
        Path(override)
        if override
        else Path(os.environ.get("RESULTS_DIR", "/fixture"))
        / model_name
        / (
            "pruned/checkpoints/push_last.pth"
            if training_phase == "pruned"
            else f"checkpoints/{training_phase}_last.pth"
        )
    )
    group_path = overrides.pop("group_checkpoint", None)
    group_name = overrides.pop("group_name", None)
    if group_path is None and group_name is not None:
        group_path = (
            Path(os.environ.get("RESULTS_DIR", "/fixture")) / group_name / ("checkpoints/th-0.05-nopush-group_last.pth")
        )

    def reference(path, name, grouped=False):
        if not root.is_dir() or not path.is_file():
            return name
        digest, _ = sha256_file(path)
        existing = next((r for r in list_models(root) if r["checkpoint"]["sha256"] == digest), None)
        return (
            existing
            or register_model(
                root=root,
                name=name,
                architecture="ScaleProtoSeg",
                dataset=data_type,
                config={"stage": training_phase, "grouped": grouped},
                checkpoint=path,
            )
        )["model_id"]

    return input_type(
        model=reference(checkpoint, model_name),
        grouping_model=reference(Path(group_path), f"{model_name}-group", True) if group_path else None,
        data_type=data_type,
        **overrides,
    )
