"""Shared part/centroid folders, parameterized result files and run descriptions."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys


def _label(value):
    value = str(value)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", value)
    if safe != value or len(safe) > 64:
        safe = safe[:48] + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]
    return safe


def _absolute(path):
    return str(Path(path).expanduser().resolve()) if path is not None else None


def output_directory(
    metric, checkpoint, data_type, *, quantile, threshold=None, std_dev=None,
    seed=None, group_checkpoint=None, official_parts_only=False,
    require_semantic_match=False, exclude_background_centroid=False, output_dir=None,
    merge_left_right_parts=False, quantile_scope="class",
):
    """Return the model's shared folder for this part/centroid configuration."""
    if quantile_scope not in ("class", "image"):
        raise ValueError("quantile_scope must be 'class' or 'image'")
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    parts = "filteredparts" if official_parts_only or require_semantic_match else "allparts"
    centroids = "foreground" if exclude_background_centroid else "background"
    model_directory = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    return model_directory / f"metrics-{parts}-{centroids}"


class MetricOutput:
    """Write human-readable JSON parameters before evaluation and its result after."""

    def __init__(self, metric, checkpoint, data_type, *, model_name, training_phase, part_groups=None, **parameters):
        self.path = output_directory(metric, checkpoint, data_type, **parameters)
        self.path.mkdir(parents=True, exist_ok=True)
        group = "ungrouped"
        if parameters.get("group_checkpoint") is not None:
            identity = hashlib.sha256(_absolute(parameters["group_checkpoint"]).encode()).hexdigest()[:10]
            group = f"grouped-{identity}"
        tokens = []
        if parameters.get("threshold") is not None:
            tokens.append(f"th_{parameters['threshold']}")
        if parameters.get("std_dev") is not None:
            tokens.append(f"std_{parameters['std_dev']}")
        tokens.append(f"qt_{parameters['quantile']}")
        if parameters.get("seed") is not None:
            tokens.append(f"seed_{parameters['seed']}")
        if parameters.get("merge_left_right_parts"):
            tokens.append("lrmerged")
        if parameters.get("quantile_scope", "class") == "image":
            tokens.append("qscope_image")
        tokens.extend((f"checkpoint_{_label(Path(checkpoint).stem)}", f"group_{group}"))
        self.file_suffix = "_".join(tokens)
        self.metric = metric
        versions = {}
        for package in ("torch", "torchvision", "numpy", "pandas", "Pillow", "opencv-python"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        self.description = {
            "metric": metric,
            "checkpoint": _absolute(checkpoint),
            "group_checkpoint": _absolute(parameters.get("group_checkpoint")),
            "model_name": model_name,
            "training_phase": training_phase,
            "dataset": data_type,
            "split": "val",
            "parts": "filteredparts" if parameters.get("official_parts_only") or parameters.get("require_semantic_match") else "allparts",
            "part_filter": (
                "official Cityscapes semantic/part pairs" if parameters.get("official_parts_only")
                else "Pascal part/VOC semantic agreement" if parameters.get("require_semantic_match")
                else "all decoded part IDs"
            ),
            "centroids": "foreground only" if parameters.get("exclude_background_centroid") else "foreground plus background",
            "parameters": {k: v for k, v in parameters.items() if k not in ("group_checkpoint", "output_dir")},
            "output_directory": str(self.path),
            "description_file": self.file(f"description_{metric}", "json").name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "packages": versions,
            "command": sys.argv,
        }
        if part_groups is not None:
            self.description["left_right_part_groups"] = part_groups
        self._write()

    def file(self, kind, extension):
        """Keep metric parameters in filenames while sharing the output directory."""
        return self.path / f"{kind}_{self.file_suffix}.{extension}"

    def record_dataset(self, dataset):
        self.description["annotations_directory"] = _absolute(dataset.annotations_dir)
        self.description["validation_images"] = len(dataset.img_ids)
        self.description["image_order_sha256"] = hashlib.sha256(
            json.dumps(list(dataset.img_ids)).encode()
        ).hexdigest()
        self._write()

    def finish(self, score, evaluated_images, observations):
        score = float(score)
        self.description.update(
            status="complete", score=score if math.isfinite(score) else None,
            evaluated_images=evaluated_images, observations=observations,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._write()

    def _write(self):
        target = self.file(f"description_{self.metric}", "json")
        temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.description, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
