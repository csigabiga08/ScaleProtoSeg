"""Plan and submit explicit coverage configurations through the metric Slurm wrapper."""

import hashlib
import json
import math
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from segmentation.catalog import table as table
from segmentation.analysis.metrics.protocol import (
    ConsistencyRunInput,
    StabilityRunInput,
)

PROTOTYPE_PATH = "model.inputs.prototype_checkpoint.path"
RUN_INPUTS = {"consistency": ConsistencyRunInput, "stability": StabilityRunInput}
BOOLEAN_PARAMETERS = {
    "official_parts_only",
    "require_semantic_match",
    "exclude_background_centroid",
    "merge_left_right_parts",
}


def measurement_arguments(configuration, frame):
    """Translate a complete matrix cell to lossless, validated Hydra overrides."""
    metric = configuration.get("metric")
    if metric not in RUN_INPUTS:
        raise ValueError("Only consistency and stability measurements can be submitted.")
    input_type = RUN_INPUTS[metric]
    parameters = BOOLEAN_PARAMETERS | {"quantile", "threshold", "std_dev", "seed"}
    allowed = set(table.IDENTITY_INPUTS) | {PROTOTYPE_PATH} | {f"parameters.{name}" for name in parameters}
    for field in configuration:
        if field not in allowed:
            raise ValueError(f"{field} cannot be set by the metric runner; remove it from Extra fields.")
    inputs = {}
    for field, argument in (
        ("model.label", "model_name"),
        ("model.training_stage", "training_phase"),
        ("dataset.name", "data_type"),
    ):
        value = configuration.get(field)
        if not value or value == table.MISSING:
            raise ValueError(f"Choose an explicit value for {field}.")
        inputs[argument] = value
    group = configuration.get("model.inputs.group_checkpoint.path", table.MISSING)
    if group != table.MISSING:
        inputs["group_checkpoint"] = str(Path(group).expanduser().resolve())
        if inputs["group_checkpoint"] != group:
            raise ValueError(f"Use the absolute group checkpoint path: {inputs['group_checkpoint']}")
    checkpoint = configuration.get(PROTOTYPE_PATH)
    if checkpoint is None:
        references = table.filter_frame(
            frame, {field: [configuration[field]] for field in ("model.label", "model.training_stage", "dataset.name")}
        )
        paths = set(references[PROTOTYPE_PATH]) - {table.MISSING} if PROTOTYPE_PATH in references else set()
        if len(paths) != 1:
            raise ValueError(
                "No unique prototype checkpoint for this model/stage/dataset. "
                f"Add {PROTOTYPE_PATH} through Extra fields and select its exact path."
            )
        checkpoint = paths.pop()
    if not checkpoint or checkpoint == table.MISSING:
        raise ValueError("An explicit prototype checkpoint is required.")
    checkpoint_path = str(Path(checkpoint).expanduser().resolve())
    if PROTOTYPE_PATH in configuration and checkpoint_path != checkpoint:
        raise ValueError(f"Use the absolute prototype checkpoint path: {checkpoint_path}")
    inputs["checkpoint" if metric == "consistency" else "checkpoint_path"] = checkpoint_path
    for field, value in configuration.items():
        if not field.startswith("parameters."):
            continue
        parameter = field[len("parameters.") :]
        if parameter not in input_type.model_fields:
            if value != table.MISSING:
                raise ValueError(f"{parameter} is not an input of {metric}; select {table.MISSING} instead.")
            continue
        if value == table.MISSING:
            raise ValueError(f"Choose an explicit value for {field}; missing metadata is not a runnable value.")
        if parameter in BOOLEAN_PARAMETERS:
            if value not in ("true", "false"):
                raise ValueError(f"{field} must be true or false.")
            inputs[parameter] = value == "true"
        elif parameter == "seed":
            inputs[parameter] = int(value)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{field} must be finite.")
            inputs[parameter] = number
        canonical = table.display_value(inputs[parameter])
        if canonical != value:
            raise ValueError(f"Use the catalog representation {canonical!r} for {field}.")
    run_input = input_type(**inputs)
    for parameter in ("quantile", "threshold"):
        value = getattr(run_input, parameter, None)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{parameter} must be between 0 and 1.")
    if metric == "stability" and run_input.std_dev < 0:
        raise ValueError("std_dev must be nonnegative.")
    for path in (run_input.resolved_checkpoint_path, run_input.resolved_group_checkpoint_path):
        if path is not None and not Path(path).is_file():
            raise ValueError(f"Checkpoint does not exist on this host: {path}")
    return [metric] + [
        f"{key}={json.dumps(value)}" for key, value in run_input.model_dump().items() if value is not None
    ]


def current_slurm_node():
    """Restrict node-local execution to the Slurm node hosting the webapp."""
    host = socket.gethostname().split(".")[0]
    result = subprocess.run(
        ["scontrol", "show", "node", host, "-o"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    match = re.search(r"(?:^|\s)NodeName=([^\s]+)", result.stdout)
    if match is None:
        raise ValueError(f"This host ({host}) is not a recognized Slurm node.")
    return match.group(1)


def validate_runtime(python):
    python = Path(python).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Select an executable evaluation Python interpreter.")
    subprocess.run(
        [str(python), "-c", "import torch, torchvision, hydra, cv2, gin, dotenv, pydantic"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return python


def submission_path(root, arguments):
    digest = hashlib.sha256(json.dumps(arguments).encode()).hexdigest()
    return Path(root) / "submissions" / f"{digest}.json"


def submission_record(root, arguments):
    path = submission_path(root, arguments)
    return json.loads(path.read_text()) if path.exists() else None


def slurm_command(checkout, root, node, arguments):
    return [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--nodelist={node}",
        f"--chdir={Path(checkout).resolve()}",
        f"--output={Path(root).resolve() / 'submissions' / 'metric-%j.out'}",
        str(Path(checkout).resolve() / "scripts/run_metric.slurm"),
        *arguments,
    ]


def submit_measurement(configuration, checkout, root, python):
    """Recheck catalog coverage and atomically claim one submission before sbatch.

    A failed or uncertain submission retains its receipt: an operator must inspect
    it before retrying, rather than risk submitting a second job after a timeout.
    """
    records, errors = table.load_records(root)
    if errors:
        raise ValueError("Unreadable catalog bundles prevent a reliable missing-measurement check.")
    fresh = table.records_frame(records)
    if all(field in fresh for field in configuration):
        if not table.filter_frame(fresh, {key: [value] for key, value in configuration.items()}).empty:
            raise ValueError("This configuration now has a catalog entry; reload Coverage.")
    else:
        raise ValueError("Catalog inputs changed; reload Coverage before submitting.")
    arguments = measurement_arguments(configuration, fresh)
    python = validate_runtime(python)
    node = current_slurm_node()
    checkout, root = Path(checkout).resolve(), Path(root).resolve()
    if not (checkout / "scripts/run_metric.slurm").is_file():
        raise ValueError("The metric Slurm wrapper is missing from this checkout.")
    if not os.access(root / "measurements", os.W_OK):
        raise ValueError("The catalog measurements directory must be writable by the job user.")
    receipt = submission_path(root, arguments)
    receipt.parent.mkdir(exist_ok=True)
    record = {
        "status": "submitting",
        "configuration": configuration,
        "arguments": arguments,
        "node": node,
        "python": str(python),
        "checkout": str(checkout),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2)
    except FileExistsError as error:
        raise ValueError(f"This configuration already has a submission receipt: {receipt}") from error
    environment = {
        **os.environ,
        "MEASUREMENT_CATALOG_ROOT": str(root),
        "SCALEPROTOSEG_ROOT": str(checkout),
        "METRIC_PYTHON": str(python),
    }
    try:
        result = subprocess.run(
            slurm_command(checkout, root, node, arguments),
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Unexpected sbatch response: {result.stdout.strip()}")
        record.update(status="submitted", job_id=job_id)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record.update(status="submission_error", error=str(error))
        raise
    finally:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(receipt)
    return record
