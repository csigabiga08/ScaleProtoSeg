"""Plan and submit explicit coverage configurations through the metric Slurm wrapper."""

import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from segmentation.catalog import table as table
from segmentation.analysis.metrics.protocol import (
    ConsistencyRunInput,
    ImpreciseActivationsRunInput,
    StabilityRunInput,
)

PROTOTYPE_PATH = "model.inputs.prototype_checkpoint.path"
RUN_INPUTS = {
    "consistency": ConsistencyRunInput,
    "stability": StabilityRunInput,
    "imprecise_activations": ImpreciseActivationsRunInput,
}
BOOLEAN_PARAMETERS = {
    "official_parts_only",
    "require_semantic_match",
    "exclude_background_centroid",
    "merge_left_right_parts",
}
RUN_INPUT_FIELDS = {"model", "grouping_model", "data_type"}
TERMINAL_JOB_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def metric_parameters(metric):
    """Return parameter fields available to a metric, excluding model identity inputs."""
    try:
        input_type = RUN_INPUTS[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported metric: {metric}") from error
    return {name: field for name, field in input_type.model_fields.items() if name not in RUN_INPUT_FIELDS}


def parameter_candidates(metric, parameter, frame):
    """Suggest observed and default values without implicitly varying a parameter."""
    fields = metric_parameters(metric)
    if parameter not in fields:
        raise ValueError(f"{parameter} is not a parameter of {metric}.")
    values = set()
    column = f"parameters.{parameter}"
    if column in frame:
        values.update(frame.loc[frame["metric"] == metric, column].tolist())
    field = fields[parameter]
    if not field.is_required():
        values.add(table.display_value(field.default))
    if field.annotation is bool:
        values.update(("false", "true"))
    return sorted(value for value in values if value != table.MISSING)


def _parameter_value(field, value):
    if field.annotation is bool:
        if value not in ("true", "false"):
            raise ValueError("Boolean parameter values must be lowercase true or false.")
        return value == "true"
    if field.annotation is int:
        return int(value)
    if field.annotation is float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Numeric parameter values must be finite.")
        return number
    return value


def plan_configurations(metric, model_references, grouping_models, parameter_values, frame, root=None):
    """Build only valid model/metric combinations; model metadata determines the dataset."""
    input_type = RUN_INPUTS.get(metric)
    if input_type is None:
        raise ValueError(f"Unsupported metric: {metric}")
    parameter_fields = metric_parameters(metric)
    unknown_parameters = set(parameter_values) - set(parameter_fields)
    if unknown_parameters:
        raise ValueError(f"Unsupported {metric} parameters: {', '.join(sorted(unknown_parameters))}")
    if not model_references:
        raise ValueError("Select at least one registered model.")
    if any(not values for values in parameter_values.values()):
        raise ValueError("Select at least one candidate value for every varied parameter.")

    parameter_names = list(parameter_values)
    typed_values = {
        name: [_parameter_value(parameter_fields[name], value) for value in parameter_values[name]]
        for name in parameter_names
    }
    parameter_combinations = (
        [dict(zip(parameter_names, values)) for values in product(*(typed_values[name] for name in parameter_names))]
        if parameter_names
        else [{}]
    )
    from segmentation.catalog.models import find_model, model_snapshot

    configurations, errors = [], []
    for reference in dict.fromkeys(model_references):
        model = find_model(reference, root)
        grouping_reference = grouping_models.get(model["dataset"])
        grouping = find_model(grouping_reference, root) if grouping_reference else None
        if grouping and grouping["dataset"] != model["dataset"]:
            raise ValueError("Main and grouping models must use the same dataset.")
        for parameters in parameter_combinations:
            try:
                run_input = input_type(
                    model=model["model_id"],
                    grouping_model=grouping["model_id"] if grouping else None,
                    data_type=model["dataset"],
                    **parameters,
                )
            except ValidationError as error:
                errors.append(f"{model['name']}: {error}")
                continue
            run_values = run_input.model_dump()
            candidate_record = {
                "metric": metric,
                "model": model_snapshot(model),
                "grouping_model": model_snapshot(grouping) if grouping else {},
                "dataset": {"name": model["dataset"]},
                "parameters": {name: run_values[name] for name in parameter_fields},
            }
            flattened = table.flatten_record(candidate_record)
            candidate_fields = table.input_columns(pd.DataFrame([flattened]))
            input_fields = list(dict.fromkeys([*table.input_columns(frame), *candidate_fields]))
            configuration = {}
            for field in input_fields:
                if field.startswith("parameters."):
                    parameter = field[len("parameters.") :]
                    value = run_values.get(parameter) if parameter in parameter_fields else None
                else:
                    value = flattened.get(field)
                configuration[field] = table.display_value(value)
            configurations.append(configuration)
    if not configurations and errors:
        raise ValueError("No valid configurations: " + "; ".join(errors))
    return configurations, errors


def matching_measurements(frame, configuration):
    """Find exact existing entries; absent new fields cannot match old catalog rows."""
    fields = table.input_columns(frame)
    if not set(fields).issubset(configuration):
        raise ValueError("The plan is missing catalog inputs; rebuild it after reloading the catalog.")
    if any(field not in frame.columns and value != table.MISSING for field, value in configuration.items()):
        return frame.iloc[0:0]
    return table.filter_frame(frame, {field: [configuration[field]] for field in fields})


def measurement_arguments(configuration, frame, root=None):
    """Translate a complete matrix cell to lossless, validated Hydra overrides."""
    metric = configuration.get("metric")
    if metric not in RUN_INPUTS:
        raise ValueError("Only consistency, stability, and imprecise activation measurements can be submitted.")
    input_type = RUN_INPUTS[metric]
    from segmentation.catalog.models import model_snapshot
    from segmentation.catalog.models import find_model

    model_reference = configuration.get("model.registration_id")
    if not model_reference or model_reference == table.MISSING:
        raise ValueError("Select a registered model; migrate legacy measurements before submitting them.")
    model = find_model(model_reference, root, verify=True)
    group_reference = configuration.get("grouping_model.registration_id", table.MISSING)
    grouping = find_model(group_reference, root, verify=True) if group_reference != table.MISSING else None
    if grouping and grouping["dataset"] != model["dataset"]:
        raise ValueError("Main and grouping models must use the same dataset.")
    expected = table.flatten_record(
        {
            "model": model_snapshot(model),
            "grouping_model": model_snapshot(grouping) if grouping else {},
            "dataset": {"name": model["dataset"]},
        }
    )
    candidate_fields = table.input_columns(pd.DataFrame([expected]))
    parameters = BOOLEAN_PARAMETERS | {"quantile", "threshold", "std_dev", "seed"}
    allowed = set(table.input_columns(frame)) | set(candidate_fields) | {f"parameters.{name}" for name in parameters}
    for field in configuration:
        if field not in allowed:
            raise ValueError(f"{field} cannot be set by the metric runner; remove it from Extra fields.")
    for field, value in configuration.items():
        if field == "metric" or field.startswith("parameters."):
            continue
        if value != table.display_value(expected.get(field)):
            raise ValueError(f"{field} does not match the selected model registration.")
    inputs = {
        "model": model["model_id"],
        "grouping_model": grouping["model_id"] if grouping else None,
        "data_type": model["dataset"],
    }
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


@contextmanager
def submission_lock(root, arguments):
    locks = Path(root) / "submissions" / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (submission_path(root, arguments).stem + ".lock")).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _missing_arguments(configuration, root):
    records, errors = table.load_records(root)
    if errors:
        raise ValueError("Unreadable catalog bundles prevent a reliable missing-measurement check.")
    fresh = table.records_frame(records)
    if not set(table.input_columns(fresh)).issubset(configuration):
        raise ValueError("Catalog inputs changed; reload Coverage before submitting.")
    if not matching_measurements(fresh, configuration).empty:
        raise ValueError("This configuration now has a catalog entry; reload Coverage.")
    return measurement_arguments(configuration, fresh, root=root)


def terminal_job_state(job_id):
    """Require an unambiguous terminal accounting record and no active queue entry."""
    job_id = str(job_id)
    if not job_id.isdigit():
        raise ValueError("The receipt has no valid Slurm job ID; inspect it manually.")
    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            f"--jobs={job_id}",
            "--format=JobIDRaw,State",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.strip().split("|") for line in accounting.stdout.splitlines() if line.strip()]
    matching = [row for row in rows if len(row) == 2 and row[0] == job_id]
    if len(matching) != 1 or len(rows) != 1:
        raise ValueError(f"No unambiguous accounting record for Slurm job {job_id}; inspect it manually.")
    parts = matching[0][1].split()
    if not parts:
        raise ValueError(f"No accounting state for Slurm job {job_id}; inspect it manually.")
    state = parts[0].rstrip("+")
    if state not in TERMINAL_JOB_STATES:
        raise ValueError(f"Slurm job {job_id} is {state}; retry is blocked.")
    queue = subprocess.run(
        ["squeue", "--noheader", "--states=all", "--format=%i|%T"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for line in queue.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 2:
            raise ValueError("Unexpected Slurm queue output; inspect the job manually.")
        if fields[0] == job_id and fields[1] not in TERMINAL_JOB_STATES:
            raise ValueError(f"Slurm job {job_id} is still {fields[1]}; retry is blocked.")
    return state


def release_submission_for_retry(configuration, root):
    """Archive a terminal job's receipt after rechecking that no measurement exists."""
    arguments = _missing_arguments(configuration, root)
    with submission_lock(root, arguments):
        if _missing_arguments(configuration, root) != arguments:
            raise ValueError("Measurement inputs changed; reload Coverage before retrying.")
        receipt_path = submission_path(root, arguments)
        receipt = submission_record(root, arguments)
        if receipt is None:
            raise ValueError("There is no submission receipt for this configuration.")
        if receipt.get("status") != "submitted" or receipt.get("arguments") != arguments:
            raise ValueError("The receipt is uncertain or does not match these inputs; inspect it manually.")
        state = terminal_job_state(receipt.get("job_id"))
        if _missing_arguments(configuration, root) != arguments:
            raise ValueError("Measurement inputs changed; reload Coverage before retrying.")
        slot = Path(root) / "submissions" / "archived" / receipt_path.stem / str(uuid.uuid4())
        slot.mkdir(parents=True)
        destination = slot / receipt_path.name
        try:
            receipt_path.rename(destination)
        except OSError:
            slot.rmdir()
            raise
        return {"job_id": receipt["job_id"], "state": state, "receipt": str(destination)}


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
    arguments = _missing_arguments(configuration, root)
    python = validate_runtime(python)
    node = current_slurm_node()
    checkout, root = Path(checkout).resolve(), Path(root).resolve()
    if not (checkout / "scripts/run_metric.slurm").is_file():
        raise ValueError("The metric Slurm wrapper is missing from this checkout.")
    if not os.access(root / "measurements", os.W_OK):
        raise ValueError("The catalog measurements directory must be writable by the job user.")
    record = {
        "status": "submitting",
        "configuration": configuration,
        "arguments": arguments,
        "node": node,
        "python": str(python),
        "checkout": str(checkout),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    environment = {
        **os.environ,
        "MEASUREMENT_CATALOG_ROOT": str(root),
        "SCALEPROTOSEG_ROOT": str(checkout),
        "METRIC_PYTHON": str(python),
    }
    with submission_lock(root, arguments):
        if _missing_arguments(configuration, root) != arguments:
            raise ValueError("Measurement inputs changed; reload Coverage before submitting.")
        receipt = submission_path(root, arguments)
        try:
            with receipt.open("x") as stream:
                json.dump(record, stream, indent=2)
        except FileExistsError as error:
            raise ValueError(f"This configuration already has a submission receipt: {receipt}") from error
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
