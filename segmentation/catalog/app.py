"""Streamlit browser and Slurm launcher for the metric measurement catalog.

Run from the repository root:
    python -m streamlit run segmentation/catalog/app.py
"""

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from segmentation.catalog import jobs as jobs
from segmentation.catalog import table as table
from segmentation.catalog import CatalogError, load_observations
from segmentation.catalog.cli import validate_measurement
from segmentation.catalog.architectures import ARCHITECTURES
from segmentation.catalog.models import list_models, register_model

st.set_page_config(page_title="Measurement catalog", layout="wide")


@st.cache_data(show_spinner="Reading result.json files…")
def load(root, signature):
    records, errors = table.load_records(root)
    return {record["measurement_id"]: record for record in records}, table.records_frame(records), errors


def csv_download(frame, name, key, label="Download CSV"):
    st.download_button(label, frame.to_csv(index=False), f"{name}.csv", "text/csv", key=key)


try:
    root = table.read_catalog_root()
except CatalogError as error:
    st.error(str(error))
    st.stop()

records, frame, errors = load(str(root), table.catalog_signature(root))
st.title("Measurement catalog")
st.caption(f"`{root}` · {len(frame)} measurements")
with st.expander("Registered models", expanded=frame.empty):
    models = list_models(root)
    st.dataframe(pd.json_normalize(models), hide_index=True)
    st.caption("Use a model's UUID or unique name in metric commands. Files are referenced, never copied.")
    architecture = st.selectbox("Architecture", sorted({name for name, _ in ARCHITECTURES}))
    st.json(ARCHITECTURES[(architecture, 1)].model_json_schema(), expanded=False)
    with st.form("register_model"):
        name = st.text_input("Unique model name")
        checkpoint = st.text_input("Existing checkpoint path")
        dataset = st.selectbox("Training dataset", ["cityscapes", "pascal"])
        config = st.text_area(
            "Architecture configuration (JSON)",
            value='{"stage": "push"}' if architecture == "ProtoSeg" else '{"stage": "push", "grouped": false}',
        )
        known_seed = st.checkbox("Training seed is known", value=False)
        training_seed = st.number_input("Training seed", value=0, step=1)
        tags = st.text_area("Tags (JSON string values)", value="{}")
        if st.form_submit_button("Register checkpoint"):
            try:
                register_model(
                    root=root,
                    name=name,
                    architecture=architecture,
                    config=json.loads(config),
                    dataset=dataset,
                    checkpoint=checkpoint,
                    seed=int(training_seed) if known_seed else None,
                    tags=json.loads(tags),
                )
            except (CatalogError, OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.rerun()
if errors:
    with st.expander(f"{len(errors)} unreadable bundles skipped"):
        st.dataframe(pd.DataFrame(errors, columns=["bundle", "error"]), hide_index=True)
if frame.empty:
    st.info("The catalog has no readable measurements yet.")
    st.stop()

inputs = table.input_columns(frame)
other_fields = [
    column
    for column in table.varying_columns(frame, list(frame.columns))
    if column not in inputs and column not in table.SCORE_COLUMNS + ("measurement_id", "score_reason")
]
if jobs.PROTOTYPE_PATH in frame and jobs.PROTOTYPE_PATH not in other_fields:
    other_fields.append(jobs.PROTOTYPE_PATH)

with st.sidebar:
    if st.button("Reload catalog"):
        st.cache_data.clear()
        st.rerun()
    statuses = st.multiselect("status", sorted(frame["status"].unique()), default=["complete"])
    extras = st.multiselect(
        "Extra fields",
        other_fields,
        help="Add any other varying result.json field as a column, filter and group-by key.",
    )
    st.subheader("Filters")
    selections = {"status": statuses}
    for column in table.varying_columns(frame, inputs) + extras:
        selections[column] = st.multiselect(column, sorted(frame[column].unique()))

view = table.filter_frame(frame, selections)
fields = inputs + extras
varying = table.varying_columns(view, fields)
common = {column: view[column].iloc[0] for column in fields if column not in varying} if len(view) else {}

runs_tab, compare_tab, aggregate_tab, coverage_tab = st.tabs(["Runs", "Compare", "Aggregate", "Coverage"])

with runs_tab:
    st.caption(f"{len(view)} of {len(frame)} measurements")
    if common:
        st.caption("Shared by all shown runs: " + " · ".join(f"`{key}={value}`" for key, value in common.items()))
    columns = [
        "score",
        *varying,
        "numerator",
        "denominator",
        "score_reason",
        "population.contributing",
        "population.observations",
        "created_at_utc",
        "measurement_id",
    ]
    shown = view[[column for column in dict.fromkeys(columns) if column in view]]
    event = st.dataframe(
        shown,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs",
        column_config={"score": st.column_config.NumberColumn(format="%.4f")},
    )
    selected = list(shown.index[event.selection.rows])
    csv_download(shown, "measurements", "runs_csv")

    if len(selected) == 1:
        measurement_id = selected[0]
        bundle = root / "measurements" / measurement_id
        st.subheader(f"Measurement {measurement_id}")
        left, right = st.columns(2)
        with left:
            st.json(records[measurement_id], expanded=1)
        with right:
            if st.button("Validate bundle (checks the observation hash)"):
                st.write(validate_measurement(bundle))
            if st.button("Load observations.csv"):
                try:
                    _, rows = load_observations(bundle, verify_hashes=False)
                except CatalogError as error:
                    st.error(str(error))
                else:
                    observations = pd.DataFrame(rows)
                    st.caption(
                        f"{len(observations)} rows · {observations['sample_id'].nunique()} samples · "
                        f"{observations['prototype_id'].nunique()} prototypes"
                    )
                    st.dataframe(observations, hide_index=True, height=400)
    else:
        st.caption("Select one row for details. The Compare tab uses all filtered runs.")

with compare_tab:
    flags = table.boolean_columns(frame, fields)
    if not flags:
        st.info("No boolean inputs are available to compare.")
    else:
        flag = st.selectbox("Compare flag", flags)
        st.caption(
            "Matches filtered catalog entries where every input and selected extra field "
            "except the chosen flag is identical. Only configurations with both Off and On entries are shown."
        )
        comparison = table.compare_flag(view, flag, fields)
        st.dataframe(comparison, hide_index=True)
        if comparison.empty:
            st.info("No matching Off/On entries for this flag in the filtered runs.")
        st.caption(
            "Score sub-columns: Off = false, On = true. Each cell lists the original run scores "
            "(including missing scores); duplicate entries are kept, never averaged. "
            "Drag column borders in the lower header row to resize."
        )
        csv_download(comparison, "compare", "compare_csv")

with aggregate_tab:
    st.caption("Groups are formed from the filtered runs; inputs left out of the grouping are averaged over.")
    default = [column for column in varying if column != "parameters.quantile"]
    by = st.multiselect("Group by", fields, default=default)
    summary = table.aggregate(view, by)
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.4f") for name in ("mean", "std", "min", "max", "pooled")
        },
    )
    st.caption("`mean`: unweighted mean of run scores · `pooled`: Σ numerator / Σ denominator.")
    csv_download(summary, "aggregate", "aggregate_csv")

    st.subheader("Pivot")
    pivot_options = [column for column in fields if column not in by] or fields
    pivot_column = st.selectbox(
        "Column",
        pivot_options,
        index=pivot_options.index("parameters.quantile") if "parameters.quantile" in pivot_options else 0,
    )
    pivot = table.pivot_means(view, by, pivot_column)
    st.dataframe(pivot, hide_index=True)
    st.caption(
        f"Cells are mean run scores; the last column averages the per-`{pivot_column}` means "
        "and is blank when a value is missing."
    )
    csv_download(pivot, "pivot", "pivot_csv")

with coverage_tab:
    st.caption(
        "Build a full configuration grid, including measurements that do not exist yet. "
        "This view uses its own filters and all catalog statuses, independently of the sidebar filters. "
        "Sidebar Extra fields are included as additional inputs."
    )
    coverage_columns = st.multiselect(
        "Compare against (column dimensions)",
        fields,
        default=["parameters.official_parts_only"] if "parameters.official_parts_only" in fields else [],
        key="coverage_columns",
    )
    fixed_fields = st.multiselect(
        "Fix inputs (filters)",
        [field for field in fields if field not in coverage_columns],
        key="coverage_fixed_fields",
    )
    fixed = {}
    for field in fixed_fields:
        fixed[field] = st.selectbox(
            f"Fixed value: {field}",
            sorted(frame[field].unique()),
            accept_new_options=True,
            key=f"coverage_fixed_{field}",
        )
    candidates_frame = table.filter_frame(frame, {field: [value] for field, value in fixed.items()})
    candidates = {}
    with st.expander("Candidate values (select existing values or type additional values)", expanded=True):
        for field in fields:
            if field in fixed:
                continue
            observed = sorted(candidates_frame[field].unique())
            options = sorted(frame[field].unique())
            candidates[field] = st.multiselect(
                f"{'Column' if field in coverage_columns else 'Row'} values: {field}",
                options,
                default=observed or options,
                accept_new_options=True,
                key=f"coverage_values_{field}",
            )
    st.caption(
        "Every non-fixed input outside the column dimensions becomes a row dimension. "
        "All selected values are crossed, even if a combination is incompatible. "
        "Use lowercase true/false for flags and the catalog's exact text representation for other inputs."
    )
    if not coverage_columns:
        st.info("Choose at least one column dimension.")
    else:
        try:
            coverage, missing = table.coverage_matrix(frame, coverage_columns, candidates, fixed)
        except ValueError as error:
            st.warning(str(error))
        else:
            st.caption(
                f"{len(coverage):,} configurations · {len(missing):,} unmeasured cells. "
                "Cells list original scores and run statuses; null means an existing run has no score. "
                "Not measured means no catalog entry exists."
            )
            st.dataframe(coverage, hide_index=True)
            csv_download(coverage, "coverage", "coverage_csv", "Download coverage CSV")
            csv_download(missing, "missing_measurements", "coverage_missing_csv", "Download missing configurations CSV")
            st.subheader("Run missing measurements")
            st.caption(
                "Select individual missing configurations below. Jobs request one GPU, eight CPUs and 32 GB "
                "for up to four hours, pinned to this webapp host's Slurm node. "
                "Invalid configurations must be corrected before submission."
            )
            selection_key = hashlib.sha256(missing.to_json(orient="split").encode()).hexdigest()
            missing_event = st.dataframe(
                missing,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key=f"coverage_missing_{selection_key}",
            )
            chosen = missing.iloc[missing_event.selection.rows].to_dict("records")
            checkout = Path(__file__).resolve().parents[2]
            python = st.text_input(
                "Evaluation Python executable",
                value=os.environ.get("METRIC_PYTHON", str(checkout / ".venv/bin/python")),
                key="coverage_python",
                help="Use a full evaluation environment, not an app-only environment.",
            )
            plans, plan_errors = [], []
            if chosen:
                try:
                    node = jobs.current_slurm_node()
                    st.caption(f"Slurm nodelist: `{node}` (not editable; node-local checkout and caches).")
                    for index, configuration in enumerate(chosen, start=1):
                        try:
                            arguments = jobs.measurement_arguments(configuration, frame, root=root)
                            prior = jobs.submission_record(root, arguments)
                            if prior is not None:
                                raise ValueError(
                                    f"Existing submission: {prior['status']}, job {prior.get('job_id', 'unknown')}. "
                                    f"Receipt: {jobs.submission_path(root, arguments)}"
                                )
                            plans.append(shlex.join(jobs.slurm_command(checkout, root, node, arguments)))
                        except (CatalogError, OSError, ValueError) as error:
                            plan_errors.append(f"Selection {index}: {error}")
                except (CatalogError, OSError, ValueError, subprocess.SubprocessError) as error:
                    plan_errors.append(f"Slurm is unavailable on this host: {error}")
            if plans:
                st.code("\n".join(plans), language="bash")
                st.caption(
                    f"Jobs use `{python}` and write to `{root}`. "
                    "Submission receipts and Slurm logs are stored under the catalog's submissions directory. "
                    "Queued jobs remain absent from the matrix until they create a measurement bundle."
                )
            for error in plan_errors:
                st.warning(error)
            if st.button(
                "Run selected missing measurements",
                disabled=not chosen or bool(plan_errors),
                key="coverage_submit",
            ):
                outcomes = []
                with st.spinner("Validating and submitting Slurm jobs…"):
                    for configuration in chosen:
                        try:
                            receipt = jobs.submit_measurement(configuration, checkout, root, python)
                            outcomes.append(f"Submitted job {receipt['job_id']} on {receipt['node']}.")
                        except (OSError, ValueError, subprocess.SubprocessError) as error:
                            details = getattr(error, "stderr", None) or str(error)
                            outcomes.append(f"Not submitted or submission uncertain: {details}")
                st.session_state["coverage_submission_outcomes"] = outcomes
                st.rerun()
            for outcome in st.session_state.get("coverage_submission_outcomes", []):
                st.write(outcome)
