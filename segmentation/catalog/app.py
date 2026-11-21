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
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from segmentation.catalog import CatalogError
from segmentation.catalog import jobs as jobs
from segmentation.catalog import table as table
from segmentation.catalog.architectures import ARCHITECTURES
from segmentation.catalog.measurement import render_measurement, support_table_row
from segmentation.catalog.models import list_models, model_snapshot, register_model

st.set_page_config(page_title="Measurement catalog", layout="wide")


@st.cache_data(show_spinner="Reading result.json files…")
def load(root, signature):
    records, errors = table.load_records(root)
    return {record["measurement_id"]: record for record in records}, table.records_frame(records), errors


def csv_download(frame, name, key, label="Download CSV"):
    st.download_button(label, frame.to_csv(index=False), f"{name}.csv", "text/csv", key=key)


def catalog_dataframe(frame, *, fields=None, column_config=None, **kwargs):
    grouped = table.grouped_frame(frame, fields=fields)
    config = {
        grouped.index.nlevels + int(grouped.columns.get_loc(table.field_header(field))): value
        for field, value in (column_config or {}).items()
    }
    return st.dataframe(grouped, column_config=config, **kwargs)


try:
    root = table.read_catalog_root()
except CatalogError as error:
    st.error(str(error))
    st.stop()

records, frame, errors = load(str(root), table.catalog_signature(root))
measurement_id = st.query_params.get("measurement")
if measurement_id:
    if st.button("Back to catalog"):
        del st.query_params["measurement"]
        st.rerun()
    if measurement_id not in records:
        st.error("Measurement not found in this catalog.")
    else:
        render_measurement(root, records[measurement_id])
    st.stop()
st.title("Measurement catalog")
st.caption(f"`{root}` · {len(frame)} measurements")
with st.expander("Registered models", expanded=frame.empty):
    models = list_models(root)
    catalog_dataframe(pd.json_normalize(models), hide_index=True)
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
        catalog_dataframe(pd.DataFrame(errors, columns=["bundle", "error"]), hide_index=True)
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
    available_statuses = sorted(frame["status"].unique())
    statuses = st.multiselect(
        "status",
        available_statuses,
        default=["complete"] if "complete" in available_statuses else [],
    )
    extras = st.multiselect(
        "Extra fields",
        other_fields,
        help="Add any other varying result.json field as a column, filter and group-by key.",
    )
    st.subheader("Filters")
    selections = {"status": statuses}
    for section, members in table.field_groups(table.varying_columns(frame, inputs) + extras).items():
        if section:
            with st.expander(section):
                for column, label in members:
                    selections[column] = st.multiselect(label, sorted(frame[column].unique()), key=f"filter_{column}")
        else:
            for column, label in members:
                selections[column] = st.multiselect(label, sorted(frame[column].unique()), key=f"filter_{column}")

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
    shown = view[[column for column in dict.fromkeys(columns) if column in view]].copy()
    if st.checkbox("Show part-support diagnostics", key="show_support_diagnostics"):
        minimum = int(
            st.number_input("Minimum supporting images", min_value=1, value=10, step=1, key="runs_support_minimum")
        )
        st.caption(
            "Derived summaries only; observations are not loaded. Recorded scores stay unchanged. "
            "Each row uses its recorded consistency threshold and a fixed observed-prototype denominator. "
            "Generate missing/stale summaries with `python -m segmentation.catalog.cli backfill-part-summaries`."
        )
        diagnostics = pd.DataFrame(
            [support_table_row(root / "measurements" / pid, records[pid], minimum) for pid in shown.index],
            index=shown.index,
        )
        shown = shown.join(diagnostics)
    linked = shown.copy()
    linked["measurement_id"] = ["?" + urlencode({"measurement": pid}) for pid in shown["measurement_id"]]
    event = catalog_dataframe(
        linked,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs",
        column_config={
            "score": st.column_config.NumberColumn(format="%.4f"),
            "measurement_id": st.column_config.LinkColumn(display_text=r"\?measurement=(.*)", alignment="left"),
        },
    )
    selected = list(shown.index[event.selection.rows])
    csv_download(shown, "measurements", "runs_csv")

    if len(selected) == 1:
        measurement_id = selected[0]
        st.link_button("Open selected measurement", "?" + urlencode({"measurement": measurement_id}))
    else:
        st.caption("Open a measurement link for details and analysis. The Compare tab uses all filtered runs.")

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
        catalog_dataframe(comparison, hide_index=True)
        if comparison.empty:
            st.info("No matching Off/On entries for this flag in the filtered runs.")
        st.caption(
            "Score sub-columns: Off = false, On = true. Each cell lists the original run scores "
            "(including missing scores); duplicate entries are kept, never averaged. "
            "Drag column borders in the lower header row to resize."
        )
        csv_download(comparison, "compare", "compare_csv")

with aggregate_tab:
    if view.empty:
        st.info("No measurements match the current filters; aggregate and pivot tables require catalog entries.")
    else:
        st.caption("Groups are formed from the filtered runs; inputs left out of the grouping are averaged over.")
        default = [column for column in varying if column != "parameters.quantile"]
        by = st.multiselect("Group by", fields, default=default)
        summary = table.aggregate(view, by)
        catalog_dataframe(
            summary,
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in ("mean", "std", "min", "max", "pooled")
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
        catalog_dataframe(pivot, fields=fields, hide_index=True)
        st.caption(
            f"Cells are mean run scores; the last column averages the per-`{pivot_column}` means "
            "and is blank when a value is missing."
        )
        csv_download(pivot, "pivot", "pivot_csv")

with coverage_tab:
    st.caption(
        "Plan an article table by choosing the registered checkpoints, metric, parameters to vary, "
        "and columns to show. A registration fixes its dataset and model metadata; only explicitly "
        "selected parameter values are crossed."
    )
    registrations = list_models(root)
    if not registrations:
        st.info("Register model checkpoints before building a measurement plan.")
    else:
        metric = st.selectbox("Metric", list(jobs.RUN_INPUTS), key="coverage_metric")
        registration_by_name = {model["name"]: model for model in registrations}
        model_names = st.multiselect(
            "Registered models",
            list(registration_by_name),
            key="coverage_models",
            help="Each selected checkpoint is evaluated only on its registered training dataset.",
        )
        selected_models = [registration_by_name[name] for name in model_names]
        selected_datasets = sorted({model["dataset"] for model in selected_models})
        grouping_models = {}
        for dataset in selected_datasets:
            eligible = [model for model in registrations if model["dataset"] == dataset]
            eligible_by_id = {model["model_id"]: model for model in eligible}
            grouping = st.selectbox(
                f"Grouping model for {dataset}",
                ["", *eligible_by_id],
                format_func=lambda reference: (
                    "None" if not reference else eligible_by_id[reference]["name"]
                ),
                key=f"coverage_grouping_{dataset}",
            )
            if grouping:
                grouping_models[dataset] = grouping

        parameter_fields = jobs.metric_parameters(metric)
        varied_parameters = st.multiselect(
            "Parameters to vary",
            list(parameter_fields),
            key=f"coverage_parameters_{metric}",
            help="Unselected parameters use their metric defaults. Selected parameters form a Cartesian product.",
        )
        parameter_values = {}
        for parameter in varied_parameters:
            candidates = jobs.parameter_candidates(metric, parameter, frame)
            parameter_values[parameter] = st.multiselect(
                f"Values: {parameter}",
                candidates,
                key=f"coverage_values_{metric}_{parameter}",
                help="Choose only the values needed for this article table.",
            )

        registration_columns = set(inputs)
        for registration in registrations:
            flattened = table.flatten_record(
                {
                    "model": model_snapshot(registration),
                    "grouping_model": model_snapshot(registration),
                }
            )
            registration_columns.update(table.input_columns(pd.DataFrame([flattened])))
        available_columns = list(dict.fromkeys([
            *inputs,
            *sorted(registration_columns),
            *[f"parameters.{name}" for name in parameter_fields],
        ]))
        display_columns = st.multiselect(
            "Table columns",
            available_columns,
            default=[field for field in ("model.name", "dataset.name") if field in available_columns],
            key=f"coverage_display_columns_{metric}",
        )
        if not display_columns:
            st.info("Choose at least one table column.")
        elif not selected_models:
            st.info("Choose at least one registered model.")
        else:
            try:
                configurations, planner_errors = jobs.plan_configurations(
                    metric,
                    [model["model_id"] for model in selected_models],
                    grouping_models,
                    parameter_values,
                    frame,
                    root=root,
                )
            except (CatalogError, ValueError) as error:
                st.warning(str(error))
            else:
                if planner_errors:
                    with st.expander(f"{len(planner_errors)} invalid combinations omitted"):
                        for error in planner_errors:
                            st.write(error)
                plan_rows, missing_configurations = [], []
                for configuration in configurations:
                    matches = jobs.matching_measurements(frame, configuration)
                    scores = [None if pd.isna(value) else value for value in matches["score"]]
                    statuses = matches["status"].tolist()
                    result = "Measured" if not matches.empty else table.NOT_MEASURED
                    plan_rows.append({
                        **{field: configuration.get(field, table.MISSING) for field in display_columns},
                        "Result": result,
                        "Scores": json.dumps(scores),
                        "Run statuses": ", ".join(statuses),
                    })
                    if matches.empty:
                        missing_configurations.append(configuration)
                plan_frame = pd.DataFrame(plan_rows)
                st.caption(
                    f"{len(configurations):,} valid configurations · "
                    f"{len(missing_configurations):,} not measured."
                )
                st.dataframe(plan_frame, hide_index=True, use_container_width=True)
                csv_download(plan_frame, "coverage_plan", "coverage_plan_csv", "Download table plan CSV")

                if not frame.empty and len(missing_configurations) == 0:
                    st.info("Every planned configuration already has a catalog entry.")
                elif missing_configurations:
                    st.subheader("Run missing measurements")
                    st.caption(
                        "Select rows to submit. Jobs are individually validated and pinned to this webapp host's "
                        "Slurm node; checkpoints and datasets are not copied."
                    )
                    missing_frame = pd.DataFrame([
                        {field: configuration.get(field, table.MISSING) for field in display_columns}
                        for configuration in missing_configurations
                    ])
                    selection_key = hashlib.sha256(missing_frame.to_json(orient="split").encode()).hexdigest()
                    missing_event = st.dataframe(
                        missing_frame,
                        hide_index=True,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="multi-row",
                        key=f"coverage_missing_{selection_key}",
                    )
                    chosen = [
                        missing_configurations[index]
                        for index in missing_event.selection.rows
                    ]
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
                                            f"Existing submission: {prior['status']}, "
                                            f"job {prior.get('job_id', 'unknown')}. "
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
                            f"Jobs use `{python}` and write to `{root}`. Submission receipts and Slurm logs are "
                            "stored under the catalog's submissions directory."
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
