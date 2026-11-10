"""Read-only Streamlit browser for the metric measurement catalog.

Run from the repository root:
    python -m streamlit run segmentation/analysis/metrics/catalog_app.py
"""

import pandas as pd
import streamlit as st

from segmentation.analysis.metrics import catalog_table as table
from segmentation.analysis.metrics.catalog import CatalogError, load_observations
from segmentation.analysis.metrics.catalog_cli import validate_measurement


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
if errors:
    with st.expander(f"{len(errors)} unreadable bundles skipped"):
        st.dataframe(pd.DataFrame(errors, columns=["bundle", "error"]), hide_index=True)
if frame.empty:
    st.info("The catalog has no readable measurements yet.")
    st.stop()

inputs = table.input_columns(frame)
other_fields = [
    column for column in table.varying_columns(frame, list(frame.columns))
    if column not in inputs and column not in table.SCORE_COLUMNS + ("measurement_id", "score_reason")
]

with st.sidebar:
    if st.button("Reload catalog"):
        st.cache_data.clear()
        st.rerun()
    statuses = st.multiselect("status", sorted(frame["status"].unique()), default=["complete"])
    extras = st.multiselect(
        "Extra fields", other_fields,
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
    columns = ["score", *varying, "numerator", "denominator", "score_reason",
               "population.contributing", "population.observations", "created_at_utc", "measurement_id"]
    shown = view[[column for column in dict.fromkeys(columns) if column in view]]
    event = st.dataframe(
        shown, hide_index=True, on_select="rerun", selection_mode="single-row", key="runs",
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
        summary, hide_index=True,
        column_config={name: st.column_config.NumberColumn(format="%.4f")
                       for name in ("mean", "std", "min", "max", "pooled")},
    )
    st.caption("`mean`: unweighted mean of run scores · `pooled`: Σ numerator / Σ denominator.")
    csv_download(summary, "aggregate", "aggregate_csv")

    st.subheader("Pivot")
    pivot_options = [column for column in fields if column not in by] or fields
    pivot_column = st.selectbox(
        "Column", pivot_options,
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
        "Compare against (column dimensions)", fields,
        default=["parameters.official_parts_only"] if "parameters.official_parts_only" in fields else [],
        key="coverage_columns",
    )
    fixed_fields = st.multiselect(
        "Fix inputs (filters)", [field for field in fields if field not in coverage_columns],
        key="coverage_fixed_fields",
    )
    fixed = {}
    for field in fixed_fields:
        fixed[field] = st.selectbox(
            f"Fixed value: {field}", sorted(frame[field].unique()),
            accept_new_options=True, key=f"coverage_fixed_{field}",
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
                options, default=observed or options, accept_new_options=True,
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
