"""Read-only measurement detail page and part-activation analysis."""

import pandas as pd
import streamlit as st

from segmentation.analysis.metrics.part_activations import prototype_part_correlation
from segmentation.catalog import CatalogError, load_observations
from segmentation.catalog.cli import validate_measurement


@st.cache_data(show_spinner="Reading and checking observations.csv...")
def _observations(bundle, signature):
    return load_observations(bundle)


def _download(frame, name, measurement_id):
    st.download_button(
        f"Download {name.replace('_', ' ')} CSV",
        frame.to_csv(),
        f"{measurement_id}_{name}.csv",
        "text/csv",
        key=f"measurement_{name}",
    )


def render_measurement(root, record):
    """Render one catalog record; only load observation evidence on request."""
    measurement_id = record["measurement_id"]
    bundle = root / "measurements" / measurement_id
    st.title(f"Measurement {measurement_id}")
    st.caption(
        f"{record['metric']} | {record['status']} | "
        f"{record['model'].get('label', '')} | {record['dataset'].get('name', '')}"
    )
    left, right = st.columns(2)
    with left:
        st.subheader("General information")
        st.json(
            {key: record.get(key) for key in ("model", "dataset", "parameters", "scores", "population")},
            expanded=1,
        )
    with right:
        with st.expander("Full measurement metadata"):
            st.json(record, expanded=1)
        if st.button("Validate bundle (checks the observation hash)"):
            try:
                st.write(validate_measurement(bundle))
            except CatalogError as error:
                st.error(str(error))

    st.subheader("Prototype part-activation correlation")
    st.caption(
        "Each prototype's part vector is averaged per part over available observations; "
        "zeros count, blanks do not. Pearson correlation uses only jointly observed parts. "
        "Fewer than two shared parts or constant vectors give undefined (blank/gray) correlations."
    )
    st.caption(
        "Part IDs in these CSVs are class-local. Analysis uses (class, part ID) coordinates, "
        "so cross-class correlations are undefined. Semantic-class-only mode keeps IDs in the "
        "class's taxonomy; existing observations cannot identify the source class of foreign "
        "part IDs. Stability measurements use the clean vectors, not noisy_part_*."
    )
    if record["status"] == "running" or "observations" not in record["artifacts"]:
        st.info("This measurement has no finalized observation evidence.")
        return
    if not st.checkbox("Load part-activation analysis", key=f"analyze_{measurement_id}"):
        return
    try:
        result, rows = _observations(str(bundle), record["artifacts"]["observations"]["sha256"])
    except CatalogError as error:
        st.error(str(error))
        return
    with st.expander("Observations"):
        observations = pd.DataFrame(rows)
        st.caption(
            f"{len(rows)} rows | {observations.get('sample_id', pd.Series(dtype=str)).nunique()} samples | "
            f"{observations.get('prototype_id', pd.Series(dtype=str)).nunique()} prototypes"
        )
        st.dataframe(observations, hide_index=True)
    consistent_only = st.checkbox("Consistent prototypes only")
    semantic_only = st.checkbox("Only parts corresponding to the prototype's semantic class")
    threshold = st.number_input(
        "Consistency threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(result["protocol"].get("consistency", {}).get("threshold", 0.8)),
        step=0.01,
        disabled=not consistent_only,
        help="Keep prototypes with any mean activation strictly above this threshold, before the part filter.",
    )
    try:
        activations, correlation, shared = prototype_part_correlation(
            result, rows, consistent_only=consistent_only, semantic_parts_only=semantic_only, threshold=threshold
        )
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Cannot analyze these observations: {error}")
        return
    st.caption(
        f"{len(correlation)} observed prototypes retained. Prototypes without observations cannot be analyzed. "
        "Consistency filtering uses all recorded object parts before semantic-class filtering."
    )
    if correlation.empty:
        st.info("No prototypes match the analysis options.")
        return
    labels = result["protocol"].get("classes", {}).get("evaluation_class_id_to_label", {})
    classes = {int(row["prototype_id"]): str(row["class_id"]) for row in rows}
    names = {pid: f"{pid} | {labels.get(classes[pid], classes[pid])}" for pid in correlation.index}
    selected = st.multiselect(
        "Heatmap prototypes (up to 200; downloads contain the full matrix)",
        list(correlation.index),
        default=list(correlation.index[:100]),
        format_func=names.get,
        max_selections=200,
    )
    if selected:
        heatmap = pd.DataFrame(
            [
                {
                    "prototype": names[first],
                    "other": names[second],
                    "correlation": correlation.loc[first, second],
                    "shared_parts": int(shared.loc[first, second]),
                }
                for first in selected
                for second in selected
            ]
        )
        order = [names[pid] for pid in selected]
        st.vega_lite_chart(
            heatmap,
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "other", "type": "nominal", "sort": order, "title": "Prototype"},
                    "y": {"field": "prototype", "type": "nominal", "sort": order, "title": "Prototype"},
                    "color": {
                        "condition": {
                            "test": "isValid(datum.correlation)",
                            "field": "correlation",
                            "type": "quantitative",
                            "scale": {"domain": [-1, 1], "scheme": "redblue"},
                        },
                        "value": "#dddddd",
                    },
                    "tooltip": [
                        {"field": "prototype", "type": "nominal"},
                        {"field": "other", "type": "nominal"},
                        {"field": "correlation", "type": "quantitative", "format": ".4f"},
                        {"field": "shared_parts", "type": "quantitative"},
                    ],
                },
                "height": 600,
            },
            width="stretch",
        )
    st.dataframe(correlation)
    _download(correlation, "correlation", measurement_id)
    with st.expander("Mean activations and shared-part counts"):
        st.dataframe(activations)
        _download(activations, "mean_activations", measurement_id)
        st.dataframe(shared)
        _download(shared, "shared_parts", measurement_id)
