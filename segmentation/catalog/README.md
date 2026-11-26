# Measurement catalog

## Registering checkpoints

Architecture schemas live in `segmentation/catalog/architectures.py`, without
importing model code or Torch. Add a versioned schema there for a new architecture;
retain existing versions so old registrations stay readable. Print the schemas with
`python -m segmentation.catalog.cli architectures`. Configs reject unknown fields.
`ProtoSeg` requires `stage`; `ScaleProtoSeg` also requires the explicit `grouped`
boolean. Optional backbone, prototype shape, scale/group counts describe known
configuration, not guessed checkpoint-path conventions.

Create a config JSON file, for example `{"stage": "push", "grouped": false}`, then:

```bash
python -m segmentation.catalog.cli register-model \
  --name scaleproto-pascal --architecture ScaleProtoSeg --dataset pascal \
  --config model-config.json --seed 0 --tags '{"experiment":"baseline"}' \
  --checkpoint /absolute/path/to/push_last.pth
python -m segmentation.catalog.cli list-models
```

Register the grouping checkpoint separately, with its own name and configuration.
Registration assigns a UUID and enforces unique names and file SHA-256 hashes
transactionally in SQLite, including concurrent registrations. Re-registering a
copied checkpoint is rejected. Names are lowercase letters/digits plus `._-`, must
start with a letter/digit, and cannot themselves be UUIDs. A training seed may be
unknown (`null`, when omitted); tags are universal string values.

Files are referenced, never moved or copied. After moving a file, use
`python -m segmentation.catalog.cli relocate-model MODEL /new/checkpoint.pth`;
its contents must match the registered hash. Both metric loaders verify contents.
The webapp's **Registered models** section lists and registers checkpoints.
Registration metadata and architecture-config fields are available as filters and
grouping dimensions, separately from metric parameters.


## Querying the measurement catalog

The metadata-only catalog CLI uses the same `MEASUREMENT_CATALOG_ROOT` environment
setting as the metric writers. Exported values take precedence over this checkout's
ignored `.env`; `.env_template` contains a placeholder to replace locally. The
Slurm wrapper uses the submit checkout from `SLURM_SUBMIT_DIR` (or `PWD`) unless
`SCALEPROTOSEG_ROOT` is explicitly set, and runs `METRIC_PYTHON` when configured
(otherwise `.venv/bin/python`).

List complete measurements without opening observation tables, checkpoints, or
model code:

```bash
.venv/bin/python -m segmentation.catalog.cli list \
  --metric consistency --dataset pascal --quantile 0.8 \
  --variant protocol.consistency.threshold=0.8
```

Running and failed measurements remain discoverable with `--all-status`; direct
non-complete status filters also require that opt-in.

Validate a record or all selected records:

```bash
.venv/bin/python -m segmentation.catalog.cli validate <measurement-id>
```

`validate` checks metadata, the observations artifact size and hash, and the
population counts against the rows.

## Browsing the measurement catalog

A Streamlit app (evaluation profile) inspects, filters, compares and
aggregates the catalog. Browsing the same `MEASUREMENT_CATALOG_ROOT` needs only
read access; launching jobs also needs write access. The existing training `.venv`
may use Python 3.9, which cannot install the Python 3.12 evaluation requirements.
Create a separate Python 3.12 environment for the browser:

```bash
python3.12 -m venv .venv-catalog
.venv-catalog/bin/python -m pip install -r requirements/requirements-catalog.txt
```

If Python 3.12 is not on `PATH`, use its full path. This lightweight environment
is for the browser only; submitted measurements require a separate full evaluation
environment (see `requirements/requirements-eval.txt`).
Start the app from the repository root and forward the port over SSH:

```bash
.venv/bin/python -m streamlit run segmentation/catalog/app.py \
  --server.headless true --server.port 8501
```

- **Runs**: one row per measurement; sidebar filters cover the inputs (model, stage,
  group checkpoint and every recorded `parameters.*` value), and *Extra fields* adds
  any other varying `result.json` field. Each row links to a measurement page with
  metadata, observations, hash validation and prototype part-activation analysis.
  The analysis averages each part over nonmissing observations (zeros count), then
  computes Pearson correlations on jointly observed parts. Options retain only
  consistent prototypes (any mean activation strictly above the threshold) and/or
  part IDs from the prototype class's taxonomy. Consistency filtering precedes the
  part filter. Stability measurements use clean `part_*` vectors.
  CSV part IDs are class-local, so analysis uses `(class, part ID)` coordinates:
  cross-class correlations, constant vectors and pairs with fewer than two shared
  parts are undefined, not zero. Existing CSVs cannot recover the source class of
  foreign part IDs included during measurement. The heatmap shows up to 200 chosen
  prototypes; CSV downloads retain the full matrix, mean vectors and shared-part counts.
  Select one or more measurements in **Runs**, or open a measurement's detail page,
  then use **Remove measurement from catalog** to archive the selected bundles. Running
  measurements cannot be archived. The bundle is moved to
  `<catalog-root>/archived_measurements/<archive-id>/<measurement-id>`;
  to restore it, move that directory back to `<catalog-root>/measurements/` only if
  no bundle with that ID exists. Checkpoints, logs and submission receipts are left
  untouched; an existing receipt still blocks resubmission.
- **Compare**: choose a boolean flag to find catalog entry pairs where every other
  input (including quantile) and selected extra field matches. The score has
  **Off**/**On** sub-columns containing lists of the original scores, preserving
  duplicate entries and missing scores without averaging. Only configurations
  present on both sides are shown, using the filtered runs.
- **Aggregate**: group by chosen inputs and average over the rest (e.g. leave
  `parameters.quantile` out to average across quantiles). `mean` is the unweighted
  mean of run scores; `pooled` is Σ numerator / Σ denominator. Groups always include
  `metric`. The pivot's last column averages per-column means and is blank when a
  column value is missing for that row.
- **Coverage**: construct an article-table plan by selecting a metric, registered
  checkpoints, the metric parameters to vary, candidate values, and display columns.
  Each registration supplies its model metadata and training dataset, so checkpoints
  are never crossed with unrelated datasets. An optional grouping model is selected
  per dataset and must match it. Only selected parameter values are crossed;
  unselected parameters use their metric defaults. The plan shows existing scores
  and statuses beside `Not measured` configurations, and can be downloaded as CSV.
  The planner works before any measurements exist.

In **Coverage**, select rows in the missing-configurations table, review the
generated commands, choose a full evaluation Python environment, and click
**Run selected missing measurements**. Each selection is submitted separately
through `scripts/run_metric.slurm`, pinned to the Slurm node hosting the app
(`--nodelist`, not user-editable), because checkout and dataset caches may be
node-local. The wrapper requests one GPU, eight CPUs, 32 GB and four hours.
Coverage submissions support consistency, stability, and imprecise activations.
`METRIC_PYTHON` supplies the interpreter default; an app-only environment cannot
run measurements. Jobs inherit the app's environment and use this checkout's
`.env` settings and catalog root.

Submission validates typed metric inputs, dataset-specific constraints and local
checkpoint registrations. The checkpoint is resolved from its registration; model
and dataset metadata are not user-entered dimensions.

Catalog coverage is rechecked before submission, including running and failed
entries. Job IDs and submission receipts are stored under
`<catalog-root>/submissions/`, with logs named `metric-<job-id>.out`. Receipts
prevent repeated submissions across reloads and app sessions; a queued job
appears in the measurement catalog only when the calculator creates its bundle.
To retry, first archive any matching measurement, select the missing configuration
in **Coverage**, and choose **Allow retry for selected submissions**. The app
rechecks the catalog and Slurm; if the job is terminal and not active in the queue, it
archives its receipt under `<catalog-root>/submissions/archived/` and permits a
new submission. Receipts with unknown job IDs, uncertain submission outcomes,
unavailable or ambiguous Slurm records, or active jobs require manual review.
Receipt changes do not cancel jobs. The catalog and its measurements directory
must be writable by the app/job user. Serve the app only to trusted operators:
the launch button submits jobs as the account running Streamlit.

