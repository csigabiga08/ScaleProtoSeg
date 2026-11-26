# Analysis: interpretability metrics

## Native consistency and stability measurements

Consistency, stability, and imprecise activations use the shared metric catalog
and runner. Consistency and stability use the part-specific evaluation loop;
imprecise activations measure full-image semantic activation:

### Imprecise activations

For prototype \(p\) assigned to class \(C_p\), the inside and outside means are
computed over valid pixels in each image. The image score is
\(\mathrm{IA}_{i,p}=o_{i,p}/(o_{i,p}+i_{i,p})\), and the prototype score is the
mean over images with a defined score. An empty region has mean zero; images
where both means are zero are excluded. The final measurement score is the mean
of defined prototype scores. Activations use continuous log-similarity maps
resized with nearest-neighbor interpolation; void labels are excluded, and
Pascal VOC background is a semantic class. Per-image activation sums, pixel
counts, and scores are retained in the observations artifact. Run it with a
registered model and dataset:

```bash
sbatch scripts/run_metric.slurm imprecise_activations \
  model=REGISTERED_MODEL data_type=pascal
```

Every invocation allocates a fresh UUID directory under the catalog root configured
by `MEASUREMENT_CATALOG_ROOT`:

```text
<catalog-root>/
  models.sqlite3
  measurements/<measurement-uuid>/
    result.json
    observations.csv
    derived/                       # complete consistency measurements
      part-summary-v1.csv
      part-summary-v1.meta.json
```

Set `MEASUREMENT_CATALOG_ROOT` in the job environment or in this checkout's ignored
`.env` file. The root must already exist and be writable; metrics do not fall back
to `RESULTS_DIR`, and there is no `--output-dir` or catalog-root command-line option.
Measurement inputs must reference registered models; checkpoint paths and training
stages are not metric overrides.
Catalog configuration uses `python-dotenv` quoting, comments, and interpolation;
an exported root takes precedence, and catalog lookup does not load other settings
into the environment. Metadata validation uses Pydantic, pinned in both dependency
profiles; filesystem integrity and scientific comparability are checked separately.

`result.json` records run status, checkpoint identities, the dataset, the expanded
metric protocol and parameters (quantile, seed and the metric flags), selected
prototype IDs, scores, the git commit and command, and the observations artifact
(path, size, SHA-256). It is first published as `running`, then atomically replaced
with a terminal `complete` or `failed` record. The original evidence is the two
top-level files; optional regenerable reports live under `derived/`.
Observations use stable sample/class/prototype/repetition keys; blank
consistency part values mean unavailable, while zero means an observed miss.
Stability stores clean and noisy part vectors rather than derived equality flags.
A sample is contributing if it appears in `observations.csv`; `population` only
counts candidates, contributing samples and observations. There
is no schema or protocol version: a changed metric definition must be added behind a
new flag. Repeated configurations are separate measurements.

Complete consistency runs automatically generate part-support counts from the
observations already in memory. Summary failure warns without changing a
successfully completed measurement. The CSV contains
`class_id,prototype_id,part_id,observations,hits`, including zero-support
coordinates for every observed prototype and declared part ID. A zero support
has an undefined mean, not a zero mean. Counts require at most one row per
sample/class/prototype; repetitions are rejected rather than counted as
independent images. Stability's missing-as-zero vectors are not used for these
support summaries.

The JSON sidecar records the source observation identity, interpretation
fingerprint, calculation/schema version, prototype population, declared analysis
part IDs, generation time, and summary CSV hash. The CSV is published atomically
before its metadata. Summaries never modify the original observations, scores,
or `result.json`. They are verified against their own CSV and the advertised
source/protocol identity when read; browsing does not rehash the original
observations. Use `validate` to recheck original evidence.

Backfill complete consistency measurements without inference (existing valid
summaries are left unchanged; missing, stale or corrupt summaries are rebuilt):

```bash
.venv-catalog/bin/python -m segmentation.catalog.cli backfill-part-summaries
.venv-catalog/bin/python -m segmentation.catalog.cli backfill-part-summaries --dataset pascal --quantile 0.8
.venv-catalog/bin/python -m segmentation.catalog.cli backfill-part-summaries <measurement-id>
```

Backfilling needs catalog write access and verifies original observations when
building a summary. Independent bundle errors are reported without stopping the
remaining batch; any error gives a nonzero exit status.

The webapp's **Show part-support diagnostics** option loads only these summaries
for filtered rows and includes diagnostics in the measurement-table CSV export.
Measurement pages show support curves, per-class and per-prototype evidence,
counts and provenance. Threshold changes reuse counts; support curves reuse each
prototype's maximum support among qualifying parts. A prototype qualifies when
some recorded valid positive part has mean **strictly above** the recorded
consistency threshold and at least the requested support. The denominator stays
fixed at all observed prototypes, including those with no supported association;
selected-but-unobserved prototypes are recorded separately and not silently added
to that denominator. Diagnostics do not replace the recorded score. Missing
summaries are shown as **Not calculated**, never zero, and browsing never writes
derived files. Per-image overlap analyses still require the original rows.

The fixed six-seed Pascal support-sensitivity study can be run without
inference once the matched cohort is complete (including Matplotlib):

```bash
.venv/bin/python -m segmentation.analysis.support_sensitivity \
  --output /path/to/new/pascal-support-sensitivity
```

This requires the complete 72-measurement consistency grid and its support
summaries. It verifies source observations, reconstructs all recorded scores,
and freezes the cohort and protocol evidence. Outputs include per-measurement
and per-prototype evidence, quantile-specific results, quantile averages within
each training seed, mean/sample SD across three seeds per architecture, and
paired O/F/R/M contrasts on native, common-prototype and common-row populations.
PDF/PNG curves accompany the tables. The output directory must not already
exist; neither prior analyses nor measurements are overwritten. Historical
scope assumptions and statistical limitations are recorded in the manifest.

The strict article cohort uses `pascalvoc_seed-42_run02`, whose training config
matches the reference and seed43 runs, instead of the resumed
`pascalvoc_seed-42` checkpoint used by the earlier Pascal exports. As of the
2026-11-20 reconciliation, run02 C/S measurements are unavailable. The older
Pascal ProtoSeg support/overlap aggregates and seed42 example are therefore
not eligible for the three-seed article tables. They remain preserved as
historical results. `--check-cohort` reports the missing cells; neither that
check nor table reconciliation launches measurements.

For the Cityscapes Original/Revised comparison, check readiness before
running the analysis:

```bash
.venv-catalog/bin/python -m segmentation.analysis.support_sensitivity --dataset cityscapes --check-cohort
.venv-catalog/bin/python -m segmentation.analysis.support_sensitivity --dataset cityscapes \
  --output /path/to/new/cityscapes-support-sensitivity
```

The check lists missing completed cells and exits nonzero until all 36 are
available. The cohort pins the three `4_16v2` ProtoSeg registrations and the
three ScaleProtoSeg registrations with their grouping checkpoints. It excludes
other configurations, the image-scope pilot, and intermediate annotation
settings. Confirm the training-seed provenance before interpreting the sample
SDs as seed variability. The Cityscapes run compares O directly with R,
matches observations by semantic label across imported and native records,
and produces tables without plots. The readiness check only reads metadata;
the full run verifies the original observations and requires valid derived
support summaries.

Profile-correlation versus positive-event overlap reuses the exact Revised
measurements from either completed support study, without inference:

```bash
.venv-catalog/bin/python -m segmentation.analysis.event_overlap --dataset pascal \
  --cohort-manifest /path/to/pascal-support-sensitivity/manifest.json \
  --output /path/to/new/pascal-profile-event-overlap
.venv-catalog/bin/python -m segmentation.analysis.event_overlap --dataset cityscapes \
  --cohort-manifest /path/to/cityscapes-support-sensitivity/manifest.json \
  --output /path/to/new/cityscapes-profile-event-overlap
```

The runner verifies frozen source identities and original observation hashes.
It compares unordered within-class pairs on unrestricted parts (at least two
jointly observed parts) and supported parts (at least five parts, each observed
in at least ten common images). Profile Pearson and mean absolute difference
use each prototype's own available-image means on the retained parts;
positive-event Jaccard intersects observed image/part coordinates and never
turns missing values into misses. Matched-image Pearson and availability overlap
are also recorded. Cityscapes' five-part subset contains car/truck/bus, excluding
four-part person/rider by design.

Outputs are tables only: per-pair evidence, per-class and pooled measurement
summaries, within-run quantile averages, mean/sample SD across three runs,
quantile-specific summaries, and deterministically selected illustrative pairs.
Primary results use all observed prototypes; both-consistent pairs are a
secondary population. Rates include explicit eligibility/undefined counts.
High Pearson means `rho >= 0.9`, low overlap `J < 0.5`, and high overlap
`J >= 0.9`. Conditional rates are calculated before averaging; empty populations
stay undefined, not zero, and incomplete quantile/run estimates are not silently
averaged. Counts and quantiles are not independent training replicates, SDs are
not confidence intervals, and no significance or prototype-deletion claim is
made. The output directory must be new; prior results and measurements remain
unchanged.

To reconcile the article's Original/Revised/Merged C/S reference tables and
per-checkpoint imprecise-activation table from existing evidence:

```bash
.venv-catalog/bin/python -m segmentation.analysis.reference_tables \
  --pascal-manifest /path/to/pascal-support-sensitivity/manifest.json \
  --cityscapes-manifest /path/to/cityscapes-support-sensitivity/manifest.json \
  --checkpoint-root /data/shared --output /path/to/new/article-reference-tables
```

The runner pins checkpoint/grouping identities from the frozen studies,
excludes the resumed Pascal seed42 model, checks the matching ProtoSeg configs
and distinct seeds, verifies original observations, and reconstructs C/S.
It averages the three quantiles within seed before calculating the mean/sample
SD across exactly three seeds. Missing measurements produce explicit empty
cells and suppress the entire three-seed aggregate, not a two-seed replacement.
Imprecise-activation ratios are independently reconstructed from recorded
inside/outside means; their SD is across prototypes, not seeds. Outputs freeze
sources and missing cells without modifying measurements or earlier studies.
Historical activation descriptions record checkpoint paths, not their original
hashes; the audit records current checkpoint identities without asserting a
retrospective hash guarantee. ScaleProtoSeg uses the existing released seed
family; its complete training configs are not included with those measurements.

Both calculators accept these part and centroid options:

| Desired configuration | Option | Folder label |
|---|---|---|
| All decoded parts | Default | `allparts` |
| Official Cityscapes semantic/part pairs | `official_parts_only=true` | `filteredparts` |
| Pascal parts matching the VOC semantic class | `require_semantic_match=true` | `filteredparts` |
| Foreground and background centroids | Default | `background` |
| Foreground part-component centroids only | `exclude_background_centroid=true` | `foreground` |

Both filtered modes require the native `annotations_SIDS` cache. Filtering is applied
in memory to both clean and noisy images and does not modify cached labels. The exact
filter is recorded in the description; Pascal semantic matching does not apply the
Cityscapes taxonomy.

Run either calculator from the repository root as a Python module. Supply config
fields as bare Hydra `key=value` overrides (not positional arguments or `--` flags).
The following Pascal examples select exact evaluated checkpoints and a prototype
filter checkpoint:


## Pascal left/right part merging

Set `merge_left_right_parts=true` for either metric to treat Pascal's 54
class-specific left/right pairs as one candidate part. Centroids remain computed
from the original masks; their binary hits are ORed per image, without modifying
cached labels. Consistency ignores a pair when neither side is annotated. Stability
merges the clean and noisy hits independently before comparing their vectors.
Unpaired parts are unchanged. The protocol records the canonical IDs and source
labels for all 54 pairs.

```bash
python -m segmentation.analysis.metrics.consistency \
  model=scaleproto-pascal grouping_model=scaleproto-pascal-grouped data_type=pascal \
  quantile=0.8 threshold=0.8 require_semantic_match=true \
  exclude_background_centroid=true merge_left_right_parts=true

python -m segmentation.analysis.metrics.stability \
  model=scaleproto-pascal grouping_model=scaleproto-pascal-grouped data_type=pascal \
  quantile=0.8 std_dev=0.2 seed=0 require_semantic_match=true \
  exclude_background_centroid=true merge_left_right_parts=true
```

The common required overrides are `model` (UUID or unique name) and `data_type`.
`grouping_model` optionally references another registered model for prototype
selection; it may reference the main model itself, in which case it is loaded once.
If omitted for consistency, the main model's own group projection is used when
present, and its registration is recorded in both roles. Stability requires an
explicit grouping reference to enable group-based filtering.
Datasets must match, and a grouping model must actually have a group projection.
Measurements snapshot both registrations and their checkpoint identities.
The metric-specific
protocol values are also Hydra overrides: consistency accepts `quantile` and
`threshold`, while stability accepts `quantile`, `std_dev`, and `seed` (default 0).
Part selection and centroid behavior are controlled by `official_parts_only`,
`require_semantic_match`, `exclude_background_centroid`, and
`merge_left_right_parts`; set the applicable boolean to `true` as shown above.
These options are recorded in the result protocol.
Pascal semantic matching applies only to Pascal, and official-parts filtering applies
only to Cityscapes. Output directories are allocated automatically from
`MEASUREMENT_CATALOG_ROOT`.

The Slurm wrapper requests one GPU on TANK, eight CPUs, and 32 GB of memory.
Override the node with `sbatch --nodelist=maxitank ...` when using that node's checkout
and preprocessed data. `/data_ssd` storage can differ between nodes: the updated code
and configured caches must be available on the node you select. The result record's
hostname and annotation paths identify which environment was used.

