# ⚖️ ScaleProtoSeg: WACV 2025
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14946772.svg)](https://zenodo.org/record/14946772)

**Prototype Learning**:
![stage-1](images/stage-1-paper-group.drawio.png)
**Grouping Mechanism**
![stage-2](images/stage-2-paper-group.drawio.png)

>  📑 Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation <br>
> ✍️ [Hugo Porta](https://scholar.google.com/citations?user=IQMApuoAAAAJ&hl=fr), [Emanuele Dalsasso](https://scholar.google.com/citations?user=4bfkX98AAAAJ&hl=it), [Diego Marcos](https://scholar.google.ch/citations?user=IUqydU0AAAAJ&hl=en) and [Devis Tuia](https://scholar.google.com/citations?user=p3iJiLIAAAAJ&hl=fr) <br>
> *[WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Porta_Multi-Scale_Grouped_Prototypes_for_Interpretable_Semantic_Segmentation_WACV_2025_paper.pdf)*

This repository is a fork of the code for [ProtoSeg](https://github.com/gmum/proto-segmentation).

## 🛠️ Installation

Install python libraries and local code:

```
pip install -r requirements/requirements.txt
```

The PyPI release `panoptic-parts 2.0rc6` incorrectly interprets Python 3.10 and
newer as Python 3.1 and raises `At least Python 3.7 is needed`. Repair its version
check using the Python interpreter from the environment you installed into:

```bash
python scripts/fix_panoptic_parts.py
python -c "import panoptic_parts"
```

The script backs up the original file in the environment's `compat-backups`
directory and can be run again safely. Repeat it after reinstalling this package.

For using Deeplab it might be necessary to run:

```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git --no-build-isolation
```

If an older PyTorch installation reports `libtorch_cpu.so: cannot enable executable stack as shared object requires`
on a newer glibc system, back up the library and clear its executable-stack flag with
[patchelf](https://github.com/NixOS/patchelf). For the project's Python 3.9 `.venv`, run from the repository root:

```bash
.venv/bin/python -m pip install patchelf
mkdir -p .venv/compat-backups
cp --no-clobber .venv/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so \
  .venv/compat-backups/libtorch_cpu.so.before-execstack-fix
.venv/bin/patchelf --clear-execstack .venv/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
env -u GLIBC_TUNABLES .venv/bin/python -c "import torch; print(torch.__version__)"
```

This repairs the installed library without changing the installed PyTorch version or requiring
`GLIBC_TUNABLES=glibc.rtld.execstack=2`. Reinstalling PyTorch can replace the repaired library;
repeat the repair if the error returns. The backup can be copied over `libtorch_cpu.so` to undo
the repair, provided it belongs to the same PyTorch installation.

⚠️ The file `.env_template` is here to provide an example of environment file necessary to run this repository.

## 💽 Data

### Downloading Data

- <u>Cityscapes:</u> Download at this [link](https://www.cityscapes-dataset.com/downloads) the images: **leftImgbit_trainvaltest.zip**, the labels: **gtFine_trainvaltest.zip**, and the parts annotations: **gtFinePanopticParts_trainval.zip**.

- <u>Pascal:</u> Follow the instructions detailed at this [link](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md) for the images and labels.
    - Pascal data: VOC2012 can be downloaded [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
    - Augmented training data: **SegmentationClassAug** can be downloaded at this [link](https://www.dropbox.com/scl/fi/xccys1fus0utdioi7nj4d/SegmentationClassAug.zip?rlkey=0wl8iz6sc40b3qf6nidun4rez&e=1&dl=0).
    - The splits: **SegmentationAug** can be found [here](https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip).
    - The data needs to be extended with the test set available [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar) when registered to the benchmark, it is just necessary to copy the test images from **JPEGImages** in the corresponding folders for train, trainaug, and val.
    - The parts annotations can be found [here](https://onedrive.live.com/?authkey=%21AHXfIdeL14IeVhI&id=F5D23EA0E1A6E588%21108&cid=F5D23EA0E1A6E588&parId=root&parQt=sharedby&o=OneUp) and the `parts.yaml` should be copied from this [file](https://github.com/pmeletis/panoptic_parts/blob/45f7121cf2d4bfe65e6dc27b3f006d1104841ae1/panoptic_parts/specs/dataset_specs/ppp_datasetspec.yaml). All those data should be placed in the same folder as the rest of the Pascal benchmark. Rename the folders in **labels** to **train** and **val**.

- <u>ADE20K:</u> The scene parsing dataset can be downloaded [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).

- <u>EM:</u> The dataset for the segmentation of neuronal structures in EM stacks challenge - ISBI 2012 can be found [here](https://downloads.imagej.net/ISBI-2012-challenge.zip)

- <u>COCO-Stuff:</u> The COCO 2017 train, and validation images can be found at this [link]([https://github.com/nightrome/cocostuff?tab=readme-ov-file#downloads]) with the Stuff+thing annotations. In the target COCO-Stuff folder move the images from **train2017.zip** and **val2017.zip** folders to **images/train** and **images/val**. Similarly, move the annotations from **stuffthingmaps_trainval2017.zip** to **annotations_png/train** and **annotations_png/val**.

### Preprocessing Data

#### Cityscapes

```
# replace <N> with the number of parallel jobs
python -m segmentation.data.preprocess_cityscapes <N> # for cityscapes
python -m segmentation.data.img_to_numpy --margin-size 0 --data-type cityscapes
python -m segmentation.data.preprocess_part_cityscapes preprocess-cityscapes <N> # for cityscapes
```

#### Pascal

```
# replace <N> with the number of parallel jobs
python -m segmentation.data.preprocess_pascal <N> # for pascal
python -m segmentation.data.img_to_numpy --margin-size 0 --data-type pascal
python -m segmentation.data.preprocess_part_pascal <N> # for pascal
```

#### ADE20K

```
# replace <N> with the number of parallel jobs
python -m segmentation.data.preprocess_ade <N> # for ade
python -m segmentation.data.img_to_numpy --margin-size 0 --data-type ade
```

#### EM

In EM preprocessing you can specify a seed to create different train/val splits from the original train data. In the paper we tested our models against two different random splits.

```
# replace <N> with the number of parallel jobs
python -m segmentation.data.preprocess_em <N> --seed 42 # for EM
python -m segmentation.data.img_to_numpy --margin-size 0 --data-type em
```

#### COCO-Stuff

```
# replace <N> with the number of parallel jobs
python -m segmentation.data.preprocess_coco <N> # for COCO
python -m segmentation.data.img_to_numpy --margin-size 0 --data-type coco
```

## 💪 Training

All the checkpoints for ScaleProtoSeg can be downloaded at the following [Zenodo Link](https://zenodo.org/record/14946772).

### ScaleProtoSeg
```
# Example for cityscapes: Prototype Learning
python -m segmentation.train_wandb_multiscale scaleproto_cityscapes <your_training_run_name>

# Example for cityscapes: Grouping Mechanism
python -m segmentation.finetune_wandb_group group_scaleproto_cityscapes <your_training_run_name>
```

In the folder **segmentation/configs** all the config files follow a similar naming pattern than the one in the example above.

You need to specify for the **grouping** the path of the checkpoints from the first phase: `train.start_checkpoint` parameter.

### ProtoSeg

```
# Example for cityscapes: Prototype Learning
python -m segmentation.train_wandb baseline_cityscapes <your_training_run_name>

# Example for cityscapes: Pruning
python -m segmentation.run_pruning baseline_cityscapes <your_training_run_name>

# Example for cityscapes: Finetuning
python -m segmentation.train_wandb baseline_cityscapes <your_training_run_name> --pruned
```

Please look at the file `train_wandb.py` and `train_wandb_multiscale.py` for manual modifications when training on EM.

## 🧪 Evaluation

### ScaleProtoSeg

```
# Setting to zero the quasi non-used prototypes
python -m segmentation.analysis.threshold_save <your_training_run_name> <training_stage> <threshold>

# Evaluation on validation set
python -m segmentation.eval_valid_multiscale <your_training_run_name> <training_stage> <batch_size> <data_type>

# Generating predictions on cityscapes or pascal test set
python -m segmentation.eval_test  <your_training_run_name> <training_stage> <batch_size> <flag_pascal>

# See "Native consistency and stability measurements" below
```

### ProtoSeg

```
# Evaluation on validation set for Pascal or Cityscapes
python -m segmentation.eval_valid <your_training_run_name> <training_stage> <flag_pascal>

# Evaluation on validation set for ADE, COCO, or EM
python -m segmentation.eval_valid_multiscale <your_training_run_name> <training_stage> <batch_size> <data_type>

# Generating predictions on cityscapes or pascal test set
python -m segmentation.eval_test  <your_training_run_name> <training_stage> <batch_size> <flag_pascal>

# See "Native consistency and stability measurements" below
```

### Native consistency and stability measurements

Every invocation allocates a fresh UUID directory under the catalog root configured
by `MEASUREMENT_CATALOG_ROOT`:

```text
<catalog-root>/
  measurements/<measurement-uuid>/
    result.json
    observations.csv
```

Set `MEASUREMENT_CATALOG_ROOT` in the job environment or in this checkout's ignored
`.env` file. The root must already exist and be writable; metrics do not fall back
to `RESULTS_DIR`, and there is no `--output-dir` or catalog-root command-line option.
`RESULTS_DIR` still locates stage-selected checkpoints and relative group names.
Catalog configuration uses `python-dotenv` quoting, comments, and interpolation;
an exported root takes precedence, and catalog lookup does not load other settings
into the environment. Metadata validation uses Pydantic, pinned in both dependency
profiles; filesystem integrity and scientific comparability are checked separately.

`result.json` records run status, checkpoint identities, the dataset, the expanded
metric protocol and parameters (quantile, seed and the metric flags), selected
prototype IDs, scores, the git commit and command, and the observations artifact
(path, size, SHA-256). It is first published as `running`, then atomically replaced
with a terminal `complete` or `failed` record. A bundle contains exactly the two
files above. Observations use stable sample/class/prototype/repetition keys; blank
consistency part values mean unavailable, while zero means an observed miss.
Stability stores clean and noisy part vectors rather than derived equality flags.
A sample is contributing if it appears in `observations.csv`; `population` only
counts candidates, contributing samples and observations. There
is no schema or protocol version: a changed metric definition must be added behind a
new flag. Repeated configurations are separate measurements.

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

### Pascal left/right part merging

Set `merge_left_right_parts=true` for either metric to treat Pascal's 54
class-specific left/right pairs as one candidate part. Centroids remain computed
from the original masks; their binary hits are ORed per image, without modifying
cached labels. Consistency ignores a pair when neither side is annotated. Stability
merges the clean and noisy hits independently before comparing their vectors.
Unpaired parts are unchanged. The protocol records the canonical IDs and source
labels for all 54 pairs.

```bash
python -m segmentation.analysis.metrics.consistency \
  model_name=scaleproto_pascal training_phase=push data_type=pascal \
  checkpoint=/data/shared/scaleprotoseg/checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  group_checkpoint=/data/shared/scaleprotoseg/checkpoints/group_scaleproto_pascal/checkpoints/th-0.05-nopush-group_last.pth \
  quantile=0.8 threshold=0.8 require_semantic_match=true \
  exclude_background_centroid=true merge_left_right_parts=true

python -m segmentation.analysis.metrics.stability \
  model_name=scaleproto_pascal training_phase=push data_type=pascal \
  checkpoint_path=/data/shared/scaleprotoseg/checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  group_checkpoint=/data/shared/scaleprotoseg/checkpoints/group_scaleproto_pascal/checkpoints/th-0.05-nopush-group_last.pth \
  quantile=0.8 std_dev=0.2 seed=0 require_semantic_match=true \
  exclude_background_centroid=true merge_left_right_parts=true
```

The common required overrides are `model_name`, `training_phase`, and `data_type`.
Without an explicit checkpoint override, each calculator selects the checkpoint for
that model and phase under `RESULTS_DIR`. Consistency uses `checkpoint=...` to select
an exact evaluated checkpoint; stability uses `checkpoint_path=...`. Either accepts
`group_checkpoint=...` for an exact prototype-filter checkpoint or `group_name=...`
to resolve the existing group checkpoint under `RESULTS_DIR`. The metric-specific
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

### Querying the measurement catalog

The metadata-only catalog CLI uses the same `MEASUREMENT_CATALOG_ROOT` environment
setting as the metric writers. Exported values take precedence over this checkout's
ignored `.env`; `.env_template` contains a placeholder to replace locally. The
Slurm wrapper uses the submit checkout from `SLURM_SUBMIT_DIR` (or `PWD`) unless
`SCALEPROTOSEG_ROOT` is explicitly set, and runs `METRIC_PYTHON` when configured
(otherwise `.venv/bin/python`).

List complete measurements without opening observation tables, checkpoints, or
model code:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli list \
  --metric consistency --dataset pascal --quantile 0.8 \
  --variant protocol.consistency.threshold=0.8
```

Running and failed measurements remain discoverable with `--all-status`; direct
non-complete status filters also require that opt-in.

Validate a record or all selected records:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli validate <measurement-id>
```

`validate` checks metadata, the observations artifact size and hash, and the
population counts against the rows.

### Browsing the measurement catalog

A read-only Streamlit app (evaluation profile) inspects, filters, compares and
aggregates the catalog. It reads the same `MEASUREMENT_CATALOG_ROOT` but needs only
read access. Start it from the repository root and forward the port over SSH:

```bash
.venv/bin/python -m streamlit run segmentation/analysis/metrics/catalog_app.py \
  --server.headless true --server.port 8501
```

- **Runs**: one row per measurement; sidebar filters cover the inputs (model, stage,
  group checkpoint and every recorded `parameters.*` value), and *Extra fields* adds
  any other varying `result.json` field. Select one row for its metadata, observations
  and hash validation.
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
- **Coverage**: choose one or more column dimensions and fix other inputs with
  this view's filters. Every remaining input automatically becomes a row dimension.
  Select candidate values or type additional values to generate the full Cartesian
  grid, including entirely unmeasured rows. Cells retain individual scores and run
  statuses; `null` is a missing score in an existing run, while `Not measured`
  means no entry exists. This view includes all statuses and ignores sidebar
  filters, but includes selected Extra fields. Download the matrix or a flat list
  of missing configurations as CSV. Fix inputs or narrow values if the grid exceeds
  100,000 cells; incompatible combinations are not automatically excluded.

## 🥇 Performance

On the validation sets, we report the results over 3 runs, while for the test sets, the results are based on [ProtoSeg](https://arxiv.org/abs/2301.12276) and our best ScaleProtoSeg validation run.

| Method                | Cityscapes (val) | Cityscapes (test) | Pascal (val) | Pascal (test) | ADE20K (val) |
|----------------------|----------------|----------------|-------------|--------------|--------------|
| DeepLabv2 [[Chen et al. 2017]](https://arxiv.org/abs/1606.00915) | 71.40 | 70.40 | 77.69 | 79.70 | 34.00 |
| ProtoSeg [[Sacha et al. 2023]](https://arxiv.org/abs/2301.12276) | 67.54 ± 0.22 | 67.04 | **71.98 ± 0.11** | **72.92** | 29.67 ± 0.23 |
| ScaleProtoSeg        | **68.97 ± 0.25** | **68.52** | 71.80 ± 0.38 | 72.35 | **34.18 ± 0.18** |

## Analysis

In the folder `segmentation/analysis` numerous scripts are available to investigate the interpretability of our model via different plotting strategy.

**Prototype Maps**:
![maps](images/samples-interpretability-maps.png)
**Grouping Mechanism**
![grouping](images/grouping-example.png)

## 🖋️ Citation
```
@InProceedings{Porta_2025_WACV,
    author    = {Porta, Hugo and Dalsasso, Emanuele and Marcos, Diego and Tuia, Devis},
    title     = {Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2869-2880}
}
```
