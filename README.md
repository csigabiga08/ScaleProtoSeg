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

# Evaluation for consistency metric
python -m segmentation.analysis.metrics.consistency <prototype_training_run_name> <training_stage> <data_type> --quantile <quantile> --threshold <threshold> --group-name <group_training_run_name>
```

### ProtoSeg

```
# Evaluation on validation set for Pascal or Cityscapes
python -m segmentation.eval_valid <your_training_run_name> <training_stage> <flag_pascal>

# Evaluation on validation set for ADE, COCO, or EM
python -m segmentation.eval_valid_multiscale <your_training_run_name> <training_stage> <batch_size> <data_type>

# Generating predictions on cityscapes or pascal test set
python -m segmentation.eval_test  <your_training_run_name> <training_stage> <batch_size> <flag_pascal>

# Evaluation for consistency metric
python -m segmentation.analysis.metrics.consistency <prototype_training_run_name> <training_stage> <data_type> --quantile <quantile> --threshold <threshold>
```

### Native consistency and stability measurements

Every invocation allocates a fresh UUID directory under the catalog root configured
by `MEASUREMENT_CATALOG_ROOT`:

```text
<catalog-root>/
  measurements/<measurement-uuid>/
    result.json
    observations.csv
    samples.csv
```

Set `MEASUREMENT_CATALOG_ROOT` in the job environment or in this checkout's ignored
`.env` file. The root must already exist and be writable; metrics do not fall back
to `RESULTS_DIR`, and there is no `--output-dir` or catalog-root command-line option.
`RESULTS_DIR` still locates stage-selected checkpoints and relative group names.
Catalog configuration uses `python-dotenv` quoting, comments, and interpolation;
an exported root takes precedence, and catalog lookup does not load other settings
into the environment. Metadata validation uses Pydantic, pinned in both dependency
profiles; filesystem integrity and scientific comparability are checked separately.

`result.json` records schema version, run status, checkpoint identities and hashes,
dataset identity coverage, the expanded metric protocol, selected and contributing
prototype IDs, population counts, scores, source/package provenance and artifact
hashes. It is first published as `running`, then atomically replaced with a terminal
`complete` or `failed` record. A native completed bundle contains exactly the three
files above. Observations use stable sample/class/prototype/repetition keys; blank
consistency part values mean unavailable, while zero means an observed miss.
Stability stores clean and noisy part vectors rather than derived equality flags.
Candidate outcome rows distinguish excluded, empty, contributing, failed and
not-attempted samples. Repeated configurations are separate measurements.

Both calculators accept these part and centroid options:

| Desired configuration | Option | Folder label |
|---|---|---|
| All decoded parts | No part-filter flag | `allparts` |
| Official Cityscapes semantic/part pairs | `--official-parts-only` | `filteredparts` |
| Pascal parts matching the VOC semantic class | `--require-semantic-match` | `filteredparts` |
| Foreground and background centroids | Default | `background` |
| Foreground part-component centroids only | `--exclude-background-centroid` | `foreground` |

Both filtered modes require the native `annotations_SIDS` cache. Filtering is applied
in memory to both clean and noisy images and does not modify cached labels. The exact
filter is recorded in the description; Pascal semantic matching does not apply the
Cityscapes taxonomy.

Select an exact evaluated checkpoint with `--checkpoint` for consistency or
`--checkpoint-path` for stability. Select a prototype-filter checkpoint using
`--group-checkpoint /path/to/group.pth` on either calculator, or retain the existing
`--group-name /path/to/group_run` option. Stability also accepts `--seed` (default 0).
Output directories are allocated automatically from `MEASUREMENT_CATALOG_ROOT`.

After configuring a writable catalog root, submit a single-quantile Pascal stability job:

```bash
sbatch --output="$PWD/stability-%j.out" scripts/run_metric.slurm \
  stability scaleproto_pascal push pascal \
  --checkpoint-path /data/shared/scaleprotoseg/checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  --group-name /data/shared/scaleprotoseg/checkpoints/group_scaleproto_pascal \
  --quantile 0.8 --std-dev 0.2
```

To use Pascal semantic matching and foreground centroids, append
`--require-semantic-match --exclude-background-centroid`. To run consistency, use:

```bash
sbatch --output="$PWD/consistency-%j.out" scripts/run_metric.slurm \
  consistency scaleproto_pascal push pascal \
  --checkpoint /data/shared/scaleprotoseg/checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  --group-name /data/shared/scaleprotoseg/checkpoints/group_scaleproto_pascal \
  --quantile 0.8 --threshold 0.8
```

The Slurm wrapper requests one GPU on TANK, eight CPUs, and 32 GB of memory.
Override the node with `sbatch --nodelist=maxitank ...` when using that node's checkout
and preprocessed data. `/data_ssd` storage can differ between nodes: the updated code
and configured caches must be available on the node you select. The result record's
hostname and annotation paths identify which environment was used.

### Querying and summarizing measurements

The metadata-only catalog CLI uses the same `MEASUREMENT_CATALOG_ROOT` environment
setting as the metric writers. Exported values take precedence over this checkout's
ignored `.env`; `.env_template` contains a placeholder to replace locally. The
Slurm wrapper uses the submit checkout from `SLURM_SUBMIT_DIR` (or `PWD`) unless
`SCALEPROTOSEG_ROOT` is explicitly set, and runs `METRIC_PYTHON` when configured
(otherwise `.venv/bin/python`).

List and filter complete measurements without opening observation tables, checkpoints,
or model code:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli list \
  --metric consistency --dataset pascal --quantile 0.8 \
  --variant protocol.consistency.threshold=0.8
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli filter \
  --all-status --status failed --provenance-gap 'part annotation'
```

Running and failed measurements remain discoverable with `--all-status`; direct
non-complete status filters also require that opt-in. Summaries default to complete
records; finalized failed records may be inspected with `summarize --all-status`,
but running records cannot be summarized. Unknown provenance and extension fields are
retained in metadata and exported CSVs. Export a flat comparison table with the
complete nested record preserved in `metadata_json`:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli export \
  --metric consistency --output /path/to/analysis/consistency.csv
```

Flat CSV exports refuse an existing destination file. Summary and plot exports
require a new output directory; choose a different path for each run rather than
reusing a directory, which avoids overwriting symlink targets or mixing stale tables.

Validate a selected record, recompute its exact metric aggregates, or save derived
summary tables outside the canonical catalog bundle:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli validate <measurement-id>
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli summarize <measurement-id> \
  --output-dir /path/to/analysis/<measurement-id>
```

Consistency summaries use the recorded protocol version and aggregation rule:
per-part means omit blank/NaN unavailable values, while observed zero remains a miss;
consistency uses strict `mean > threshold` and weights class/prototype rows as the
recorded evaluator does. Stability summaries compare paired clean/noisy part vectors
and do not substitute missing values with zero. Unknown aggregation or comparison
protocols are rejected rather than reinterpreted. The known historical evaluator
profile is identified explicitly as `scaleprotoseg-v1`; its recorded aggregation
descriptor must still match. Legacy records without complete
candidate membership remain visible with explicit diagnostics; summaries use only
available observation artifacts and do not invent excluded samples.

### Catalog visualizations

Plot explicit consistency measurement IDs; quantiles are not auto-selected and
repeated attempts remain separate:

```bash
.venv/bin/python -m segmentation.analysis.metrics.catalog_cli plot \
  <measurement-id-q70> <measurement-id-q80> \
  --output-dir /path/to/analysis/consistency-figures
```

The command writes four conventional views (prototype/part heatmaps, per-class
consistency, prototype maximum-part scores across quantiles, and usable-image
support), individual PNG/PDF files, a combined PDF, and exact-value CSV tables.
Comparisons require matching known checkpoint, dataset, and non-quantile protocol
identities. Support sample sets are checked across selections; when they differ, the
support view and CSV retain separate values per selected quantile/attempt. Derived
exports are refused anywhere under canonical `measurements/` bundles.

The view choices are conceptually informed by ReProSeg's consistency visualization
methods. Its checked-out source had no license file, so these catalog plots are an
independent implementation and do not import or copy that source.

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
