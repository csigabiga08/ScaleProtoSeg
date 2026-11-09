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

### Activation quantile scope

Both consistency and stability accept `--quantile-scope class|image`:

- `--quantile-scope class` (default) computes each prototype's quantile using only
  its ground-truth semantic-class pixels and activates only pixels in that class.
- `--quantile-scope image` computes each prototype's quantile using **all pixels in
  its upsampled activation map** and permits active pixels anywhere in the image.
  There is no semantic mask on the resulting activation area.

For both modes, each prototype and image has its own threshold, computed with
linear interpolation; pixels must be **strictly greater** than the threshold.
Ties are inactive. Part filtering, centroid selection, and left/right merging
remain separate options. Stability computes a threshold independently for clean
and noisy activations using the same selected scope.

Append `--quantile-scope image` to an existing Python or `scripts/run_metric.slurm`
command to use the whole-image version. Whole-image runs add `qscope_image` to
all result and description filenames; class-mode filenames retain their existing
names. JSON descriptions record `parameters.quantile_scope`. Both modes use the
same configuration folder unless `--output-dir` is supplied.

### Consistency and stability result folders

Both calculators save into one of four folders in the evaluated checkpoint's model
directory. For `<model>/checkpoints/push_last.pth`, these folders are directly under
`<model>/`; if the checkpoint is stored directly in a model directory, the folders
are created there. Both metrics share the selected folder:

```text
<model>/
  checkpoints/
    push_last.pth
  metrics-allparts-background/
  metrics-allparts-foreground/
  metrics-filteredparts-background/
  metrics-filteredparts-foreground/
```

Only folders for configurations that you actually run are created. Folder names
contain only part selection and centroid selection. `background` means foreground
part centroids **plus** OpenCV's background centroid. `foreground` means only actual
part-component centroids. Existing results in older directories are left in place.

Quantiles and thresholds are included in result filenames. Stability filenames
include the noise standard deviation and seed. Filenames also identify the evaluated
checkpoint and grouping filter, so different checkpoint/filter configurations can
coexist in the same folder. For example, an ungrouped run writes:

```text
metrics-allparts-background/
  consistency_score_th_0.8_qt_0.8_checkpoint_push_last_group_ungrouped.txt
  part_presence_th_0.8_qt_0.8_checkpoint_push_last_group_ungrouped.csv
  part_presence_mean_th_0.8_qt_0.8_checkpoint_push_last_group_ungrouped.csv
  description_consistency_th_0.8_qt_0.8_checkpoint_push_last_group_ungrouped.json
  stability_score_std_0.2_qt_0.8_seed_0_checkpoint_push_last_group_ungrouped.txt
  tot_presence_stab_std_0.2_qt_0.8_seed_0_checkpoint_push_last_group_ungrouped.csv
  description_stability_std_0.2_qt_0.8_seed_0_checkpoint_push_last_group_ungrouped.json
```

Grouped runs use `group_grouped-<identity>` instead of `group_ungrouped` in their
filenames. The identity is computed from the resolved grouping checkpoint path;
its full path is recorded in the description.

Each run has its own `description_<metric>_<parameters>.json`, which records the
exact checkpoint and grouping paths, filtering flags, quantile, threshold/noise/seed,
Python and package versions, hostname and Slurm job ID, annotation location,
image-order digest, and completion status. Completed runs also record the score
and the number of evaluated images and observations. Descriptions for consistency,
stability, and different quantiles coexist. Repeating an identical configuration
uses the same filenames; use `--output-dir` with a separate folder to retain repeats.
The scripts print the result directory in their log.

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

### Pascal consistency with left/right parts merged

Add `--merge-left-right-parts` to the **consistency** command for Pascal VOC.
It is not a Cityscapes option and does not change stability evaluation. The default
continues to treat left/right labels separately.

This pools 54 left/right pairs across 11 Pascal object classes, using the normalized
part IDs from `parts.yaml`. For example, a person's left and right eyes become one
candidate part; front/back and upper/lower distinctions remain separate. Unpaired
parts are unchanged. The cached labels and masks are never modified.

Connected-component centroids are still computed separately for each original part
label. For each prototype/image, a paired part is hit if **any** centroid of either
side is inside that prototype's activation map. The side hits are ORed **before**
computing the across-image mean. An image counts toward that pair's average when
at least one side is annotated; when neither side is annotated the observation is
NaN and is ignored. The existing centroid option still applies independently to
each original mask: use `--exclude-background-centroid` for actual part-component
centroids only.

The consistency decision otherwise stays the same: a prototype is consistent if
at least one candidate part's mean hit frequency is **strictly greater than**
`--threshold`. The score is the proportion of consistent prototypes. This measures
consistency with a bilateral part category, rather than a particular side. Because
the paired category uses the union of the sides' annotated images, the final score
is not guaranteed to increase.

For example, if a prototype hits only the left eye in image A and only the right
eye in image B, with both eyes annotated in each image, the old frequencies are
0.5 and 0.5. The merged eye frequency is 1.0, exceeding the default 0.8 threshold.

```text
for each image, object class, prototype:
    compute centroids separately for every ORIGINAL part label
    compute binary centroid hits using the prototype activation map
    for each left/right pair:
        if either side is annotated:
            hit[pair] = hit[left] OR hit[right]
        else:
            hit[pair] = NaN
    keep unpaired part hits unchanged

for each prototype:
    frequency[part] = mean of non-NaN hits across images
    consistent = any(frequency[part] > consistency_threshold)
score = mean(consistent across prototypes)
```

The merged candidate uses the left member's `part_<id>` CSV column; the right
member's column becomes NaN to avoid counting the pair twice. The description JSON
records the option and the complete class-specific canonical-ID/member-label map.
The existing four output folder names are unchanged. Merged-run result and
description filenames include `lrmerged` so they coexist with separate-side runs.

```bash
sbatch --output="$PWD/consistency-lrmerged-%j.out" scripts/run_metric.slurm \
  consistency pascalvoc push_best pascal \
  --checkpoint /data/shared/protoseg/checkpoints/pascalvoc/checkpoints/push_best.pth \
  --require-semantic-match --exclude-background-centroid \
  --merge-left-right-parts --quantile 0.8 --threshold 0.8
```

Select an exact evaluated checkpoint with `--checkpoint` for consistency or
`--checkpoint-path` for stability. Select a prototype-filter checkpoint using
`--group-checkpoint /path/to/group.pth` on either calculator, or retain the existing
`--group-name /path/to/group_run` option. Stability also accepts `--seed` (default 0).
Use `--output-dir /exact/result/folder` to override the default model-level folder;
parameterized result files and descriptions are still written there. `RESULTS_DIR`
continues to locate stage-selected checkpoints and relative group names; it no longer
selects the default result directory.

From `~/ScaleProtoSeg-ssd`, submit a single-quantile Pascal stability job:

```bash
sbatch --output="$PWD/stability-%j.out" scripts/run_metric.slurm \
  stability scaleproto_pascal push pascal \
  --checkpoint-path /data/shared/protoseg/scaleproto_checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  --group-name /data/shared/protoseg/scaleproto_checkpoints/group_scaleproto_pascal \
  --quantile 0.8 --std-dev 0.2
```

To use Pascal semantic matching and foreground centroids, append
`--require-semantic-match --exclude-background-centroid`. To run consistency, use:

```bash
sbatch --output="$PWD/consistency-%j.out" scripts/run_metric.slurm \
  consistency scaleproto_pascal push pascal \
  --checkpoint /data/shared/protoseg/scaleproto_checkpoints/scaleproto_pascal/checkpoints/push_last.pth \
  --group-name /data/shared/protoseg/scaleproto_checkpoints/group_scaleproto_pascal \
  --quantile 0.8 --threshold 0.8
```

The Slurm wrapper requests one GPU on TANK, eight CPUs, and 32 GB of memory.
Override the node with `sbatch --nodelist=maxitank ...` when using that node's checkout
and preprocessed data. `/data_ssd` storage can differ between nodes: the updated code
and configured caches must be available on the node you select. The description's
hostname and annotation path identify which environment was used.

For testing it is necessary to then properly arrange the output folder to match the benchmarks requirements online.

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
