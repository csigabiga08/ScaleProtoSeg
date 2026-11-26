# Interpretability Metrics for Prototype-Based Semantic Segmentation

This repository extends [ScaleProtoSeg](README_original.md) (WACV 2025) with tooling for
evaluating the interpretability of prototype-based segmentation models (ProtoSeg and
ScaleProtoSeg) on part-annotated datasets (PASCAL VOC / Pascal-Part, Cityscapes).

The work accompanies the paper *Towards Better Interpretability Metrics for Explainable
Semantic Segmentation* (Szenkovits, Portik, Bodó; submitted to ESANN 2027). It shows how
annotation and evaluation choices (rare parts, fine-grained taxonomies, centroid
computation, activation scope) change consistency and stability scores for identical models.

## Documentation index

| Topic | Document |
|---|---|
| Original ScaleProtoSeg: installation, data, training, evaluation, results | [README_original.md](README_original.md) |
| Metrics (consistency, stability, imprecise activations) and part-annotation handling | [segmentation/analysis/README.md](segmentation/analysis/README.md) |
| Measurement catalog: registering checkpoints, querying, web app, job submission | [segmentation/catalog/README.md](segmentation/catalog/README.md) |

## Paper contributions → documentation

| Paper section | Where to look |
|---|---|
| Replication and experimental protocol | [README_original.md](README_original.md) (setup, training, evaluation), [catalog README](segmentation/catalog/README.md) (registered checkpoints and measurements) |
| Part-annotation considerations (incl. merging Pascal left/right parts) | [analysis README – Pascal left/right part merging](segmentation/analysis/README.md#pascal-leftright-part-merging) |
| Support-aware consistency | [analysis README](segmentation/analysis/README.md#native-consistency-and-stability-measurements) |
| Correlation between prototype–part activation profiles | [analysis README](segmentation/analysis/README.md#native-consistency-and-stability-measurements) |
| Imprecise activations | [analysis README – Imprecise activations](segmentation/analysis/README.md#imprecise-activations) |

## Quick start

1. Set up the environment and data following [README_original.md](README_original.md).
2. Train or obtain checkpoints, then run the metrics as described in the
   [analysis README](segmentation/analysis/README.md).
3. Register checkpoints and browse or query results with the
   [measurement catalog](segmentation/catalog/README.md).

## Citation

If you use the original method, cite ScaleProtoSeg (see [README_original.md](README_original.md)).
The citation for the metrics paper will be added once the BibTeX is final.
