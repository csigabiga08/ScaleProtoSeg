"""Class-specific normalized Pascal Part IDs used by preprocess_part_pascal.

Labels follow SOURCE_DATA_PATH_PASCAL/parts.yaml (PASCAL Panoptic Parts v2).
IDs restart at 1 in each class; repeated part instances are already folded by
preprocessing. This module does not modify masks or compute new centroids.
"""

import numpy as np


PASCAL_PART_LABELS = {
    "aeroplane": ("body", "stern", "lwing", "rwing", "tail", "engine", "wheel"),
    "bicycle": ("fwheel", "bwheel", "saddle", "handlebar", "chainwheel", "headlight"),
    "bird": (
        "head", "leye", "reye", "beak", "torso", "neck",
        "lwing", "rwing", "lleg", "lfoot", "rleg", "rfoot",
        "tail",
    ),
    "boat": (),
    "bottle": ("cap", "body"),
    "bus": (
        "frontside", "leftside", "rightside", "backside", "roofside", "leftmirror",
        "rightmirror", "fliplate", "bliplate", "door", "wheel", "headlight",
        "window",
    ),
    "car": (
        "frontside", "leftside", "rightside", "backside", "roofside", "leftmirror",
        "rightmirror", "fliplate", "bliplate", "door", "wheel", "headlight",
        "window",
    ),
    "cat": (
        "head", "leye", "reye", "lear", "rear", "nose",
        "torso", "neck", "lfleg", "lfpa", "rfleg", "rfpa",
        "lbleg", "lbpa", "rbleg", "rbpa", "tail",
    ),
    "chair": (),
    "cow": (
        "head", "leye", "reye", "lear", "rear", "muzzle",
        "lhorn", "rhorn", "torso", "neck", "lfuleg", "lflleg",
        "rfuleg", "rflleg", "lbuleg", "lblleg", "rbuleg", "rblleg",
        "tail",
    ),
    "diningtable": (),
    "dog": (
        "head", "leye", "reye", "lear", "rear", "nose",
        "torso", "neck", "lfleg", "lfpa", "rfleg", "rfpa",
        "lbleg", "lbpa", "rbleg", "rbpa", "tail", "muzzle",
    ),
    "horse": (
        "head", "leye", "reye", "lear", "rear", "muzzle",
        "lfho", "rfho", "lbho", "rbho", "torso", "neck",
        "lfuleg", "lflleg", "rfuleg", "rflleg", "lbuleg", "lblleg",
        "rbuleg", "rblleg", "tail",
    ),
    "motorbike": ("fwheel", "bwheel", "handlebar", "saddle", "headlight"),
    "person": (
        "head", "leye", "reye", "lear", "rear", "lebrow",
        "rebrow", "nose", "mouth", "hair", "torso", "neck",
        "llarm", "luarm", "lhand", "rlarm", "ruarm", "rhand",
        "llleg", "luleg", "lfoot", "rlleg", "ruleg", "rfoot",
    ),
    "pottedplant": ("pot", "plant"),
    "sheep": (
        "head", "leye", "reye", "lear", "rear", "muzzle",
        "lhorn", "rhorn", "torso", "neck", "lfuleg", "lflleg",
        "rfuleg", "rflleg", "lbuleg", "lblleg", "rbuleg", "rblleg",
        "tail",
    ),
    "sofa": (),
    "train": (
        "head", "hfrontside", "hleftside", "hrightside", "hbackside", "hroofside",
        "headlight", "coach", "cfrontside", "cleftside", "crightside", "cbackside",
        "croofside",
    ),
    "tvmonitor": ("screen", "frame"),
}


def _left_right_pairs(labels):
    pairs = []
    for left_id, label in enumerate(labels, 1):
        if "left" in label:
            counterpart = label.replace("left", "right")
        elif label.startswith("l"):
            counterpart = "r" + label[1:]
        else:
            continue
        if counterpart in labels:
            pairs.append((left_id, labels.index(counterpart) + 1))
    return tuple(pairs)


PASCAL_LEFT_RIGHT_PAIRS = {
    name: _left_right_pairs(labels) for name, labels in PASCAL_PART_LABELS.items()
}


def merge_left_right_presence(presence, class_name):
    """OR each pair's per-image hits, retaining its left ID as the canonical ID.

    Centroids have already been computed independently for the original labels.
    A missing side is ignored, both missing remains NaN, and the right-hand column
    is set to NaN so each pair contributes one candidate to the consistency score.
    Unpaired labels (and any extra decoded IDs) retain their original values.
    """
    merged = list(presence)
    for left_id, right_id in PASCAL_LEFT_RIGHT_PAIRS[class_name]:
        observed = [presence[pid] for pid in (left_id, right_id) if not np.isnan(presence[pid])]
        merged[left_id] = int(any(value != 0 for value in observed)) if observed else np.nan
        merged[right_id] = np.nan
    return merged


def left_right_part_groups():
    """Describe canonical CSV IDs and both original labels for run metadata."""
    return {
        name: [
            {
                "canonical_part_id": left,
                "source_part_ids": [left, right],
                "source_labels": [PASCAL_PART_LABELS[name][left - 1], PASCAL_PART_LABELS[name][right - 1]],
            }
            for left, right in pairs
        ]
        for name, pairs in PASCAL_LEFT_RIGHT_PAIRS.items() if pairs
    }
