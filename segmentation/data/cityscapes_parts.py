"""Filtering for the official Cityscapes Panoptic Parts taxonomy."""

import numpy as np


def filter_official_parts(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """Keep official semantic/part pairs, using -1 for no part.

    ``sids`` contains native panoptic semantic IDs, not training IDs.
    Neither input is modified.
    Source: https://github.com/pmeletis/panoptic_parts/blob/master/
    panoptic_parts/specs/dataset_specs/cpp_datasetspec.yaml
    """
    if sids.shape != pids.shape:
        raise ValueError("Cityscapes semantic and part masks must have matching shapes")
    # Person/rider: torso, head, arm, leg (1-4).
    human_parts = np.isin(sids, (24, 25)) & np.isin(pids, (1, 2, 3, 4))
    # Car/truck/bus: window, wheel, light, license plate, chassis (1-5).
    vehicle_parts = np.isin(sids, (26, 27, 28)) & np.isin(pids, (1, 2, 3, 4, 5))
    return np.where(human_parts | vehicle_parts, pids, -1)
