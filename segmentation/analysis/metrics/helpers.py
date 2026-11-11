"""Helpers shared by the consistency and stability metrics."""

from typing import List, Optional, Sequence

import numpy as np
import torch

from segmentation.model.model_multiscale_group import PPNetMultiScale


def proto_filter(ppnet: PPNetMultiScale) -> List[int]:
    """Return prototype IDs used by the model's grouping mechanism."""
    filtered_ids = []
    for class_id in range(len(ppnet.group_projection)):
        local_ids = torch.nonzero(ppnet.group_projection[class_id].weight.data.sum(dim=0))
        local_ids = local_ids.flatten().tolist()
        class_ids = ppnet.prototype_class_identity[:, class_id].flatten().nonzero()
        class_ids = class_ids.flatten().tolist()
        filtered_ids.extend(class_ids[local_id] for local_id in local_ids)
    return filtered_ids


def selected_prototypes(
    model,
    allowed_ids: Optional[Sequence[int]],
    filtered_classes: Sequence[int] = (),
) -> List[int]:
    identity = getattr(model, "prototype_class_identity", None)
    if not isinstance(identity, torch.Tensor) or identity.ndim != 2:
        return []

    identity = identity.detach().cpu().numpy()
    allowed = None if allowed_ids is None else set(int(value) for value in allowed_ids)
    filtered = set(filtered_classes)
    return [
        int(proto_id)
        for class_index in range(identity.shape[1])
        if class_index + 1 not in filtered
        for proto_id in np.flatnonzero(identity[:, class_index])
        if allowed is None or int(proto_id) in allowed
    ]
