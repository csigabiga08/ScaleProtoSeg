"""Loading of trusted, serialized model checkpoints."""

import inspect
from pathlib import Path
from typing import Union

import torch

from segmentation import checkpoint_pickle


def load_model_checkpoint(path: Union[str, Path]) -> torch.nn.Module:
    """Load a trusted complete model on CPU, including on newer PyTorch.

    ScaleProtoSeg checkpoints contain model objects, not just state dicts.
    Explicitly disable weights-only loading when the installed PyTorch
    supports it; older releases do not accept that argument. Resolve the
    historical top-level model module names used by released checkpoints.
    """
    load_kwargs = {"map_location": "cpu", "pickle_module": checkpoint_pickle}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)
