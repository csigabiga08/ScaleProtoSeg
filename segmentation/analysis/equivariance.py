"""Activation-map thresholding shared by the interpretability metrics."""

import torch


def quantile_map(activation_maps: torch.Tensor, quantile: float, class_mask: torch.Tensor) -> torch.Tensor:
    """Threshold each activation channel using only pixels in a semantic class.

    Args:
        activation_maps: Floating-point activations shaped ``(C, H, W)``.
        quantile: Quantile in ``[0, 1]``, computed independently per channel
            with linear interpolation.
        class_mask: Boolean mask shaped ``(H, W)`` selecting the semantic
            class. Zero-valued activations within this mask are included.

    Returns:
        A boolean tensor shaped ``(H, W, C)`` on the activations' device.
        Pixels must belong to the class and be strictly greater than their
        channel's quantile to be active. Ties at the threshold are inactive;
        an empty class mask returns an all-false map.
    """
    if activation_maps.ndim != 3:
        raise ValueError("activation_maps must have shape (C, H, W)")
    if not activation_maps.is_floating_point():
        raise TypeError("activation_maps must contain floating-point values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    class_mask = torch.as_tensor(class_mask, dtype=torch.bool, device=activation_maps.device)
    if class_mask.shape != activation_maps.shape[1:]:
        raise ValueError("class_mask must have shape (H, W) matching activation_maps")

    binary_maps = torch.zeros_like(activation_maps, dtype=torch.bool)
    class_values = activation_maps[:, class_mask]
    if class_values.shape[1] > 0:
        if not torch.isfinite(class_values).all():
            raise ValueError("activations within class_mask must be finite")
        # torch.quantile supports float32/float64; promote lower precision.
        if class_values.dtype not in (torch.float32, torch.float64):
            class_values = class_values.float()
        thresholds = torch.quantile(class_values, quantile, dim=1, keepdim=True)
        binary_maps[:, class_mask] = class_values > thresholds

    return binary_maps.permute(1, 2, 0)
