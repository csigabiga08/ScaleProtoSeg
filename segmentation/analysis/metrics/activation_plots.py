"""Render the activation maps used by whole-image consistency and stability."""

from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.analysis.metrics.output import _label


def save_activation_plot(
    directory: Path,
    image: Image.Image,
    image_id: str,
    class_name: str,
    prototype_id: int,
    raw_activation: np.ndarray,
    binary_mask: np.ndarray,
    quantile: float,
    *,
    class_mask: np.ndarray,
    noisy: bool = False,
) -> None:
    """Save both activation maps with the ground-truth class outline."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    pass_name = "noisy" if noisy else "clean"
    path = directory / _label(image_id) / f"{_label(class_name)}_prototype_{prototype_id}_{pass_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12, 3.8), dpi=120, constrained_layout=True)
    FigureCanvasAgg(figure)
    try:
        original_ax, raw_ax, mask_ax = figure.subplots(1, 3)
        original_ax.imshow(image)
        original_ax.set_title("Original image")
        heatmap = raw_ax.imshow(raw_activation, cmap="viridis", interpolation="nearest")
        raw_ax.set_title("Raw activation")
        figure.colorbar(heatmap, ax=raw_ax, label="Log similarity", shrink=0.75)
        mask_ax.imshow(binary_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        mask_ax.set_title(f"Binary mask (q={quantile:g}, {binary_mask.mean():.1%} active)")
        if class_mask.any():
            height, width = class_mask.shape
            # The one-pixel border also outlines classes that touch the image edge.
            padded_mask = np.pad(class_mask, 1, constant_values=False)
            outline_style = {"colors": "#ff47bc", "linewidths": 1.1}
            raw_ax.set_autoscale_on(False)
            mask_ax.set_autoscale_on(False)
            contour = raw_ax.contour(
                np.arange(-1, width + 1), np.arange(-1, height + 1),
                padded_mask, levels=[0.5], **outline_style,
            )
            # Reuse the same paths on the binary panel to keep the outlines aligned.
            mask_contour = LineCollection(contour.allsegs[0], **outline_style)
            mask_ax.add_collection(mask_contour, autolim=False)
            figure.supxlabel("Pink outline shows the ground-truth class mask", fontsize=9)
        for ax in (original_ax, raw_ax, mask_ax):
            ax.set_axis_off()
        figure.suptitle(f"{image_id}  |  {class_name}  |  Prototype {prototype_id}  |  {pass_name}")
        figure.savefig(path)
    finally:
        figure.clear()
