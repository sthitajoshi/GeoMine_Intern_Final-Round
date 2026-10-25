"""Colour tables and map rendering shared by figures, exports and the app."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MINERAL_COLOURS = {"alunite": "#d62728", "kaolinite": "#1f77b4", "muscovite": "#2ca02c",
                   "buddingtonite": "#9467bd", "calcite": "#ff7f0e", "chalcedony": "#17becf"}


def class_palette(n: int) -> np.ndarray:
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    return base[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19][:n]]


def colourise(labels: np.ndarray, palette: np.ndarray, background=(0, 0, 0)) -> np.ndarray:
    """labels: int map with negative = no class. Returns RGB float image."""
    out = np.zeros(labels.shape + (3,))
    out[:] = background
    m = labels >= 0
    out[m] = palette[labels[m]]
    return out


def mineral_rgb(lab: np.ndarray, names: list[str]) -> np.ndarray:
    pal = np.array([matplotlib.colors.to_rgb(MINERAL_COLOURS[n]) for n in names])
    out = np.full(lab.shape + (3,), 0.15)
    out[lab == -1] = 0.85                       # unknown / none = light grey
    m = lab >= 0
    out[m] = pal[lab[m]]
    return out


def scalar_rgb(v: np.ndarray, cmap="viridis", vmin=None, vmax=None) -> np.ndarray:
    vmin = np.nanpercentile(v, 2) if vmin is None else vmin
    vmax = np.nanpercentile(v, 98) if vmax is None else vmax
    n = np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = plt.get_cmap(cmap)(np.nan_to_num(n))[..., :3]
    rgb[~np.isfinite(v)] = 0
    return rgb


def save_map_png(path, img, title, legend: list[tuple[str, tuple]] | None = None, cbar=None, note=None):
    """Write an RGB map with a legend (categorical) or colour bar (cbar=(cmap, vmin, vmax, label))."""
    h, w = img.shape[:2]
    fig_w = 7.5 if legend else 6.5
    fig, ax = plt.subplots(figsize=(fig_w, max(3.5, 6.0 * h / w * (5.0 / fig_w) + 1)))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    if legend:
        ax.legend(handles=[Patch(color=c, label=l) for l, c in legend], loc="center left",
                  bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    if cbar:
        cm_, vmin, vmax, label = cbar
        sm = plt.cm.ScalarMappable(cmap=cm_, norm=plt.Normalize(vmin, vmax))
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label=label)
    if note:
        fig.text(0.01, 0.01, note, fontsize=6, color="0.3")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
