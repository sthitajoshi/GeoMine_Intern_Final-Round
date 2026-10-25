"""Layer export helpers (PNG with legend, plain TIFF arrays).

The .mat scenes carry no georeferencing, so exports are image-space only: we do not invent a CRS
or geotransform. TIFFs store the raw label / value array for use in GIS as an ungeoreferenced
raster (pixel size noted in the filename metadata only).
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .visualization import save_map_png


def png_bytes(img: np.ndarray, title: str, legend=None, cbar=None, note=None) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "layer.png"
        save_map_png(p, img, title, legend, cbar, note)
        return p.read_bytes()


def tiff_bytes(arr: np.ndarray, description: str) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.asarray(arr), description=description + " | NOT georeferenced (source .mat has no geo metadata)")
    return buf.getvalue()
