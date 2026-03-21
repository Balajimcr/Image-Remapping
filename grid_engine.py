#!/usr/bin/env python3
"""
Fixed Grid Image Remapping Engine
==================================
Core algorithm for sparse-grid-based image remapping using OpenCV.

Algorithm flow:
  1. Receive a sparse (grid_rows × grid_cols) displacement grid (dx, dy).
  2. Interpolate displacements to full image resolution via bicubic spline.
  3. Build per-pixel remap maps (map_x, map_y) for cv2.remap (inverse mapping).
  4. Apply cv2.remap with caller-specified interpolation and border mode.

Inverse mapping convention (required by cv2.remap):
  map_x[dst_y, dst_x] = src_x  →  source column to sample
  map_y[dst_y, dst_x] = src_y  →  source row    to sample

  So: src = dst + displacement
      map_x = col_grid + dense_dx
      map_y = row_grid + dense_dy

Author: Balaji R
License: MIT
"""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageArray = np.ndarray          # uint8 BGR or grayscale
DisplacementGrid = np.ndarray    # float32, shape (grid_rows, grid_cols)
RemapMap = np.ndarray            # float32, shape (H, W)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear":  cv2.INTER_LINEAR,
    "cubic":   cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

_CV2_BORDER = {
    "constant":  cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect":   cv2.BORDER_REFLECT_101,
    "wrap":      cv2.BORDER_WRAP,
}


# ---------------------------------------------------------------------------
# Grid Remap Engine
# ---------------------------------------------------------------------------
class GridRemapEngine:
    """
    Fixed-grid image remapping engine.

    Usage
    -----
    engine = GridRemapEngine()
    dx_grid, dy_grid = DistortionPresets.barrel((480, 640), 11, 11, k1=0.4)
    map_x, map_y     = engine.build_remap_maps((480, 640), dx_grid, dy_grid)
    result           = engine.apply_remap(image, map_x, map_y)
    """

    def build_remap_maps(
        self,
        image_shape: Tuple[int, int],
        grid_dx: DisplacementGrid,
        grid_dy: DisplacementGrid,
        grid_interp: str = "bicubic",
    ) -> Tuple[RemapMap, RemapMap]:
        """
        Interpolate sparse displacement grid to dense per-pixel remap maps.

        Parameters
        ----------
        image_shape : (H, W)
            Target image dimensions.
        grid_dx : ndarray, shape (grid_rows, grid_cols)
            Horizontal displacement at each grid node (pixels).
        grid_dy : ndarray, shape (grid_rows, grid_cols)
            Vertical displacement at each grid node (pixels).
        grid_interp : {"bicubic", "linear"}
            Interpolation method for grid upsampling.

        Returns
        -------
        map_x, map_y : ndarray float32, shape (H, W)
            Source pixel coordinates for cv2.remap.
        """
        if grid_dx.shape != grid_dy.shape:
            raise ValueError(
                f"grid_dx and grid_dy must have the same shape; "
                f"got {grid_dx.shape} vs {grid_dy.shape}"
            )

        H, W = image_shape
        grid_rows, grid_cols = grid_dx.shape

        # Grid node positions in pixel space
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        x_nodes = np.linspace(0.0, W - 1, grid_cols)

        # Full-resolution pixel coordinates
        y_pixels = np.arange(H, dtype=np.float64)
        x_pixels = np.arange(W, dtype=np.float64)

        # Suppress scipy spline edge-warning (expected for small grids)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            if grid_interp == "bicubic":
                kx = ky = min(3, grid_cols - 1, grid_rows - 1)  # cubic clamped to data size
                spline_dx = RectBivariateSpline(
                    y_nodes, x_nodes, grid_dx.astype(np.float64), kx=kx, ky=kx, s=0
                )
                spline_dy = RectBivariateSpline(
                    y_nodes, x_nodes, grid_dy.astype(np.float64), kx=kx, ky=ky, s=0
                )
                dense_dx = spline_dx(y_pixels, x_pixels)
                dense_dy = spline_dy(y_pixels, x_pixels)

            else:  # linear
                from scipy.interpolate import RegularGridInterpolator
                interp_dx = RegularGridInterpolator(
                    (y_nodes, x_nodes), grid_dx.astype(np.float64), method="linear"
                )
                interp_dy = RegularGridInterpolator(
                    (y_nodes, x_nodes), grid_dy.astype(np.float64), method="linear"
                )
                yy, xx = np.meshgrid(y_pixels, x_pixels, indexing="ij")
                pts = np.column_stack([yy.ravel(), xx.ravel()])
                dense_dx = interp_dx(pts).reshape(H, W)
                dense_dy = interp_dy(pts).reshape(H, W)

        # Build inverse remap maps
        col_grid, row_grid = np.meshgrid(x_pixels, y_pixels)   # both (H, W)
        map_x = (col_grid + dense_dx).astype(np.float32)
        map_y = (row_grid + dense_dy).astype(np.float32)

        return map_x, map_y

    def apply_remap(
        self,
        image: ImageArray,
        map_x: RemapMap,
        map_y: RemapMap,
        interpolation: str = "linear",
        border_mode: str = "constant",
        border_value: int = 0,
    ) -> ImageArray:
        """
        Apply remap maps to an image using cv2.remap.

        Parameters
        ----------
        image : ndarray uint8
            Source image (BGR or grayscale).
        map_x, map_y : ndarray float32, shape (H, W)
            Source coordinates from build_remap_maps().
        interpolation : {"nearest", "linear", "cubic", "lanczos"}
        border_mode   : {"constant", "replicate", "reflect", "wrap"}
        border_value  : Fill value when border_mode is "constant".

        Returns
        -------
        remapped : ndarray uint8, same shape as image.
        """
        if interpolation not in _CV2_INTERP:
            raise ValueError(f"Unknown interpolation '{interpolation}'. Choose from {list(_CV2_INTERP)}")
        if border_mode not in _CV2_BORDER:
            raise ValueError(f"Unknown border_mode '{border_mode}'. Choose from {list(_CV2_BORDER)}")

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=_CV2_INTERP[interpolation],
            borderMode=_CV2_BORDER[border_mode],
            borderValue=border_value,
        )

    def overlay_grid(
        self,
        image: ImageArray,
        grid_rows: int,
        grid_cols: int,
        map_x: RemapMap,
        map_y: RemapMap,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
        viz_type: str = "lines",
        dot_radius: int = 3,
    ) -> ImageArray:
        """
        Draw remapped grid visualization onto an image for distortion verification.

        Samples node positions from map_x/map_y and renders them as either
        connected lines or dots, showing how the regular grid is deformed.

        Parameters
        ----------
        viz_type : {"lines", "dots"}
            "lines" - draw connected grid lines (default)
            "dots"  - draw grid points as circles
        dot_radius : int
            Radius of dots when viz_type="dots"
        """
        H, W = image.shape[:2]
        canvas = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        y_nodes = np.linspace(0, H - 1, grid_rows).astype(int)
        x_nodes = np.linspace(0, W - 1, grid_cols).astype(int)

        if viz_type == "dots":
            # Draw grid points as circles
            for gy in y_nodes:
                for gx in x_nodes:
                    cx, cy = int(map_x[gy, gx]), int(map_y[gy, gx])
                    cv2.circle(canvas, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)
        else:
            # Draw connected lines (default)
            # Horizontal lines
            for gy in y_nodes:
                pts = np.array(
                    [[int(map_x[gy, gx]), int(map_y[gy, gx])] for gx in x_nodes],
                    dtype=np.int32,
                )
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

            # Vertical lines
            for gx in x_nodes:
                pts = np.array(
                    [[int(map_x[gy, gx]), int(map_y[gy, gx])] for gy in y_nodes],
                    dtype=np.int32,
                )
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

        return canvas


# ---------------------------------------------------------------------------
# Distortion Presets — generate sparse (grid_rows × grid_cols) displacement grids
# ---------------------------------------------------------------------------
class DistortionPresets:
    """
    Factory methods for common distortion displacement grids.

    All methods return:
      dx_grid, dy_grid : ndarray float32, shape (grid_rows, grid_cols)

    Displacements are in the inverse-mapping sense:
      source pixel = destination pixel + displacement
    """

    @staticmethod
    def identity(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """Zero displacement — output equals input."""
        zeros = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        return zeros.copy(), zeros.copy()

    @staticmethod
    def barrel(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        k1: float = 0.4,
        k2: float = 0.0,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """
        Radial barrel / pincushion distortion.

        k1 > 0 : barrel  (edges bow outward)
        k1 < 0 : pincushion (edges bow inward)
        k2     : higher-order radial coefficient
        """
        H, W = image_shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        # Normalisation radius — half-diagonal so corners reach ~1
        r_norm = math.sqrt(cx**2 + cy**2)

        x_nodes = np.linspace(0.0, W - 1, grid_cols)
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        xx, yy  = np.meshgrid(x_nodes, y_nodes)

        nx = (xx - cx) / r_norm
        ny = (yy - cy) / r_norm
        r2 = nx**2 + ny**2

        # Radial scale: src = dst * (1 + k1*r^2 + k2*r^4)
        scale = 1.0 + k1 * r2 + k2 * (r2 ** 2)

        src_x = cx + nx * scale * r_norm
        src_y = cy + ny * scale * r_norm

        dx = (src_x - xx).astype(np.float32)
        dy = (src_y - yy).astype(np.float32)
        return dx, dy

    @staticmethod
    def pincushion(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        k1: float = 0.4,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """Pincushion distortion (convenience wrapper, k1 negated)."""
        return DistortionPresets.barrel(image_shape, grid_rows, grid_cols, k1=-abs(k1))

    @staticmethod
    def swirl(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        strength: float = 2.0,
        radius: float | None = None,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """
        Swirl / vortex distortion.

        Pixels rotate by an angle proportional to their distance from centre.
        strength : maximum rotation angle (radians) at the centre.
        radius   : influence radius (defaults to min(W, H) / 2).
        """
        H, W = image_shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        if radius is None:
            radius = min(W, H) / 2.0

        x_nodes = np.linspace(0.0, W - 1, grid_cols)
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        xx, yy  = np.meshgrid(x_nodes, y_nodes)

        rel_x = xx - cx
        rel_y = yy - cy
        r     = np.sqrt(rel_x**2 + rel_y**2)

        # Rotation angle decreases linearly from centre → edge
        angle  = strength * np.maximum(0.0, 1.0 - r / radius)
        cos_a  = np.cos(angle)
        sin_a  = np.sin(angle)

        src_x  = cx + rel_x * cos_a - rel_y * sin_a
        src_y  = cy + rel_x * sin_a + rel_y * cos_a

        dx = (src_x - xx).astype(np.float32)
        dy = (src_y - yy).astype(np.float32)
        return dx, dy

    @staticmethod
    def wave(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        amplitude_x: float = 15.0,
        frequency_x: float = 2.0,
        amplitude_y: float = 10.0,
        frequency_y: float = 3.0,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """
        Sinusoidal wave distortion.

        amplitude_* : peak displacement in pixels.
        frequency_* : number of complete cycles across the image.
        """
        H, W = image_shape

        x_nodes = np.linspace(0.0, W - 1, grid_cols)
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        xx, yy  = np.meshgrid(x_nodes, y_nodes)

        dx = (amplitude_x * np.sin(2 * math.pi * frequency_x * yy / H)).astype(np.float32)
        dy = (amplitude_y * np.sin(2 * math.pi * frequency_y * xx / W)).astype(np.float32)
        return dx, dy

    @staticmethod
    def fisheye(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        fov_scale: float = 0.6,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """
        Simplified equidistant fisheye projection.

        Maps straight lines to curved ones by applying an equidistant
        fisheye remapping (θ / sin(θ) scaling).

        fov_scale : controls effective field-of-view (0.4 – 0.9 range).
        """
        H, W = image_shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        r_norm = min(cx, cy)

        x_nodes = np.linspace(0.0, W - 1, grid_cols)
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        xx, yy  = np.meshgrid(x_nodes, y_nodes)

        rel_x = xx - cx
        rel_y = yy - cy
        r     = np.sqrt(rel_x**2 + rel_y**2)
        r_clip = np.clip(r / (r_norm * fov_scale), 0, 1.0)

        theta = r_clip * (math.pi / 2)
        # Avoid division by zero at r=0
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r < 1e-6, 1.0, np.sin(theta) / (r / (r_norm * fov_scale)))

        src_x = cx + rel_x * scale
        src_y = cy + rel_y * scale

        dx = (src_x - xx).astype(np.float32)
        dy = (src_y - yy).astype(np.float32)
        return dx, dy

    @staticmethod
    def pinhole_correction(
        image_shape: Tuple[int, int],
        grid_rows: int,
        grid_cols: int,
        k1: float = -0.3,
        k2: float = 0.05,
        p1: float = 0.001,
        p2: float = 0.001,
    ) -> Tuple[DisplacementGrid, DisplacementGrid]:
        """
        Approximates OpenCV camera model distortion correction grid.

        Applies radial (k1, k2) + tangential (p1, p2) correction.
        Useful for camera calibration preview without actual intrinsics.
        """
        H, W = image_shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        fx = fy = max(W, H) * 0.9   # approximate focal length

        x_nodes = np.linspace(0.0, W - 1, grid_cols)
        y_nodes = np.linspace(0.0, H - 1, grid_rows)
        xx, yy  = np.meshgrid(x_nodes, y_nodes)

        # Normalised camera coords
        xn = (xx - cx) / fx
        yn = (yy - cy) / fy
        r2 = xn**2 + yn**2

        radial    = 1 + k1 * r2 + k2 * (r2 ** 2)
        tan_x     = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
        tan_y     = p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

        src_x = cx + fx * (xn * radial + tan_x)
        src_y = cy + fy * (yn * radial + tan_y)

        dx = (src_x - xx).astype(np.float32)
        dy = (src_y - yy).astype(np.float32)
        return dx, dy
