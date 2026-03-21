#!/usr/bin/env python3
"""
Sample Image Generator
======================
Generates synthetic test images suitable for verifying remapping correctness.

Patterns with strong geometric structure (grids, circles, checkerboards) make
remapping artefacts immediately visible, which is essential for ISP / camera
calibration work.

Author: Balaji R
License: MIT
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageArray = np.ndarray   # uint8 BGR
Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# SampleImageGenerator
# ---------------------------------------------------------------------------
class SampleImageGenerator:
    """
    Generates structured synthetic images for remapping validation.

    All methods return uint8 BGR images of the requested size.
    """

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def checkerboard(
        width: int,
        height: int,
        cell_size: int = 40,
        color_a: Color = (255, 255, 255),
        color_b: Color = (30, 30, 30),
    ) -> ImageArray:
        """
        Classic checkerboard pattern.

        Ideal for detecting local scale/shear distortions — any deformation
        makes square cells appear curved or skewed.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = color_b

        rows = (height + cell_size - 1) // cell_size
        cols = (width  + cell_size - 1) // cell_size

        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    y1 = r * cell_size
                    x1 = c * cell_size
                    y2 = min(y1 + cell_size, height)
                    x2 = min(x1 + cell_size, width)
                    canvas[y1:y2, x1:x2] = color_a

        return canvas

    @staticmethod
    def color_checkerboard(
        width: int,
        height: int,
    ) -> ImageArray:
        """
        Macbeth ColorChecker chart filling the entire image.

        Classic 6x4 grid with skin tones, nature colors, primaries,
        secondaries, and grayscale patches - tiles fill the whole image.
        """
        # Macbeth 24 standard colors (RGB format)
        macbeth_rgb = [
            (115, 82, 68),   (194, 150, 130), (98, 122, 157), (87, 108, 67),
            (133, 128, 177), (103, 189, 170),
            (214, 126, 44),  (80, 91, 166),   (193, 90, 99),   (94, 60, 108),
            (157, 188, 64),  (224, 163, 46),
            (56, 61, 150),   (70, 148, 73),   (175, 54, 60),   (231, 199, 31),
            (187, 86, 149),  (8, 133, 161),
            (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121),
            (85, 85, 85),    (35, 35, 35)
        ]

        # Convert RGB → BGR (OpenCV format)
        macbeth_bgr = [tuple(reversed(c)) for c in macbeth_rgb]

        rows, cols = 4, 6

        # Border/gap size proportional to image
        gap = max(1, min(width, height) // 80)

        # Calculate available space for patches
        available_w = width - (cols + 1) * gap
        available_h = height - (rows + 1) * gap

        # Calculate base patch size and remainder
        base_patch_w = available_w // cols
        base_patch_h = available_h // rows
        extra_w = available_w % cols
        extra_h = available_h % rows

        # Create background with gap color (dark gray)
        canvas = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)

        idx = 0
        for r in range(rows):
            # Distribute extra pixels to bottom rows
            y1 = gap + r * (base_patch_h + gap)
            if r < extra_h:
                y1 += r
            else:
                y1 += extra_h
            patch_h = base_patch_h + (1 if r < extra_h else 0)

            for c in range(cols):
                # Distribute extra pixels to right columns
                x1 = gap + c * (base_patch_w + gap)
                if c < extra_w:
                    x1 += c
                else:
                    x1 += extra_w
                patch_w = base_patch_w + (1 if c < extra_w else 0)

                x2 = min(x1 + patch_w, width)
                y2 = min(y1 + patch_h, height)

                cv2.rectangle(canvas, (x1, y1), (x2, y2), macbeth_bgr[idx], -1)
                idx += 1

        return canvas

    @staticmethod
    def color_wheel(
        width: int,
        height: int,
        grid_cells: int = 16,
        border_color: Color = (220, 220, 220),
        border_thickness: int = 1,
    ) -> ImageArray:
        """
        Tiled color wheel - grid of solid color squares with color wheel range.

        Divides image into grid_cells x grid_cells squares.
        Each square gets a single HSV color based on its position
        (hue from angle, saturation from distance to center).
        """
        canvas = np.full((height, width, 3), border_color, dtype=np.uint8)

        # Calculate tile positions to evenly distribute across image
        # Total border space: (grid_cells + 1) borders
        total_border_h = (grid_cells + 1) * border_thickness
        total_border_w = (grid_cells + 1) * border_thickness

        # Available space for tiles
        available_h = height - total_border_h
        available_w = width - total_border_w

        # Calculate base tile size and remainder
        base_tile_h = available_h // grid_cells
        base_tile_w = available_w // grid_cells
        extra_h = available_h % grid_cells
        extra_w = available_w % grid_cells

        # Center of the image
        cx, cy = width / 2, height / 2
        max_dist = np.hypot(cx, cy)

        for row in range(grid_cells):
            # Distribute extra pixels to bottom rows
            y1 = border_thickness + row * (base_tile_h + border_thickness)
            if row < extra_h:
                y1 += row
            else:
                y1 += extra_h
            tile_h = base_tile_h + (1 if row < extra_h else 0)
            y2 = min(y1 + tile_h, height)

            for col in range(grid_cells):
                # Distribute extra pixels to right columns
                x1 = border_thickness + col * (base_tile_w + border_thickness)
                if col < extra_w:
                    x1 += col
                else:
                    x1 += extra_w
                tile_w = base_tile_w + (1 if col < extra_w else 0)
                x2 = min(x1 + tile_w, width)

                # Cell center for color calculation
                cell_cx = (x1 + x2) / 2
                cell_cy = (y1 + y2) / 2

                # Polar coordinates from center
                dx = cell_cx - cx
                dy = cell_cy - cy
                dist = np.hypot(dx, dy)
                angle = np.arctan2(dy, dx)  # -pi to pi

                # HSV values
                hue = int(((angle + np.pi) / (2 * np.pi)) * 179)
                saturation = int(np.clip((dist / max_dist) * 255, 0, 255))
                value = 255

                # Convert to BGR and fill cell
                color = (hue, saturation, value)
                bgr = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0, 0]
                canvas[y1:y2, x1:x2] = bgr

        return canvas

    @staticmethod
    def grid_lines(
        width: int,
        height: int,
        spacing: int = 40,
        line_color: Color = (220, 220, 220),
        bg_color: Color = (30, 30, 80),
        thickness: int = 1,
    ) -> ImageArray:
        """
        Uniform grid-line pattern on a dark background.

        Straight grid lines become curved under radial or swirl distortion,
        making the distortion magnitude visually obvious.
        """
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

        for x in range(0, width, spacing):
            cv2.line(canvas, (x, 0), (x, height - 1), line_color, thickness, cv2.LINE_AA)
        for y in range(0, height, spacing):
            cv2.line(canvas, (0, y), (width - 1, y), line_color, thickness, cv2.LINE_AA)

        return canvas

    @staticmethod
    def concentric_circles(
        width: int,
        height: int,
        ring_spacing: int = 30,
        line_color: Color = (80, 200, 120),
        bg_color: Color = (15, 15, 35),
        thickness: int = 1,
    ) -> ImageArray:
        """
        Concentric circles centred on the image.

        Under radial distortion the circles remain circles but their spacing
        changes, cleanly quantifying the radial displacement magnitude.
        Under swirl, perfect circles become spirals.
        """
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
        cx, cy = width // 2, height // 2
        max_r  = int(np.hypot(cx, cy)) + ring_spacing

        for r in range(ring_spacing, max_r, ring_spacing):
            cv2.circle(canvas, (cx, cy), r, line_color, thickness, cv2.LINE_AA)

        # Cross-hair for reference
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (200, 200, 50), 1)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (200, 200, 50), 1)

        return canvas

    @staticmethod
    def color_gradient_chart(
        width: int,
        height: int,
    ) -> ImageArray:
        """
        4-corner colour gradient chart.

        Each corner maps to a primary colour; interpolated across the image.
        Colour shifts after remapping indicate interpolation artefacts.
        """
        canvas = np.zeros((height, width, 3), dtype=np.float32)

        # Corner colours: TL=red, TR=green, BL=blue, BR=yellow
        tl = np.array([0,   0,   200], dtype=np.float32)
        tr = np.array([0,   200, 0  ], dtype=np.float32)
        bl = np.array([200, 0,   0  ], dtype=np.float32)
        br = np.array([200, 200, 0  ], dtype=np.float32)

        for y in range(height):
            fy = y / (height - 1)
            for x in range(width):
                fx = x / (width - 1)
                c = (
                    tl * (1 - fx) * (1 - fy)
                    + tr * fx * (1 - fy)
                    + bl * (1 - fx) * fy
                    + br * fx * fy
                )
                canvas[y, x] = c

        return canvas.astype(np.uint8)

    @staticmethod
    def dot_grid(
        width: int,
        height: int,
        spacing: int = 40,
        radius: int = 4,
        dot_color: Color = (50, 220, 255),
        bg_color: Color = (20, 20, 20),
    ) -> ImageArray:
        """
        Regular dot-grid pattern.

        Easier than grid lines for tracking point displacement:
        each dot corresponds to one control point whose movement
        can be measured precisely.
        """
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                cv2.circle(canvas, (x, y), radius, dot_color, -1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def radial_spokes(
        width: int,
        height: int,
        num_spokes: int = 24,
        line_color: Color = (240, 180, 50),
        bg_color: Color = (15, 15, 30),
        thickness: int = 1,
    ) -> ImageArray:
        """
        Radial spoke pattern from image centre.

        Straight spokes become curved under barrel/pincushion distortion,
        while swirl rotates the spokes — useful for distinguishing distortion types.
        """
        import math
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
        cx, cy = width // 2, height // 2
        r      = int(np.hypot(cx, cy)) + 10

        for i in range(num_spokes):
            angle  = 2 * math.pi * i / num_spokes
            x_end  = int(cx + r * math.cos(angle))
            y_end  = int(cy + r * math.sin(angle))
            cv2.line(canvas, (cx, cy), (x_end, y_end), line_color, thickness, cv2.LINE_AA)

        # Centre dot
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)
        return canvas

    @staticmethod
    def resolution_chart(
        width: int,
        height: int,
    ) -> ImageArray:
        """
        Simplified resolution / calibration chart combining multiple patterns.

        Combines: grid lines, concentric circles, dot corners, and text labels.
        Useful as a single comprehensive test image.
        """
        canvas = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
        cx, cy = width // 2, height // 2

        # Outer border
        cv2.rectangle(canvas, (10, 10), (width - 11, height - 11), (200, 200, 200), 2)

        # Grid lines (coarse)
        spacing = max(width, height) // 8
        for x in range(0, width, spacing):
            cv2.line(canvas, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, spacing):
            cv2.line(canvas, (0, y), (width, y), (60, 60, 60), 1)

        # Concentric circles
        for r in range(spacing, min(cx, cy), spacing):
            cv2.circle(canvas, (cx, cy), r, (80, 120, 80), 1, cv2.LINE_AA)

        # Corner targets
        target_r = 20
        for (tx, ty) in [(40, 40), (width - 40, 40), (40, height - 40), (width - 40, height - 40)]:
            cv2.circle(canvas, (tx, ty), target_r, (200, 80, 80), 2)
            cv2.circle(canvas, (tx, ty), 3, (200, 80, 80), -1)
            cv2.line(canvas, (tx - target_r, ty), (tx + target_r, ty), (200, 80, 80), 1)
            cv2.line(canvas, (tx, ty - target_r), (tx, ty + target_r), (200, 80, 80), 1)

        # Centre cross
        cv2.drawMarker(canvas, (cx, cy), (50, 180, 240),
                       cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)

        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "FIXED GRID REMAP TEST CHART"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.putText(canvas, label, (cx - tw // 2, height - 18),
                    font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        return canvas


# ---------------------------------------------------------------------------
# Convenience map: name → generator callable
# ---------------------------------------------------------------------------
SAMPLE_IMAGES = {
    "checkerboard":         SampleImageGenerator.checkerboard,
    "color_checkerboard":   SampleImageGenerator.color_checkerboard,
    "color_wheel":          SampleImageGenerator.color_wheel,
    "grid_lines":           SampleImageGenerator.grid_lines,
    "circles":              SampleImageGenerator.concentric_circles,
    "color_chart":          SampleImageGenerator.color_gradient_chart,
    "dot_grid":             SampleImageGenerator.dot_grid,
    "radial_spokes":        SampleImageGenerator.radial_spokes,
    "resolution_chart":     SampleImageGenerator.resolution_chart,
}
