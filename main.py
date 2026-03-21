#!/usr/bin/env python3
"""
Fixed Grid Image Remapping — CLI Application
=============================================

Applies a sparse fixed-grid displacement field to an input image via cv2.remap.

Usage examples
--------------
# Run all presets on all sample patterns → output/<preset>/<pattern>.png
python main.py --batch

# Single run: barrel distortion on checkerboard (display)
python main.py --transform barrel --sample checkerboard --show

# Process an external image
python main.py --transform swirl --input path/to/photo.jpg --output out.png

# Grid overlay to verify deformation
python main.py --transform wave --sample grid_lines --grid-overlay --show

# Custom grid size and strength
python main.py --transform barrel --sample circles --grid 13 13 --k1 0.6 --show

CLI flags
---------
--transform   : barrel | pincushion | swirl | wave | fisheye | correction | identity
--sample      : checkerboard | grid_lines | circles | color_chart | dot_grid |
                radial_spokes | resolution_chart
--input       : path to input image (overrides --sample)
--output      : output file path (default: output/<transform>_<sample>.png)
--size        : image size  WxH  (default: 640x480)
--grid        : grid size  ROWS COLS  (default: 11 11)
--interp      : remap interpolation — nearest | linear | cubic | lanczos (default: cubic)
--grid-interp : grid upsampling method — bicubic | linear (default: bicubic)
--border      : border mode — constant | replicate | reflect | wrap (default: constant)
--k1          : barrel/pincushion strength (default: 0.4)
--strength    : swirl rotation strength in radians (default: 2.5)
--amplitude   : wave amplitude in pixels (default: 15)
--grid-overlay: overlay deformed grid lines on output image
--batch       : run all presets × all sample patterns and save to output/
--show        : display result using matplotlib (non-blocking)
--quiet       : suppress progress messages

Author: Balaji R
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless-safe default; overridden to TkAgg if --show
import matplotlib.pyplot as plt
import numpy as np

from grid_engine import GridRemapEngine, DistortionPresets
from sample_images import SampleImageGenerator, SAMPLE_IMAGES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SIZE        = (640, 480)          # (W, H)
DEFAULT_GRID        = (11, 11)            # (rows, cols)
DEFAULT_INTERP      = "cubic"
DEFAULT_GRID_INTERP = "bicubic"
DEFAULT_BORDER      = "constant"

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def load_or_generate_image(
    input_path: str | None,
    sample_name: str,
    width: int,
    height: int,
    quiet: bool,
) -> np.ndarray:
    """Return uint8 BGR image from file or generated sample."""
    if input_path:
        img = cv2.imread(input_path)
        if img is None:
            sys.exit(f"[ERROR] Cannot read image: {input_path}")
        img = cv2.resize(img, (width, height))
        if not quiet:
            print(f"  [image]  loaded '{input_path}' → resized to {width}×{height}")
        return img

    generator = SAMPLE_IMAGES.get(sample_name)
    if generator is None:
        sys.exit(f"[ERROR] Unknown sample pattern '{sample_name}'. "
                 f"Choose from: {list(SAMPLE_IMAGES)}")

    img = generator(width, height)
    if not quiet:
        print(f"  [image]  generated '{sample_name}' {width}×{height}")
    return img


def build_grid(
    transform: str,
    image_shape: Tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    params: dict,
    quiet: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (dx_grid, dy_grid) for the requested transform."""
    H, W = image_shape

    dispatch = {
        "barrel":     lambda: DistortionPresets.barrel(
                          (H, W), grid_rows, grid_cols, k1=params["k1"]),
        "pincushion": lambda: DistortionPresets.pincushion(
                          (H, W), grid_rows, grid_cols, k1=params["k1"]),
        "swirl":      lambda: DistortionPresets.swirl(
                          (H, W), grid_rows, grid_cols, strength=params["strength"]),
        "wave":       lambda: DistortionPresets.wave(
                          (H, W), grid_rows, grid_cols,
                          amplitude_x=params["amplitude"],
                          amplitude_y=params["amplitude"] * 0.7),
        "fisheye":    lambda: DistortionPresets.fisheye(
                          (H, W), grid_rows, grid_cols),
        "correction": lambda: DistortionPresets.pinhole_correction(
                          (H, W), grid_rows, grid_cols,
                          k1=params["k1"], k2=params["k1"] * 0.1),
        "identity":   lambda: DistortionPresets.identity(
                          (H, W), grid_rows, grid_cols),
    }

    if transform not in dispatch:
        sys.exit(f"[ERROR] Unknown transform '{transform}'. "
                 f"Choose from: {list(dispatch)}")

    dx_grid, dy_grid = dispatch[transform]()
    if not quiet:
        print(f"  [grid]   transform='{transform}' "
              f"grid={grid_rows}×{grid_cols} "
              f"dx=[{dx_grid.min():.1f}, {dx_grid.max():.1f}] "
              f"dy=[{dy_grid.min():.1f}, {dy_grid.max():.1f}]")
    return dx_grid, dy_grid


def run_pipeline(
    image: np.ndarray,
    transform: str,
    grid_rows: int,
    grid_cols: int,
    params: dict,
    interp: str,
    grid_interp: str,
    border: str,
    grid_overlay: bool,
    quiet: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Execute the full Fixed Grid Remapping pipeline.

    Returns
    -------
    input_image, output_image, overlay_image, info_dict
    """
    H, W = image.shape[:2]
    engine = GridRemapEngine()

    t0 = time.perf_counter()

    # 1. Build sparse displacement grid
    dx_grid, dy_grid = build_grid(
        transform, (H, W), grid_rows, grid_cols, params, quiet
    )

    # 2. Interpolate grid → dense per-pixel maps
    map_x, map_y = engine.build_remap_maps(
        (H, W), dx_grid, dy_grid, grid_interp=grid_interp
    )
    if not quiet:
        print(f"  [maps]   built map_x/map_y ({H}×{W} float32) "
              f"via {grid_interp} interpolation")

    # 3. Apply cv2.remap
    output = engine.apply_remap(image, map_x, map_y,
                                interpolation=interp, border_mode=border)

    t1 = time.perf_counter()
    if not quiet:
        print(f"  [remap]  cv2.remap done in {(t1-t0)*1000:.1f} ms "
              f"(interp={interp}, border={border})")

    # 4. Optional grid overlay
    overlay = engine.overlay_grid(output, grid_rows, grid_cols, map_x, map_y) \
        if grid_overlay else output

    info = {
        "transform":   transform,
        "grid":        f"{grid_rows}×{grid_cols}",
        "grid_interp": grid_interp,
        "remap_interp": interp,
        "border":      border,
        "size":        f"{W}×{H}",
        "elapsed_ms":  (t1 - t0) * 1000,
        "dx_range":    (float(dx_grid.min()), float(dx_grid.max())),
        "dy_range":    (float(dy_grid.min()), float(dy_grid.max())),
    }
    return image, output, overlay, info


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def build_comparison_figure(
    input_img: np.ndarray,
    output_img: np.ndarray,
    overlay_img: np.ndarray,
    dx_grid: np.ndarray,
    dy_grid: np.ndarray,
    info: dict,
    show_overlay: bool,
) -> plt.Figure:
    """
    Build a 2×3 (or 2×2) matplotlib figure with:
      Row 1: original | remapped | (overlay)
      Row 2: dx heatmap | dy heatmap | displacement magnitude
    """
    cols     = 3 if show_overlay else 2
    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 9))
    fig.patch.set_facecolor("#111111")

    def _bgr2rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    title_kw = dict(fontsize=11, color="white", pad=6)
    
    # --- Row 0: images ---
    axes[0, 0].imshow(_bgr2rgb(input_img))
    axes[0, 0].set_title("Original", **title_kw)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_bgr2rgb(output_img))
    axes[0, 1].set_title(f"Remapped  [{info['transform']}]", **title_kw)
    axes[0, 1].axis("off")

    if show_overlay and cols == 3:
        axes[0, 2].imshow(_bgr2rgb(overlay_img))
        axes[0, 2].set_title("Grid Overlay", **title_kw)
        axes[0, 2].axis("off")

    # --- Row 1: displacement heatmaps ---
    im_dx = axes[1, 0].imshow(dx_grid, cmap="RdBu_r", aspect="auto")
    axes[1, 0].set_title(f"ΔX grid  [{dx_grid.min():.1f}, {dx_grid.max():.1f}] px", **title_kw)
    axes[1, 0].axis("off")
    plt.colorbar(im_dx, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_dy = axes[1, 1].imshow(dy_grid, cmap="RdBu_r", aspect="auto")
    axes[1, 1].set_title(f"ΔY grid  [{dy_grid.min():.1f}, {dy_grid.max():.1f}] px", **title_kw)
    axes[1, 1].axis("off")
    plt.colorbar(im_dy, ax=axes[1, 1], fraction=0.046, pad=0.04)

    if cols == 3:
        mag = np.sqrt(dx_grid**2 + dy_grid**2)
        im_m = axes[1, 2].imshow(mag, cmap="plasma", aspect="auto")
        axes[1, 2].set_title(f"Displacement magnitude  max={mag.max():.1f} px", **title_kw)
        axes[1, 2].axis("off")
        plt.colorbar(im_m, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        mag = np.sqrt(dx_grid**2 + dy_grid**2)
        # Re-use spare slot (doesn't exist; skip) — keep 2-col layout clean
        pass

    # Metadata strip
    meta = (
        f"Transform: {info['transform']}  │  Grid: {info['grid']}  │  "
        f"Interp: {info['remap_interp']}  │  Grid-interp: {info['grid_interp']}  │  "
        f"Border: {info['border']}  │  Size: {info['size']}  │  "
        f"Time: {info['elapsed_ms']:.1f} ms"
    )
    fig.text(0.5, 0.01, meta, ha="center", fontsize=8.5, color="#aaaaaa",
             fontfamily="monospace")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#111111")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_batch(args: argparse.Namespace) -> None:
    """Run all transforms × all sample patterns and save to output/."""
    transforms = ["barrel", "pincushion", "swirl", "wave", "fisheye", "correction"]
    samples    = list(SAMPLE_IMAGES.keys())
    W, H       = args.size

    params = {"k1": args.k1, "strength": args.strength, "amplitude": args.amplitude}

    total = len(transforms) * len(samples)
    done  = 0

    for transform in transforms:
        out_dir = OUTPUT_DIR / transform
        out_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            done += 1
            print(f"  [{done:02d}/{total}] {transform:12s} × {sample}")

            image = SAMPLE_IMAGES[sample](W, H)
            inp, outp, overlay, info = run_pipeline(
                image, transform,
                args.grid[0], args.grid[1],
                params, args.interp, args.grid_interp,
                args.border, True, quiet=True,
            )

            # Rebuild grid for heatmap
            dx_grid, dy_grid = build_grid(
                transform, (H, W), args.grid[0], args.grid[1], params, quiet=True
            )

            fig = build_comparison_figure(inp, outp, overlay, dx_grid, dy_grid, info, True)
            save_path = out_dir / f"{sample}.png"
            fig.savefig(save_path, dpi=110, bbox_inches="tight",
                        facecolor="#111111")
            plt.close(fig)

    print(f"\n[batch] Saved {total} comparison images → {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fixed Grid Image Remapping — applies sparse grid displacements via cv2.remap",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Transform selection
    p.add_argument("--transform", "-t",
                   choices=["barrel", "pincushion", "swirl", "wave",
                            "fisheye", "correction", "identity"],
                   default="barrel",
                   help="Distortion preset (default: barrel)")

    # Input
    p.add_argument("--sample", "-s",
                   choices=list(SAMPLE_IMAGES),
                   default="checkerboard",
                   help="Built-in sample pattern (default: checkerboard)")
    p.add_argument("--input", "-i",
                   default=None,
                   help="Path to input image (overrides --sample)")

    # Output
    p.add_argument("--output", "-o",
                   default=None,
                   help="Output image path (default: output/<transform>_<sample>.png)")

    # Image / grid sizing
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"),
                   default=list(DEFAULT_SIZE),
                   help="Image size (default: 640 480)")
    p.add_argument("--grid", nargs=2, type=int, metavar=("ROWS", "COLS"),
                   default=list(DEFAULT_GRID),
                   help="Sparse grid dimensions (default: 11 11)")

    # Interpolation / border
    p.add_argument("--interp",
                   choices=["nearest", "linear", "cubic", "lanczos"],
                   default=DEFAULT_INTERP,
                   help="Pixel interpolation for cv2.remap (default: cubic)")
    p.add_argument("--grid-interp",
                   choices=["bicubic", "linear"],
                   default=DEFAULT_GRID_INTERP,
                   help="Grid upsampling method (default: bicubic)")
    p.add_argument("--border",
                   choices=["constant", "replicate", "reflect", "wrap"],
                   default=DEFAULT_BORDER,
                   help="Border mode for cv2.remap (default: constant)")

    # Transform parameters
    p.add_argument("--k1", type=float, default=0.4,
                   help="Barrel/pincushion radial coefficient (default: 0.4)")
    p.add_argument("--strength", type=float, default=2.5,
                   help="Swirl rotation strength in radians (default: 2.5)")
    p.add_argument("--amplitude", type=float, default=15.0,
                   help="Wave amplitude in pixels (default: 15.0)")

    # Flags
    p.add_argument("--grid-overlay", action="store_true",
                   help="Draw deformed grid lines on output image")
    p.add_argument("--batch", action="store_true",
                   help="Run all transforms × all samples and save to output/")
    p.add_argument("--show", action="store_true",
                   help="Display result with matplotlib")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress messages")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    W, H = args.size
    params = {
        "k1":       args.k1,
        "strength": args.strength,
        "amplitude": args.amplitude,
    }

    if not args.quiet:
        print("=" * 60)
        print("  Fixed Grid Image Remapping")
        print("=" * 60)

    # --- Batch mode ---
    if args.batch:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print("[batch] Running all transforms × sample patterns …\n")
        run_batch(args)
        return

    # --- Single run ---
    image = load_or_generate_image(args.input, args.sample, W, H, args.quiet)
    sample_name = Path(args.input).stem if args.input else args.sample

    inp, outp, overlay, info = run_pipeline(
        image,
        args.transform,
        args.grid[0], args.grid[1],
        params,
        args.interp,
        args.grid_interp,
        args.border,
        args.grid_overlay,
        args.quiet,
    )

    # Rebuild grid for visualisation (cheap, already computed inside run_pipeline)
    dx_grid, dy_grid = build_grid(
        args.transform, (H, W), args.grid[0], args.grid[1], params, quiet=True
    )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{args.transform}_{sample_name}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save remapped image
    cv2.imwrite(str(out_path), overlay)
    if not args.quiet:
        print(f"  [saved]  remapped image → {out_path}")

    # Save comparison figure
    fig_path = out_path.with_name(out_path.stem + "_comparison.png")
    fig = build_comparison_figure(
        inp, outp, overlay, dx_grid, dy_grid, info, args.grid_overlay
    )
    fig.savefig(str(fig_path), dpi=120, bbox_inches="tight", facecolor="#111111")
    if not args.quiet:
        print(f"  [saved]  comparison figure → {fig_path}")

    if args.show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        plt.show()

    plt.close("all")

    if not args.quiet:
        print(f"\n  Transform   : {info['transform']}")
        print(f"  Grid        : {info['grid']} nodes")
        print(f"  Grid interp : {info['grid_interp']}")
        print(f"  Remap interp: {info['remap_interp']}")
        print(f"  Border      : {info['border']}")
        print(f"  Image size  : {info['size']}")
        print(f"  ΔX range    : {info['dx_range'][0]:.1f} … {info['dx_range'][1]:.1f} px")
        print(f"  ΔY range    : {info['dy_range'][0]:.1f} … {info['dy_range'][1]:.1f} px")
        print(f"  Elapsed     : {info['elapsed_ms']:.1f} ms")
        print("=" * 60)


if __name__ == "__main__":
    main()
