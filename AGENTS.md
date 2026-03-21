# Fixed Grid Image Remapping

> **Project type**: Computer Vision / Image Processing  
> **Language**: Python 3.13+  
> **License**: MIT  
> **Author**: Balaji R

---

## Project Overview

This project implements **Fixed Grid Image Remapping** — a computer vision technique that applies sparse displacement grids to deform images using OpenCV's `cv2.remap`. The core algorithm:

1. Defines a sparse grid of displacement vectors (dx, dy) at control points
2. Interpolates the sparse grid to full image resolution using bicubic splines (via SciPy)
3. Builds inverse remap maps for `cv2.remap`
4. Applies the transformation with configurable interpolation and border modes

**Use cases**: Camera distortion correction, lens simulation, fisheye effects, artistic image warping, ISP validation, and geometric calibration workflows.

---

## Technology Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.13+** | Core language (uses `from __future__ import annotations`) |
| **OpenCV (cv2)** | Image I/O, `cv2.remap`, visualization primitives |
| **NumPy** | Array operations, meshgrids, numerical computing |
| **SciPy** | Grid interpolation (`RectBivariateSpline`, `RegularGridInterpolator`) |
| **Matplotlib** | Comparison figures, heatmap visualizations (CLI mode) |
| **Pillow (PIL)** | Tkinter image display support (GUI mode) |
| **Tkinter** | Desktop GUI framework |

---

## Project Structure

```
Image-Remapping/
├── grid_engine.py          # Core remapping engine + distortion presets
├── sample_images.py        # Synthetic test pattern generators
├── main.py                 # CLI application entry point
├── gui_app.py              # Tkinter GUI application
├── files/                  # Sample outputs and reference images
├── output/                 # Default output directory for processed images
└── __pycache__/            # Python bytecode cache
```

### Module Breakdown

#### `grid_engine.py` (~450 lines)
- **`GridRemapEngine`** — Core class with three main methods:
  - `build_remap_maps()` — Interpolates sparse displacement grid to dense per-pixel maps
  - `apply_remap()` — Applies `cv2.remap` with specified interpolation/border modes
  - `overlay_grid()` — Draws deformed grid lines for visual verification
- **`DistortionPresets`** — Factory class providing preset displacement grids:
  - `barrel` / `pincushion` — Radial distortion (k1, k2 coefficients)
  - `swirl` — Vortex rotation effect
  - `wave` — Sinusoidal displacement
  - `fisheye` — Equidistant fisheye projection
  - `pinhole_correction` — Camera model distortion correction
  - `identity` — Pass-through (zero displacement)

#### `sample_images.py` (~275 lines)
- **`SampleImageGenerator`** — Static methods for synthetic test patterns:
  - `checkerboard` — Classic calibration pattern
  - `grid_lines` — Uniform grid for distortion visualization
  - `concentric_circles` — Radial symmetry test
  - `color_gradient_chart` — 4-corner color gradient for interpolation testing
  - `dot_grid` — Point tracking pattern
  - `radial_spokes` — Angular distortion test
  - `resolution_chart` — Combined calibration chart
- **`SAMPLE_IMAGES`** — Dictionary mapping names to generator callables

#### `main.py` (~510 lines)
- CLI argument parsing with `argparse`
- Pipeline orchestration: `load_or_generate_image()` → `build_grid()` → `run_pipeline()`
- Batch processing mode (`--batch`) — runs all transforms × all samples
- Visualization: side-by-side comparison figures with displacement heatmaps

#### `gui_app.py` (~1020 lines)
- **`RemapGUI`** — Main Tkinter application class
- **`LabelledScale`** — Custom slider widget with numeric readout
- **`ImageCanvas`** — Auto-scaling image display component
- Three-tab notebook: Images | Heatmaps | Info
- Live preview with debounced updates (300ms)
- Background threading for non-blocking processing

---

## Running the Application

### Prerequisites

```bash
pip install opencv-python numpy scipy matplotlib Pillow
```

### CLI Mode

```bash
# Basic usage — barrel distortion on checkerboard
python main.py --transform barrel --sample checkerboard --show

# Process external image
python main.py --transform swirl --input photo.jpg --output result.png

# Batch mode — all transforms × all samples
python main.py --batch

# Custom parameters
python main.py --transform wave --grid 15 15 --amplitude 25 --grid-overlay --show
```

**Key CLI flags:**
- `--transform` — `barrel` | `pincushion` | `swirl` | `wave` | `fisheye` | `correction` | `identity`
- `--sample` — `checkerboard` | `grid_lines` | `circles` | `color_chart` | `dot_grid` | `radial_spokes` | `resolution_chart`
- `--grid ROWS COLS` — Sparse grid dimensions (default: 11×11)
- `--interp` — Pixel interpolation: `nearest` | `linear` | `cubic` | `lanczos`
- `--grid-interp` — Grid upsampling: `bicubic` | `linear`
- `--k1`, `--strength`, `--amplitude` — Transform-specific parameters

### GUI Mode

```bash
# Launch GUI with default sample
python gui_app.py

# Launch with external image
python gui_app.py --image path/to/photo.jpg
```

**GUI features:**
- Left panel: Transform selection, grid configuration, parameter sliders
- Right panel: Tabbed view (Images / Heatmaps / Info)
- Live preview toggle with 300ms debounce
- Export all presets to directory
- Save remapped images and comparison figures

---

## Code Style Guidelines

### Naming Conventions
- **Classes**: `PascalCase` (`GridRemapEngine`, `DistortionPresets`)
- **Functions/Methods**: `snake_case` (`build_remap_maps`, `apply_remap`)
- **Constants**: `UPPER_SNAKE_CASE` (`DEFAULT_SIZE`, `_CV2_INTERP`)
- **Private members**: Leading underscore (`_bgr_to_photoimage`, `_on_pipeline_done`)
- **Type aliases**: `CamelCase` suffix with `Array` (`ImageArray`, `DisplacementGrid`)

### Type Annotations
- Full type hints throughout (Python 3.9+ style)
- Use `from __future__ import annotations` for forward references
- Common types: `np.ndarray`, `Tuple[int, int]`, `Optional[str]`

### Documentation Style
- NumPy-style docstrings with Parameters/Returns sections
- Module-level docstrings explaining purpose and usage
- Inline comments for algorithm steps and non-obvious logic

### Code Organization
- Group imports: stdlib → third-party → local modules
- Constants defined near top of files
- Helper classes defined before main classes
- Static methods for stateless operations

---

## Development Workflow

### No Formal Build System
This is a pure Python project with no build step required.

### No Test Suite
The project relies on:
1. **Visual verification** — Use `--show` flag or GUI to inspect results
2. **Sample patterns** — Geometric patterns make distortion artefacts obvious
3. **Batch mode** — `python main.py --batch` generates comprehensive comparison set

### Recommended Validation Workflow
```bash
# 1. Run batch to verify all transforms work
python main.py --batch

# 2. Inspect output/ directory for expected files
ls output/*/

# 3. Test specific transform with grid overlay
python main.py --transform barrel --sample grid_lines --grid-overlay --show
```

---

## Algorithm Notes

### Inverse Mapping Convention
`cv2.remap` uses **inverse mapping** (source coordinates):
```
map_x[dst_y, dst_x] = src_x  # source column to sample
map_y[dst_y, dst_x] = src_y  # source row to sample
```
Therefore: `src = dst + displacement`

### Grid Interpolation
- **Bicubic** (default): `RectBivariateSpline` with kx=ky=3 (clamped to grid size)
- **Linear**: `RegularGridInterpolator` with `method="linear"`

### Coordinate Systems
- Images: `(H, W)` shape, `(row, col)` indexing
- Grid nodes: evenly spaced from `0` to `H-1` / `W-1`
- Displacement units: pixels (float32)

---

## Security Considerations

- **File I/O**: Uses `pathlib.Path` for path handling; no user input directly passed to shell
- **Image loading**: OpenCV's `cv2.imread()` — validates image format via OpenCV
- **No network operations**: Pure offline image processing
- **GUI file dialogs**: Tkinter native dialogs with standard file type filters

---

## Dependencies

Minimal core dependencies:
```
opencv-python
numpy
scipy
matplotlib
Pillow
```

Optional for headless environments:
```
opencv-python-headless
```

---

## Common Tasks

### Add New Distortion Preset
1. Add static method to `DistortionPresets` in `grid_engine.py`
2. Return `(dx_grid, dy_grid)` as `float32` arrays
3. Update `TRANSFORM_PARAMS` in `gui_app.py` if parameters needed
4. Add to `dispatch` dict in `main.py` and `gui_app.py`

### Add New Sample Pattern
1. Add static method to `SampleImageGenerator` in `sample_images.py`
2. Return `uint8` BGR array of requested `(width, height)`
3. Register in `SAMPLE_IMAGES` dictionary

### Adjust Default Parameters
- CLI defaults: `DEFAULT_*` constants in `main.py`
- GUI defaults: Variable initializers in `RemapGUI.__init__()`

---

## File Outputs

- **Remapped images**: `{transform}_{sample}.png` (BGR, via `cv2.imwrite`)
- **Comparison figures**: `{transform}_{sample}_comparison.png` (matplotlib, dark theme)
- **Batch output structure**: `output/{transform}/{sample}.png`
