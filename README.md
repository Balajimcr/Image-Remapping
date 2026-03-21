# Fixed Grid Image Remapping with Folded Geometric Transforms

A computer vision research tool for evaluating grid-based image remapping pipelines with geometric transform composition.

## Problem Statement

Modern image processing pipelines frequently apply geometric distortions (e.g., barrel, swirl, wave) using dense per-pixel mappings via `cv2.remap`. In many practical workflows, additional geometric transforms such as flips and rotations are applied after remapping as separate post-processing steps.

This leads to two alternative implementations:

### Method A — Sequential Pipeline

```
Sparse Grid → Interpolation → Dense Map → cv2.remap → Flip/Rotate
```

- Remapping is performed first
- Flip/rotation is applied as a separate image-space operation
- Requires multiple passes over the image

### Method B — Grid-Folded Pipeline

```
Sparse Grid → Fold Flip/Rotate into Grid → Interpolation → Dense Map → cv2.remap
```

- Flip/rotation is mathematically folded into the displacement grid
- Entire transformation is executed in a single remap operation
- Eliminates post-processing passes

## Objective

Evaluate whether Method B (Grid-Folded) can:

1. **Produce results equivalent to Method A** within acceptable numerical error
2. **Reduce computational overhead and memory bandwidth**
3. **Provide a more optimal formulation** for real-time or hardware pipelines

## Key Challenge

> Does folding geometric transforms into a sparse displacement grid,
> followed by interpolation, produce the same result as applying those
> transforms after dense remapping?

This is non-trivial because:

- **Interpolation is nonlinear**
- **Transform composition is non-commutative**
- **Grid sampling introduces approximation error**

## Why Method B is Better

### 1. Single-Pass Execution

| Aspect | Method A | Method B |
|--------|----------|----------|
| Operations | Remap → full traversal, Flip/Rotate → second traversal | Only one `cv2.remap` |
| Memory Bandwidth | Higher | Reduced ✔ |
| Cache Locality | Lower | Improved ✔ |

### 2. Hardware Efficiency (ISP / GPU / FPGA)

Method B aligns with how real systems operate:

- Image Signal Processors (ISP)
- GPU fragment shaders
- Hardware remap engines

**Advantages:**
- ✔ One mapping function per pixel
- ✔ Avoids chaining multiple kernels
- ✔ Reduces latency
- ✔ Easier pipeline scheduling

### 3. Mathematical Unification

Method B converts:
```
Post-processing transforms → Coordinate-space transforms
```

This gives:
```
Final mapping = f(remap ∘ transform)
```

- ✔ Cleaner formulation
- ✔ Easier to reason about
- ✔ Extensible to more transforms

### 4. Better Scalability

| Method | New Transform Cost |
|--------|-------------------|
| Method A | New image pass |
| Method B | Composed into grid ✔ |

**Ideal for:**
- Multi-stage pipelines
- Complex warps
- Chained transformations

### 5. Enables Advanced Optimizations

Once everything is in the grid:
- Precompute maps
- Quantize for hardware
- Compress displacement fields
- Use LUT-based pipelines

✔ **Critical for real-time systems**

## Known Trade-Off

Method B introduces **interpolation error** due to:
```
Fold → Interpolate ≠ Interpolate → Fold
```

**Implications:**
- Results are not bit-exact
- Small residual differences appear

**However:**
- ✔ Errors are typically sub-pixel
- ✔ Acceptable for most imaging pipelines

## Evaluation Strategy

To validate correctness, compare outputs using the built-in Comparison tab:

| Metric | Description |
|--------|-------------|
| **Method A Result** | Sequential pipeline output |
| **Method B Result** | Grid-folded pipeline output |
| **Absolute Difference** | `\|A - B\|` heatmap |
| **RMSE** | Root Mean Square Error |
| **Max Error** | Peak absolute difference |

## Features

### Distortion Presets
- **Barrel / Pincushion** - Radial distortion
- **Swirl** - Vortex rotation effect
- **Wave** - Sinusoidal displacement
- **Fisheye** - Equidistant fisheye projection
- **Pinhole Correction** - Camera model correction

### Post-Processing
- **Flip** - Horizontal and/or Vertical
- **Rotate** - 0°, 90°, 180°, 270°
- **ROI** - Region of Interest crop with visual overlay

### Visualization
- **Images Tab** - Original vs Remapped side-by-side
- **Transform Tab** - Before/After flip-rotate comparison
- **Heatmaps Tab** - dX, dY, Magnitude displacement maps
- **Comparison Tab** - Method A vs Method B with residual heatmap
- **Info Tab** - Detailed processing statistics and grid values

## Running the Application

### Prerequisites

```bash
pip install opencv-python numpy scipy matplotlib Pillow
```

### Launch GUI

```bash
# Default launch
python gui_app.py

# With image
python gui_app.py --image path/to/photo.jpg

# Using batch file (Windows)
launch_gui.bat
```

### CLI Mode

```bash
# Basic usage
python main.py --transform barrel --sample checkerboard --show

# Batch mode
python main.py --batch
```

## Project Structure

```
Image-Remapping/
├── gui_app.py          # Tkinter GUI application
├── main.py             # CLI entry point
├── grid_engine.py      # Core remapping engine
├── sample_images.py    # Test pattern generators
├── launch_gui.bat      # Windows launcher
├── def_values_gui.json # Saved GUI settings
├── output/             # Default output directory
└── files/              # Sample/reference images
```

## Conclusion

The Grid-Folded approach (Method B):

- ✅ Reduces computation to a single remap pass
- ✅ Aligns with hardware-efficient architectures
- ✅ Provides a unified mathematical framework

While introducing minor interpolation error, it offers a significantly more scalable and performant design, especially for **real-time and embedded imaging systems**.

## License

MIT License - Author: Balaji R
