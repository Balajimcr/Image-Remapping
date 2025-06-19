# Image Remapping Suite

A comprehensive, modular toolkit for geometric image transformations, lens distortion simulation, and advanced correction algorithms.

## 🚀 Features

### 🔍 **Lens Distortion Simulation & Correction**
- **Brown-Conrady Model**: Complete implementation with radial (K1, K2, K3) and tangential (P1, P2) distortion coefficients
- **Multiple Correction Algorithms**: 
  - Iterative Newton-Raphson (most accurate)
  - Analytical closed-form solution (fastest for simple distortions)
  - Polynomial approximation (balanced speed/accuracy)
- **Quality Assessment**: PSNR, SSIM, correlation, and geometric error metrics

### 🔧 **Geometric Transformations**
- **Affine Transformations**: Translation, rotation, scaling, shearing
- **Projective Transformations**: Full homography support
- **Polynomial Transformations**: Flexible deformation modeling
- **Thin Plate Splines**: Smooth interpolation between control points

### 📊 **Visualization & Analysis**
- **Interactive Grid Visualization**: Real-time distortion mapping
- **Distortion Heatmaps**: Magnitude visualization across image regions
- **Quality Comparison Plots**: Side-by-side method comparisons
- **Vector Field Visualization**: Complete distortion field analysis

### 📁 **Data Export & Hardware Integration**
- **Multiple Export Formats**: CSV, JSON, XML, YAML
- **GDC Format Support**: Hardware-ready fixed-point values for ISP/FPGA
- **Calibration Packages**: Complete parameter sets with metadata
- **Binary Map Export**: Efficient storage for production systems

## 📁 Project Structure

```
image_remapping_suite/
├── main.py                           # Application entry point
├── config/
│   └── settings.py                   # Global configuration and defaults
├── core/
│   ├── remapping_engine.py           # Generic remapping operations
│   └── transform_models.py           # Transformation model implementations
├── lens_distortion/
│   ├── simulator.py                  # Lens distortion simulation
│   └── corrector.py                  # Advanced correction algorithms
├── data_io/
│   ├── image_utils.py                # Image loading/saving utilities
│   └── exporters.py                  # Data export functions
├── visualization/
│   └── visualizer.py                 # Visualization and plotting functions
├── application/
│   └── processor.py                  # Processing orchestration
├── interfaces/
│   ├── gradio_main_interface.py      # Main web interface
│   └── gradio_gdc_utility.py         # GDC grid processing utility
├── utils/
│   └── math_helpers.py               # Mathematical utility functions
├── requirements.txt                  # Python dependencies
└── README.md                         # This documentation
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd image_remapping_suite
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   # Main interface (lens distortion simulation & correction)
   python main.py

   # GDC grid utility
   python main.py --interface gdc
   ```

### Basic Usage

```python
from lens_distortion.simulator import LensDistortionSimulator
from lens_distortion.corrector import LensDistortionCorrector

# Create simulator
simulator = LensDistortionSimulator()
simulator.set_parameters(
    image_width=1280, image_height=720,
    grid_rows=7, grid_cols=9,
    k1=-0.2, k2=0.05  # Barrel distortion
)

# Generate test image and apply distortion
original_image = simulator.create_sample_image("checkerboard")
distorted_image = simulator.apply_distortion_to_image(original_image)

# Correct distortion
corrector = LensDistortionCorrector(simulator)
corrected_image = corrector.correct_image(distorted_image, method="iterative")

# Validate quality
metrics = corrector.validate_correction(original_image, corrected_image)
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

## 🎯 Usage Examples

### Lens Distortion Correction

```python
# Basic barrel distortion
simulator.set_parameters(1920, 1080, 9, 7, k1=-0.2, k2=0.05)

# Complex fisheye distortion  
simulator.set_parameters(1920, 1080, 9, 7, k1=-0.4, k2=0.1, k3=-0.02)

# Pincushion distortion
simulator.set_parameters(1920, 1080, 9, 7, k1=0.15, k2=-0.03)
```

### Geometric Transformations

```python
from core.transform_models import AffineTransform, ProjectiveTransform
from core.remapping_engine import RemappingEngine

# Affine transformation
affine = AffineTransform(rotation=45, scale_x=1.2, scale_y=0.8)
engine = RemappingEngine()

# Generate mapping arrays
map_x, map_y = engine.generate_mapping_vectorized(
    height, width, affine.transform_points
)

# Apply to image
transformed = engine.apply_remapping(image, map_x, map_y)
```

### Data Export

```python
from data_io.exporters import export_calibration_data

# Export complete calibration package
exported_files = export_calibration_data(simulator, corrector, "./output")

# Export specific formats
csv_data = export_grid_to_csv(simulator)
gdc_data = export_gdc_grid_to_csv(simulator, 8192, 6144)
```

## 🔧 Configuration

Edit `config/settings.py` to customize:

- **Default image dimensions and grid sizes**
- **Distortion coefficient ranges and validation**
- **Quality assessment thresholds**
- **Visualization parameters**
- **Export format settings**

## 🌐 Web Interface Features

### Main Interface
- **Real-time parameter adjustment** with immediate visual feedback
- **Multiple test patterns**: checkerboard, grid, circles, text
- **Quality validation** with comprehensive metrics
- **Method comparison** across all available algorithms
- **Custom image processing** for user-uploaded images

### GDC Utility
- **Grid file parsing** with automatic format detection
- **Bicubic interpolation** for resolution enhancement  
- **Multiple export formats** including hardware-ready formats
- **Visual comparison** of original vs. interpolated grids

## 📊 Quality Metrics

The suite provides comprehensive quality assessment:

- **PSNR**: Peak Signal-to-Noise Ratio (>30dB good, >40dB excellent)
- **Correlation**: Image correlation coefficient (>0.95 good, >0.99 excellent)
- **Geometric Error**: Pixel-level accuracy (<0.5px excellent, <1.0px good)
- **MSE/MAE**: Mean squared/absolute error metrics

## 🔬 Algorithm Guide

| Method | Best For | Speed | Accuracy | Use Case |
|--------|----------|-------|----------|----------|
| **Iterative** | Complex distortions | Medium | Excellent | Research, high-quality correction |
| **Analytical** | K1-only distortion | Fast | Good | Real-time applications |
| **Polynomial** | Moderate distortions | Medium | Good | Balanced speed/quality |
| **Original** | Basic correction | Fast | Fair | Visualization, prototyping |

## 🔧 Hardware Integration

### GDC Format
The suite supports GDC (Geometric Distortion Correction) format for hardware implementation:
- **Fixed-point arithmetic** with configurable bit-shifting
- **Memory-optimized** grid representations
- **FPGA/ISP ready** output formats

### Export Options
- **Binary maps** for efficient runtime loading
- **Parameter files** for calibration storage
- **Validation datasets** for quality assurance

## 🧪 Testing & Validation

Run comprehensive tests:

```bash
# Test correction accuracy across methods
python -c "
from application.processor import processor
print(processor.test_correction_accuracy())
"

# Validate correction quality
python -c "
from application.processor import processor  
print(processor.validate_correction_quality('checkerboard', 'iterative'))
"
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the modular architecture
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

### Development Guidelines
- Follow the **modular architecture** with clear separation of concerns
- Use **type hints** and comprehensive docstrings
- Implement **error handling** and input validation
- Add **unit tests** for new algorithms
- Update **configuration** files for new parameters

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Brown-Conrady distortion model** implementation
- **OpenCV** for core image processing operations
- **SciPy** for advanced mathematical functions
- **Gradio** for the interactive web interface
- **NumPy/Matplotlib** for numerical computing and visualization

## 📞 Support

- **Documentation**: Check the comprehensive docstrings in each module
- **Examples**: See the `examples/` directory for usage patterns
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas

---

**Image Remapping Suite** - Professional-grade geometric image transformation toolkit