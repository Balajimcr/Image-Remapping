# Image Remapping Suite - Agent Guide

A comprehensive, modular toolkit for geometric image transformations, lens distortion simulation, and GDC (Geometric Distortion Correction) grid processing.

**Version:** 2.0.0  
**Author:** Balaji R  
**License:** MIT  
**Python Required:** 3.8+

---

## Project Overview

This project provides professional-grade tools for:

1. **Lens Distortion Simulation & Correction**: Brown-Conrady model implementation with radial (K1, K2, K3) and tangential (P1, P2) distortion coefficients
2. **GDC Grid Processing**: Advanced bicubic interpolation of geometric distortion correction grids for hardware implementation
3. **Geometric Transformations**: Affine, projective, and polynomial transformations
4. **Quality Assessment**: PSNR, SSIM, correlation, and geometric error metrics
5. **Hardware Integration**: Export to FPGA/ISP-ready formats

---

## Project Structure

```
image_remapping_suite/
├── main.py                           # Application entry point (from project root)
├── Readme.md                         # Main documentation
├── requirements.txt                  # Python dependencies
│
├── image_remapping/                  # Main package directory
│   ├── main.py                       # Launcher with CLI args (--interface main/gdc/integrated)
│   ├── Readme.md                     # Package-level documentation
│   ├── requirements.txt              # Package dependencies
│   ├── Test.py                       # Quick test script
│   │
│   ├── config/
│   │   └── settings.py               # Global configuration, defaults, validation ranges
│   │
│   ├── core/                         # Core transformation engine
│   │   ├── remapping_engine.py       # Generic remapping operations using OpenCV
│   │   └── transform_models.py       # Affine, projective, radial distortion transforms
│   │
│   ├── lens_distortion/              # Lens distortion module
│   │   ├── simulator.py              # Brown-Conrady distortion simulation
│   │   └── corrector.py              # Correction algorithms (iterative, analytical, polynomial)
│   │
│   ├── data_io/                      # Data I/O operations
│   │   ├── exporters.py              # CSV, GDC, JSON, XML export functions
│   │   └── image_utils.py            # Image loading/saving utilities
│   │
│   ├── visualization/                # Visualization components
│   │   └── visualizer.py             # Grid visualization, heatmaps, comparison plots
│   │
│   ├── application/                  # Application orchestration
│   │   └── processor.py              # Main processor with GDCGridProcessor class
│   │
│   ├── interfaces/                   # Web interfaces (Gradio)
│   │   ├── gradio_main.py            # Main lens distortion interface (3-tab design)
│   │   └── gradio_gdc.py             # GDC grid processing interface
│   │
│   ├── utils/                        # Utility modules
│   │   ├── math_helpers.py           # Mathematical utility functions
│   │   ├── gdc_grid.py               # GDC grid utilities
│   │   ├── gdc_hex_converter.py      # Hex conversion utilities
│   │   └── distortion_visualizer.py  # Distortion visualization tools
│   │
│   ├── tests/                        # Testing modules
│   │   └── standalone_test_console.py # Interactive test console
│   │
│   ├── docs/                         # Documentation
│   │   ├── integration_guide.md      # GDC integration examples and API guide
│   │   └── Project Structure.pdf     # Architecture documentation
│   │
│   ├── Data/                         # Sample data
│   │   ├── gdc_grid.txt              # Sample GDC grid file
│   │   └── gdc_grid_data_7x9_Grid.csv # Sample grid data
│   │
│   ├── debug/                        # Debug output images
│   └── debug_images/                 # Additional debug images
│
└── scripts/
    ├── launch.bat                    # Main launcher with interactive menu
    ├── launch-main.bat               # Quick launch: Main Lens Distortion
    ├── launch-gdc.bat                # Quick launch: GDC Grid Processing
    └── launch-integrated.bat         # Quick launch: Integrated Interface
    
└── legacy/                         # Legacy standalone GDC files (optional)
    └── gdc_*.py files
```

---

## Technology Stack

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical computing |
| opencv-python | >=4.5.0 | Image processing |
| scipy | >=1.7.0 | Interpolation, optimization |
| pillow | >=8.0.0 | Image I/O |

### Web Interface
| Package | Version | Purpose |
|---------|---------|---------|
| gradio | >=4.0.0 | Web UI framework |

### Visualization & Analysis
| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >=3.5.0 | Plotting and visualization |
| seaborn | >=0.11.0 | Statistical visualization |
| pandas | >=1.3.0 | Data processing |
| scikit-image | >=0.19.0 | Advanced image processing |

---

## Build and Run Commands

### Installation
```bash
# From project root
pip install -r requirements.txt
```

### Running the Application

#### Main Lens Distortion Interface
```bash
cd image_remapping
python main.py
# OR
python main.py --interface main
```

#### GDC Grid Processing Interface
```bash
cd image_remapping
python main.py --interface gdc
```

#### Integrated Interface (both capabilities)
```bash
cd image_remapping
python main.py --interface integrated
```

#### Standalone GDC Interface (alternative)
```bash
cd image_remapping
python gdc_main.py
```

### CLI Options
```bash
python main.py --help          # Show all options
python main.py --features      # Show feature overview
python main.py --check         # Check dependencies
python main.py --version       # Show version
```

### Testing
```bash
# Interactive test console
cd image_remapping/tests
python standalone_test_console.py

# Quick test
python standalone_test_console.py --quick

# Stress test
python standalone_test_console.py --stress

# Method comparison
python standalone_test_console.py --compare
```

---

## Code Style Guidelines

### Python Style
- **Type hints**: Use type annotations for function parameters and return values
- **Docstrings**: Use triple-double-quote docstrings for all public functions and classes
- **Imports**: Group imports in order: stdlib, third-party, local modules
- **Constants**: Use UPPER_SNAKE_CASE for module-level constants (defined in `config/settings.py`)
- **Classes**: Use PascalCase for class names
- **Functions/Variables**: Use snake_case for functions and variables

### Architecture Patterns
- **Modular design**: Each module has a single responsibility
- **Separation of concerns**: Core logic separate from UI, data I/O separate from processing
- **Configuration centralization**: All defaults and validation ranges in `config/settings.py`
- **Error handling**: Use try-except with specific exception types; log warnings but continue processing when possible

### Example Code Pattern
```python
from typing import Tuple, Optional, Dict, Any
import numpy as np

class MyProcessor:
    """
    Brief description of the class.
    
    Longer description with usage examples if needed.
    """
    
    def __init__(self):
        self.cache = {}
        
    def process_data(self, data: np.ndarray, param: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process the input data.
        
        Args:
            data: Input array to process
            param: Processing parameter
            
        Returns:
            Tuple of (processed_array, metadata_dict)
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None or data.size == 0:
            raise ValueError("Data cannot be empty")
        
        # Processing logic here
        result = data * param
        
        metadata = {
            'input_shape': data.shape,
            'param_used': param
        }
        
        return result, metadata
```

---

## Testing Instructions

### Running Tests

1. **Interactive Test Console** (recommended for development):
   ```bash
   cd image_remapping/tests
   python standalone_test_console.py
   ```
   Menu options:
   - Quick Validation Test - Standard barrel distortion
   - Stress Test - Extreme parameters
   - Method Comparison - Compare algorithms
   - Custom Test - User-defined parameters

2. **Programmatic Testing**:
   ```python
   from application.processor import processor
   
   # Test correction accuracy
   print(processor.test_correction_accuracy())
   
   # Validate correction quality
   print(processor.validate_correction_quality('checkerboard', 'iterative'))
   ```

### Quality Metrics Thresholds
| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| PSNR | >40 dB | >30 dB | >25 dB | <25 dB |
| Correlation | >0.99 | >0.95 | >0.90 | <0.90 |
| Geometric Error | <0.5 px | <1.0 px | <2.0 px | >2.0 px |
| Round-trip Error | <0.1 px | <0.5 px | <1.0 px | >1.0 px |

### Test Grade Scale
- **A** (<0.1 px): Excellent
- **B** (<0.5 px): Good  
- **C** (<1.0 px): Fair
- **D** (>1.0 px): Poor

---

## Key Configuration Values

Located in `image_remapping/config/settings.py`:

### Default Dimensions
- Image: 1280x720
- Grid: 7 rows x 9 columns
- GDC: 8192x6144

### Distortion Coefficient Ranges (Brown-Conrady)
- K1: [-0.5, 0.5]
- K2: [-0.2, 0.2]
- K3: [-0.1, 0.1]
- P1, P2: [-0.1, 0.1]

### Default Coefficients
- K1: -0.2 (barrel distortion)
- K2: 0.05
- K3: 0.0
- P1, P2: 0.0

### Algorithm Parameters
- Max iterations: 10
- Convergence tolerance: 1e-6
- Default interpolation: linear

---

## Algorithm Guide

| Method | Best For | Speed | Accuracy | Use Case |
|--------|----------|-------|----------|----------|
| **Iterative** | Complex distortions | Medium | Excellent | Research, high-quality correction |
| **Analytical** | K1-only distortion | Fast | Good | Real-time applications |
| **Polynomial** | Moderate distortions | Medium | Good | Balanced speed/quality |
| **Original** | Basic correction | Fast | Fair | Visualization, prototyping |

---

## GDC Data Format

GDC files use the following format:

```
yuv_gdc_grid_dx_0_0 -48221    # X displacement for element 0
yuv_gdc_grid_dx_0_1 137272    # X displacement for element 1
...
yuv_gdc_grid_dy_0_0 5678      # Y displacement for element 0
yuv_gdc_grid_dy_0_1 5679      # Y displacement for element 1
...
```

Each line contains: `<element_name> <integer_value>`

- Element naming: `yuv_gdc_grid_{dx|dy}_0_{index}`
- Index is zero-based and sequential
- Values are typically integers representing fixed-point displacements

---

## Security Considerations

### File Upload Restrictions
- Max upload size: 50 MB
- Allowed extensions: .txt, .csv, .dat
- Max lines per file: 1,000,000
- Content scanning enabled

### Input Validation
- All grid dimensions validated against ranges in settings.py
- Coefficient values checked against VALIDATION_RANGES
- File size limits enforced before processing

---

## Development Workflow

1. **Make changes** to relevant module in appropriate subdirectory
2. **Update configuration** in `config/settings.py` if adding new parameters
3. **Test changes** using the test console or validation functions
4. **Update docstrings** and documentation as needed
5. **Follow the modular architecture** - maintain separation of concerns

---

## Common Issues and Solutions

### Import Errors
```python
# Add project root to path if needed
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### Gradio Port Conflicts
```bash
# Specify custom port
python main.py --port 7861
```

### Memory Issues with Large Grids
- Reduce grid dimensions in processing
- Use chunked processing for very large files
- Monitor memory usage with `psutil` if available

---

## Additional Resources

- **Integration Guide**: `image_remapping/docs/integration_guide.md`
- **Sample Data**: `image_remapping/Data/`
- **Main README**: `Readme.md` (project root)
