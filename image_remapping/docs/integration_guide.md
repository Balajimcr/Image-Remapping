# GDC Grid Processing Integration Guide

## Overview

The Image Remapping Suite now includes comprehensive GDC (Geometric Distortion Correction) grid processing capabilities, seamlessly integrated with the existing lens distortion correction toolkit. This guide demonstrates how to use both features independently and together.

## Quick Start

### Launch Interfaces

```bash
# Main lens distortion interface
python main.py --interface main

# GDC grid processing interface
python main.py --interface gdc

# Show all available features
python main.py --features
```

### Basic GDC Processing Workflow

1. **Prepare your GDC file** with format:
   ```
   yuv_gdc_grid_dx_0_0 -48221
   yuv_gdc_grid_dx_0_1 137272
   ...
   yuv_gdc_grid_dy_0_0 5678
   yuv_gdc_grid_dy_0_1 5679
   ...
   ```

2. **Upload and configure** grid dimensions

3. **Process and download** results

## Programming Interface

### Using GDC Processor Directly

```python
from application.processor import GDCGridProcessor
import numpy as np

# Create processor
gdc_processor = GDCGridProcessor()

# Sample GDC file content
file_content = """
yuv_gdc_grid_dx_0_0 -1000
yuv_gdc_grid_dx_0_1 -500
yuv_gdc_grid_dx_0_2 0
yuv_gdc_grid_dx_0_3 500
yuv_gdc_grid_dx_0_4 1000
yuv_gdc_grid_dx_0_5 -800
yuv_gdc_grid_dy_0_0 100
yuv_gdc_grid_dy_0_1 200
yuv_gdc_grid_dy_0_2 300
yuv_gdc_grid_dy_0_3 400
yuv_gdc_grid_dy_0_4 500
yuv_gdc_grid_dy_0_5 600
"""

# Process the grid
try:
    result = gdc_processor.process_file(
        file_content,
        original_shape=(2, 3),  # 2 rows, 3 columns
        target_shape=(6, 9)     # 6 rows, 9 columns
    )
    
    print("Processing successful!")
    print(f"Original shape: {result['metadata']['original_shape']}")
    print(f"Target shape: {result['metadata']['target_shape']}")
    print(f"Files created: {len(result['export'])} export files")
    
    # Access interpolated grids
    dx_interpolated = result['grids']['dx_interpolated']
    dy_interpolated = result['grids']['dy_interpolated']
    
    print(f"DX grid shape: {dx_interpolated.shape}")
    print(f"DY grid shape: {dy_interpolated.shape}")
    
except Exception as e:
    print(f"Processing failed: {e}")
```

### Using Integrated Processor

```python
from application.processor import processor

# The main processor now includes GDC capabilities
print("Processor capabilities:")
print(f"- Lens distortion simulation: ✅")
print(f"- Multiple correction algorithms: ✅")
print(f"- GDC grid processing: ✅")
print(f"- Quality assessment: ✅")

# Access the GDC processor
gdc_processor = processor.gdc_processor

# Use both lens distortion and GDC processing
distortion_summary = processor.get_processing_summary()
print(distortion_summary)
```

## Advanced Usage Examples

### Example 1: Complete Grid Processing Pipeline

```python
import tempfile
import zipfile
from application.processor import GDCGridProcessor

def process_gdc_file_complete(file_path, original_rows, original_cols, target_rows, target_cols):
    """Complete GDC processing pipeline example"""
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Initialize processor
    processor = GDCGridProcessor()
    
    # Process
    result = processor.process_file(
        content,
        original_shape=(original_rows, original_cols),
        target_shape=(target_rows, target_cols)
    )
    
    # Extract results
    stats = result['statistics']
    export_files = result['export']
    
    # Print comprehensive report
    print("=" * 60)
    print("GDC GRID PROCESSING REPORT")
    print("=" * 60)
    
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"Original Grid: {original_rows} x {original_cols}")
    print(f"Target Grid: {target_rows} x {target_cols}")
    print(f"Interpolation Factor: {(target_rows * target_cols) / (original_rows * original_cols):.1f}x")
    
    print(f"\n📈 DX GRID STATISTICS:")
    print(f"Original  - Min: {stats['dx_original']['min']:.3f}, Max: {stats['dx_original']['max']:.3f}")
    print(f"          - Mean: {stats['dx_original']['mean']:.3f}, Std: {stats['dx_original']['std']:.3f}")
    print(f"Interpolated - Min: {stats['dx_interpolated']['min']:.3f}, Max: {stats['dx_interpolated']['max']:.3f}")
    print(f"             - Mean: {stats['dx_interpolated']['mean']:.3f}, Std: {stats['dx_interpolated']['std']:.3f}")
    
    print(f"\n📉 DY GRID STATISTICS:")
    print(f"Original  - Min: {stats['dy_original']['min']:.3f}, Max: {stats['dy_original']['max']:.3f}")
    print(f"          - Mean: {stats['dy_original']['mean']:.3f}, Std: {stats['dy_original']['std']:.3f}")
    print(f"Interpolated - Min: {stats['dy_interpolated']['min']:.3f}, Max: {stats['dy_interpolated']['max']:.3f}")
    print(f"             - Mean: {stats['dy_interpolated']['mean']:.3f}, Std: {stats['dy_interpolated']['std']:.3f}")
    
    print(f"\n📁 EXPORTED FILES:")
    for file_type, file_path in export_files.items():
        if file_path and 'zip' not in file_type:
            print(f"  • {file_type}: {file_path.split('/')[-1]}")
    
    print(f"\n📦 Complete package: {export_files.get('zip_path', 'Not available')}")
    
    return result

# Example usage
# process_gdc_file_complete('sample_gdc.txt', 9, 7, 33, 33)
```

### Example 2: Quality Assessment and Validation

```python
def assess_interpolation_quality(original_grid, interpolated_grid):
    """Assess the quality of grid interpolation"""
    import numpy as np
    from scipy import ndimage
    
    # Resize original to match interpolated for comparison
    zoom_factors = (
        interpolated_grid.shape[0] / original_grid.shape[0],
        interpolated_grid.shape[1] / original_grid.shape[1]
    )
    
    # Simple nearest-neighbor upsampling for comparison
    nearest_upsampled = ndimage.zoom(original_grid, zoom_factors, order=0)
    
    # Calculate quality metrics
    mse_bicubic = np.mean((interpolated_grid - nearest_upsampled) ** 2)
    
    # Smoothness metric (gradient variance)
    grad_x = np.gradient(interpolated_grid, axis=1)
    grad_y = np.gradient(interpolated_grid, axis=0)
    smoothness = np.var(grad_x) + np.var(grad_y)
    
    print("INTERPOLATION QUALITY ASSESSMENT:")
    print(f"MSE vs Nearest Neighbor: {mse_bicubic:.3f}")
    print(f"Gradient Variance (smoothness): {smoothness:.3f}")
    print(f"Original shape: {original_grid.shape}")
    print(f"Interpolated shape: {interpolated_grid.shape}")
    
    return {
        'mse': mse_bicubic,
        'smoothness': smoothness,
        'zoom_factors': zoom_factors
    }

# Usage with GDC processor results
# quality_dx = assess_interpolation_quality(
#     result['grids']['dx_original'],
#     result['grids']['dx_interpolated']
# )
```

### Example 3: Custom Export Formats

```python
def export_custom_format(result, output_format='matlab'):
    """Export grid data in custom formats"""
    import json
    import scipy.io
    
    dx_interpolated = result['grids']['dx_interpolated']
    dy_interpolated = result['grids']['dy_interpolated']
    metadata = result['metadata']
    
    if output_format == 'matlab':
        # Export as MATLAB .mat file
        matlab_data = {
            'dx_grid': dx_interpolated,
            'dy_grid': dy_interpolated,
            'original_shape': metadata['original_shape'],
            'target_shape': metadata['target_shape'],
            'interpolation_method': metadata['interpolation_method']
        }
        
        scipy.io.savemat('gdc_grids.mat', matlab_data)
        print("✅ Exported to MATLAB format: gdc_grids.mat")
        
    elif output_format == 'json':
        # Export as JSON (with lists instead of numpy arrays)
        json_data = {
            'dx_grid': dx_interpolated.tolist(),
            'dy_grid': dy_interpolated.tolist(),
            'metadata': metadata,
            'statistics': {
                key: {k: float(v) for k, v in stats.items()}
                for key, stats in result['statistics'].items()
            }
        }
        
        with open('gdc_grids.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print("✅ Exported to JSON format: gdc_grids.json")
        
    elif output_format == 'numpy':
        # Export as NumPy binary files
        np.save('dx_grid.npy', dx_interpolated)
        np.save('dy_grid.npy', dy_interpolated)
        
        # Save metadata separately
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Exported to NumPy format: dx_grid.npy, dy_grid.npy, metadata.json")

# Usage
# export_custom_format(result, 'matlab')
# export_custom_format(result, 'json')
# export_custom_format(result, 'numpy')
```

## Integration with Lens Distortion Correction

### Example 4: Combined Workflow

```python
def combined_distortion_gdc_workflow():
    """Demonstrate combined lens distortion and GDC processing"""
    from application.processor import processor
    
    # 1. Set up lens distortion parameters
    processor.simulator.set_parameters(
        image_width=1920, image_height=1080,
        grid_rows=9, grid_cols=7,
        k1=-0.3, k2=0.1, k3=-0.02,  # Strong barrel distortion
        p1=0.01, p2=0.005           # Slight tangential distortion
    )
    
    # 2. Generate distortion grid and export
    source_coords, target_coords = processor.simulator.generate_sparse_distortion_grid()
    
    # 3. Create GDC-compatible data from distortion simulation
    displacement_x = source_coords[:, :, 0] - target_coords[:, :, 0]
    displacement_y = source_coords[:, :, 1] - target_coords[:, :, 1]
    
    # 4. Format as GDC data
    gdc_content = []
    rows, cols = displacement_x.shape
    
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            dx_value = int(displacement_x[row, col] * 1000)  # Scale and convert to int
            dy_value = int(displacement_y[row, col] * 1000)
            
            gdc_content.append(f"yuv_gdc_grid_dx_0_{index} {dx_value}")
            gdc_content.append(f"yuv_gdc_grid_dy_0_{index} {dy_value}")
    
    gdc_file_content = "\n".join(gdc_content)
    
    # 5. Process through GDC pipeline
    gdc_result = processor.gdc_processor.process_file(
        gdc_file_content,
        original_shape=(rows, cols),
        target_shape=(rows * 3, cols * 3)  # 3x interpolation
    )
    
    print("🔄 COMBINED WORKFLOW COMPLETE")
    print(f"✅ Lens distortion simulation: {rows}x{cols} grid")
    print(f"✅ GDC interpolation: {gdc_result['metadata']['target_shape']} grid")
    print(f"✅ Files exported: {len(gdc_result['export'])} formats")
    
    return gdc_result

# Run combined workflow
# combined_result = combined_distortion_gdc_workflow()
```

## Batch Processing

### Example 5: Process Multiple GDC Files

```python
import os
import glob
from pathlib import Path

def batch_process_gdc_files(input_directory, output_directory, config):
    """Process multiple GDC files in batch"""
    
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Find all GDC files
    gdc_files = glob.glob(os.path.join(input_directory, "*.txt"))
    gdc_files.extend(glob.glob(os.path.join(input_directory, "*.dat")))
    
    processor = GDCGridProcessor()
    results = []
    
    print(f"🔄 Processing {len(gdc_files)} GDC files...")
    
    for i, file_path in enumerate(gdc_files, 1):
        try:
            print(f"[{i}/{len(gdc_files)}] Processing: {os.path.basename(file_path)}")
            
            # Read file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Process
            result = processor.process_file(
                content,
                original_shape=config['original_shape'],
                target_shape=config['target_shape']
            )
            
            # Move output files to organized directory
            file_base = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_directory, file_base)
            Path(file_output_dir).mkdir(exist_ok=True)
            
            # Copy zip file to output directory
            if 'zip_path' in result['export']:
                import shutil
                new_zip_path = os.path.join(file_output_dir, f"{file_base}_processed.zip")
                shutil.copy2(result['export']['zip_path'], new_zip_path)
            
            results.append({
                'file': file_path,
                'status': 'success',
                'output_dir': file_output_dir,
                'statistics': result['statistics']
            })
            
            print(f"  ✅ Success: {file_base}")
            
        except Exception as e:
            print(f"  ❌ Failed: {os.path.basename(file_path)} - {e}")
            results.append({
                'file': file_path,
                'status': 'failed',
                'error': str(e)
            })
    
    # Generate batch report
    generate_batch_report(results, output_directory)
    
    return results

def generate_batch_report(results, output_directory):
    """Generate a comprehensive batch processing report"""
    import json
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    report = {
        'summary': {
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0
        },
        'successful_files': successful,
        'failed_files': failed
    }
    
    # Save report
    report_path = os.path.join(output_directory, 'batch_processing_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 BATCH PROCESSING REPORT")
    print(f"Total files: {report['summary']['total_files']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Report saved: {report_path}")

# Example usage
# config = {
#     'original_shape': (9, 7),
#     'target_shape': (33, 33)
# }
# batch_process_gdc_files('./input_gdc_files', './processed_output', config)
```

## Error Handling and Debugging

### Example 6: Robust Error Handling

```python
def robust_gdc_processing(file_content, original_shape, target_shape):
    """GDC processing with comprehensive error handling"""
    
    try:
        processor = GDCGridProcessor()
        
        # Validate inputs
        if not file_content or not file_content.strip():
            raise ValueError("File content is empty")
        
        if any(dim <= 0 for dim in original_shape + target_shape):
            raise ValueError("Grid dimensions must be positive")
        
        # Check interpolation factor
        interpolation_factor = (target_shape[0] * target_shape[1]) / (original_shape[0] * original_shape[1])
        if interpolation_factor > 100:
            raise ValueError(f"Interpolation factor too high: {interpolation_factor:.1f}x")
        
        # Process with error recovery
        result = processor.process_file(file_content, original_shape, target_shape)
        
        # Validate results
        if not result or 'grids' not in result:
            raise RuntimeError("Processing failed to produce valid results")
        
        # Check output grid shapes
        expected_shape = target_shape
        actual_dx_shape = result['grids']['dx_interpolated'].shape
        actual_dy_shape = result['grids']['dy_interpolated'].shape
        
        if actual_dx_shape != expected_shape or actual_dy_shape != expected_shape:
            raise RuntimeError(f"Output shape mismatch: expected {expected_shape}, got DX={actual_dx_shape}, DY={actual_dy_shape}")
        
        print("✅ Processing completed successfully with validation")
        return result
        
    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        return None
        
    except RuntimeError as e:
        print(f"❌ Processing error: {e}")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage with error handling
# result = robust_gdc_processing(file_content, (9, 7), (33, 33))
# if result:
#     print("Processing successful!")
# else:
#     print("Processing failed, check errors above")
```

## Performance Optimization

### Example 7: Memory-Efficient Processing

```python
def memory_efficient_gdc_processing(file_path, original_shape, target_shape, chunk_size=1000):
    """Memory-efficient processing for large grids"""
    
    processor = GDCGridProcessor()
    
    # Check if we need chunked processing
    total_elements = target_shape[0] * target_shape[1]
    
    if total_elements > chunk_size:
        print(f"⚠️  Large grid detected ({total_elements} elements)")
        print(f"🔧 Using memory-efficient processing...")
        
        # For very large grids, you might want to:
        # 1. Process in chunks
        # 2. Use memory mapping
        # 3. Use sparse representations
        # 4. Stream processing
        
        # This is a simplified example - in practice you'd implement
        # actual chunked processing based on your memory constraints
        
    # Read file in chunks if it's very large
    file_size = os.path.getsize(file_path)
    if file_size > 100 * 1024 * 1024:  # 100MB
        print(f"⚠️  Large file detected ({file_size / 1024 / 1024:.1f}MB)")
        print(f"🔧 Using streaming file reader...")
        
        # Implement streaming file reading here
        # This is a placeholder for actual streaming implementation
        
    # Regular processing for manageable sizes
    with open(file_path, 'r') as f:
        content = f.read()
    
    result = processor.process_file(content, original_shape, target_shape)
    
    print(f"✅ Memory-efficient processing completed")
    return result

# Monitor memory usage
def monitor_memory_usage():
    """Monitor memory usage during processing"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"💾 Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"💾 Virtual memory: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    return memory_info

# Usage
# monitor_memory_usage()
# result = memory_efficient_gdc_processing('large_gdc_file.txt', (100, 100), (500, 500))
# monitor_memory_usage()
```

## Testing and Validation

### Example 8: Comprehensive Testing

```python
def test_gdc_integration():
    """Comprehensive test suite for GDC integration"""
    
    print("🧪 Running GDC Integration Tests...")
    
    # Test 1: Basic functionality
    print("\n1️⃣ Testing basic functionality...")
    try:
        processor = GDCGridProcessor()
        
        # Create minimal test data
        test_content = """
yuv_gdc_grid_dx_0_0 100
yuv_gdc_grid_dx_0_1 200
yuv_gdc_grid_dy_0_0 300
yuv_gdc_grid_dy_0_1 400
"""
        
        result = processor.process_file(test_content, (1, 2), (2, 4))
        assert result is not None
        assert 'grids' in result
        assert result['grids']['dx_interpolated'].shape == (2, 4)
        print("   ✅ Basic functionality test passed")
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False
    
    # Test 2: Error handling
    print("\n2️⃣ Testing error handling...")
    try:
        # Test empty content
        try:
            processor.process_file("", (1, 1), (2, 2))
            print("   ❌ Error handling test failed: should have raised exception")
            return False
        except ValueError:
            print("   ✅ Empty content error handling passed")
        
        # Test invalid dimensions
        try:
            processor.process_file(test_content, (0, 1), (2, 2))
            print("   ❌ Error handling test failed: should have raised exception")
            return False
        except (ValueError, ZeroDivisionError):
            print("   ✅ Invalid dimensions error handling passed")
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # Test 3: Integration with main processor
    print("\n3️⃣ Testing integration with main processor...")
    try:
        from application.processor import processor as main_processor
        
        # Verify GDC processor is available
        assert hasattr(main_processor, 'gdc_processor')
        assert main_processor.gdc_processor is not None
        
        # Test summary includes GDC capabilities
        summary = main_processor.get_processing_summary()
        assert 'GDC' in summary
        
        print("   ✅ Integration test passed")
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False
    
    # Test 4: Performance benchmark
    print("\n4️⃣ Running performance benchmark...")
    try:
        import time
        
        # Create larger test data
        large_test_content = []
        for i in range(100):  # 10x10 grid
            large_test_content.append(f"yuv_gdc_grid_dx_0_{i} {i}")
            large_test_content.append(f"yuv_gdc_grid_dy_0_{i} {i + 1000}")
        
        large_content = "\n".join(large_test_content)
        
        start_time = time.time()
        result = processor.process_file(large_content, (10, 10), (30, 30))
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"   ✅ Performance test: {processing_time:.2f} seconds for 10x10 → 30x30 interpolation")
        
        if processing_time > 10:  # More than 10 seconds is concerning
            print("   ⚠️  Performance warning: processing took longer than expected")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! GDC integration is working correctly.")
    return True

# Run tests
# test_gdc_integration()
```

This comprehensive integration guide demonstrates how to use the new GDC grid processing capabilities both standalone and integrated with the existing lens distortion correction toolkit. The examples cover basic usage, advanced workflows, error handling, and performance optimization.