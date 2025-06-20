"""
Application processing orchestrator with integrated GDC Grid Processing

This module coordinates the overall distortion/correction pipeline and includes
comprehensive GDC grid processing capabilities.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from lens_distortion.simulator import LensDistortionSimulator
from lens_distortion.corrector import LensDistortionCorrector
from visualization.visualizer import (
    visualize_distortion_grid, create_distortion_heatmap,
    create_correction_comparison, create_quality_metrics_plot,
    create_method_comparison_plot
)
from data_io.exporters import (
    export_grid_to_csv, export_gdc_grid_to_csv,
    export_distortion_parameters
)

# Import the new GDC Grid Processor
import csv
import re
import warnings
import os
import tempfile
import zipfile
import json
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns

from utils.math_helpers import compute_grid_statistics
from config.settings import (
    CSV_DECIMAL_PRECISION, GRID_DEFAULTS, GDC_PARSING_CONFIG,
    EXPORT_FORMATS, VISUALIZATION_CONFIG
)


class GDCGridProcessor:
    """
    Advanced GDC grid processor with interpolation and export capabilities
    """
    
    def __init__(self):
        self.parsed_data = []
        self.dx_values = []
        self.dy_values = []
        self.original_shape = None
        self.target_shape = None
        self.dx_grid_2d = None
        self.dy_grid_2d = None
        self.dx_interpolated = None
        self.dy_interpolated = None
        
    def parse_grid_data_from_content(self, file_content: str) -> List[Tuple[str, int]]:
        """Parse grid data from file content string."""
        parsed_data_ordered = []
        lines = file_content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if line.strip():
                try:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        element_name = parts[0].strip()
                        value = int(parts[1].strip())
                        parsed_data_ordered.append((element_name, value))
                    else:
                        print(f"Warning: Skipping malformed line {line_num}: '{line.strip()}'")
                except ValueError:
                    print(f"Warning: Could not convert value to integer for line {line_num}: '{line.strip()}'")
                except Exception as e:
                    print(f"An unexpected error occurred while parsing line {line_num}: {e}")
        
        self.parsed_data = parsed_data_ordered
        return parsed_data_ordered
    
    def extract_and_sort_grid_values(self) -> Tuple[List[int], List[int]]:
        """Extract and sort DX and DY values from parsed data."""
        index_pattern = re.compile(GDC_PARSING_CONFIG['element_pattern'])
        
        dx_elements = []
        dy_elements = []
        
        for name, value in self.parsed_data:
            match = index_pattern.search(name)
            if match:
                index = int(match.group(2))
                if "yuv_gdc_grid_dx" in name:
                    dx_elements.append((index, value))
                elif "yuv_gdc_grid_dy" in name:
                    dy_elements.append((index, value))
        
        dx_elements.sort(key=lambda x: x[0])
        dy_elements.sort(key=lambda x: x[0])
        
        self.dx_values = [item[1] for item in dx_elements]
        self.dy_values = [item[1] for item in dy_elements]
        
        return self.dx_values, self.dy_values
    
    def interpolate_grid_bicubic(self, grid_2d: np.ndarray, 
                                target_rows: int, target_cols: int) -> np.ndarray:
        """Interpolate 2D grid using bicubic interpolation."""
        original_rows, original_cols = grid_2d.shape
        
        x_orig = np.linspace(0, 1, original_cols)
        y_orig = np.linspace(0, 1, original_rows)
        x_target = np.linspace(0, 1, target_cols)
        y_target = np.linspace(0, 1, target_rows)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = RectBivariateSpline(y_orig, x_orig, grid_2d, kx=3, ky=3, s=0)
            interpolated_grid = spline(y_target, x_target)
        
        return interpolated_grid
    
    def create_visualization(self, grid_2d: np.ndarray, title: str, 
                           colormap: str = 'viridis') -> str:
        """Create a visualization of the grid data."""
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        sns.heatmap(grid_2d, annot=False, cmap=colormap, cbar=True)
        plt.title(title)
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=VISUALIZATION_CONFIG['figure_dpi'], 
                   bbox_inches=VISUALIZATION_CONFIG['bbox_inches'])
        plt.close()
        
        return temp_file.name
    
    def create_csv_file(self, grid_2d: np.ndarray, filename: str) -> str:
        """Create a CSV file from 2D grid data."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv', 
            prefix=filename.replace('.csv', '_')
        )
        
        writer = csv.writer(temp_file, delimiter=EXPORT_FORMATS['csv']['delimiter'])
        for row in grid_2d:
            rounded_row = [round(val, EXPORT_FORMATS['csv']['decimal_places']) for val in row]
            writer.writerow(rounded_row)
        
        temp_file.close()
        return temp_file.name
    
    def create_gdc_format_file(self, grid_2d: np.ndarray, grid_type: str, 
                              original_shape: Tuple[int, int]) -> str:
        """Create GDC format file with proper field names."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt',
            prefix=f"gdc_{grid_type}_"
        )
        
        rows, cols = grid_2d.shape
        naming_template = EXPORT_FORMATS['gdc']['naming_convention']
        
        with open(temp_file.name, 'w') as f:
            for row in range(rows):
                for col in range(cols):
                    value = int(round(grid_2d[row, col]))
                    element_name = naming_template.format(type=grid_type, index=row * cols + col)
                    f.write(f"{element_name} {value}\n")
        
        return temp_file.name
    
    def create_zip_file(self, file_paths: List[str], zip_name: str = "gdc_output") -> str:
        """Create a zip file containing all specified files."""
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    base_name = os.path.basename(file_path)
                    zipf.write(file_path, base_name)
        
        temp_zip.close()
        return temp_zip.name
    
    def create_grid_comparison_plot(self, original_grid: np.ndarray, 
                                  interpolated_grid: np.ndarray, title: str) -> str:
        """Create a side-by-side comparison plot of original and interpolated grids."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VISUALIZATION_CONFIG['comparison_figure_size'])
        
        # Original grid
        im1 = ax1.imshow(original_grid, cmap=VISUALIZATION_CONFIG['comparison_colormap'], aspect='auto')
        ax1.set_title(f'Original Grid ({original_grid.shape[0]}x{original_grid.shape[1]})')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
        plt.colorbar(im1, ax=ax1)
        
        # Interpolated grid
        im2 = ax2.imshow(interpolated_grid, cmap=VISUALIZATION_CONFIG['comparison_colormap'], aspect='auto')
        ax2.set_title(f'Interpolated Grid ({interpolated_grid.shape[0]}x{interpolated_grid.shape[1]})')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        plt.colorbar(im2, ax=ax2)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=VISUALIZATION_CONFIG['figure_dpi'], 
                   bbox_inches=VISUALIZATION_CONFIG['bbox_inches'])
        plt.close()
        
        return temp_file.name
    
    def process_file(self, file_content: str, 
                    original_shape: Tuple[int, int],
                    target_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Complete processing pipeline for GDC grid file."""
        self.original_shape = original_shape
        self.target_shape = target_shape
        
        # Parse the data
        parsed_data = self.parse_grid_data_from_content(file_content)
        
        if not parsed_data:
            raise ValueError("No valid data found in file")
        
        # Extract and sort values
        dx_values, dy_values = self.extract_and_sort_grid_values()
        
        # Validate data
        original_rows, original_cols = original_shape
        expected_elements = original_rows * original_cols
        
        if len(dx_values) < expected_elements or len(dy_values) < expected_elements:
            raise ValueError(
                f"Insufficient data: Expected {expected_elements} elements for "
                f"{original_rows}x{original_cols} grid. "
                f"Found {len(dx_values)} DX and {len(dy_values)} DY elements."
            )
        
        # Reshape to 2D grids
        self.dx_grid_2d = np.array(dx_values[:expected_elements]).reshape(original_rows, original_cols)
        self.dy_grid_2d = np.array(dy_values[:expected_elements]).reshape(original_rows, original_cols)
        
        # Perform interpolation
        target_rows, target_cols = target_shape
        self.dx_interpolated = self.interpolate_grid_bicubic(self.dx_grid_2d, target_rows, target_cols)
        self.dy_interpolated = self.interpolate_grid_bicubic(self.dy_grid_2d, target_rows, target_cols)
        
        # Generate statistics
        stats = {
            'dx_original': compute_grid_statistics(self.dx_grid_2d),
            'dy_original': compute_grid_statistics(self.dy_grid_2d),
            'dx_interpolated': compute_grid_statistics(self.dx_interpolated),
            'dy_interpolated': compute_grid_statistics(self.dy_interpolated)
        }
        
        # Create export files
        export_files = self._create_export_files()
        
        # Create visualizations
        visualizations = self._create_visualizations()
        
        # Package results
        results = {
            'metadata': {
                'original_shape': original_shape,
                'target_shape': target_shape,
                'num_dx_elements': len(dx_values),
                'num_dy_elements': len(dy_values),
                'interpolation_method': GRID_DEFAULTS['interpolation_method']
            },
            'statistics': stats,
            'export': export_files,
            'visualizations': visualizations,
            'grids': {
                'dx_original': self.dx_grid_2d,
                'dy_original': self.dy_grid_2d,
                'dx_interpolated': self.dx_interpolated,
                'dy_interpolated': self.dy_interpolated
            }
        }
        
        return results
    
    def _create_export_files(self) -> Dict[str, str]:
        """Create all export files and return their paths."""
        export_files = {}
        file_paths = []
        
        # CSV files
        export_files['csv_dx_original'] = self.create_csv_file(
            self.dx_grid_2d, f"original_dx_{self.original_shape[0]}x{self.original_shape[1]}.csv"
        )
        export_files['csv_dy_original'] = self.create_csv_file(
            self.dy_grid_2d, f"original_dy_{self.original_shape[0]}x{self.original_shape[1]}.csv"
        )
        export_files['csv_dx_interpolated'] = self.create_csv_file(
            self.dx_interpolated, f"interpolated_dx_{self.target_shape[0]}x{self.target_shape[1]}.csv"
        )
        export_files['csv_dy_interpolated'] = self.create_csv_file(
            self.dy_interpolated, f"interpolated_dy_{self.target_shape[0]}x{self.target_shape[1]}.csv"
        )
        
        # GDC format files
        export_files['gdc_dx_interpolated'] = self.create_gdc_format_file(
            self.dx_interpolated, 'dx', self.target_shape
        )
        export_files['gdc_dy_interpolated'] = self.create_gdc_format_file(
            self.dy_interpolated, 'dy', self.target_shape
        )
        
        # Combined GDC file
        if EXPORT_FORMATS['gdc']['include_combined']:
            export_files['gdc_combined'] = self._create_combined_gdc_file()
        
        # Collect all files for zipping
        file_paths = list(export_files.values())
        
        # Create zip file
        export_files['zip_path'] = self.create_zip_file(file_paths, "gdc_interpolation_results")
        
        return export_files
    
    def _create_combined_gdc_file(self) -> str:
        """Create a combined GDC file with both DX and DY grids."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt', prefix="gdc_combined_"
        )
        
        rows, cols = self.dx_interpolated.shape
        naming_template = EXPORT_FORMATS['gdc']['naming_convention']
        
        with open(temp_file.name, 'w') as f:
            # Write DX values
            for row in range(rows):
                for col in range(cols):
                    value = int(round(self.dx_interpolated[row, col]))
                    element_name = naming_template.format(type='dx', index=row * cols + col)
                    f.write(f"{element_name} {value}\n")
            
            # Write DY values
            for row in range(rows):
                for col in range(cols):
                    value = int(round(self.dy_interpolated[row, col]))
                    element_name = naming_template.format(type='dy', index=row * cols + col)
                    f.write(f"{element_name} {value}\n")
        
        return temp_file.name
    
    def _create_visualizations(self) -> Dict[str, str]:
        """Create all visualization plots and return their paths."""
        visualizations = {}
        
        try:
            visualizations['dx_original'] = self.create_visualization(
                self.dx_grid_2d, 
                f"Original DX Grid ({self.original_shape[0]}x{self.original_shape[1]})", 
                VISUALIZATION_CONFIG['comparison_colormap']
            )
            
            visualizations['dy_original'] = self.create_visualization(
                self.dy_grid_2d, 
                f"Original DY Grid ({self.original_shape[0]}x{self.original_shape[1]})", 
                VISUALIZATION_CONFIG['comparison_colormap']
            )
            
            visualizations['dx_interpolated'] = self.create_visualization(
                self.dx_interpolated, 
                f"Interpolated DX Grid ({self.target_shape[0]}x{self.target_shape[1]})", 
                VISUALIZATION_CONFIG['comparison_colormap']
            )
            
            visualizations['dy_interpolated'] = self.create_visualization(
                self.dy_interpolated, 
                f"Interpolated DY Grid ({self.target_shape[0]}x{self.target_shape[1]})", 
                VISUALIZATION_CONFIG['comparison_colormap']
            )
            
            visualizations['comparison'] = self.create_grid_comparison_plot(
                self.dx_grid_2d, self.dx_interpolated, "DX Grid Comparison"
            )
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        return visualizations


class ImageRemappingProcessor:
    """
    Main processing orchestrator for image remapping operations
    Enhanced with GDC grid processing capabilities
    """
    
    def __init__(self):
        self.simulator = LensDistortionSimulator()
        self.corrector = LensDistortionCorrector(self.simulator)
        self.gdc_processor = GDCGridProcessor()
        self._processing_history = []
    
    def process_distortion(self, image_width: int, image_height: int, 
                          grid_rows: int, grid_cols: int, 
                          k1: float, k2: float, k3: float, 
                          p1: float, p2: float, 
                          pattern_type: str, correction_method: str = "iterative") -> Tuple:
        """
        Complete distortion processing pipeline
        
        Args:
            image_width, image_height: Image dimensions
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            pattern_type: Test pattern type
            correction_method: Correction algorithm
            
        Returns:
            Tuple of (original_image, distorted_image, corrected_image, grid_vis, heatmap_vis)
        """
        # Update simulator parameters
        self.simulator.set_parameters(
            image_width, image_height, grid_rows, grid_cols, 
            k1, k2, k3, p1, p2
        )
        
        # Create sample image
        original_image = self.simulator.create_sample_image(pattern_type)
        
        # Apply distortion
        distorted_image = self.simulator.apply_distortion_to_image(original_image)
        
        # Correct distortion using selected method
        if correction_method == "iterative":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="iterative", recompute_maps=True
            )
        elif correction_method == "analytical":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="analytical", recompute_maps=True
            )
        elif correction_method == "polynomial":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="polynomial", recompute_maps=True
            )
        else:  # original/basic method
            corrected_image = self.simulator.correct_distortion(distorted_image)
        
        # Create visualizations
        grid_vis = visualize_distortion_grid(self.simulator)
        heatmap_vis = create_distortion_heatmap(self.simulator)
        
        # Store processing info
        processing_info = {
            'timestamp': self._get_timestamp(),
            'parameters': self.simulator.get_distortion_info(),
            'method': correction_method,
            'pattern': pattern_type
        }
        self._processing_history.append(processing_info)
        
        return original_image, distorted_image, corrected_image, grid_vis, heatmap_vis
    
    # [Keep all existing methods from the original processor...]
    
    def export_grid_csv(self) -> str:
        """Export current grid parameters to CSV format"""
        return export_grid_to_csv(self.simulator)
    
    def export_gdc_grid_csv(self, gdc_width: int, gdc_height: int) -> str:
        """Export GDC grid parameters to CSV format"""
        return export_gdc_grid_to_csv(self.simulator, gdc_width, gdc_height)
    
    def test_correction_accuracy(self) -> str:
        """Test the accuracy of different correction methods"""
        # Define test points
        test_points = [
            (self.simulator.image_width//2, self.simulator.image_height//2),  # Center
            (100, 100),  # Corner
            (self.simulator.image_width-100, self.simulator.image_height-100), # Opposite corner
            (self.simulator.image_width//2, 100),  # Top edge
            (100, self.simulator.image_height//2),  # Left edge
        ]
        
        results = []
        for x_orig, y_orig in test_points:
            # Apply distortion
            x_dist, y_dist = self.simulator.apply_barrel_distortion(x_orig, y_orig)
            
            # Test different correction methods
            x_corr_orig, y_corr_orig = self.simulator.apply_inverse_barrel_distortion(x_dist, y_dist)
            x_corr_iter, y_corr_iter = self.corrector.iterative_inverse_mapping(x_dist, y_dist)
            x_corr_anal, y_corr_anal = self.corrector.analytical_inverse_mapping(x_dist, y_dist)
            x_corr_poly, y_corr_poly = self.corrector.polynomial_inverse_mapping(x_dist, y_dist)
            
            # Calculate errors
            error_orig = np.sqrt((x_orig - x_corr_orig)**2 + (y_orig - y_corr_orig)**2)
            error_iter = np.sqrt((x_orig - x_corr_iter)**2 + (y_orig - y_corr_iter)**2)
            error_anal = np.sqrt((x_orig - x_corr_anal)**2 + (y_orig - y_corr_anal)**2)
            error_poly = np.sqrt((x_orig - x_corr_poly)**2 + (y_orig - y_corr_poly)**2)
            
            results.append({
                'point': f"({x_orig}, {y_orig})",
                'distorted': f"({x_dist:.2f}, {y_dist:.2f})",
                'error_original': f"{error_orig:.4f}",
                'error_iterative': f"{error_iter:.4f}",
                'error_analytical': f"{error_anal:.4f}",
                'error_polynomial': f"{error_poly:.4f}"
            })
        
        # Format results as text
        result_text = "Correction Method Accuracy Test:\n" + "="*60 + "\n\n"
        result_text += f"{'Point':<15} {'Distorted':<20} {'Original':<10} {'Iterative':<10} {'Analytical':<10} {'Polynomial':<10}\n"
        result_text += f"{'Original':<15} {'Position':<20} {'Error':<10} {'Error':<10} {'Error':<10} {'Error':<10}\n"
        result_text += "-" * 95 + "\n"
        
        for r in results:
            result_text += (f"{r['point']:<15} {r['distorted']:<20} {r['error_original']:<10} "
                          f"{r['error_iterative']:<10} {r['error_analytical']:<10} {r['error_polynomial']:<10}\n")
        
        result_text += "\n" + "="*60 + "\n"
        result_text += "Lower error values indicate better correction accuracy.\n"
        result_text += "Iterative method should generally be most accurate for complex distortions.\n"
        result_text += "Analytical method is fastest but only works well for simple radial distortion (k1 only).\n"
        result_text += "Polynomial method offers a good balance between speed and accuracy."
        
        return result_text
    
    def validate_correction_quality(self, pattern_type: str, correction_method: str) -> str:
        """Validate the quality of correction using comprehensive metrics"""
        # Create test image
        original_image = self.simulator.create_sample_image(pattern_type)
        
        # Apply distortion
        distorted_image = self.simulator.apply_distortion_to_image(original_image)
        
        # Apply correction using selected method
        if correction_method == "iterative":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="iterative", recompute_maps=True
            )
        elif correction_method == "analytical":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="analytical", recompute_maps=True
            )
        elif correction_method == "polynomial":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="polynomial", recompute_maps=True
            )
        else:  # original method
            corrected_image = self.simulator.correct_distortion(distorted_image)
        
        # Create grid points for geometric validation
        grid_points = []
        rows, cols = 5, 7
        for r in range(rows):
            for c in range(cols):
                x = (c + 1) * self.simulator.image_width // (cols + 1)
                y = (r + 1) * self.simulator.image_height // (rows + 1)
                grid_points.append([x, y])
        
        # Validate correction
        metrics = self.corrector.validate_correction(
            original_image, corrected_image, np.array(grid_points)
        )
        
        # Format results
        result_text = f"Correction Quality Validation ({correction_method}):\n" + "="*70 + "\n\n"
        
        if 'psnr' in metrics:
            psnr_quality = "Excellent" if metrics['psnr'] > 40 else "Good" if metrics['psnr'] > 30 else "Fair"
            result_text += f"Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB ({psnr_quality})\n"
            result_text += f"  - > 30 dB: Good quality\n"
            result_text += f"  - > 40 dB: Excellent quality\n\n"
        
        if 'correlation' in metrics:
            corr_quality = "Excellent" if metrics['correlation'] > 0.99 else "Good" if metrics['correlation'] > 0.95 else "Fair"
            result_text += f"Image Correlation: {metrics['correlation']:.4f} ({corr_quality})\n"
            result_text += f"  - > 0.95: Good correlation\n"
            result_text += f"  - > 0.99: Excellent correlation\n\n"
        
        if 'mean_geometric_error' in metrics:
            geo_quality = ("Excellent" if metrics['mean_geometric_error'] < 0.5 else 
                          "Good" if metrics['mean_geometric_error'] < 1.0 else 
                          "Acceptable" if metrics['mean_geometric_error'] < 2.0 else "Poor")
            
            result_text += f"Geometric Accuracy ({geo_quality}):\n"
            result_text += f"  - Mean Error: {metrics['mean_geometric_error']:.4f} pixels\n"
            result_text += f"  - Max Error: {metrics['max_geometric_error']:.4f} pixels\n"
            result_text += f"  - Std Error: {metrics['std_geometric_error']:.4f} pixels\n"
            result_text += f"  - Median Error: {metrics.get('median_geometric_error', 0):.4f} pixels\n\n"
            result_text += f"  - < 0.5 pixels: Excellent geometric accuracy\n"
            result_text += f"  - < 1.0 pixels: Good geometric accuracy\n"
            result_text += f"  - < 2.0 pixels: Acceptable geometric accuracy\n\n"
        
        # Add MSE and MAE if available
        if 'mse' in metrics:
            result_text += f"Mean Squared Error: {metrics['mse']:.4f}\n"
        if 'mae' in metrics:
            result_text += f"Mean Absolute Error: {metrics['mae']:.4f}\n\n"
        
        # Distortion parameters summary
        distortion_info = self.simulator.get_distortion_info()
        result_text += f"Distortion Parameters:\n"
        result_text += f"  - Type: {distortion_info['distortion_type'].replace('_', ' ').title()}\n"
        result_text += f"  - Severity: {distortion_info['severity'].title()}\n"
        result_text += f"  - K1 (Primary Radial): {self.simulator.k1:.4f}\n"
        result_text += f"  - K2 (Secondary Radial): {self.simulator.k2:.4f}\n"
        result_text += f"  - K3 (Tertiary Radial): {self.simulator.k3:.4f}\n"
        result_text += f"  - P1 (Tangential): {self.simulator.p1:.4f}\n"
        result_text += f"  - P2 (Tangential): {self.simulator.p2:.4f}\n\n"
        
        # Method recommendations
        optimal_method = self.corrector.get_optimal_method(distortion_info)
        result_text += "Method Recommendations:\n"
        result_text += f"  - Recommended method for this distortion: {optimal_method.title()}\n"
        result_text += "  - Use 'Iterative' for complex distortions with multiple coefficients\n"
        result_text += "  - Use 'Analytical' for simple radial distortion (K1 only) - fastest\n"
        result_text += "  - Use 'Polynomial' for moderate distortions - good speed/accuracy balance\n"
        result_text += "  - Use 'Original' for basic correction and grid visualization"
        
        return result_text
    
    # [Additional methods would continue here...]
    
    def process_custom_image(self, image: np.ndarray, 
                           correction_method: str = "iterative") -> Tuple[np.ndarray, np.ndarray]:
        """Process a custom user-provided image"""
        # Apply distortion
        distorted_image = self.simulator.apply_distortion_to_image(image)
        
        # Apply correction
        if correction_method == "iterative":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="iterative", recompute_maps=True
            )
        elif correction_method == "analytical":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="analytical", recompute_maps=True
            )
        elif correction_method == "polynomial":
            corrected_image = self.corrector.correct_image(
                distorted_image, method="polynomial", recompute_maps=True
            )
        else:  # original method
            corrected_image = self.simulator.correct_distortion(distorted_image)
        
        return distorted_image, corrected_image
    
    def get_processing_summary(self) -> str:
        """Get summary of current processing configuration"""
        info = self.simulator.get_distortion_info()
        
        summary_lines = [
            "Current Processing Configuration",
            "=" * 35,
            "",
            f"Image Dimensions: {info['image_dimensions'][0]} x {info['image_dimensions'][1]}",
            f"Grid Dimensions: {info['grid_dimensions'][0]} x {info['grid_dimensions'][1]}",
            f"Principal Point: ({info['principal_point'][0]:.1f}, {info['principal_point'][1]:.1f})",
            "",
            "Distortion Parameters:",
            f"  Type: {info['distortion_type'].replace('_', ' ').title()}",
            f"  Severity: {info['severity'].title()}",
            "",
            "Coefficients:",
            f"  K1 (Radial): {info['radial_coefficients']['k1']:.4f}",
            f"  K2 (Radial): {info['radial_coefficients']['k2']:.4f}",
            f"  K3 (Radial): {info['radial_coefficients']['k3']:.4f}",
            f"  P1 (Tangential): {info['tangential_coefficients']['p1']:.4f}",
            f"  P2 (Tangential): {info['tangential_coefficients']['p2']:.4f}",
            "",
            f"Processing History: {len(self._processing_history)} operations",
            "",
            "GDC Grid Processor: ✅ Available"
        ]
        
        return "\n".join(summary_lines)
    
    def validate_parameters(self, k1: float, k2: float, k3: float, 
                           p1: float, p2: float) -> Dict[str, Any]:
        """Validate distortion parameters"""
        from config.settings import VALIDATION_RANGES
        
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check ranges
        params = {'k1': k1, 'k2': k2, 'k3': k3, 'p1': p1, 'p2': p2}
        
        for param, value in params.items():
            if param in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[param]
                if value < min_val or value > max_val:
                    results['warnings'].append(
                        f"{param.upper()} value {value:.4f} is outside typical range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Check for extreme distortions
        total_radial = abs(k1) + abs(k2) + abs(k3)
        total_tangential = abs(p1) + abs(p2)
        
        if total_radial > 0.5:
            results['warnings'].append("Very high radial distortion - may cause instability")
        
        if total_tangential > 0.1:
            results['warnings'].append("High tangential distortion detected")
        
        # Recommendations
        if abs(k2) < 1e-6 and abs(k3) < 1e-6 and abs(p1) < 1e-6 and abs(p2) < 1e-6:
            results['recommendations'].append("Simple K1-only distortion - analytical method recommended")
        
        if total_radial > 0.3:
            results['recommendations'].append("Complex distortion - iterative method recommended")
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Global processor instance for interface modules
processor = ImageRemappingProcessor()