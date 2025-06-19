"""
Application processing orchestrator

This module coordinates the overall distortion/correction pipeline,
managing the flow between simulation, correction, and visualization components.
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

class ImageRemappingProcessor:
    """
    Main processing orchestrator for image remapping operations
    """
    
    def __init__(self):
        self.simulator = LensDistortionSimulator()
        self.corrector = LensDistortionCorrector(self.simulator)
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
    
    def compare_all_correction_methods(self, pattern_type: str) -> Dict[str, Any]:
        """
        Compare all available correction methods
        
        Args:
            pattern_type: Test pattern type
            
        Returns:
            Comprehensive comparison results
        """
        # Create test image
        original_image = self.simulator.create_sample_image(pattern_type)
        distorted_image = self.simulator.apply_distortion_to_image(original_image)
        
        # Compare methods
        comparison_results = self.corrector.compare_correction_methods(
            distorted_image, original_image
        )
        
        # Add distortion information
        comparison_results['distortion_info'] = self.simulator.get_distortion_info()
        
        # Add recommendation
        optimal_method = self.corrector.get_optimal_method(
            comparison_results['distortion_info']
        )
        comparison_results['recommended_method'] = optimal_method
        
        return comparison_results
    
    def process_custom_image(self, image: np.ndarray, 
                           correction_method: str = "iterative") -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a custom user-provided image
        
        Args:
            image: Input image to process
            correction_method: Correction method to use
            
        Returns:
            Tuple of (distorted_image, corrected_image)
        """
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
    
    def export_calibration_package(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Export comprehensive calibration data package
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of exported files
        """
        from data_io.exporters import export_calibration_data
        return export_calibration_data(self.simulator, self.corrector, output_dir)
    
    def get_processing_summary(self) -> str:
        """
        Get summary of current processing configuration
        
        Returns:
            Summary text
        """
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
            f"Processing History: {len(self._processing_history)} operations"
        ]
        
        return "\n".join(summary_lines)
    
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        from config.settings import (
            DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
            DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS,
            DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2
        )
        
        self.simulator.set_parameters(
            DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
            DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS,
            DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2
        )
        
        # Clear correction maps
        self.corrector.map_computed = False
        self.corrector.map_x = None
        self.corrector.map_y = None
        
        # Clear history
        self._processing_history.clear()
    
    def apply_preset(self, preset_name: str):
        """
        Apply a predefined distortion preset
        
        Args:
            preset_name: Name of the preset to apply
        """
        self.simulator.create_preset_distortion(preset_name)
        
        # Clear cached correction maps since parameters changed
        self.corrector.map_computed = False
        self.corrector.map_x = None
        self.corrector.map_y = None
    
    def get_available_presets(self) -> List[str]:
        """Get list of available distortion presets"""
        return ['barrel', 'pincushion', 'fisheye', 'mild_barrel', 'complex', 'identity']
    
    def validate_parameters(self, k1: float, k2: float, k3: float, 
                           p1: float, p2: float) -> Dict[str, Any]:
        """
        Validate distortion parameters
        
        Args:
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            Validation results
        """
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
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history"""
        return self._processing_history.copy()
    
    def clear_history(self):
        """Clear processing history"""
        self._processing_history.clear()

# Global processor instance for interface modules
processor = ImageRemappingProcessor()