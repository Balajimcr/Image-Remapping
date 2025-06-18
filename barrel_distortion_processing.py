import numpy as np
from barrel_distortion_simulator import BarrelDistortionSimulator
from barrel_distortion_corrector import BarrelDistortionCorrector
from barrel_distortion_visualization import (
    visualize_distortion_grid, 
    create_distortion_heatmap,
    export_grid_to_csv,
    export_gdc_grid_to_csv,
    analyze_distortion_quality
)

# Global instances
simulator = BarrelDistortionSimulator()
corrector = BarrelDistortionCorrector(simulator)

def process_distortion(image_width, image_height, grid_rows, grid_cols, 
                      k1, k2, k3, p1, p2, pattern_type, correction_method="original"):
    """
    Process distortion with given parameters and return visualizations
    """
    global simulator, corrector
    
    # Update simulator parameters
    simulator.set_parameters(image_width, image_height, grid_rows, grid_cols, k1, k2, k3, p1, p2)
    
    # Create sample image
    original_image = simulator.create_sample_image(pattern_type)
    
    # Apply distortion
    distorted_image = simulator.apply_distortion_to_image(original_image)
    
    # Correct distortion using selected method
    if correction_method == "advanced_iterative":
        corrected_image = corrector.correct_image(distorted_image, method="iterative", recompute_maps=True)
    elif correction_method == "advanced_analytical":
        corrected_image = corrector.correct_image(distorted_image, method="analytical", recompute_maps=True)
    else:  # original method
        corrected_image = simulator.correct_distortion(distorted_image)
    
    # Create visualizations
    grid_vis = visualize_distortion_grid(simulator)
    heatmap_vis = create_distortion_heatmap(simulator)
    
    return original_image, distorted_image, corrected_image, grid_vis, heatmap_vis

def export_grid_csv():
    """
    Export grid parameters to CSV format
    """
    global simulator
    csv_data = export_grid_to_csv(simulator)
    return csv_data

def export_gdc_grid_csv(gdc_width, gdc_height):
    """
    Export GDC grid parameters to CSV format
    """
    global simulator
    csv_data = export_gdc_grid_to_csv(simulator, gdc_width, gdc_height)
    return csv_data

def test_correction_accuracy():
    """
    Test the accuracy of different correction methods
    """
    global simulator, corrector
    
    # Test points
    test_points = [
        (simulator.image_width//2, simulator.image_height//2),  # Center
        (100, 100),  # Corner
        (simulator.image_width-100, simulator.image_height-100), # Opposite corner
        (simulator.image_width//2, 100),  # Top edge
        (100, simulator.image_height//2),  # Left edge
    ]
    
    results = []
    for x_orig, y_orig in test_points:
        # Apply distortion
        x_dist, y_dist = simulator.apply_barrel_distortion(x_orig, y_orig)
        
        # Test different correction methods
        x_corr_orig, y_corr_orig = simulator.apply_inverse_barrel_distortion(x_dist, y_dist)
        x_corr_iter, y_corr_iter = corrector.iterative_inverse_mapping(x_dist, y_dist)
        x_corr_anal, y_corr_anal = corrector.analytical_inverse_mapping(x_dist, y_dist)
        
        # Calculate errors
        error_orig = np.sqrt((x_orig - x_corr_orig)**2 + (y_orig - y_corr_orig)**2)
        error_iter = np.sqrt((x_orig - x_corr_iter)**2 + (y_orig - y_corr_iter)**2)
        error_anal = np.sqrt((x_orig - x_corr_anal)**2 + (y_orig - y_corr_anal)**2)
        
        results.append({
            'point': f"({x_orig}, {y_orig})",
            'distorted': f"({x_dist:.2f}, {y_dist:.2f})",
            'error_original': f"{error_orig:.4f}",
            'error_iterative': f"{error_iter:.4f}",
            'error_analytical': f"{error_anal:.4f}"
        })
    
    # Format results as text
    result_text = "Correction Method Accuracy Test:\n" + "="*50 + "\n\n"
    result_text += f"{'Point':<15} {'Distorted':<20} {'Original':<12} {'Iterative':<12} {'Analytical':<12}\n"
    result_text += f"{'Original':<15} {'Position':<20} {'Error':<12} {'Error':<12} {'Error':<12}\n"
    result_text += "-" * 85 + "\n"
    
    for r in results:
        result_text += f"{r['point']:<15} {r['distorted']:<20} {r['error_original']:<12} {r['error_iterative']:<12} {r['error_analytical']:<12}\n"
    
    result_text += "\n" + "="*50 + "\n"
    result_text += "Lower error values indicate better correction accuracy.\n"
    result_text += "Iterative method should generally be most accurate for complex distortions.\n"
    result_text += "Analytical method is fastest but only works well for simple radial distortion (k1 only)."
    
    return result_text

def validate_correction_quality(pattern_type, correction_method):
    """
    Validate the quality of correction using different methods
    """
    global simulator, corrector
    
    # Create test image
    original_image = simulator.create_sample_image(pattern_type)
    
    # Apply distortion
    distorted_image = simulator.apply_distortion_to_image(original_image)
    
    # Apply correction using selected method
    if correction_method == "advanced_iterative":
        corrected_image = corrector.correct_image(distorted_image, method="iterative", recompute_maps=True)
    elif correction_method == "advanced_analytical":
        corrected_image = corrector.correct_image(distorted_image, method="analytical", recompute_maps=True)
    else:  # original method
        corrected_image = simulator.correct_distortion(distorted_image)
    
    # Create grid points for geometric validation
    grid_points = []
    rows, cols = 5, 7
    for r in range(rows):
        for c in range(cols):
            x = (c + 1) * simulator.image_width // (cols + 1)
            y = (r + 1) * simulator.image_height // (rows + 1)
            grid_points.append([x, y])
    
    # Validate correction
    metrics = corrector.validate_correction(original_image, corrected_image, np.array(grid_points))
    
    # Format results
    result_text = f"Correction Quality Validation ({correction_method}):\n" + "="*60 + "\n\n"
    
    if 'psnr' in metrics:
        result_text += f"Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB\n"
        result_text += f"  - > 30 dB: Good quality\n"
        result_text += f"  - > 40 dB: Excellent quality\n\n"
    
    if 'correlation' in metrics:
        result_text += f"Image Correlation: {metrics['correlation']:.4f}\n"
        result_text += f"  - > 0.95: Good correlation\n"
        result_text += f"  - > 0.99: Excellent correlation\n\n"
    
    if 'mean_geometric_error' in metrics:
        result_text += f"Geometric Accuracy:\n"
        result_text += f"  - Mean Error: {metrics['mean_geometric_error']:.4f} pixels\n"
        result_text += f"  - Max Error: {metrics['max_geometric_error']:.4f} pixels\n"
        result_text += f"  - Std Error: {metrics['std_geometric_error']:.4f} pixels\n\n"
        result_text += f"  - < 0.5 pixels: Excellent geometric accuracy\n"
        result_text += f"  - < 1.0 pixels: Good geometric accuracy\n"
        result_text += f"  - < 2.0 pixels: Acceptable geometric accuracy\n\n"
    
    # Distortion parameters summary
    result_text += f"Distortion Parameters:\n"
    result_text += f"  - K1 (Primary Radial): {simulator.k1:.4f}\n"
    result_text += f"  - K2 (Secondary Radial): {simulator.k2:.4f}\n"
    result_text += f"  - K3 (Tertiary Radial): {simulator.k3:.4f}\n"
    result_text += f"  - P1 (Tangential): {simulator.p1:.4f}\n"
    result_text += f"  - P2 (Tangential): {simulator.p2:.4f}\n\n"
    
    result_text += "Method Recommendations:\n"
    result_text += "- Use 'Advanced Iterative' for complex distortions with multiple coefficients\n"
    result_text += "- Use 'Advanced Analytical' for simple radial distortion (K1 only) - fastest\n"
    result_text += "- Use 'Original' for basic correction and grid visualization"
    
    return result_text