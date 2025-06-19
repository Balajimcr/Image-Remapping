"""
Visualization functions for distortion grids, heatmaps, and analysis plots

This module provides comprehensive visualization capabilities for
distortion analysis, correction validation, and quality assessment.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import io
from typing import Tuple, Optional, List, Dict, Any

from config.settings import (
    VISUALIZATION_SCALE_FACTOR, GRID_LINE_THICKNESS, VECTOR_LINE_THICKNESS,
    POINT_RADIUS, COLORS
)

def visualize_distortion_grid(simulator, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create visualization of the distortion grid showing target and distorted points
    
    Args:
        simulator: Lens distortion simulator instance
        save_path: Optional path to save the visualization
        
    Returns:
        Visualization image as numpy array
    """
    source_coords, target_coords = simulator.generate_sparse_distortion_grid()
    
    # Create visualization canvas
    display_width = int(simulator.image_width * VISUALIZATION_SCALE_FACTOR)
    display_height = int(simulator.image_height * VISUALIZATION_SCALE_FACTOR)
    
    img = np.ones((display_height, display_width, 3), dtype=np.uint8) * 255
    
    # Extract colors
    target_color = COLORS['target_grid']
    source_color = COLORS['source_distorted']
    grid_color = COLORS['grid_lines']
    vector_color = COLORS['distortion_vectors']
    
    # Scale coordinates for display
    def scale_coord(coord):
        return (int(coord[0] * VISUALIZATION_SCALE_FACTOR), 
                int(coord[1] * VISUALIZATION_SCALE_FACTOR))
    
    # Draw grid lines (target grid)
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols - 1):  # Horizontal lines
            pt1 = scale_coord(target_coords[row, col])
            pt2 = scale_coord(target_coords[row, col + 1])
            cv2.line(img, pt1, pt2, grid_color, 1)
    
    for row in range(simulator.grid_rows - 1):  # Vertical lines
        for col in range(simulator.grid_cols):
            pt1 = scale_coord(target_coords[row, col])
            pt2 = scale_coord(target_coords[row + 1, col])
            cv2.line(img, pt1, pt2, grid_color, 1)
    
    # Draw distortion vectors and points
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            target_pt = scale_coord(target_coords[row, col])
            source_pt = scale_coord(source_coords[row, col])
            
            # Draw distortion vector (target -> source)
            vector_length = np.linalg.norm(np.array(source_pt) - np.array(target_pt))
            if vector_length > 2:
                cv2.arrowedLine(img, target_pt, source_pt, vector_color, 
                              VECTOR_LINE_THICKNESS, tipLength=0.1)
            
            # Draw target point (blue)
            cv2.circle(img, target_pt, POINT_RADIUS, target_color, -1)
            
            # Draw source point (red)
            cv2.circle(img, source_pt, POINT_RADIUS-1, source_color, -1)
    
    # Add legend
    _add_grid_legend(img)
    
    # Add title
    title = f"Distortion Grid ({simulator.grid_cols}x{simulator.grid_rows})"
    cv2.putText(img, title, (display_width//2 - 120, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
    
    if save_path:
        cv2.imwrite(save_path, img)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _add_grid_legend(img: np.ndarray):
    """Add legend to grid visualization"""
    legend_y = 30
    
    # Target points
    cv2.circle(img, (20, legend_y), POINT_RADIUS, COLORS['target_grid'], -1)
    cv2.putText(img, "Target Grid", (35, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
    
    legend_y += 25
    # Distorted points
    cv2.circle(img, (20, legend_y), POINT_RADIUS-1, COLORS['source_distorted'], -1)
    cv2.putText(img, "Distorted Points", (35, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
    
    legend_y += 25
    # Distortion vectors
    cv2.arrowedLine(img, (15, legend_y), (25, legend_y), COLORS['distortion_vectors'], 
                    VECTOR_LINE_THICKNESS, tipLength=0.3)
    cv2.putText(img, "Distortion Vectors", (35, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)

def create_distortion_heatmap(simulator, colormap: str = 'jet') -> np.ndarray:
    """
    Create heatmap showing distortion magnitude across the grid
    
    Args:
        simulator: Lens distortion simulator instance
        colormap: Matplotlib colormap name
        
    Returns:
        Heatmap image as numpy array
    """
    source_coords, target_coords = simulator.generate_sparse_distortion_grid()
    
    # Calculate distortion magnitudes
    distortion_magnitudes = np.zeros((simulator.grid_rows, simulator.grid_cols))
    
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            delta = source_coords[row, col] - target_coords[row, col]
            magnitude = np.linalg.norm(delta)
            distortion_magnitudes[row, col] = magnitude
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(distortion_magnitudes, cmap=colormap, 
                   interpolation='bilinear', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Distortion Magnitude (pixels)')
    
    # Formatting
    ax.set_title(f'Distortion Magnitude Heatmap ({simulator.grid_cols}x{simulator.grid_rows} Grid)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    
    # Set ticks
    ax.set_xticks(range(simulator.grid_cols))
    ax.set_yticks(range(simulator.grid_rows))
    
    # Convert to image
    return _matplotlib_to_image(fig)

def create_correction_comparison(original_image: np.ndarray, 
                               distorted_image: np.ndarray,
                               corrected_image: np.ndarray,
                               titles: Optional[List[str]] = None) -> np.ndarray:
    """
    Create side-by-side comparison of original, distorted, and corrected images
    
    Args:
        original_image: Original undistorted image
        distorted_image: Distorted image
        corrected_image: Corrected image
        titles: Optional custom titles
        
    Returns:
        Comparison visualization
    """
    if titles is None:
        titles = ['Original', 'Distorted', 'Corrected']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images = [original_image, distorted_image, corrected_image]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = img
        
        axes[i].imshow(display_img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(title, fontsize=12, weight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    return _matplotlib_to_image(fig)

def create_quality_metrics_plot(metrics: Dict[str, Any]) -> np.ndarray:
    """
    Create visualization of correction quality metrics
    
    Args:
        metrics: Dictionary containing quality metrics
        
    Returns:
        Quality metrics visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Correction Quality Metrics', fontsize=16, weight='bold')
    
    # PSNR visualization
    if 'psnr' in metrics:
        psnr = metrics['psnr']
        ax = axes[0, 0]
        bars = ax.bar(['PSNR'], [psnr], color='skyblue')
        ax.set_ylabel('dB')
        ax.set_title('Peak Signal-to-Noise Ratio')
        ax.axhline(y=30, color='orange', linestyle='--', label='Good (30dB)')
        ax.axhline(y=40, color='green', linestyle='--', label='Excellent (40dB)')
        ax.legend()
        ax.set_ylim(0, max(50, psnr + 5))
        
        # Add value label on bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom')
    
    # Correlation visualization
    if 'correlation' in metrics:
        corr = metrics['correlation']
        ax = axes[0, 1]
        bars = ax.bar(['Correlation'], [corr], color='lightgreen')
        ax.set_ylabel('Coefficient')
        ax.set_title('Image Correlation')
        ax.axhline(y=0.95, color='orange', linestyle='--', label='Good (0.95)')
        ax.axhline(y=0.99, color='green', linestyle='--', label='Excellent (0.99)')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value label
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # Geometric error visualization
    if 'mean_geometric_error' in metrics:
        errors = ['Mean', 'Max', 'Std']
        values = [
            metrics.get('mean_geometric_error', 0),
            metrics.get('max_geometric_error', 0),
            metrics.get('std_geometric_error', 0)
        ]
        
        ax = axes[1, 0]
        bars = ax.bar(errors, values, color=['lightcoral', 'orange', 'lightyellow'])
        ax.set_ylabel('Pixels')
        ax.set_title('Geometric Errors')
        ax.axhline(y=0.5, color='green', linestyle='--', label='Excellent (<0.5)')
        ax.axhline(y=1.0, color='orange', linestyle='--', label='Good (<1.0)')
        ax.axhline(y=2.0, color='red', linestyle='--', label='Acceptable (<2.0)')
        ax.legend()
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # MSE and MAE visualization
    if 'mse' in metrics or 'mae' in metrics:
        ax = axes[1, 1]
        error_types = []
        error_values = []
        
        if 'mse' in metrics:
            error_types.append('MSE')
            error_values.append(metrics['mse'])
        
        if 'mae' in metrics:
            error_types.append('MAE')
            error_values.append(metrics['mae'])
        
        bars = ax.bar(error_types, error_values, color='lightsteelblue')
        ax.set_ylabel('Error Value')
        ax.set_title('Image Error Metrics')
        
        # Add value labels
        for bar, value in zip(bars, error_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(error_values) * 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return _matplotlib_to_image(fig)

def create_distortion_field_visualization(simulator, density: int = 50) -> np.ndarray:
    """
    Create vector field visualization showing distortion across the entire image
    
    Args:
        simulator: Lens distortion simulator instance
        density: Grid density for vector field
        
    Returns:
        Vector field visualization
    """
    # Create dense grid
    x = np.linspace(0, simulator.image_width, density)
    y = np.linspace(0, simulator.image_height, density)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distortion vectors
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            x_orig, y_orig = X[i, j], Y[i, j]
            x_dist, y_dist = simulator.apply_barrel_distortion(x_orig, y_orig)
            U[i, j] = x_dist - x_orig
            V[i, j] = y_dist - y_orig
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create quiver plot
    magnitude = np.sqrt(U**2 + V**2)
    quiver = ax.quiver(X, Y, U, V, magnitude, scale_units='xy', scale=1, 
                      cmap='viridis', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, label='Distortion Magnitude (pixels)')
    
    # Formatting
    ax.set_xlim(0, simulator.image_width)
    ax.set_ylim(0, simulator.image_height)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Distortion Vector Field', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return _matplotlib_to_image(fig)

def create_radial_distortion_profile(simulator, num_points: int = 100) -> np.ndarray:
    """
    Create radial distortion profile plot
    
    Args:
        simulator: Lens distortion simulator instance
        num_points: Number of points for profile
        
    Returns:
        Radial profile visualization
    """
    # Calculate radial distances
    center_x, center_y = simulator.cx, simulator.cy
    max_radius = min(center_x, center_y, 
                    simulator.image_width - center_x, 
                    simulator.image_height - center_y)
    
    radii = np.linspace(0, max_radius, num_points)
    distortions = []
    
    for r in radii:
        # Test point at this radius
        x = center_x + r
        y = center_y
        
        # Apply distortion
        x_dist, y_dist = simulator.apply_barrel_distortion(x, y)
        
        # Calculate radial distortion
        r_dist = np.sqrt((x_dist - center_x)**2 + (y_dist - center_y)**2)
        distortion = r_dist - r
        distortions.append(distortion)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(radii, distortions, 'b-', linewidth=2, label='Radial Distortion')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlabel('Radial Distance (pixels)')
    ax.set_ylabel('Distortion (pixels)')
    ax.set_title('Radial Distortion Profile', fontsize=14, weight='bold')
    ax.legend()
    
    # Add distortion type annotation
    distortion_info = simulator.get_distortion_info()
    ax.text(0.05, 0.95, f"Type: {distortion_info['distortion_type'].title()}", 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return _matplotlib_to_image(fig)

def create_method_comparison_plot(comparison_results: Dict[str, Dict]) -> np.ndarray:
    """
    Create visualization comparing different correction methods
    
    Args:
        comparison_results: Results from method comparison
        
    Returns:
        Method comparison visualization
    """
    methods = list(comparison_results.keys())
    available_methods = [m for m in methods if comparison_results[m].get('method_available', False)]
    
    if not available_methods:
        # Create empty plot with error message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No methods available for comparison', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Method Comparison - Error')
        return _matplotlib_to_image(fig)
    
    # Extract metrics
    metrics_to_plot = ['psnr', 'correlation', 'mean_geometric_error', 'correction_time']
    metric_labels = ['PSNR (dB)', 'Correlation', 'Geometric Error (px)', 'Time (s)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Correction Method Comparison', fontsize=16, weight='bold')
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[i // 2, i % 2]
        
        values = []
        method_names = []
        
        for method in available_methods:
            if metric in comparison_results[method]:
                values.append(comparison_results[method][metric])
                method_names.append(method.title())
        
        if values:
            bars = ax.bar(method_names, values, 
                         color=['skyblue', 'lightgreen', 'lightcoral'][:len(values)])
            ax.set_title(label)
            ax.set_ylabel(label.split('(')[0].strip())
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(method_names) > 2:
                ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _matplotlib_to_image(fig)

def create_before_after_grid(original_image: np.ndarray, 
                           processed_image: np.ndarray,
                           grid_overlay: bool = True) -> np.ndarray:
    """
    Create before/after comparison with optional grid overlay
    
    Args:
        original_image: Original image
        processed_image: Processed image
        grid_overlay: Whether to add grid overlay
        
    Returns:
        Before/after comparison image
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (img, title) in enumerate(zip([original_image, processed_image], 
                                        ['Before', 'After'])):
        # Convert BGR to RGB if needed
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = img
        
        axes[i].imshow(display_img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(title, fontsize=14, weight='bold')
        axes[i].axis('off')
        
        # Add grid overlay if requested
        if grid_overlay:
            h, w = img.shape[:2]
            grid_spacing = min(h, w) // 10
            
            for x in range(0, w, grid_spacing):
                axes[i].axvline(x=x, color='red', alpha=0.3, linewidth=0.5)
            for y in range(0, h, grid_spacing):
                axes[i].axhline(y=y, color='red', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    return _matplotlib_to_image(fig)

def _matplotlib_to_image(fig) -> np.ndarray:
    """
    Convert matplotlib figure to numpy image array
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Image array
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Read image data
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    
    # Decode image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def analyze_distortion_quality(original: np.ndarray, corrected: np.ndarray) -> dict:
    """
    Analyze the quality of distortion correction using image metrics
    
    Args:
        original: Original undistorted image
        corrected: Corrected image
        
    Returns:
        Dictionary with quality metrics
    """
    # Convert to grayscale for analysis
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original
    corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY) if len(corrected.shape) == 3 else corrected
    
    # Calculate PSNR
    mse = np.mean((original_gray.astype(float) - corrected_gray.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate simplified SSIM
    mu1 = np.mean(original_gray)
    mu2 = np.mean(corrected_gray)
    sigma1 = np.var(original_gray)
    sigma2 = np.var(corrected_gray)
    sigma12 = np.mean((original_gray - mu1) * (corrected_gray - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse
    }