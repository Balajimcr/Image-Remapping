#!/usr/bin/env python3
"""
Enhanced Distortion Map Visualizer with GDC Export - Fixed Version
Real-time interactive visualization of lens distortion parameters with GDC grid export capability

High-performance vectorized implementation for real-time visualization
of distortion maps using the Brown-Conrady model with hardware-ready export options.

Author: Balaji R
License: MIT
"""

import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import io
import math
import csv
import zipfile
import tempfile
import os
from typing import Tuple, Optional
from functools import lru_cache
import time
from datetime import datetime

class OptimizedDistortionVisualizerWithGDC:
    """High-performance vectorized distortion map visualizer with GDC export capabilities"""
    
    def __init__(self):
        self.image_width = 1280
        self.image_height = 720
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        # Pre-compute common values for optimization
        self._update_normalization_factors()
        
        # Cache for grid coordinates to avoid recomputation
        self._cached_grids = {}
        
        # GDC export configuration
        self.default_gdc_width = 8192
        self.default_gdc_height = 6144
        
    def _update_normalization_factors(self):
        """Pre-compute normalization factors for better performance"""
        self.norm_x = 1.0 / self.image_width
        self.norm_y = 1.0 / self.image_height
        
    @lru_cache(maxsize=32)
    def _get_cached_grid_coordinates(self, grid_rows: int, grid_cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and cache grid coordinates for given dimensions
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            
        Returns:
            Cached grid coordinate arrays (X, Y)
        """
        x_positions = np.linspace(0, self.image_width, grid_cols)
        y_positions = np.linspace(0, self.image_height, grid_rows)
        
        # Create meshgrid for vectorized operations
        X, Y = np.meshgrid(x_positions, y_positions)
        
        return X, Y
    
    def apply_brown_conrady_distortion_vectorized(self, X: np.ndarray, Y: np.ndarray, 
                                                k1: float, k2: float, k3: float,
                                                p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized Brown-Conrady distortion model
        
        Args:
            X, Y: Input coordinate arrays
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            Distorted coordinates (X_dist, Y_dist)
        """
        # Normalize coordinates relative to principal point (vectorized)
        x_norm = (X - self.cx) * self.norm_x
        y_norm = (Y - self.cy) * self.norm_y
        
        # Calculate radial distance squared (vectorized)
        r2 = x_norm**2 + y_norm**2
        
        # Compute higher order terms only if needed
        if abs(k2) > 1e-10 or abs(k3) > 1e-10:
            r4 = r2 * r2
            if abs(k3) > 1e-10:
                r6 = r2 * r4
                radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
            else:
                radial_factor = 1 + k1 * r2 + k2 * r4
        else:
            radial_factor = 1 + k1 * r2
        
        # Apply radial distortion (vectorized)
        x_radial = x_norm * radial_factor
        y_radial = y_norm * radial_factor
        
        # Tangential distortion (vectorized) - only if needed
        if abs(p1) > 1e-10 or abs(p2) > 1e-10:
            xy = x_norm * y_norm
            x_tangential = 2 * p1 * xy + p2 * (r2 + 2 * x_norm**2)
            y_tangential = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * xy
            
            x_dist_norm = x_radial + x_tangential
            y_dist_norm = y_radial + y_tangential
        else:
            x_dist_norm = x_radial
            y_dist_norm = y_radial
        
        # Convert back to pixel coordinates (vectorized)
        X_dist = x_dist_norm * self.image_width + self.cx
        Y_dist = y_dist_norm * self.image_height + self.cy
        
        return X_dist, Y_dist
    
    def generate_distortion_grid_vectorized(self, grid_rows: int, grid_cols: int,
                                          k1: float, k2: float, k3: float,
                                          p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized distortion grid generation
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            source_coords, target_coords: Grid coordinate arrays
        """
        # Get cached grid coordinates
        target_X, target_Y = self._get_cached_grid_coordinates(grid_rows, grid_cols)
        
        # Apply distortion to entire grid at once (vectorized)
        source_X, source_Y = self.apply_brown_conrady_distortion_vectorized(
            target_X, target_Y, k1, k2, k3, p1, p2
        )
        
        return np.stack([source_X, source_Y], axis=-1), np.stack([target_X, target_Y], axis=-1)
    
    def export_gdc_format_only(self, grid_rows: int, grid_cols: int,
                              k1: float, k2: float, k3: float, p1: float, p2: float,
                              gdc_width: int = None, gdc_height: int = None) -> str:
        """
        Export distortion grid in GDC (Geometric Distortion Correction) format only
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            gdc_width, gdc_height: Target GDC dimensions
            
        Returns:
            Path to created CSV file with GDC format
        """
        if gdc_width is None:
            gdc_width = self.default_gdc_width
        if gdc_height is None:
            gdc_height = self.default_gdc_height
            
        # Generate distortion grid (simple, fast)
        source_coords, target_coords = self.generate_distortion_grid_vectorized(
            grid_rows, grid_cols, k1, k2, k3, p1, p2
        )
        
        # Calculate displacement vectors (simple subtraction)
        displacement = source_coords - target_coords
        displacement_x = displacement[:, :, 0]
        displacement_y = displacement[:, :, 1]
        
        # Create temporary file for GDC export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        gdc_file = os.path.join(temp_dir, f"gdc_grid_data_{timestamp}.csv")
        
        try:
            # Convert displacement to GDC format (simple conversion only)
            grid_distort_x, grid_distort_y = self._convert_to_gdc_format(
                displacement_x, displacement_y, gdc_width, gdc_height
            )
            
            # Write simple CSV file
            with open(gdc_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header with metadata
                writer.writerow([f"# GDC Grid Data (Hardware Ready Format) - Generated: {datetime.now().isoformat()}"])
                writer.writerow([f"# Parameters: K1={k1:.6f}, K2={k2:.6f}, K3={k3:.6f}, P1={p1:.6f}, P2={p2:.6f}"])
                writer.writerow([f"# Original Grid: {grid_rows}x{grid_cols}, Target Resolution: {gdc_width}x{gdc_height}"])
                writer.writerow([f"# Image Size: {self.image_width}x{self.image_height}, Center: ({self.cx:.1f}, {self.cy:.1f})"])
                writer.writerow([f"# Scale Factors: X={int((gdc_width * (2**14)) / self.image_width)}, Y={int((gdc_height * (2**14)) / self.image_height)}"])
                writer.writerow([])
                
                # Write GDC X values (direct output, no interpolation)
                rows, cols = grid_distort_x.shape
                for row in range(rows):
                    for col in range(cols):
                        # Calculate 1D index
                        gdc_index = row * cols + col
                        value = int(grid_distort_x[row, col])
                        hex_value = self._format_hex_value(value, signed=True)
                        # Use 1D index in the output format
                        writer.writerow([f"yuv_gdc_grid_dx_0_{gdc_index}", value, hex_value])
                
                # Write GDC Y values (direct output, no interpolation)
                for row in range(rows):
                    for col in range(cols):
                        # Calculate 1D index
                        gdc_index = row * cols + col
                        value = int(grid_distort_y[row, col])
                        hex_value = self._format_hex_value(value, signed=True)
                        # Use 1D index in the output format
                        writer.writerow([f"yuv_gdc_grid_dy_0_{gdc_index}", value, hex_value])
            
            return gdc_file
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(gdc_file):
                os.remove(gdc_file)
            raise e
    
    def _convert_to_gdc_format(self, displacement_x: np.ndarray, displacement_y: np.ndarray,
                              gdc_width: int, gdc_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert displacement arrays to GDC format using the hardware conversion formula
        
        Args:
            displacement_x, displacement_y: Displacement arrays
            gdc_width, gdc_height: Target GDC dimensions
            
        Returns:
            GDC format arrays (grid_distort_x, grid_distort_y)
        """
        rows, cols = displacement_x.shape
        
        # Initialize output grids
        grid_distort_x = np.zeros((rows, cols), dtype=np.int32)
        grid_distort_y = np.zeros((rows, cols), dtype=np.int32)
        
        for row in range(rows):
            for col in range(cols):
                # Get displacement values
                delta_x = float(displacement_x[row, col])
                delta_y = float(displacement_y[row, col])
                
                # Apply GDC conversion formula for X
                # grid_distort_x = (ceil(((Delta_x * round((gdc_width << 14) / image_width)) << 2) / (2^14))) << 7
                scale_factor_x = int(round((gdc_width * (2 ** 14)) / self.image_width))
                scaled_delta_x = delta_x * scale_factor_x
                shifted_delta_x = scaled_delta_x * 4  # << 2
                intermediate_x = math.ceil(shifted_delta_x / (2 ** 14))
                grid_distort_x[row, col] = int(intermediate_x) << 7
                
                # Apply GDC conversion formula for Y
                # grid_distort_y = (ceil(((Delta_y * round((gdc_height << 14) / image_height)) << 2) / (2^14))) << 7
                scale_factor_y = int(round((gdc_height * (2 ** 14)) / self.image_height))
                scaled_delta_y = delta_y * scale_factor_y
                shifted_delta_y = scaled_delta_y * 4  # << 2
                intermediate_y = math.ceil(shifted_delta_y / (2 ** 14))
                grid_distort_y[row, col] = int(intermediate_y) << 7
        
        return grid_distort_x, grid_distort_y
    
    def _format_hex_value(self, value: int, signed: bool = True) -> str:
        """Format integer value as hexadecimal string"""
        if signed and value < 0:
            # Two's complement representation for negative values
            hex_value = f"0x{value & 0xFFFFFFFF:08X}"
        else:
            hex_value = f"0x{value:08X}"
        return hex_value
    
    def create_grid_visualization_optimized(self, grid_rows: int, grid_cols: int,
                                          k1: float, k2: float, k3: float,
                                          p1: float, p2: float) -> np.ndarray:
        """
        Optimized grid visualization using LineCollection for better performance
        """
        # Generate distortion grid (vectorized)
        source_coords, target_coords = self.generate_distortion_grid_vectorized(
            grid_rows, grid_cols, k1, k2, k3, p1, p2
        )
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Set up the plot
        ax.set_xlim(0, self.image_width)
        ax.set_ylim(0, self.image_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Prepare line segments for target grid (vectorized)
        target_lines = []
        source_lines = []
        
        # Horizontal and vertical lines for target grid
        for row in range(grid_rows):
            for col in range(grid_cols - 1):
                line = [target_coords[row, col], target_coords[row, col + 1]]
                target_lines.append(line)
        
        for row in range(grid_rows - 1):
            for col in range(grid_cols):
                line = [target_coords[row, col], target_coords[row + 1, col]]
                target_lines.append(line)
        
        # Horizontal and vertical lines for distorted grid
        for row in range(grid_rows):
            for col in range(grid_cols - 1):
                line = [source_coords[row, col], source_coords[row, col + 1]]
                source_lines.append(line)
        
        for row in range(grid_rows - 1):
            for col in range(grid_cols):
                line = [source_coords[row, col], source_coords[row + 1, col]]
                source_lines.append(line)
        
        # Create LineCollections for efficient rendering
        if target_lines:
            target_collection = LineCollection(target_lines, colors='lightgray', linewidths=1, alpha=0.6)
            ax.add_collection(target_collection)
        
        if source_lines:
            source_collection = LineCollection(source_lines, colors='red', linewidths=2, alpha=0.8)
            ax.add_collection(source_collection)
        
        # Calculate distortion vectors (vectorized)
        displacement = source_coords - target_coords
        magnitude = np.linalg.norm(displacement, axis=-1)
        
        # Only show significant vectors (vectorized filtering)
        significant_mask = magnitude > 2
        if np.any(significant_mask):
            # Get positions where mask is True
            sig_rows, sig_cols = np.where(significant_mask)
            
            # Extract significant vectors
            sig_target = target_coords[sig_rows, sig_cols]
            sig_displacement = displacement[sig_rows, sig_cols]
            
            # Draw arrows using quiver (more efficient than individual arrows)
            ax.quiver(sig_target[:, 0], sig_target[:, 1], 
                     sig_displacement[:, 0], sig_displacement[:, 1],
                     angles='xy', scale_units='xy', scale=1,
                     color='blue', alpha=0.7, width=0.003, headwidth=3)
        
        # Plot points (vectorized)
        ax.scatter(target_coords[:, :, 0].flatten(), target_coords[:, :, 1].flatten(), 
                  c='blue', s=15, alpha=0.8, zorder=5)
        ax.scatter(source_coords[:, :, 0].flatten(), source_coords[:, :, 1].flatten(), 
                  c='red', s=15, alpha=0.8, zorder=5)
        
        # Add title and labels
        distortion_type = self._classify_distortion_fast(k1, k2, k3, p1, p2)
        ax.set_title(f'Distortion Grid Visualization - {distortion_type}\n'
                    f'K1={k1:.3f}, K2={k2:.3f}, K3={k3:.3f}, P1={p1:.3f}, P2={p2:.3f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        # Optimized legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', linewidth=1, label='Target Grid'),
            Line2D([0], [0], color='red', linewidth=2, label='Distorted Grid'),
            Line2D([0], [0], marker='o', color='blue', linewidth=0, markersize=4, label='Target Points'),
            Line2D([0], [0], marker='o', color='red', linewidth=0, markersize=4, label='Distorted Points'),
            Line2D([0], [0], color='blue', linewidth=2, label='Distortion Vectors')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return self._fig_to_array_optimized(fig)
    
    def create_magnitude_heatmap_optimized(self, grid_rows: int, grid_cols: int,
                                         k1: float, k2: float, k3: float,
                                         p1: float, p2: float) -> np.ndarray:
        """
        Optimized heatmap generation using vectorized operations
        """
        # Generate distortion grid (vectorized)
        source_coords, target_coords = self.generate_distortion_grid_vectorized(
            grid_rows, grid_cols, k1, k2, k3, p1, p2
        )
        
        # Calculate distortion magnitudes (vectorized)
        displacement = source_coords - target_coords
        distortion_magnitudes = np.linalg.norm(displacement, axis=-1)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Create heatmap with optimized colormap
        im = ax.imshow(distortion_magnitudes, cmap='hot', 
                      interpolation='bilinear', aspect='auto', origin='upper')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Distortion Magnitude (pixels)', shrink=0.8)
        
        # Formatting
        distortion_type = self._classify_distortion_fast(k1, k2, k3, p1, p2)
        ax.set_title(f'Distortion Magnitude Heatmap - {distortion_type}\n'
                    f'Grid: {grid_cols}×{grid_rows}, Max: {np.max(distortion_magnitudes):.2f}px', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        
        # Optimized tick placement
        x_ticks = np.linspace(0, grid_cols-1, min(10, grid_cols))
        y_ticks = np.linspace(0, grid_rows-1, min(10, grid_rows))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])
        ax.set_yticklabels([f'{int(y)}' for y in y_ticks])
        
        plt.tight_layout()
        return self._fig_to_array_optimized(fig)
    
    def create_radial_profile_optimized(self, k1: float, k2: float, k3: float,
                                      p1: float, p2: float, num_points: int = 100) -> np.ndarray:
        """
        Optimized radial profile generation using vectorized operations
        """
        # Calculate max radius
        max_radius = min(self.cx, self.cy, self.image_width - self.cx, self.image_height - self.cy)
        
        # Create radius array (vectorized)
        radii = np.linspace(0, max_radius, num_points)
        
        # Create test points along horizontal axis (vectorized)
        x_points = self.cx + radii
        y_points = np.full_like(radii, self.cy)
        
        # Apply distortion to all points at once (vectorized)
        x_dist, y_dist = self.apply_brown_conrady_distortion_vectorized(
            x_points, y_points, k1, k2, k3, p1, p2
        )
        
        # Calculate radial distortion (vectorized)
        r_dist = np.sqrt((x_dist - self.cx)**2 + (y_dist - self.cy)**2)
        distortions = r_dist - radii
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        ax.plot(radii, distortions, 'b-', linewidth=3, label='Radial Distortion')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Radial Distance from Center (pixels)', fontsize=12)
        ax.set_ylabel('Distortion (pixels)', fontsize=12)
        
        distortion_type = self._classify_distortion_fast(k1, k2, k3, p1, p2)
        max_distortion = np.max(np.abs(distortions))
        ax.set_title(f'Radial Distortion Profile - {distortion_type}\n'
                    f'K1={k1:.3f}, K2={k2:.3f}, K3={k3:.3f}, Max: {max_distortion:.2f}px', 
                    fontsize=14, fontweight='bold')
        
        # Add distortion info
        ax.text(0.05, 0.95, f"Max Distortion: {max_distortion:.2f} px\nAt radius: {radii[np.argmax(np.abs(distortions))]:.0f} px", 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return self._fig_to_array_optimized(fig)
    
    def _classify_distortion_fast(self, k1: float, k2: float, k3: float, 
                                p1: float, p2: float) -> str:
        """Fast distortion classification using optimized logic"""
        # Pre-compute absolute values
        abs_k1, abs_k2, abs_k3 = abs(k1), abs(k2), abs(k3)
        abs_p1, abs_p2 = abs(p1), abs(p2)
        
        radial_magnitude = abs_k1 + abs_k2 + abs_k3
        tangential_magnitude = abs_p1 + abs_p2
        
        # Fast classification using thresholds
        if radial_magnitude < 1e-6 and tangential_magnitude < 1e-6:
            return "No Distortion"
        elif radial_magnitude < 1e-6:
            return "Tangential Only"
        elif k1 < -1e-6:
            return "Severe Barrel" if radial_magnitude > 0.3 else "Moderate Barrel" if radial_magnitude > 0.1 else "Mild Barrel"
        elif k1 > 1e-6:
            return "Severe Pincushion" if radial_magnitude > 0.3 else "Moderate Pincushion" if radial_magnitude > 0.1 else "Mild Pincushion"
        else:
            return "Complex Distortion"
    
    def _fig_to_array_optimized(self, fig) -> np.ndarray:
        """Optimized figure to array conversion with memory management"""
        # Use tight layout and optimized DPI
        fig.tight_layout(pad=1.0)
        
        # Create buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Read image data
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        # Close figure to free memory
        plt.close(fig)
        
        # Decode image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_enhanced_distortion_visualizer():
    """Create the enhanced Gradio interface with GDC export capability"""
    
    visualizer = OptimizedDistortionVisualizerWithGDC()
    
    def update_visualizations(k1, k2, k3, p1, p2, grid_rows, grid_cols):
        """Update all visualizations with performance timing - FIXED to prevent loops"""
        try:
            start_time = time.time()
            
            # Generate visualizations using optimized methods
            grid_viz = visualizer.create_grid_visualization_optimized(
                grid_rows, grid_cols, k1, k2, k3, p1, p2
            )
            
            heatmap_viz = visualizer.create_magnitude_heatmap_optimized(
                grid_rows, grid_cols, k1, k2, k3, p1, p2
            )
            
            profile_viz = visualizer.create_radial_profile_optimized(k1, k2, k3, p1, p2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create optimized parameter summary
            distortion_type = visualizer._classify_distortion_fast(k1, k2, k3, p1, p2)
            total_distortion = abs(k1) + abs(k2) + abs(k3) + abs(p1) + abs(p2)
            
            # Calculate some quick statistics
            max_radius = min(visualizer.cx, visualizer.cy)
            
            summary = f"""
**⚡ Real-Time Distortion Analysis** (Updated in {processing_time:.3f}s)

**🎯 Type:** {distortion_type}  
**📊 Total Magnitude:** {total_distortion:.4f}  
**🎪 Max Analysis Radius:** {max_radius:.0f} pixels

**🔴 Radial Coefficients:**
- **K1** (Primary): {k1:.4f} {'🪣 Barrel' if k1 < -0.01 else '📐 Pincushion' if k1 > 0.01 else '⚪ Minimal'}
- **K2** (Secondary): {k2:.4f} {'✅ Active' if abs(k2) > 0.01 else '💤 Inactive'}
- **K3** (Tertiary): {k3:.4f} {'✅ Active' if abs(k3) > 0.005 else '💤 Inactive'}

**🎯 Tangential Coefficients:**
- **P1**: {p1:.4f} {'✅ Active' if abs(p1) > 0.005 else '💤 Inactive'}
- **P2**: {p2:.4f} {'✅ Active' if abs(p2) > 0.005 else '💤 Inactive'}

**📋 Grid Configuration:**
- **Size:** {grid_rows} × {grid_cols} = {grid_rows * grid_cols} points
- **Performance:** {'🚀 Fast' if grid_rows * grid_cols < 100 else '⚡ Medium' if grid_rows * grid_cols < 400 else '🐌 Slow'} update rate

**🖼️ Image Setup:**
- **Resolution:** {visualizer.image_width} × {visualizer.image_height} pixels
- **Center:** ({visualizer.cx:.0f}, {visualizer.cy:.0f})
            """
            
            return grid_viz, heatmap_viz, profile_viz, summary
            
        except Exception as e:
            error_msg = f"⚠️ **Error generating visualizations:** {str(e)}\n\nTry reducing grid size or adjusting parameters."
            return None, None, None, error_msg
    
    def export_gdc_data(k1, k2, k3, p1, p2, grid_rows, grid_cols, gdc_width, gdc_height):
        """Export GDC grid data with progress feedback - FIXED to return file properly"""
        try:
            start_time = time.time()
            
            # Generate GDC export
            gdc_file_path = visualizer.export_gdc_format_only(
                grid_rows, grid_cols, k1, k2, k3, p1, p2,
                int(gdc_width), int(gdc_height)
            )
            
            export_time = time.time() - start_time
            
            # Generate export summary
            distortion_type = visualizer._classify_distortion_fast(k1, k2, k3, p1, p2)
            total_points = grid_rows * grid_cols
            
            summary = f"""
**✅ GDC Export Completed Successfully!** (Generated in {export_time:.2f}s)

**📦 Export Details:**
- **File:** {os.path.basename(gdc_file_path)}
- **Distortion Type:** {distortion_type}
- **Grid Size:** {grid_rows} × {grid_cols} = {total_points} points
- **Target Resolution:** {int(gdc_width)} × {int(gdc_height)} pixels
- **Format:** Hardware-Ready GDC with Hex Values

**🎯 Parameters Used:**
- K1={k1:.4f}, K2={k2:.4f}, K3={k3:.4f}
- P1={p1:.4f}, P2={p2:.4f}

**🔧 Hardware Integration:**
- **Field Names:** yuv_gdc_grid_dx_R_C, yuv_gdc_grid_dy_R_C
- **Fixed-Point:** Bit-shifted values for ISP/FPGA
- **Hex Encoding:** Two's complement for negative values
- **Scale Factors:** X={int((gdc_width * (2**14)) / visualizer.image_width)}, Y={int((gdc_height * (2**14)) / visualizer.image_height)}

**📊 File Structure:**
- Header with metadata and parameters
- DX grid values: {total_points} entries
- DY grid values: {total_points} entries
- Total entries: {total_points * 2}

**⚡ Performance:** Fast direct conversion - no interpolation
Download the CSV file below for hardware integration.
            """
            
            return gdc_file_path, summary
            
        except Exception as e:
            error_summary = f"""
**❌ GDC Export Failed**

**Error:** {str(e)}

**Troubleshooting:**
- Reduce grid size if memory issues occur
- Check that parameters are within valid ranges
- Ensure sufficient disk space for export files

**Valid Ranges:**
- Grid size: 3-30 rows/columns  
- GDC dimensions: 1024-16384 pixels
- Distortion coefficients: See parameter tooltips
            """
            return None, error_summary
    
    # Create Gradio interface with enhanced export functionality
    with gr.Blocks(
        title="⚡ Enhanced Distortion Visualizer with GDC Export", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1600px !important;
        }
        #summary_output {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }
        #export_summary {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🔍 **Enhanced Distortion Visualizer with GDC Export**
        
        Interactive visualization and export tool for lens distortion using the **Brown-Conrady model**.
        Features real-time distortion mapping with **hardware-ready GDC format export** capabilities.
        
        📊 **Features:**
        - **Real-time Visualization**: Grid view, heatmap, and radial profile
        - **GDC Export**: Hardware-ready format for ISP/FPGA implementation
        - **Performance Optimized**: Vectorized calculations for smooth interaction
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🌀 **Distortion Parameters**")
                
                # Optimized parameter sliders with better step sizes
                k1 = gr.Slider(
                    minimum=-0.5, maximum=0.5, step=0.005, value=-0.2,
                    label="K1 (Primary Radial)", 
                    info="Negative=Barrel, Positive=Pincushion"
                )
                
                k2 = gr.Slider(
                    minimum=-0.2, maximum=0.2, step=0.005, value=0.05,
                    label="K2 (Secondary Radial)", 
                    info="Higher-order correction"
                )
                
                k3 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.002, value=0.0,
                    label="K3 (Tertiary Radial)", 
                    info="Extreme distortion fine-tuning"
                )
                
                p1 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.002, value=0.0,
                    label="P1 (Tangential)", 
                    info="Lens decentering compensation"
                )
                
                p2 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.002, value=0.0,
                    label="P2 (Tangential)", 
                    info="Secondary decentering correction"
                )
                
                gr.Markdown("### 🎯 **Grid Configuration**")
                
                grid_rows = gr.Slider(
                    minimum=3, maximum=30, step=1, value=7,
                    label="Grid Rows", 
                    info="Vertical resolution"
                )
                
                grid_cols = gr.Slider(
                    minimum=3, maximum=30, step=1, value=9,
                    label="Grid Columns", 
                    info="Horizontal resolution"
                )
                
                gr.Markdown("### 🎛️ **Quick Presets**")
                
                # Optimized preset functions
                def apply_barrel_preset():
                    return -0.2, 0.05, 0.0, 0.0, 0.0
                
                def apply_pincushion_preset():
                    return 0.15, -0.03, 0.0, 0.0, 0.0
                
                def apply_fisheye_preset():
                    return -0.4, 0.1, -0.02, 0.0, 0.0
                
                def apply_complex_preset():
                    return -0.3, 0.08, -0.01, 0.02, 0.01
                
                def apply_identity_preset():
                    return 0.0, 0.0, 0.0, 0.0, 0.0
                
                with gr.Row():
                    barrel_btn = gr.Button("🪣 Barrel", variant="secondary", size="sm")
                    pincushion_btn = gr.Button("📐 Pincushion", variant="secondary", size="sm")
                
                with gr.Row():
                    fisheye_btn = gr.Button("🐟 Fisheye", variant="secondary", size="sm")
                    complex_btn = gr.Button("🌀 Complex", variant="secondary", size="sm")
                
                with gr.Row():
                    identity_btn = gr.Button("🔄 Reset", variant="primary", size="sm")
                
                gr.Markdown("### 📊 **Real-Time Analysis**")
                summary_output = gr.Markdown(
                    "🔄 **Initializing optimizer...** Adjust parameters to see real-time analysis.",
                    elem_id="summary_output"
                )
                
                gr.Markdown("### 📦 **GDC Export Configuration**")
                
                gdc_width = gr.Number(
                    value=8192, minimum=1024, maximum=16384, step=1,
                    label="GDC Target Width",
                    info="Hardware target width (typical: 8192)"
                )
                
                gdc_height = gr.Number(
                    value=6144, minimum=768, maximum=12288, step=1,
                    label="GDC Target Height", 
                    info="Hardware target height (typical: 6144)"
                )
                
                export_btn = gr.Button(
                    "📦 Export GDC Grid", 
                    variant="primary", 
                    size="lg"
                )
                
                export_file = gr.File(
                    label="📁 Download GDC File",
                    visible=False
                )
                
                export_summary = gr.Markdown(
                    "**💡 Ready to Export:** Configure parameters above and click 'Export GDC Grid' to generate hardware-ready file.",
                    elem_id="export_summary"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 **Optimized Real-Time Visualizations**")
                
                with gr.Tab("🌐 Grid Distortion"):
                    grid_output = gr.Image(
                        label="Vectorized Grid Visualization",
                        height=500,
                        show_download_button=True
                    )
                    gr.Markdown("""
                    **⚡ Optimized Rendering:**
                    - **LineCollection**: Fast batch line rendering
                    - **Vectorized Arrows**: Efficient displacement vectors  
                    - **Smart Filtering**: Only significant distortions shown
                    - **Cached Coordinates**: Reused grid calculations
                    """)
                
                with gr.Tab("🌡️ Magnitude Heatmap"):
                    heatmap_output = gr.Image(
                        label="Vectorized Magnitude Heatmap",
                        height=500,
                        show_download_button=True
                    )
                    gr.Markdown("""
                    **⚡ Performance Features:**
                    - **NumPy Norm**: Vectorized magnitude calculation
                    - **Optimized Colormap**: Fast hot colormap rendering
                    - **Bilinear Interpolation**: Smooth value transitions
                    - **Efficient Memory**: Reduced array copying
                    """)
                
                with gr.Tab("📈 Radial Profile"):
                    profile_output = gr.Image(
                        label="Vectorized Radial Profile",
                        height=500,
                        show_download_button=True
                    )
                    gr.Markdown("""
                    **⚡ Optimization Details:**
                    - **Vectorized Points**: Batch coordinate processing
                    - **Single Distortion Call**: Process all radii at once
                    - **NumPy Operations**: Fast array-based calculations
                    - **Memory Efficient**: Minimal temporary arrays
                    """)
                
                with gr.Tab("📦 GDC Format Info"):
                    gr.Markdown("""
                    ### 🔧 **GDC Hardware Format Details**
                    
                    **What is GDC Format?**
                    - **Geometric Distortion Correction** format for hardware implementation
                    - **Fixed-point arithmetic** with bit-shifting for ISP/FPGA
                    - **Two's complement** hex encoding for negative values
                    - **Field names** ready for direct integration: `yuv_gdc_grid_dx_R_C`, `yuv_gdc_grid_dy_R_C`
                    
                    **Hardware Conversion Formula:**
                    ```
                    grid_distort_x = (ceil(((Delta_x * scale_factor_x) << 2) / 2^14)) << 7
                    grid_distort_y = (ceil(((Delta_y * scale_factor_y) << 2) / 2^14)) << 7
                    
                    where:
                    scale_factor_x = round((gdc_width << 14) / image_width)
                    scale_factor_y = round((gdc_height << 14) / image_height)
                    ```
                    
                    **Export File Contents:**
                    - **Header**: Metadata with parameters and image info
                    - **DX Values**: `yuv_gdc_grid_dx_row_col, decimal_value, hex_value`
                    - **DY Values**: `yuv_gdc_grid_dy_row_col, decimal_value, hex_value`
                    - **Direct Conversion**: Original grid resolution only - no interpolation
                    - **Ready to Use**: Direct copy-paste into hardware configurations
                    
                    **Typical Use Cases:**
                    - 📷 Camera ISP distortion correction modules
                    - 🔧 FPGA-based real-time image processing
                    - 🎯 Embedded vision system calibration
                    - 📊 Hardware validation and quality control
                    
                    **Integration Example:**
                    ```c
                    // Hardware register values from exported file
                    yuv_gdc_grid_dx_0_0 = -1536;  // 0xFFFFFA00
                    yuv_gdc_grid_dy_0_0 = 2048;   // 0x00000800
                    ```
                    """)
        
        # Event handlers for visualization updates - FIXED to prevent infinite loops
        def update_export_file_visibility(file_path, summary):
            """Update export file visibility based on export success"""
            if file_path is not None:
                return gr.update(visible=True, value=file_path)
            else:
                return gr.update(visible=False)
        
        # Connect preset buttons
        barrel_btn.click(
            fn=apply_barrel_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        pincushion_btn.click(
            fn=apply_pincushion_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        fisheye_btn.click(
            fn=apply_fisheye_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        complex_btn.click(
            fn=apply_complex_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        identity_btn.click(
            fn=apply_identity_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        # Connect export functionality - FIXED to prevent loops
        export_btn.click(
            fn=export_gdc_data,
            inputs=[k1, k2, k3, p1, p2, grid_rows, grid_cols, gdc_width, gdc_height],
            outputs=[export_file, export_summary]
        ).then(
            fn=update_export_file_visibility,
            inputs=[export_file, export_summary],
            outputs=[export_file]
        )
        
        # FIXED: Real-time updates without infinite loops
        visualization_inputs = [k1, k2, k3, p1, p2, grid_rows, grid_cols]
        visualization_outputs = [grid_output, heatmap_output, profile_output, summary_output]
        
        # Connect parameter changes to visualization updates only (no circular dependencies)
        for input_component in visualization_inputs:
            input_component.change(
                fn=update_visualizations,
                inputs=visualization_inputs,
                outputs=visualization_outputs,
                show_progress=False
            )
        
        # Initial load
        interface.load(
            fn=update_visualizations,
            inputs=visualization_inputs,
            outputs=visualization_outputs
        )
        
    return interface

def launch_enhanced_visualizer():
    """Launch the enhanced distortion visualizer application with GDC export"""
    interface = create_enhanced_distortion_visualizer()
    
    interface.launch(
        server_name="localhost",
        server_port=7862,
        share=False,
        show_error=True,
        quiet=False,
        debug=False,
        max_threads=4
    )

if __name__ == "__main__":
    print("⚡ Starting ENHANCED Distortion Visualizer with GDC Export...")
    print("=" * 70)
    print("🚀 High-Performance Vectorized Implementation")
    print("📊 Real-time visualization with advanced optimizations")
    print("📦 GDC Export: Hardware-ready format generation")
    print("🎯 Fixed infinite loop issue - stable performance")
    print("=" * 70)
    print("🌐 Navigate to the URL shown below...")
    print("⚡ Performance improvements:")
    print("   • Vectorized Brown-Conrady model")
    print("   • Cached grid generation")
    print("   • Optimized matplotlib rendering")
    print("   • Efficient memory management")
    print("📦 GDC Export features:")
    print("   • Hardware-ready CSV format only")
    print("   • Fixed-point arithmetic with bit-shifting")
    print("   • Two's complement hex encoding")
    print("   • Direct ISP/FPGA integration")
    print("🔧 Fixes applied:")
    print("   • Removed infinite loop in visualization updates")
    print("   • Simplified export to GDC format only")
    print("   • Optimized event handling")
    print("=" * 70)
    
    launch_enhanced_visualizer()