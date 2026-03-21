#!/usr/bin/env python3
"""
GDC Visualization Module

Comprehensive visualization capabilities for GDC grid processing and analysis.
Provides various plot types for grid analysis, comparison, and quality assessment.

Author: Balaji R
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tempfile
from typing import Optional, Tuple, Dict, Any
from matplotlib import cm
from matplotlib import colors as mcolors


class GDCVisualizer:
    """
    Visualization engine for GDC grid processing and image remapping.
    
    Provides comprehensive visualization capabilities including:
    - Grid heatmaps and comparisons
    - Displacement field analysis
    - Statistical plots
    - Quality assessment visualizations
    """
    
    def __init__(self):
        self.default_figsize = (12, 8)
        self.default_dpi = 120
        self.colormap = 'RdBu_r'
        
    def create_grid_heatmap(self, dx_grid: np.ndarray, dy_grid: np.ndarray, 
                          title: str = "Grid Visualization") -> str:
        """
        Create side-by-side heatmap visualization of DX and DY grids.
        
        Args:
            dx_grid: DX displacement grid
            dy_grid: DY displacement grid
            title: Plot title
            
        Returns:
            Path to saved visualization file
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.default_figsize)
            
            # DX Grid heatmap
            sns.heatmap(dx_grid, ax=ax1, cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.8})
            ax1.set_title(f'{title} - DX Displacement')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            
            # DY Grid heatmap
            sns.heatmap(dy_grid, ax=ax2, cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.8})
            ax2.set_title(f'{title} - DY Displacement')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            
            # Add grid statistics as text
            dx_stats = f"DX: {np.min(dx_grid):.1f} to {np.max(dx_grid):.1f}"
            dy_stats = f"DY: {np.min(dy_grid):.1f} to {np.max(dy_grid):.1f}"
            fig.suptitle(f"{title}\n{dx_stats} | {dy_stats}", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating grid heatmap: {e}")
            plt.close('all')
            return None
    
    def create_grid_comparison(self, original_dx: np.ndarray, original_dy: np.ndarray,
                             interpolated_dx: np.ndarray, interpolated_dy: np.ndarray) -> str:
        """
        Create comprehensive grid comparison visualization.
        
        Args:
            original_dx: Original DX grid
            original_dy: Original DY grid
            interpolated_dx: Interpolated DX grid
            interpolated_dy: Interpolated DY grid
            
        Returns:
            Path to saved visualization file
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original DX
            sns.heatmap(original_dx, ax=axes[0,0], cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.6})
            axes[0,0].set_title(f'Original DX ({original_dx.shape[0]}×{original_dx.shape[1]})')
            axes[0,0].set_xlabel('Column')
            axes[0,0].set_ylabel('Row')
            
            # Interpolated DX
            sns.heatmap(interpolated_dx, ax=axes[0,1], cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.6})
            axes[0,1].set_title(f'Interpolated DX ({interpolated_dx.shape[0]}×{interpolated_dx.shape[1]})')
            axes[0,1].set_xlabel('Column')
            axes[0,1].set_ylabel('Row')
            
            # Original DY
            sns.heatmap(original_dy, ax=axes[1,0], cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.6})
            axes[1,0].set_title(f'Original DY ({original_dy.shape[0]}×{original_dy.shape[1]})')
            axes[1,0].set_xlabel('Column')
            axes[1,0].set_ylabel('Row')
            
            # Interpolated DY
            sns.heatmap(interpolated_dy, ax=axes[1,1], cmap=self.colormap, center=0, 
                       annot=False, cbar=True, cbar_kws={'shrink': 0.6})
            axes[1,1].set_title(f'Interpolated DY ({interpolated_dy.shape[0]}×{interpolated_dy.shape[1]})')
            axes[1,1].set_xlabel('Column')
            axes[1,1].set_ylabel('Row')
            
            # Calculate interpolation factors
            scale_x = interpolated_dx.shape[1] / original_dx.shape[1]
            scale_y = interpolated_dx.shape[0] / original_dx.shape[0]
            
            plt.suptitle(f'Grid Comparison: Original vs Interpolated\n'
                        f'Scale Factor: {scale_x:.1f}× (cols) | {scale_y:.1f}× (rows)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating grid comparison: {e}")
            plt.close('all')
            return None
    
    def create_displacement_analysis(self, dx_grid: np.ndarray, dy_grid: np.ndarray,
                                   scale_factor: float = 0.1) -> str:
        """
        Create displacement field visualization with vectors and magnitude.
        
        Args:
            dx_grid: DX displacement grid
            dy_grid: DY displacement grid
            scale_factor: Scale factor for vector visualization
            
        Returns:
            Path to saved visualization file
        """
        try:
            rows, cols = dx_grid.shape
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: Vector field
            target_x, target_y = np.meshgrid(np.arange(cols), np.arange(rows))
            displaced_x = target_x + dx_grid * scale_factor
            displaced_y = target_y + dy_grid * scale_factor
            
            # Plot original grid points
            ax1.scatter(target_x, target_y, c='blue', s=30, alpha=0.7, label='Target Points')
            
            # Plot displaced points
            ax1.scatter(displaced_x, displaced_y, c='red', s=25, alpha=0.7, label='Displaced Points')
            
            # Draw displacement vectors (subsample for clarity)
            step = max(1, max(rows, cols) // 15)  # Adaptive step size
            for i in range(0, rows, step):
                for j in range(0, cols, step):
                    dx = displaced_x[i,j] - target_x[i,j]
                    dy = displaced_y[i,j] - target_y[i,j]
                    if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only show significant displacements
                        ax1.arrow(target_x[i,j], target_y[i,j], dx, dy, 
                                head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.6)
            
            ax1.set_aspect('equal')
            ax1.legend()
            ax1.set_title('Displacement Vector Field')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Displacement magnitude
            magnitude = np.sqrt(dx_grid**2 + dy_grid**2)
            im = ax2.imshow(magnitude, cmap='hot', interpolation='bilinear')
            plt.colorbar(im, ax=ax2, label='Displacement Magnitude')
            ax2.set_title('Displacement Magnitude')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            
            # Add statistics
            max_displacement = np.max(magnitude)
            mean_displacement = np.mean(magnitude)
            fig.suptitle(f'Displacement Analysis\n'
                        f'Max: {max_displacement:.2f} | Mean: {mean_displacement:.2f} | Scale: {scale_factor}×',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating displacement analysis: {e}")
            plt.close('all')
            return None
    
    def create_grid_statistics_plot(self, statistics: Dict[str, Any]) -> str:
        """
        Create statistical analysis plots for grid data.
        
        Args:
            statistics: Dictionary containing grid statistics
            
        Returns:
            Path to saved visualization file
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Extract statistics for different grids
            grid_types = ['dx_original', 'dy_original', 'dx_interpolated', 'dy_interpolated']
            available_stats = [stats for stats in grid_types if stats in statistics]
            
            if not available_stats:
                # Create empty plot with message
                fig.text(0.5, 0.5, 'No grid statistics available', 
                        ha='center', va='center', fontsize=16)
                return self._save_plot(fig)
            
            # Plot 1: Range comparison
            ax1 = axes[0, 0]
            ranges = []
            labels = []
            for stat_key in available_stats:
                stat_data = statistics[stat_key]
                ranges.append([stat_data['min'], stat_data['max']])
                labels.append(stat_key.replace('_', ' ').title())
            
            if ranges:
                ranges = np.array(ranges)
                x_pos = np.arange(len(labels))
                ax1.bar(x_pos, ranges[:, 1] - ranges[:, 0], bottom=ranges[:, 0], alpha=0.7)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(labels, rotation=45)
                ax1.set_title('Value Ranges')
                ax1.set_ylabel('Displacement Value')
            
            # Plot 2: Mean and Standard Deviation
            ax2 = axes[0, 1]
            means = []
            stds = []
            for stat_key in available_stats:
                stat_data = statistics[stat_key]
                means.append(stat_data['mean'])
                stds.append(stat_data['std'])
            
            if means:
                x_pos = np.arange(len(labels))
                ax2.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, capthick=2)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels, rotation=45)
                ax2.set_title('Mean ± Standard Deviation')
                ax2.set_ylabel('Displacement Value')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Shape comparison
            ax3 = axes[1, 0]
            shapes = []
            total_elements = []
            for stat_key in available_stats:
                stat_data = statistics[stat_key]
                if 'shape' in stat_data:
                    shape = stat_data['shape']
                    shapes.append(f"{shape[0]}×{shape[1]}")
                    total_elements.append(stat_data.get('total_elements', shape[0] * shape[1]))
            
            if shapes:
                ax3.bar(range(len(shapes)), total_elements, alpha=0.7)
                ax3.set_xticks(range(len(shapes)))
                ax3.set_xticklabels(shapes, rotation=45)
                ax3.set_title('Grid Sizes')
                ax3.set_ylabel('Total Elements')
            
            # Plot 4: Distribution comparison (if available)
            ax4 = axes[1, 1]
            percentiles = ['percentile_25', 'percentile_75']
            if all(perc in statistics.get(available_stats[0], {}) for perc in percentiles):
                box_data = []
                for stat_key in available_stats:
                    stat_data = statistics[stat_key]
                    # Approximate box plot data from statistics
                    q1 = stat_data.get('percentile_25', stat_data['mean'] - stat_data['std'])
                    q3 = stat_data.get('percentile_75', stat_data['mean'] + stat_data['std'])
                    median = stat_data.get('median', stat_data['mean'])
                    box_data.append([stat_data['min'], q1, median, q3, stat_data['max']])
                
                # Simple box-like visualization
                for i, (label, data) in enumerate(zip(labels, box_data)):
                    ax4.plot([i, i], [data[0], data[4]], 'k-', alpha=0.5)  # Min-max line
                    ax4.plot([i-0.2, i+0.2], [data[1], data[1]], 'b-', linewidth=2)  # Q1
                    ax4.plot([i-0.2, i+0.2], [data[2], data[2]], 'r-', linewidth=3)  # Median
                    ax4.plot([i-0.2, i+0.2], [data[3], data[3]], 'b-', linewidth=2)  # Q3
                
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, rotation=45)
                ax4.set_title('Distribution Summary')
                ax4.set_ylabel('Displacement Value')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Distribution data\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.suptitle('Grid Statistics Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating statistics plot: {e}")
            plt.close('all')
            return None
    
    def create_processing_flow_diagram(self, processing_history: list) -> str:
        """
        Create processing flow visualization.
        
        Args:
            processing_history: List of processing operations
            
        Returns:
            Path to saved visualization file
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if not processing_history:
                ax.text(0.5, 0.5, 'No processing history available', 
                       ha='center', va='center', fontsize=16)
                ax.set_title('Processing Flow')
                return self._save_plot(fig)
            
            # Create timeline of operations
            operations = [op['operation'] for op in processing_history]
            statuses = [op['status'] for op in processing_history]
            timestamps = [op['timestamp'] for op in processing_history]
            
            # Color mapping for status
            colors = ['green' if status == 'success' else 'red' for status in statuses]
            
            # Create horizontal timeline
            y_pos = np.arange(len(operations))
            bars = ax.barh(y_pos, [1] * len(operations), color=colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([op.replace('_', ' ').title() for op in operations])
            ax.set_xlabel('Processing Steps')
            ax.set_title('Processing Flow Timeline')
            
            # Add status indicators
            for i, (bar, status) in enumerate(zip(bars, statuses)):
                symbol = '✓' if status == 'success' else '✗'
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       symbol, ha='left', va='center', fontweight='bold', fontsize=12)
            
            # Add legend
            success_patch = plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Success')
            error_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Error')
            ax.legend(handles=[success_patch, error_patch], loc='lower right')
            
            ax.set_xlim(0, 1.2)
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating processing flow diagram: {e}")
            plt.close('all')
            return None
    
    def create_image_comparison(self, input_image: np.ndarray, 
                              output_image: np.ndarray) -> str:
        """
        Create side-by-side image comparison.
        
        Args:
            input_image: Original input image
            output_image: Processed output image
            
        Returns:
            Path to saved visualization file
        """
        try:
            if input_image is None and output_image is None:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Input image
            if input_image is not None:
                if len(input_image.shape) == 3:
                    axes[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
                else:
                    axes[0].imshow(input_image, cmap='gray')
                axes[0].set_title(f'Input Image\n{input_image.shape}')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, 'No Input Image', ha='center', va='center')
                axes[0].set_title('Input Image')
            
            # Output image
            if output_image is not None:
                if len(output_image.shape) == 3:
                    axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                else:
                    axes[1].imshow(output_image, cmap='gray')
                axes[1].set_title(f'Output Image\n{output_image.shape}')
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'No Output Image', ha='center', va='center')
                axes[1].set_title('Output Image')
            
            plt.suptitle('Image Processing Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return self._save_plot(fig)
            
        except Exception as e:
            print(f"Error creating image comparison: {e}")
            plt.close('all')
            return None
    
    def _save_plot(self, fig) -> str:
        """
        Save matplotlib figure to temporary file.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Path to saved file
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            fig.savefig(temp_file.name, dpi=self.default_dpi, 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            return temp_file.name
        except Exception as e:
            print(f"Error saving plot: {e}")
            plt.close(fig)
            return None