#!/usr/bin/env python3
"""
Standalone Distortion Map Visualizer
Real-time interactive visualization of lens distortion parameters

This utility provides real-time visualization of distortion maps using 
the Brown-Conrady model with adjustable parameters.

Author: Balaji R
License: MIT
"""

import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import io
from typing import Tuple, Optional

class DistortionVisualizer:
    """Real-time distortion map visualizer"""
    
    def __init__(self):
        self.image_width = 1280
        self.image_height = 720
        
    def apply_brown_conrady_distortion(self, x: np.ndarray, y: np.ndarray, 
                                     k1: float, k2: float, k3: float,
                                     p1: float, p2: float,
                                     cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Brown-Conrady distortion model to coordinates
        
        Args:
            x, y: Input coordinates
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            cx, cy: Principal point coordinates
            
        Returns:
            Distorted coordinates (x_dist, y_dist)
        """
        # Normalize coordinates relative to principal point
        x_norm = (x - cx) / self.image_width
        y_norm = (y - cy) / self.image_height
        
        # Calculate radial distance squared
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion factor
        radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        # Tangential distortion
        tangential_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
        tangential_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
        
        # Apply distortion
        x_dist_norm = x_norm * radial_factor + tangential_x
        y_dist_norm = y_norm * radial_factor + tangential_y
        
        # Convert back to pixel coordinates
        x_dist = x_dist_norm * self.image_width + cx
        y_dist = y_dist_norm * self.image_height + cy
        
        return x_dist, y_dist
    
    def generate_distortion_grid(self, grid_rows: int, grid_cols: int,
                               k1: float, k2: float, k3: float,
                               p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate distortion grid with current parameters
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            source_coords, target_coords: Grid coordinate arrays
        """
        # Principal point (image center)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        
        # Create target grid (undistorted)
        x_positions = np.linspace(0, self.image_width, grid_cols)
        y_positions = np.linspace(0, self.image_height, grid_rows)
        
        target_coords = np.zeros((grid_rows, grid_cols, 2))
        source_coords = np.zeros((grid_rows, grid_cols, 2))
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Target coordinates (undistorted grid)
                target_x = x_positions[col]
                target_y = y_positions[row]
                target_coords[row, col] = [target_x, target_y]
                
                # Apply distortion to get source coordinates
                source_x, source_y = self.apply_brown_conrady_distortion(
                    np.array([target_x]), np.array([target_y]),
                    k1, k2, k3, p1, p2, cx, cy
                )
                source_coords[row, col] = [source_x[0], source_y[0]]
        
        return source_coords, target_coords
    
    def create_grid_visualization(self, grid_rows: int, grid_cols: int,
                                k1: float, k2: float, k3: float,
                                p1: float, p2: float) -> np.ndarray:
        """
        Create grid visualization showing distortion
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            Visualization image as numpy array
        """
        # Generate distortion grid
        source_coords, target_coords = self.generate_distortion_grid(
            grid_rows, grid_cols, k1, k2, k3, p1, p2
        )
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the plot
        ax.set_xlim(0, self.image_width)
        ax.set_ylim(0, self.image_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
        
        # Draw grid lines (target grid) in light gray
        for row in range(grid_rows):
            for col in range(grid_cols - 1):  # Horizontal lines
                x_coords = [target_coords[row, col, 0], target_coords[row, col + 1, 0]]
                y_coords = [target_coords[row, col, 1], target_coords[row, col + 1, 1]]
                ax.plot(x_coords, y_coords, 'lightgray', linewidth=1, alpha=0.6)
        
        for row in range(grid_rows - 1):  # Vertical lines
            for col in range(grid_cols):
                x_coords = [target_coords[row, col, 0], target_coords[row + 1, col, 0]]
                y_coords = [target_coords[row, col, 1], target_coords[row + 1, col, 1]]
                ax.plot(x_coords, y_coords, 'lightgray', linewidth=1, alpha=0.6)
        
        # Draw distorted grid lines in red
        for row in range(grid_rows):
            for col in range(grid_cols - 1):  # Horizontal lines
                x_coords = [source_coords[row, col, 0], source_coords[row, col + 1, 0]]
                y_coords = [source_coords[row, col, 1], source_coords[row, col + 1, 1]]
                ax.plot(x_coords, y_coords, 'red', linewidth=2, alpha=0.8)
        
        for row in range(grid_rows - 1):  # Vertical lines
            for col in range(grid_cols):
                x_coords = [source_coords[row, col, 0], source_coords[row + 1, col, 0]]
                y_coords = [source_coords[row, col, 1], source_coords[row + 1, col, 1]]
                ax.plot(x_coords, y_coords, 'red', linewidth=2, alpha=0.8)
        
        # Draw distortion vectors
        for row in range(grid_rows):
            for col in range(grid_cols):
                target_pt = target_coords[row, col]
                source_pt = source_coords[row, col]
                
                # Calculate vector
                dx = source_pt[0] - target_pt[0]
                dy = source_pt[1] - target_pt[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                
                # Only draw significant vectors
                if magnitude > 2:
                    ax.arrow(target_pt[0], target_pt[1], dx, dy,
                           head_width=8, head_length=8, fc='blue', ec='blue', alpha=0.7)
                
                # Draw target points
                ax.plot(target_pt[0], target_pt[1], 'bo', markersize=3, alpha=0.8)
                # Draw distorted points
                ax.plot(source_pt[0], source_pt[1], 'ro', markersize=3, alpha=0.8)
        
        # Add title and labels
        distortion_type = self._classify_distortion(k1, k2, k3, p1, p2)
        ax.set_title(f'Distortion Grid Visualization - {distortion_type}\n'
                    f'K1={k1:.3f}, K2={k2:.3f}, K3={k3:.3f}, P1={p1:.3f}, P2={p2:.3f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='lightgray', linewidth=1, label='Target Grid'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Distorted Grid'),
            plt.Line2D([0], [0], marker='o', color='blue', linewidth=0, markersize=4, label='Target Points'),
            plt.Line2D([0], [0], marker='o', color='red', linewidth=0, markersize=4, label='Distorted Points'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Distortion Vectors')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Convert to image
        return self._fig_to_array(fig)
    
    def create_magnitude_heatmap(self, grid_rows: int, grid_cols: int,
                               k1: float, k2: float, k3: float,
                               p1: float, p2: float) -> np.ndarray:
        """
        Create heatmap showing distortion magnitude
        
        Args:
            grid_rows, grid_cols: Grid dimensions
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            Heatmap image as numpy array
        """
        # Generate distortion grid
        source_coords, target_coords = self.generate_distortion_grid(
            grid_rows, grid_cols, k1, k2, k3, p1, p2
        )
        
        # Calculate distortion magnitudes
        distortion_magnitudes = np.zeros((grid_rows, grid_cols))
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                delta = source_coords[row, col] - target_coords[row, col]
                magnitude = np.linalg.norm(delta)
                distortion_magnitudes[row, col] = magnitude
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(distortion_magnitudes, cmap='hot', 
                      interpolation='bilinear', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Distortion Magnitude (pixels)')
        
        # Formatting
        distortion_type = self._classify_distortion(k1, k2, k3, p1, p2)
        ax.set_title(f'Distortion Magnitude Heatmap - {distortion_type}\n'
                    f'Grid: {grid_cols}×{grid_rows}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        
        # Set ticks
        ax.set_xticks(range(0, grid_cols, max(1, grid_cols//10)))
        ax.set_yticks(range(0, grid_rows, max(1, grid_rows//10)))
        
        return self._fig_to_array(fig)
    
    def create_radial_profile(self, k1: float, k2: float, k3: float,
                            p1: float, p2: float) -> np.ndarray:
        """
        Create radial distortion profile plot
        
        Args:
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            
        Returns:
            Profile plot as numpy array
        """
        # Calculate radial distances
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        max_radius = min(cx, cy, self.image_width - cx, self.image_height - cy)
        
        radii = np.linspace(0, max_radius, 100)
        distortions = []
        
        for r in radii:
            # Test point at this radius
            x = cx + r
            y = cy
            
            # Apply distortion
            x_dist, y_dist = self.apply_brown_conrady_distortion(
                np.array([x]), np.array([y]), k1, k2, k3, p1, p2, cx, cy
            )
            
            # Calculate radial distortion
            r_dist = np.sqrt((x_dist[0] - cx)**2 + (y_dist[0] - cy)**2)
            distortion = r_dist - r
            distortions.append(distortion)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(radii, distortions, 'b-', linewidth=3, label='Radial Distortion')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Radial Distance from Center (pixels)', fontsize=12)
        ax.set_ylabel('Distortion (pixels)', fontsize=12)
        
        distortion_type = self._classify_distortion(k1, k2, k3, p1, p2)
        ax.set_title(f'Radial Distortion Profile - {distortion_type}\n'
                    f'K1={k1:.3f}, K2={k2:.3f}, K3={k3:.3f}', 
                    fontsize=14, fontweight='bold')
        
        # Add distortion info
        max_distortion = max(abs(min(distortions)), abs(max(distortions)))
        ax.text(0.05, 0.95, f"Max Distortion: {max_distortion:.2f} px", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return self._fig_to_array(fig)
    
    def _classify_distortion(self, k1: float, k2: float, k3: float, 
                           p1: float, p2: float) -> str:
        """Classify distortion type based on coefficients"""
        radial_magnitude = abs(k1) + abs(k2) + abs(k3)
        tangential_magnitude = abs(p1) + abs(p2)
        
        if radial_magnitude < 1e-6 and tangential_magnitude < 1e-6:
            return "No Distortion"
        elif radial_magnitude < 1e-6:
            return "Tangential Only"
        elif k1 < -1e-6:
            if radial_magnitude > 0.3:
                return "Severe Barrel"
            elif radial_magnitude > 0.1:
                return "Moderate Barrel"
            else:
                return "Mild Barrel"
        elif k1 > 1e-6:
            if radial_magnitude > 0.3:
                return "Severe Pincushion"
            elif radial_magnitude > 0.1:
                return "Moderate Pincushion"
            else:
                return "Mild Pincushion"
        else:
            return "Complex Distortion"
    
    def _fig_to_array(self, fig) -> np.ndarray:
        """Convert matplotlib figure to numpy array"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Read image data
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        
        # Decode image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_distortion_visualizer_interface():
    """Create the Gradio interface for distortion visualization"""
    
    visualizer = DistortionVisualizer()
    
    def update_visualizations(k1, k2, k3, p1, p2, grid_rows, grid_cols):
        """Update all visualizations based on current parameters"""
        try:
            # Generate visualizations
            grid_viz = visualizer.create_grid_visualization(
                grid_rows, grid_cols, k1, k2, k3, p1, p2
            )
            
            heatmap_viz = visualizer.create_magnitude_heatmap(
                grid_rows, grid_cols, k1, k2, k3, p1, p2
            )
            
            profile_viz = visualizer.create_radial_profile(k1, k2, k3, p1, p2)
            
            # Create parameter summary
            distortion_type = visualizer._classify_distortion(k1, k2, k3, p1, p2)
            total_distortion = abs(k1) + abs(k2) + abs(k3) + abs(p1) + abs(p2)
            
            summary = f"""
**Distortion Analysis Summary**

**Type:** {distortion_type}
**Total Magnitude:** {total_distortion:.4f}

**Radial Coefficients:**
- K1 (Primary): {k1:.4f}
- K2 (Secondary): {k2:.4f}  
- K3 (Tertiary): {k3:.4f}

**Tangential Coefficients:**
- P1: {p1:.4f}
- P2: {p2:.4f}

**Grid Configuration:**
- Rows: {grid_rows}
- Columns: {grid_cols}
- Total Points: {grid_rows * grid_cols}

**Image Dimensions:**
- Width: {visualizer.image_width}px
- Height: {visualizer.image_height}px
            """
            
            return grid_viz, heatmap_viz, profile_viz, summary
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            return None, None, None, error_msg
    
    # Create Gradio interface
    with gr.Blocks(title="Distortion Map Visualizer", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🔍 Real-Time Distortion Map Visualizer
        
        Interactive visualization of lens distortion using the **Brown-Conrady model**.
        Adjust parameters using the sliders below to see real-time changes in the distortion maps.
        
        **Features:**
        - **Grid Visualization**: Shows how distortion affects a regular grid
        - **Magnitude Heatmap**: Color-coded distortion strength across the image
        - **Radial Profile**: Distortion as a function of distance from center
        - **Real-time Updates**: All visualizations update as you move the sliders
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🌀 Distortion Parameters")
                
                # Distortion parameter sliders
                k1 = gr.Slider(
                    minimum=-0.5, maximum=0.5, step=0.01, value=-0.2,
                    label="K1 (Primary Radial)", 
                    info="Main barrel/pincushion distortion"
                )
                
                k2 = gr.Slider(
                    minimum=-0.2, maximum=0.2, step=0.01, value=0.05,
                    label="K2 (Secondary Radial)", 
                    info="Higher-order radial correction"
                )
                
                k3 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="K3 (Tertiary Radial)", 
                    info="Extreme distortion correction"
                )
                
                p1 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="P1 (Tangential)", 
                    info="Lens decentering correction"
                )
                
                p2 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="P2 (Tangential)", 
                    info="Secondary decentering correction"
                )
                
                gr.Markdown("### 🎯 Grid Configuration")
                
                grid_rows = gr.Slider(
                    minimum=3, maximum=50, step=1, value=7,
                    label="Grid Rows", 
                    info="Number of horizontal grid lines"
                )
                
                grid_cols = gr.Slider(
                    minimum=3, maximum=50, step=1, value=9,
                    label="Grid Columns", 
                    info="Number of vertical grid lines"
                )
                
                gr.Markdown("### 🎛️ Quick Presets")
                
                def apply_barrel_preset():
                    return -0.2, 0.05, 0.0, 0.0, 0.0
                
                def apply_pincushion_preset():
                    return 0.15, -0.03, 0.0, 0.0, 0.0
                
                def apply_fisheye_preset():
                    return -0.4, 0.1, -0.02, 0.0, 0.0
                
                def apply_identity_preset():
                    return 0.0, 0.0, 0.0, 0.0, 0.0
                
                with gr.Row():
                    barrel_btn = gr.Button("📷 Barrel", variant="secondary")
                    pincushion_btn = gr.Button("📐 Pincushion", variant="secondary")
                
                with gr.Row():
                    fisheye_btn = gr.Button("🐟 Fisheye", variant="secondary")
                    identity_btn = gr.Button("🔄 Reset", variant="secondary")
                
                # Analysis summary
                gr.Markdown("### 📊 Analysis Summary")
                summary_output = gr.Markdown(
                    "Adjust parameters to see distortion analysis...",
                    elem_id="summary_output"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Real-Time Visualizations")
                
                with gr.Tab("Grid Distortion"):
                    grid_output = gr.Image(
                        label="Distortion Grid Visualization",
                        height=500
                    )
                    gr.Markdown("""
                    **Grid Visualization Legend:**
                    - **Gray lines**: Original undistorted grid
                    - **Red lines**: Distorted grid showing deformation
                    - **Blue dots**: Original grid intersection points
                    - **Red dots**: Distorted grid intersection points
                    - **Blue arrows**: Distortion vectors showing displacement
                    """)
                
                with gr.Tab("Magnitude Heatmap"):
                    heatmap_output = gr.Image(
                        label="Distortion Magnitude Heatmap",
                        height=500
                    )
                    gr.Markdown("""
                    **Heatmap Interpretation:**
                    - **Dark regions**: Low distortion (< 1 pixel)
                    - **Bright regions**: High distortion (> 5 pixels)
                    - **Color scale**: From black (no distortion) to white (maximum distortion)
                    - **Pattern**: Shows how distortion varies across the image field
                    """)
                
                with gr.Tab("Radial Profile"):
                    profile_output = gr.Image(
                        label="Radial Distortion Profile",
                        height=500
                    )
                    gr.Markdown("""
                    **Profile Analysis:**
                    - **X-axis**: Distance from image center (pixels)
                    - **Y-axis**: Distortion magnitude (pixels)
                    - **Positive values**: Outward distortion (pincushion effect)
                    - **Negative values**: Inward distortion (barrel effect)
                    - **Zero line**: No distortion at that radius
                    """)
        
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
        
        identity_btn.click(
            fn=apply_identity_preset,
            outputs=[k1, k2, k3, p1, p2]
        )
        
        # Connect real-time updates
        inputs = [k1, k2, k3, p1, p2, grid_rows, grid_cols]
        outputs = [grid_output, heatmap_output, profile_output, summary_output]
        
        # Update on any parameter change
        for input_component in inputs:
            input_component.change(
                fn=update_visualizations,
                inputs=inputs,
                outputs=outputs
            )
        
        # Initial load
        interface.load(
            fn=update_visualizations,
            inputs=inputs,
            outputs=outputs
        )
        
        gr.Markdown("""
        ### 📖 Understanding Distortion Parameters
        
        **Radial Distortion (K1, K2, K3):**
        - **K1 < 0**: Barrel distortion (image appears to bulge outward)
        - **K1 > 0**: Pincushion distortion (image appears to pinch inward)
        - **K2, K3**: Higher-order corrections for severe distortions
        
        **Tangential Distortion (P1, P2):**
        - Caused by lens elements not being perfectly centered
        - Creates asymmetric distortion patterns
        - Usually smaller in magnitude than radial distortion
        
        **Grid Configuration:**
        - More grid points provide finer detail but slower updates
        - Recommended: 7-15 rows/columns for good balance
        
        **Real-time Performance:**
        - Optimized for interactive use with immediate visual feedback
        - All calculations use vectorized NumPy operations for speed
        """)
    
    return interface

def launch_visualizer():
    """Launch the distortion visualizer application"""
    interface = create_distortion_visualizer_interface()
    
    interface.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        debug=False
    )

if __name__ == "__main__":
    print("🔍 Starting Distortion Map Visualizer...")
    print("=" * 50)
    print("Real-time interactive visualization of lens distortion")
    print("Navigate to the URL shown below to access the interface")
    print("=" * 50)
    
    launch_visualizer()
