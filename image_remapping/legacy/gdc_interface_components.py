#!/usr/bin/env python3
"""
GDC Interface Components Module

Contains individual interface components and utilities for the GDC remapping interface.
Provides reusable components for tabs, forms, and interactive elements.

Author: Balaji R
License: MIT
"""

import gradio as gr
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class GDCInterfaceComponents:
    """
    Collection of reusable interface components for GDC processing.
    
    This class provides:
    - Tab creation methods
    - Form components
    - Status indicators
    - Sample data generators
    """
    
    def __init__(self):
        self.default_grid_sizes = [7, 9, 11, 15, 21, 33]
        self.default_image_sizes = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080)]
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def create_import_tab_components(self) -> Dict[str, Any]:
        components = {}
        
        # Grid dimensions
        with gr.Group():
            gr.Markdown("**Input Grid Dimensions**")
            with gr.Row():
                components['input_grid_rows'] = gr.Number(
                    label="Grid Rows",
                    value=7,
                    minimum=1,
                    maximum=50,
                    step=1,
                    info="Number of rows in the input grid"
                )
                components['input_grid_cols'] = gr.Number(
                    label="Grid Columns", 
                    value=9,
                    minimum=1,
                    maximum=50,
                    step=1,
                    info="Number of columns in the input grid"
                )
            
            components['expected_elements'] = gr.Textbox(
                label="Expected Data Elements",
                value="63 total (9 DX + 7 DY)",
                interactive=False,
                info="Automatically calculated based on grid dimensions"
            )
        
        # GDC data input
        components['gdc_text_input'] = gr.Textbox(
            label="GDC Grid Data",
            lines=15,
            placeholder=self._get_gdc_placeholder_text(
                components['input_grid_rows'].value,
                components['input_grid_cols'].value
            ),
            show_copy_button=True
        )
        
        # Update placeholder when grid dimensions change
        for component in [components['input_grid_rows'], components['input_grid_cols']]:
            component.change(
                fn=lambda rows, cols: self._get_gdc_placeholder_text(rows, cols),
                inputs=[components['input_grid_rows'], components['input_grid_cols']],
                outputs=components['gdc_text_input']
            )
        
        components['gdc_file_input'] = gr.File(
            label="Upload GDC File",
            file_types=[".txt", ".dat"],
            file_count="single",
        )
        
        # Status and results
        components['import_status'] = gr.Textbox(
            label="Import Status",
            lines=12,
            interactive=False,
            placeholder="Import status will appear here..."
        )
        
        components['import_stats'] = gr.JSON(
            label="Import Statistics",
            visible=False
        )
        
        components['data_preview'] = gr.Dataframe(
            label="Data Preview",
            headers=["Index", "DX Value", "DY Value"],
            datatype=["number", "number", "number"],
            row_count=(1, "dynamic"),
            visible=False
        )
        
        # Action buttons
        components['import_btn'] = gr.Button(
            "🔄 Import GDC Data",
            variant="primary",
            size="lg"
        )
        
        return components
    
    def create_grid_tab_components(self) -> Dict[str, Any]:
        """
        Create components for the grid processing tab.
        
        Returns:
            Dictionary containing all grid tab components
        """
        components = {}
        
        # Original grid dimensions
        with gr.Group():
            gr.Markdown("**Original Grid**")
            with gr.Row():
                components['orig_rows'] = gr.Dropdown(
                    choices=self.default_grid_sizes,
                    value=3,
                    label="Rows",
                    info="Number of rows in original grid"
                )
                components['orig_cols'] = gr.Dropdown(
                    choices=self.default_grid_sizes,
                    value=3,
                    label="Columns",
                    info="Number of columns in original grid"
                )
        
        # Target grid dimensions
        with gr.Group():
            gr.Markdown("**Target Grid (After Interpolation)**")
            with gr.Row():
                components['target_rows'] = gr.Dropdown(
                    choices=self.default_grid_sizes,
                    value=9,
                    label="Rows",
                    info="Target rows after interpolation"
                )
                components['target_cols'] = gr.Dropdown(
                    choices=self.default_grid_sizes,
                    value=9,
                    label="Columns",
                    info="Target columns after interpolation"
                )
        
        # Interpolation settings
        components['interp_method'] = gr.Dropdown(
            choices=["bicubic", "linear"],
            value="bicubic",
            label="Interpolation Method",
            info="Bicubic for smoother results"
        )
        
        components['preserve_boundaries'] = gr.Checkbox(
            label="Preserve Boundaries",
            value=True,
            info="Maintain edge values during interpolation"
        )
        
        components['interp_factor'] = gr.Textbox(
            label="Interpolation Factor",
            value="3.0×",
            interactive=False,
            info="Automatically calculated"
        )
        
        # Visualization outputs
        components['original_grid_viz'] = gr.Image(
            label="Original DX/DY Grids",
            height=400,
            show_download_button=True
        )
        
        components['interpolated_grid_viz'] = gr.Image(
            label="Interpolated DX/DY Grids",
            height=400,
            show_download_button=True
        )
        
        components['grid_comparison_viz'] = gr.Image(
            label="Side-by-Side Comparison",
            height=400,
            show_download_button=True
        )
        
        components['grid_stats'] = gr.Textbox(
            label="Grid Statistics",
            lines=6,
            interactive=False,
            placeholder="Grid statistics will appear here..."
        )
        
        # Action buttons
        components['setup_grid_btn'] = gr.Button(
            "⚙️ Setup Grid",
            variant="primary",
            size="lg"
        )
        
        return components
    
    def create_remapping_tab_components(self) -> Dict[str, Any]:
        """
        Create components for the image remapping tab.
        
        Returns:
            Dictionary containing all remapping tab components
        """
        components = {}
        
        # Image input
        components['input_image'] = gr.Image(
            label="Upload Image",
            type="numpy",
            height=250
        )
        
        # Sample pattern generation
        components['sample_pattern'] = gr.Dropdown(
            choices=["grid", "checkerboard", "circles", "text"],
            value="grid",
            label="Pattern"
        )
        
        components['generate_sample_btn'] = gr.Button(
            "🎨 Generate",
            variant="secondary"
        )
        
        # Image settings
        components['img_width'] = gr.Dropdown(
            choices=[size[0] for size in self.default_image_sizes],
            value=640,
            label="Width",
            allow_custom_value=True
        )
        
        components['img_height'] = gr.Dropdown(
            choices=[size[1] for size in self.default_image_sizes],
            value=480,
            label="Height",
            allow_custom_value=True
        )
        
        # Remapping parameters
        components['scale_factor'] = gr.Slider(
            minimum=0.001,
            maximum=2.0,
            value=0.1,
            step=0.001,
            label="Displacement Scale Factor",
            info="Scales the displacement magnitude"
        )
        
        components['cv_interpolation'] = gr.Dropdown(
            choices=["linear", "cubic", "nearest", "lanczos"],
            value="linear",
            label="CV2 Interpolation",
            info="OpenCV interpolation method"
        )
        
        components['border_mode'] = gr.Dropdown(
            choices=["constant", "reflect", "wrap", "replicate"],
            value="constant",
            label="Border Handling",
            info="How to handle pixels outside image"
        )
        
        # Results
        components['image_comparison'] = gr.Image(
            label="Input vs Remapped",
            height=350,
            show_download_button=True
        )
        
        components['output_image'] = gr.Image(
            label="Remapped Result",
            height=350,
            show_download_button=True
        )
        
        components['displacement_viz'] = gr.Image(
            label="Displacement Field Analysis",
            height=350,
            show_download_button=True
        )
        
        components['processing_status'] = gr.Textbox(
            label="Processing Status",
            lines=4,
            interactive=False,
            placeholder="Processing status will appear here..."
        )
        
        components['quality_metrics'] = gr.JSON(
            label="Quality Metrics",
            visible=False
        )
        
        # Action buttons
        components['setup_remapping_btn'] = gr.Button(
            "⚙️ Setup Remapping",
            variant="secondary"
        )
        
        components['process_image_btn'] = gr.Button(
            "🔄 Process Image",
            variant="primary"
        )
        
        return components
    
    def create_analysis_tab_components(self) -> Dict[str, Any]:
        """
        Create components for the analysis and export tab.
        
        Returns:
            Dictionary containing all analysis tab components
        """
        components = {}
        
        # Analysis options
        components['analysis_type'] = gr.Dropdown(
            choices=[
                "Grid Statistics",
                "Displacement Analysis", 
                "Processing Flow",
                "Quality Assessment"
            ],
            value="Grid Statistics",
            label="Analysis Type"
        )
        
        components['create_analysis_btn'] = gr.Button(
            "📊 Generate Analysis",
            variant="primary"
        )
        
        # Export options
        components['export_format'] = gr.Dropdown(
            choices=["PNG", "JPEG", "TIFF", "All Formats"],
            value="PNG",
            label="Image Format"
        )
        
        components['export_metadata'] = gr.Checkbox(
            label="Include Metadata",
            value=True,
            info="Export processing information"
        )
        
        components['export_visualizations'] = gr.Checkbox(
            label="Export Visualizations",
            value=True,
            info="Export all generated plots"
        )
        
        # Batch processing
        components['batch_folder'] = gr.Textbox(
            label="Input Folder",
            placeholder="Path to folder with images..."
        )
        
        # Results
        components['analysis_viz'] = gr.Image(
            label="Analysis Visualization",
            height=400,
            show_download_button=True
        )
        
        components['processing_summary'] = gr.Textbox(
            label="Processing Summary",
            lines=12,
            show_copy_button=True,
            interactive=False,
            placeholder="Processing summary will appear here..."
        )
        
        components['export_status'] = gr.Textbox(
            label="Export Status",
            lines=4,
            interactive=False,
            placeholder="Export status will appear here..."
        )
        
        components['download_files'] = gr.File(
            label="📥 Download Results",
            file_count="multiple",
            visible=False
        )
        
        # Action buttons
        components['export_single_btn'] = gr.Button(
            "💾 Export Current",
            variant="secondary"
        )
        
        components['export_all_btn'] = gr.Button(
            "📦 Export Grid Data",
            variant="primary"
        )
        
        components['batch_process_btn'] = gr.Button(
            "🔄 Process Batch",
            variant="secondary"
        )
        
        return components
    
    def create_sample_buttons(self, text_input_component) -> Dict[str, Any]:
        """
        Create sample data generation buttons.
        
        Args:
            text_input_component: Text input component to update
            
        Returns:
            Dictionary containing sample buttons
        """
        buttons = {}
        
        with gr.Row():
            buttons['sample_3x3_btn'] = gr.Button("📋 3×3 Sample", variant="secondary", size="sm")
            buttons['sample_5x5_btn'] = gr.Button("📋 5×5 Sample", variant="secondary", size="sm")
            buttons['clear_input_btn'] = gr.Button("🗑️ Clear", variant="secondary", size="sm")
        
        # Wire up the buttons
        buttons['sample_3x3_btn'].click(
            fn=lambda: self.generate_traditional_gdc_sample(3, 3),
            outputs=text_input_component
        )
        
        buttons['sample_5x5_btn'].click(
            fn=lambda: self.generate_traditional_gdc_sample(5, 5),
            outputs=text_input_component
        )
        
        buttons['clear_input_btn'].click(
            fn=lambda: "",
            outputs=text_input_component
        )
        
        return buttons
    
    def generate_traditional_gdc_sample(self, rows: int, cols: int) -> str:
        """
        Generate sample GDC data in traditional format.
        
        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            
        Returns:
            Sample GDC data string
        """
        lines = []
        lines.append(f"# Sample GDC data for {rows}×{cols} grid")
        lines.append(f"# Generated for testing purposes")
        lines.append("")
        
        # Generate DX values
        for i in range(rows * cols):
            # Simple pattern: create some displacement
            dx_value = int(np.sin(i * 0.5) * 50 + np.random.randint(-10, 11))
            lines.append(f"yuv_gdc_grid_dx_0_{i}    {dx_value}")
        
        lines.append("")
        
        # Generate DY values
        for i in range(rows * cols):
            # Simple pattern: create some displacement
            dy_value = int(np.cos(i * 0.3) * 30 + np.random.randint(-5, 6))
            lines.append(f"yuv_gdc_grid_dy_0_{i}    {dy_value}")
        
        return "\n".join(lines)
    
    def update_expected_elements(self, rows: int, cols: int) -> str:
        """
        Update expected elements display.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Formatted string showing expected elements
        """
        if rows and cols:
            total_grid_elements = rows * cols
            total_elements = total_grid_elements * 2  # DX + DY
            return f"{total_elements} total ({total_grid_elements} DX + {total_grid_elements} DY)"
        return "Invalid dimensions"
    
    def update_interpolation_factor(self, orig_rows: int, orig_cols: int, 
                                  target_rows: int, target_cols: int) -> str:
        """
        Calculate and format interpolation factor.
        
        Args:
            orig_rows: Original grid rows
            orig_cols: Original grid columns
            target_rows: Target grid rows
            target_cols: Target grid columns
            
        Returns:
            Formatted interpolation factor string
        """
        try:
            if orig_rows and orig_cols and target_rows and target_cols:
                factor_x = target_cols / orig_cols
                factor_y = target_rows / orig_rows
                avg_factor = (factor_x + factor_y) / 2
                return f"{avg_factor:.1f}×"
        except (TypeError, ZeroDivisionError):
            pass
        return "N/A"
    
    def generate_status_html(self, processing_state: Dict[str, bool]) -> str:
        """
        Generate HTML for status indicators.
        
        Args:
            processing_state: Dictionary with processing state flags
            
        Returns:
            HTML string for status display
        """
        status_items = [
            ('gdc_loaded', '📁 GDC Data'),
            ('grid_setup', '🌐 Grid Setup'),
            ('remapping_ready', '🎯 Remapping'),
            ('image_processed', '🖼️ Image')
        ]
        
        html = '<div id="status_indicators">'
        html += '<h3>🎯 Processing Pipeline Status</h3>'
        
        for key, label in status_items:
            status = processing_state.get(key, False)
            css_class = 'status-ready' if status else 'status-pending'
            status_text = 'Ready' if status else 'Pending'
            html += f'<span class="status-item {css_class}">{label}: {status_text}</span>'
        
        html += '</div>'
        return html
    
    def get_custom_css(self) -> str:
        """
        Get custom CSS for interface styling.
        
        Returns:
            CSS string
        """
        return """
        #status_indicators {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .status-item {
            display: inline-block;
            margin: 5px 15px;
            padding: 5px 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            font-weight: bold;
        }
        
        .status-ready {
            background: rgba(76, 175, 80, 0.8) !important;
        }
        
        .status-pending {
            background: rgba(255, 152, 0, 0.8) !important;
        }
        
        .gradio-container {
            max-width: 1400px !important;
        }
        
        .gr-form {
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        
        .analysis-tab .gr-textbox {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9em;
        }
        """
    
    def _get_gdc_placeholder_text(self, rows: int = 7, cols: int = 9) -> str:
        """Get placeholder text for GDC input."""
        placeholder_text = [
            f"Enter GDC data in traditional format for {rows}×{cols} grid:",
            ""
        ]
        for i in range(min(3, rows * cols)):  # Show first 3 elements as example
            placeholder_text.append(f"yuv_gdc_grid_dx_0_{i}    -50")
        placeholder_text.append("...")
        for i in range(min(3, rows * cols)):
            placeholder_text.append(f"yuv_gdc_grid_dy_0_{i}    25")
        placeholder_text.append("...")
        placeholder_text.append("")
        placeholder_text.append("Note: Multiple spaces between name and value are allowed.")
        placeholder_text.append("Order of elements doesn't matter - they will be sorted by index.")
        return "\n".join(placeholder_text)
        