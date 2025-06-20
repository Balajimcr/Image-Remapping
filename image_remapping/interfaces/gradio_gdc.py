"""
Enhanced Gradio interface specifically for GDC grid processing
Integrated with the main Image Remapping Suite architecture
"""

import gradio as gr
from typing import Optional

# Import the updated processor with GDC capabilities
from application.processor import processor, GDCGridProcessor
from config.settings import (
    APP_CONFIG, GRID_DEFAULTS, GDC_VALIDATION_RANGES,
    SECURITY_CONFIG, TEMP_FILE_CONFIG
)


def process_gdc_file(
    file,
    original_rows: int,
    original_cols: int,
    target_rows: int,
    target_cols: int,
    show_visualizations: bool = True
):
    """
    Main processing function for Gradio interface.
    
    Args:
        file: Uploaded file object
        original_rows: Number of rows in original grid
        original_cols: Number of columns in original grid
        target_rows: Target number of rows for interpolation
        target_cols: Target number of columns for interpolation
        show_visualizations: Whether to generate visualization plots
        
    Returns:
        Tuple of (summary, zip_file, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison)
    """
    if file is None:
        return "No file uploaded", None, None, None, None, None, None
    
    try:
        # Validate grid dimensions
        validation_error = validate_grid_dimensions(original_rows, original_cols, target_rows, target_cols)
        if validation_error:
            return validation_error, None, None, None, None, None, None
        
        # Read file content
        if hasattr(file, 'read'):
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        else:
            with open(file.name, 'r') as f:
                content = f.read()
        
        # Validate file size and content
        content_validation = validate_file_content(content)
        if content_validation:
            return content_validation, None, None, None, None, None, None
        
        # Process the file using the integrated GDC processor
        gdc_processor = GDCGridProcessor()
        result = gdc_processor.process_file(
            content,
            original_shape=(original_rows, original_cols),
            target_shape=(target_rows, target_cols)
        )
        
        # Create comprehensive summary
        summary = create_processing_summary(result, show_visualizations)
        
        # Get visualization paths (only if requested)
        vis = result['visualizations'] if show_visualizations else {}
        
        return (
            summary,
            result['export'].get('zip_path'),
            vis.get('dx_original'),
            vis.get('dy_original'),
            vis.get('dx_interpolated'),
            vis.get('dy_interpolated'),
            vis.get('comparison')
        )
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"GDC Processing Error: {e}")  # Log for debugging
        return error_msg, None, None, None, None, None, None


def validate_grid_dimensions(original_rows: int, original_cols: int, 
                           target_rows: int, target_cols: int) -> Optional[str]:
    """
    Validate grid dimension parameters.
    
    Returns:
        Error message if validation fails, None if valid
    """
    # Check original grid dimensions
    min_rows, max_rows = GDC_VALIDATION_RANGES['grid_rows']
    min_cols, max_cols = GDC_VALIDATION_RANGES['grid_cols']
    
    if not (min_rows <= original_rows <= max_rows):
        return f"Original rows ({original_rows}) must be between {min_rows} and {max_rows}"
    
    if not (min_cols <= original_cols <= max_cols):
        return f"Original columns ({original_cols}) must be between {min_cols} and {max_cols}"
    
    # Check target grid dimensions
    min_target_rows, max_target_rows = GDC_VALIDATION_RANGES['target_rows']
    min_target_cols, max_target_cols = GDC_VALIDATION_RANGES['target_cols']
    
    if not (min_target_rows <= target_rows <= max_target_rows):
        return f"Target rows ({target_rows}) must be between {min_target_rows} and {max_target_rows}"
    
    if not (min_target_cols <= target_cols <= max_target_cols):
        return f"Target columns ({target_cols}) must be between {min_target_cols} and {max_target_cols}"
    
    # Check interpolation factor
    interpolation_factor = (target_rows * target_cols) / (original_rows * original_cols)
    min_factor, max_factor = GDC_VALIDATION_RANGES['interpolation_factor']
    
    if not (min_factor <= interpolation_factor <= max_factor):
        return f"Interpolation factor ({interpolation_factor:.1f}x) must be between {min_factor}x and {max_factor}x"
    
    return None


def validate_file_content(content: str) -> Optional[str]:
    """
    Validate uploaded file content.
    
    Returns:
        Error message if validation fails, None if valid
    """
    if not content or not content.strip():
        return "File appears to be empty"
    
    # Check file size
    content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
    if content_size_mb > SECURITY_CONFIG['max_upload_size_mb']:
        return f"File too large ({content_size_mb:.1f}MB). Maximum allowed: {SECURITY_CONFIG['max_upload_size_mb']}MB"
    
    # Check number of lines
    lines = content.split('\n')
    if len(lines) > SECURITY_CONFIG['max_lines_per_file']:
        return f"File has too many lines ({len(lines)}). Maximum allowed: {SECURITY_CONFIG['max_lines_per_file']}"
    
    # Basic format validation
    valid_lines = 0
    for line in lines[:100]:  # Check first 100 lines for format
        line = line.strip()
        if line and ' ' in line:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                try:
                    int(parts[1])
                    valid_lines += 1
                except ValueError:
                    pass
    
    if valid_lines == 0:
        return "No valid GDC data lines found. Expected format: 'element_name integer_value'"
    
    return None


def create_processing_summary(result: dict, show_visualizations: bool) -> str:
    """
    Create a comprehensive processing summary.
    
    Args:
        result: Processing results dictionary
        show_visualizations: Whether visualizations were generated
        
    Returns:
        Formatted summary string
    """
    stats = result['statistics']
    metadata = result['metadata']
    
    summary_lines = [
        "🎯 GDC Grid Processing Complete",
        "=" * 50,
        "",
        "📊 Processing Summary:",
        f"✅ Successfully processed {metadata['num_dx_elements']} DX and {metadata['num_dy_elements']} DY elements",
        f"📐 Original grid shape: {metadata['original_shape'][0]} x {metadata['original_shape'][1]}",
        f"🎯 Target grid shape: {metadata['target_shape'][0]} x {metadata['target_shape'][1]}",
        f"🔧 Interpolation method: {metadata['interpolation_method'].title()}",
        f"📈 Resolution increase: {(metadata['target_shape'][0] * metadata['target_shape'][1]) / (metadata['original_shape'][0] * metadata['original_shape'][1]):.1f}x",
        "",
        "📈 DX Grid Statistics:",
        "  Original Grid:",
        f"    • Min: {stats['dx_original']['min']:.3f}",
        f"    • Max: {stats['dx_original']['max']:.3f}",
        f"    • Mean: {stats['dx_original']['mean']:.3f}",
        f"    • Std Dev: {stats['dx_original']['std']:.3f}",
        "",
        "  Interpolated Grid:",
        f"    • Min: {stats['dx_interpolated']['min']:.3f}",
        f"    • Max: {stats['dx_interpolated']['max']:.3f}",
        f"    • Mean: {stats['dx_interpolated']['mean']:.3f}",
        f"    • Std Dev: {stats['dx_interpolated']['std']:.3f}",
        "",
        "📉 DY Grid Statistics:",
        "  Original Grid:",
        f"    • Min: {stats['dy_original']['min']:.3f}",
        f"    • Max: {stats['dy_original']['max']:.3f}",
        f"    • Mean: {stats['dy_original']['mean']:.3f}",
        f"    • Std Dev: {stats['dy_original']['std']:.3f}",
        "",
        "  Interpolated Grid:",
        f"    • Min: {stats['dy_interpolated']['min']:.3f}",
        f"    • Max: {stats['dy_interpolated']['max']:.3f}",
        f"    • Mean: {stats['dy_interpolated']['mean']:.3f}",
        f"    • Std Dev: {stats['dy_interpolated']['std']:.3f}",
        "",
        "📁 Generated Files:",
        "  📊 CSV Format Files:",
        f"    • original_dx_{metadata['original_shape'][0]}x{metadata['original_shape'][1]}.csv",
        f"    • original_dy_{metadata['original_shape'][0]}x{metadata['original_shape'][1]}.csv",
        f"    • interpolated_dx_{metadata['target_shape'][0]}x{metadata['target_shape'][1]}.csv",
        f"    • interpolated_dy_{metadata['target_shape'][0]}x{metadata['target_shape'][1]}.csv",
        "",
        "  🎯 GDC Format Files (Hardware Ready):",
        f"    • gdc_dx_interpolated_{metadata['target_shape'][0]}x{metadata['target_shape'][1]}.txt",
        f"    • gdc_dy_interpolated_{metadata['target_shape'][0]}x{metadata['target_shape'][1]}.txt",
        "    • gdc_combined_interpolated.txt (both DX and DY)",
        "",
    ]
    
    if show_visualizations:
        summary_lines.extend([
            "🎨 Visualizations Generated:",
            "    • Original DX/DY grid heatmaps",
            "    • Interpolated DX/DY grid heatmaps", 
            "    • Side-by-side comparison plots",
            "",
        ])
    
    summary_lines.extend([
        f"📦 All files packaged in: {result['export'].get('zip_path', 'export.zip').split('/')[-1]}",
        "",
        "✅ Processing completed successfully!",
        "",
        "💡 Next Steps:",
        "  1. Download the ZIP file containing all results",
        "  2. Use CSV files for analysis in spreadsheet applications",
        "  3. Use GDC format files for hardware implementation",
        "  4. Review visualizations for quality assessment"
    ])
    
    return "\n".join(summary_lines)


def create_gdc_interface():
    """Create the enhanced GDC processing Gradio interface."""
    
    with gr.Blocks(title="GDC Grid Interpolation Tool", theme=gr.themes.Soft()) as interface:
        
        # Header with comprehensive description
        gr.Markdown("""
        # 🎯 GDC Grid Interpolation Tool
        
        **Advanced Geometric Distortion Correction Grid Processor**
        
        This professional tool parses GDC (Geometric Distortion Correction) grid files and performs high-quality 
        bicubic interpolation to increase resolution. Perfect for hardware implementation pipelines and analysis workflows.
        
        ## ✨ Key Features:
        - 🔧 **Bicubic Interpolation**: Smooth, high-quality grid upsampling
        - 📊 **Multiple Export Formats**: CSV for analysis, GDC format for hardware
        - 🎨 **Advanced Visualizations**: Heatmaps and comparison plots
        - 📈 **Statistical Analysis**: Comprehensive grid statistics
        - 🔒 **Robust Validation**: Input validation and error handling
        - 📦 **Complete Package**: All outputs bundled in convenient ZIP files
        """)
        
        with gr.Row():
            # Left Column - Input Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Configuration")
                
                file_input = gr.File(
                    label="Upload GDC Grid File",
                    file_types=[".txt", ".dat", ".csv"],
                    file_count="single"
                )
                
                gr.Markdown("""
                📋 **Supported File Formats**: 
                - `.txt` - Standard GDC format
                - `.dat` - Alternative GDC format  
                - `.csv` - Comma-separated values
                
                📝 **Expected Format**: `element_name integer_value` per line
                """)
                
                gr.Markdown("### 📐 Grid Dimensions")
                
                with gr.Row():
                    original_rows = gr.Number(
                        label="Original Rows",
                        value=GRID_DEFAULTS['original_rows'],
                        precision=0,
                        minimum=GDC_VALIDATION_RANGES['grid_rows'][0],
                        maximum=GDC_VALIDATION_RANGES['grid_rows'][1],
                        info="Number of rows in input grid"
                    )
                    original_cols = gr.Number(
                        label="Original Columns",
                        value=GRID_DEFAULTS['original_cols'],
                        precision=0,
                        minimum=GDC_VALIDATION_RANGES['grid_cols'][0],
                        maximum=GDC_VALIDATION_RANGES['grid_cols'][1],
                        info="Number of columns in input grid"
                    )
                
                with gr.Row():
                    target_rows = gr.Number(
                        label="Target Rows",
                        value=GRID_DEFAULTS['target_rows'],
                        precision=0,
                        minimum=GDC_VALIDATION_RANGES['target_rows'][0],
                        maximum=GDC_VALIDATION_RANGES['target_rows'][1],
                        info="Desired output rows"
                    )
                    target_cols = gr.Number(
                        label="Target Columns",
                        value=GRID_DEFAULTS['target_cols'],
                        precision=0,
                        minimum=GDC_VALIDATION_RANGES['target_cols'][0],
                        maximum=GDC_VALIDATION_RANGES['target_cols'][1],
                        info="Desired output columns"
                    )
                
                # Dynamic interpolation factor display
                interpolation_factor = gr.Textbox(
                    label="Interpolation Factor",
                    value="3.7x",
                    interactive=False,
                    info="Automatically calculated resolution increase"
                )
                
                gr.Markdown("### 🎨 Output Options")
                
                show_vis = gr.Checkbox(
                    label="Generate Visualizations",
                    value=True,
                    info="Create heatmap visualizations and comparison plots"
                )
                
                advanced_options = gr.Checkbox(
                    label="Advanced Processing",
                    value=False,
                    info="Enable additional quality checks and metadata"
                )
                
                gr.Markdown("### 🚀 Processing")
                
                process_btn = gr.Button(
                    "🔄 Process Grid",
                    variant="primary",
                    size="lg"
                )
                
                # Quick validation display
                validation_status = gr.Textbox(
                    label="Validation Status",
                    value="Ready to process",
                    interactive=False,
                    lines=2
                )
            
            # Right Column - Results and Output
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Processing Results")
                
                summary_output = gr.Textbox(
                    label="Processing Summary",
                    lines=28,
                    max_lines=35,
                    show_copy_button=True,
                    placeholder="Processing results will appear here after clicking 'Process Grid'..."
                )
                
                download_output = gr.File(
                    label="📦 Download Complete Results Package (ZIP)",
                    visible=False
                )
        
        # Visualization Section
        with gr.Row():
            gr.Markdown("### 🎨 Grid Visualizations")
        
        gr.Markdown("""
        **Understanding the Visualizations:**
        - **Heatmaps**: Color intensity represents grid values (red=high, blue=low)
        - **Original Grids**: Input data visualization
        - **Interpolated Grids**: High-resolution output visualization
        - **Comparison Plots**: Side-by-side before/after analysis
        """)
        
        with gr.Row():
            with gr.Column():
                vis_orig_dx = gr.Image(
                    label="📈 Original DX Grid", 
                    visible=False,
                    height=300
                )
                vis_interp_dx = gr.Image(
                    label="🎯 Interpolated DX Grid", 
                    visible=False,
                    height=300
                )
            
            with gr.Column():
                vis_orig_dy = gr.Image(
                    label="📉 Original DY Grid", 
                    visible=False,
                    height=300
                )
                vis_interp_dy = gr.Image(
                    label="🎯 Interpolated DY Grid", 
                    visible=False,
                    height=300
                )
        
        with gr.Row():
            comparison_plot = gr.Image(
                label="⚖️ Grid Comparison Analysis", 
                visible=False,
                height=400
            )
        
        # Documentation and Examples Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 📝 Example Input Format
                
                Your GDC grid file should contain lines in the format:
                ```
                yuv_gdc_grid_dx_0_0 -48221
                yuv_gdc_grid_dx_0_1 137272
                yuv_gdc_grid_dx_0_2 -111019
                yuv_gdc_grid_dx_0_3 89445
                ...
                yuv_gdc_grid_dy_0_0 5678
                yuv_gdc_grid_dy_0_1 5679
                yuv_gdc_grid_dy_0_2 5680
                yuv_gdc_grid_dy_0_3 5681
                ...
                ```
                
                **Naming Convention:**
                - `yuv_gdc_grid_dx_0_N` for X-direction displacements
                - `yuv_gdc_grid_dy_0_N` for Y-direction displacements
                - `N` is the sequential index (0, 1, 2, ...)
                """)
            
            with gr.Column():
                gr.Markdown("""
                ### 📦 Output Files Description
                
                **CSV Format Files:**
                - `original_dx_RxC.csv` - Raw DX grid data  
                - `original_dy_RxC.csv` - Raw DY grid data
                - `interpolated_dx_RxC.csv` - High-res DX grid
                - `interpolated_dy_RxC.csv` - High-res DY grid
                
                **GDC Format Files (Hardware Ready):**
                - `gdc_dx_interpolated.txt` - DX values in GDC format
                - `gdc_dy_interpolated.txt` - DY values in GDC format  
                - `gdc_combined.txt` - Both DX and DY in single file
                
                **Visualizations:**
                - High-quality PNG heatmap plots
                - Comparison analysis charts
                - Statistical distribution plots
                """)
        
        # Event Handlers and Dynamic Updates
        def update_interpolation_factor(orig_rows, orig_cols, target_rows, target_cols):
            """Update interpolation factor display."""
            try:
                factor = (target_rows * target_cols) / (orig_rows * orig_cols)
                return f"{factor:.1f}x"
            except:
                return "Invalid"
        
        def update_validation_status(orig_rows, orig_cols, target_rows, target_cols):
            """Update validation status based on current parameters."""
            validation_error = validate_grid_dimensions(orig_rows, orig_cols, target_rows, target_cols)
            if validation_error:
                return f"❌ {validation_error}"
            else:
                return "✅ Parameters valid, ready to process"
        
        def update_visibility(summary, zip_file, vis1, vis2, vis3, vis4, comp):
            """Update component visibility based on processing results."""
            has_results = summary and "Error" not in summary and "No file uploaded" not in summary
            has_visualizations = has_results and vis1 is not None
            
            return [
                gr.update(visible=has_results),  # download_output
                gr.update(visible=has_visualizations),  # vis_orig_dx
                gr.update(visible=has_visualizations),  # vis_orig_dy
                gr.update(visible=has_visualizations),  # vis_interp_dx
                gr.update(visible=has_visualizations),  # vis_interp_dy
                gr.update(visible=has_visualizations),  # comparison_plot
            ]
        
        # Wire up dynamic updates
        for component in [original_rows, original_cols, target_rows, target_cols]:
            component.change(
                fn=update_interpolation_factor,
                inputs=[original_rows, original_cols, target_rows, target_cols],
                outputs=interpolation_factor
            )
            component.change(
                fn=update_validation_status,
                inputs=[original_rows, original_cols, target_rows, target_cols],
                outputs=validation_status
            )
        
        # Main processing event
        process_btn.click(
            fn=process_gdc_file,
            inputs=[file_input, original_rows, original_cols, target_rows, target_cols, show_vis],
            outputs=[summary_output, download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot]
        ).then(
            fn=update_visibility,
            inputs=[summary_output, download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot],
            outputs=[download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot]
        )
        
        # Footer with additional information
        gr.Markdown("""
        ---
        ### 🔧 Technical Notes
        
        **Interpolation Method:** Bicubic spline interpolation using SciPy's RectBivariateSpline
        
        **Performance:** Optimized for grids up to 1000x1000 elements
        
        **Accuracy:** Maintains sub-pixel precision in interpolated values
        
        **Compatibility:** Output files compatible with standard ISP/FPGA toolchains
        """)
    
    return interface


def launch_gdc_utility():
    """Launch the enhanced GDC utility interface."""
    interface = create_gdc_interface()
    
    interface.launch(
        server_name=APP_CONFIG['server_name'],
        server_port=APP_CONFIG['port'],
        share=APP_CONFIG['share'],
        debug=APP_CONFIG['debug'],
        show_error=True,
        quiet=False
    )


# For standalone execution
if __name__ == "__main__":
    print("🚀 Launching GDC Grid Interpolation Tool...")
    launch_gdc_utility()