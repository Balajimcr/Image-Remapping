"""
Gradio interface specifically for GDC grid processing
"""

import gradio as gr
from application.processor import GDCGridProcessor
from config.settings import APP_CONFIG, GRID_DEFAULTS
from typing import Optional

def process_gdc_file(
    file,
    original_rows: int,
    original_cols: int,
    target_rows: int,
    target_cols: int,
    show_visualizations: bool = True
):
    """Main processing function for Gradio interface."""
    if file is None:
        return "No file uploaded", None, None, None, None, None, None
    
    try:
        # Read file content
        if hasattr(file, 'read'):
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        else:
            with open(file.name, 'r') as f:
                content = f.read()
        
        # Process the file
        processor = GDCGridProcessor()
        result = processor.process_file(
            content,
            original_shape=(original_rows, original_cols),
            target_shape=(target_rows, target_cols)
        )
        
        # Create summary
        stats = result['statistics']
        metadata = result['metadata']
        
        summary = f"""
Successfully processed {metadata['num_dx_elements']} DX and {metadata['num_dy_elements']} DY elements.

Original grid shape: {metadata['original_shape']}
Target grid shape: {metadata['target_shape']}
Interpolation method: Bicubic

DX Grid Statistics (Original):
- Min: {stats['dx_original']['min']:.3f}
- Max: {stats['dx_original']['max']:.3f}
- Mean: {stats['dx_original']['mean']:.3f}
- Std: {stats['dx_original']['std']:.3f}

DY Grid Statistics (Original):
- Min: {stats['dy_original']['min']:.3f}
- Max: {stats['dy_original']['max']:.3f}
- Mean: {stats['dy_original']['mean']:.3f}
- Std: {stats['dy_original']['std']:.3f}

DX Grid Statistics (Interpolated):
- Min: {stats['dx_interpolated']['min']:.3f}
- Max: {stats['dx_interpolated']['max']:.3f}
- Mean: {stats['dx_interpolated']['mean']:.3f}
- Std: {stats['dx_interpolated']['std']:.3f}

DY Grid Statistics (Interpolated):
- Min: {stats['dy_interpolated']['min']:.3f}
- Max: {stats['dy_interpolated']['max']:.3f}
- Mean: {stats['dy_interpolated']['mean']:.3f}
- Std: {stats['dy_interpolated']['std']:.3f}

Files generated:
- CSV format files for analysis
- GDC format files with field names (ready to use)
- Combined interpolated file with both DX and DY grids
- All files packaged in: {result['export'].get('zip_path', 'export.zip')}
        """
        
        # Get visualization paths
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
        return f"Error processing file: {str(e)}", None, None, None, None, None, None


def create_gdc_interface():
    """Create the GDC processing Gradio interface."""
    with gr.Blocks(title="GDC Grid Interpolation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # GDC Grid Interpolation Tool
        
        This tool parses GDC (Geometric Distortion Correction) grid files and interpolates them from low resolution to high resolution using bicubic interpolation.
        
        **Instructions:**
        1. Upload your GDC grid text file
        2. Configure the original and target grid dimensions
        3. Click "Process Grid" to generate interpolated grids
        4. Download the results and view visualizations
        
        **Features:**
        - Bicubic interpolation for smooth results
        - Multiple export formats (CSV and GDC)
        - Visualization of grid patterns
        - Statistical analysis of grids
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")
                
                file_input = gr.File(
                    label="Upload GDC Grid File (.txt)",
                    file_types=[".txt"]
                )
                
                gr.Markdown("📁 **File Format**: Upload a text file with GDC grid data in format: `element_name value`")
                
                with gr.Row():
                    original_rows = gr.Number(
                        label="Original Rows",
                        value=GRID_DEFAULTS['original_rows'],
                        precision=0,
                        minimum=1,
                        maximum=100
                    )
                    original_cols = gr.Number(
                        label="Original Columns",
                        value=GRID_DEFAULTS['original_cols'],
                        precision=0,
                        minimum=1,
                        maximum=100
                    )
                
                with gr.Row():
                    target_rows = gr.Number(
                        label="Target Rows",
                        value=GRID_DEFAULTS['target_rows'],
                        precision=0,
                        minimum=1,
                        maximum=1000
                    )
                    target_cols = gr.Number(
                        label="Target Columns",
                        value=GRID_DEFAULTS['target_cols'],
                        precision=0,
                        minimum=1,
                        maximum=1000
                    )
                
                show_vis = gr.Checkbox(
                    label="Generate Visualizations",
                    value=True
                )
                gr.Markdown("🎨 **Visualizations**: Create heatmap visualizations of the grids")
                
                process_btn = gr.Button(
                    "Process Grid",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Processing Results")
                
                summary_output = gr.Textbox(
                    label="Processing Summary",
                    lines=25,
                    max_lines=30,
                    show_copy_button=True
                )
                
                download_output = gr.File(
                    label="Download All Results (ZIP)",
                    visible=False
                )
        
        with gr.Row():
            gr.Markdown("### Grid Visualizations")
        
        with gr.Row():
            with gr.Column():
                vis_orig_dx = gr.Image(label="Original DX Grid", visible=False)
                vis_interp_dx = gr.Image(label="Interpolated DX Grid", visible=False)
            
            with gr.Column():
                vis_orig_dy = gr.Image(label="Original DY Grid", visible=False)
                vis_interp_dy = gr.Image(label="Interpolated DY Grid", visible=False)
        
        with gr.Row():
            comparison_plot = gr.Image(label="Grid Comparison", visible=False)
        
        # Event handlers
        def update_visibility(summary, zip_file, vis1, vis2, vis3, vis4, comp):
            """Update component visibility based on processing results."""
            has_results = summary and "Error" not in summary
            return [
                gr.update(visible=has_results),  # download_output
                gr.update(visible=has_results and vis1 is not None),  # vis_orig_dx
                gr.update(visible=has_results and vis2 is not None),  # vis_orig_dy
                gr.update(visible=has_results and vis3 is not None),  # vis_interp_dx
                gr.update(visible=has_results and vis4 is not None),  # vis_interp_dy
                gr.update(visible=has_results and comp is not None),  # comparison_plot
            ]
        
        process_btn.click(
            fn=process_gdc_file,
            inputs=[file_input, original_rows, original_cols, target_rows, target_cols, show_vis],
            outputs=[summary_output, download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot]
        ).then(
            fn=update_visibility,
            inputs=[summary_output, download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot],
            outputs=[download_output, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, comparison_plot]
        )
        
        # Example section
        with gr.Row():
            gr.Markdown("""
            ### Example Input Format
            
            Your GDC grid file should contain lines in the format:
            ```
            yuv_gdc_grid_dx_0_0 -48221
            yuv_gdc_grid_dx_0_1 137272
            yuv_gdc_grid_dx_0_2 -111019
            ...
            yuv_gdc_grid_dy_0_0 5678
            yuv_gdc_grid_dy_0_1 5679
            yuv_gdc_grid_dy_0_2 5680
            ...
            ```
            
            ### Output Files
            
            The tool generates multiple output formats:
            
            **CSV Format:**
            - Grid data in CSV format for spreadsheet analysis
            
            **GDC Format:**
            - Original format with field names (e.g., `yuv_gdc_grid_dx_0_0 value`)
            - Ready to use in your GDC processing pipeline
            - Combined file with both DX and DY grids
            
            **Visualizations:**
            - Heatmap plots showing grid value distributions
            - Comparison plots for before/after analysis
            """)
    
    return interface


def launch_gdc_utility():
    """Launch the GDC utility interface."""
    interface = create_gdc_interface()
    
    interface.launch(
        server_name=APP_CONFIG['server_name'],
        server_port=APP_CONFIG['port'],
        share=APP_CONFIG['share'],
        debug=APP_CONFIG['debug']
    )