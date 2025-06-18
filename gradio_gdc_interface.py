import gradio as gr
import csv
import re
import numpy as np
from scipy.interpolate import RectBivariateSpline
import warnings
import os
import tempfile
import zipfile
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GDCGridProcessor:
    def __init__(self):
        self.parsed_data = []
        self.dx_values = []
        self.dy_values = []
        
    def parse_grid_data_from_content(self, file_content):
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
    
    def extract_and_sort_grid_values(self):
        """Extract and sort DX and DY values from parsed data."""
        index_pattern = re.compile(r"_(dx|dy)_0_(\d+)$")
        
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
    
    def interpolate_grid_bicubic(self, grid_2d, target_rows, target_cols):
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
    
    def create_visualization(self, grid_2d, title, colormap='viridis'):
        """Create a visualization of the grid data."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(grid_2d, annot=False, cmap=colormap, cbar=True)
        plt.title(title)
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_file.name

def create_input_format_file(grid_2d, grid_type, filename):
    """Create a text file in the same format as the input file with field names."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', 
                                          prefix=filename.replace('.txt', '_'))
    
    # Flatten the 2D grid in row-major order
    flattened_grid = grid_2d.flatten()
    
    # Write each value with its corresponding field name
    for idx, value in enumerate(flattened_grid):
        # Round to integer as in the original format
        int_value = int(round(value))
        field_name = f"yuv_gdc_grid_{grid_type}_0_{idx}"
        temp_file.write(f"{field_name} {int_value}\n")
    
    temp_file.close()
    return temp_file.name

def process_gdc_file(file, original_rows, original_cols, target_rows, target_cols, 
                    show_visualizations=True):
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
        
        processor = GDCGridProcessor()
        
        # Parse the data
        parsed_data = processor.parse_grid_data_from_content(content)
        
        if not parsed_data:
            return "No valid data found in file", None, None, None, None, None, None
        
        # Extract and sort values
        dx_values, dy_values = processor.extract_and_sort_grid_values()
        
        # Validate data
        expected_elements = original_rows * original_cols
        if len(dx_values) != expected_elements or len(dy_values) != expected_elements:
            warning_msg = f"Warning: Expected {expected_elements} elements for {original_rows}x{original_cols} grid.\n"
            warning_msg += f"Found {len(dx_values)} DX elements and {len(dy_values)} DY elements.\n"
        else:
            warning_msg = f"Successfully parsed {len(dx_values)} DX and {len(dy_values)} DY elements.\n"
        
        # Reshape to 2D grids
        try:
            dx_grid_2d = np.array(dx_values[:expected_elements]).reshape(original_rows, original_cols)
            dy_grid_2d = np.array(dy_values[:expected_elements]).reshape(original_rows, original_cols)
        except ValueError as e:
            return f"Error reshaping grid data: {e}", None, None, None, None, None, None
        
        # Perform interpolation
        dx_interpolated = processor.interpolate_grid_bicubic(dx_grid_2d, target_rows, target_cols)
        dy_interpolated = processor.interpolate_grid_bicubic(dy_grid_2d, target_rows, target_cols)
        
        # Create files in different formats
        all_files = []
        
        # Original grids - CSV format
        original_dx_csv = create_csv_file(dx_grid_2d, f"original_dx_{original_rows}x{original_cols}.csv")
        original_dy_csv = create_csv_file(dy_grid_2d, f"original_dy_{original_rows}x{original_cols}.csv")
        
        # Interpolated grids - CSV format
        interp_dx_csv = create_csv_file(dx_interpolated, f"interpolated_dx_{target_rows}x{target_cols}.csv")
        interp_dy_csv = create_csv_file(dy_interpolated, f"interpolated_dy_{target_rows}x{target_cols}.csv")
        
        # Original grids - Input format
        original_dx_txt = create_input_format_file(dx_grid_2d, "dx", f"original_dx_{original_rows}x{original_cols}_formatted.txt")
        original_dy_txt = create_input_format_file(dy_grid_2d, "dy", f"original_dy_{original_rows}x{original_cols}_formatted.txt")
        
        # Interpolated grids - Input format (MAIN ADDITION)
        interp_dx_txt = create_input_format_file(dx_interpolated, "dx", f"interpolated_dx_{target_rows}x{target_cols}_formatted.txt")
        interp_dy_txt = create_input_format_file(dy_interpolated, "dy", f"interpolated_dy_{target_rows}x{target_cols}_formatted.txt")
        
        # Combined interpolated grid file in input format
        combined_interp_txt = create_combined_input_format_file(dx_interpolated, dy_interpolated, 
                                                              f"interpolated_combined_{target_rows}x{target_cols}_formatted.txt")
        
        all_files = [
            original_dx_csv, original_dy_csv, interp_dx_csv, interp_dy_csv,
            original_dx_txt, original_dy_txt, interp_dx_txt, interp_dy_txt,
            combined_interp_txt
        ]
        
        # Create zip file with all outputs
        zip_file = create_zip_file(all_files)
        
        # Create visualizations if requested
        vis_orig_dx = vis_orig_dy = vis_interp_dx = vis_interp_dy = None
        
        if show_visualizations:
            vis_orig_dx = processor.create_visualization(dx_grid_2d, f"Original DX Grid ({original_rows}x{original_cols})", 'RdBu_r')
            vis_orig_dy = processor.create_visualization(dy_grid_2d, f"Original DY Grid ({original_rows}x{original_cols})", 'RdBu_r')
            vis_interp_dx = processor.create_visualization(dx_interpolated, f"Interpolated DX Grid ({target_rows}x{target_cols})", 'RdBu_r')
            vis_interp_dy = processor.create_visualization(dy_interpolated, f"Interpolated DY Grid ({target_rows}x{target_cols})", 'RdBu_r')
        
        # Create summary
        summary = f"""
{warning_msg}
Original grid shape: {dx_grid_2d.shape}
Target grid shape: ({target_rows}, {target_cols})
Interpolation method: Bicubic
        
DX Grid Statistics (Original):
- Min: {np.min(dx_grid_2d):.3f}
- Max: {np.max(dx_grid_2d):.3f}
- Mean: {np.mean(dx_grid_2d):.3f}
- Std: {np.std(dx_grid_2d):.3f}

DY Grid Statistics (Original):
- Min: {np.min(dy_grid_2d):.3f}
- Max: {np.max(dy_grid_2d):.3f}
- Mean: {np.mean(dy_grid_2d):.3f}
- Std: {np.std(dy_grid_2d):.3f}

DX Grid Statistics (Interpolated):
- Min: {np.min(dx_interpolated):.3f}
- Max: {np.max(dx_interpolated):.3f}
- Mean: {np.mean(dx_interpolated):.3f}
- Std: {np.std(dx_interpolated):.3f}

DY Grid Statistics (Interpolated):
- Min: {np.min(dy_interpolated):.3f}
- Max: {np.max(dy_interpolated):.3f}
- Mean: {np.mean(dy_interpolated):.3f}
- Std: {np.std(dy_interpolated):.3f}

Files generated:
CSV Format:
- Original DX CSV: original_dx_{original_rows}x{original_cols}.csv
- Original DY CSV: original_dy_{original_rows}x{original_cols}.csv
- Interpolated DX CSV: interpolated_dx_{target_rows}x{target_cols}.csv
- Interpolated DY CSV: interpolated_dy_{target_rows}x{target_cols}.csv

Input Format (with field names):
- Original DX: original_dx_{original_rows}x{original_cols}_formatted.txt
- Original DY: original_dy_{original_rows}x{original_cols}_formatted.txt
- Interpolated DX: interpolated_dx_{target_rows}x{target_cols}_formatted.txt
- Interpolated DY: interpolated_dy_{target_rows}x{target_cols}_formatted.txt
- Combined Interpolated: interpolated_combined_{target_rows}x{target_cols}_formatted.txt
        """
        
        return summary, zip_file, vis_orig_dx, vis_orig_dy, vis_interp_dx, vis_interp_dy, create_grid_comparison_plot(dx_grid_2d, dx_interpolated, "DX Grid Comparison")
        
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None, None, None, None, None

def create_combined_input_format_file(dx_grid_2d, dy_grid_2d, filename):
    """Create a single text file with both DX and DY grids in input format."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', 
                                          prefix=filename.replace('.txt', '_'))
    
    # Write DX values first
    dx_flattened = dx_grid_2d.flatten()
    for idx, value in enumerate(dx_flattened):
        int_value = int(round(value))
        field_name = f"yuv_gdc_grid_dx_0_{idx}"
        temp_file.write(f"{field_name} {int_value}\n")
    
    # Write DY values
    dy_flattened = dy_grid_2d.flatten()
    for idx, value in enumerate(dy_flattened):
        int_value = int(round(value))
        field_name = f"yuv_gdc_grid_dy_0_{idx}"
        temp_file.write(f"{field_name} {int_value}\n")
    
    temp_file.close()
    return temp_file.name

def create_csv_file(grid_2d, filename):
    """Create a CSV file from 2D grid data."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', 
                                          prefix=filename.replace('.csv', '_'))
    
    writer = csv.writer(temp_file)
    for row in grid_2d:
        rounded_row = [round(val, 6) for val in row]
        writer.writerow(rounded_row)
    
    temp_file.close()
    return temp_file.name

def create_zip_file(all_files):
    """Create a zip file containing all output files."""
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for file_path in all_files:
            # Get the original filename from the temp file path
            base_name = os.path.basename(file_path)
            # Clean up the temp prefix
            if base_name.startswith('tmp'):
                parts = base_name.split('_', 1)
                if len(parts) > 1:
                    base_name = parts[1]
            zipf.write(file_path, base_name)
    
    temp_zip.close()
    return temp_zip.name

def create_grid_comparison_plot(original_grid, interpolated_grid, title):
    """Create a side-by-side comparison plot of original and interpolated grids."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original grid
    im1 = ax1.imshow(original_grid, cmap='RdBu_r', aspect='auto')
    ax1.set_title(f'Original Grid ({original_grid.shape[0]}x{original_grid.shape[1]})')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    plt.colorbar(im1, ax=ax1)
    
    # Interpolated grid
    im2 = ax2.imshow(interpolated_grid, cmap='RdBu_r', aspect='auto')
    ax2.set_title(f'Interpolated Grid ({interpolated_grid.shape[0]}x{interpolated_grid.shape[1]})')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_file.name

# Create the Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="GDC Grid Interpolation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # GDC Grid Interpolation Tool
        
        This tool parses GDC (Geometric Distortion Correction) grid files and interpolates them from low resolution to high resolution using bicubic interpolation.
        
        **Instructions:**
        1. Upload your GDC grid text file
        2. Configure the original and target grid dimensions
        3. Click "Process Grid" to generate interpolated grids
        4. Download the results and view visualizations
        
        **New Feature:** Interpolated grids are now exported in the same format as the input file with proper field names!
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
                        value=9,
                        precision=0,
                        minimum=1,
                        maximum=100
                    )
                    original_cols = gr.Number(
                        label="Original Columns",
                        value=7,
                        precision=0,
                        minimum=1,
                        maximum=100
                    )
                
                with gr.Row():
                    target_rows = gr.Number(
                        label="Target Rows",
                        value=33,
                        precision=0,
                        minimum=1,
                        maximum=1000
                    )
                    target_cols = gr.Number(
                        label="Target Columns",
                        value=33,
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
            - Original and interpolated grids as CSV files for easy analysis
            
            **Input Format (NEW):**
            - Original grids with field names (e.g., `yuv_gdc_grid_dx_0_0 value`)
            - **Interpolated grids with field names** - same format as input file
            - Combined file with both DX and DY interpolated grids
            
            **Visualizations:**
            - Heatmap plots showing grid patterns and distributions
            
            All files are bundled in a ZIP package for easy download.
            """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    # Required dependencies check
    try:
        import gradio as gr
        import numpy as np
        from scipy.interpolate import RectBivariateSpline
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install gradio numpy scipy matplotlib seaborn pandas")
        exit(1)
    
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="LocalHost",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=True,            # Set to True for public sharing
        debug=True              # Enable debug mode
    )