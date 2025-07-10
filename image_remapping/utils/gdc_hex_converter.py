import gradio as gr
import csv
import re
import numpy as np
import warnings
import os
import tempfile
import zipfile
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

# --- Constants ---
DEFAULT_COLORMAP = 'RdBu_r'

# --- Core Logic Classes ---

class GDCGridProcessor:
    """
    Handles parsing and converting GDC grid data to hex format.
    """
    def __init__(self):
        self.parsed_data = []
        self.dx_values = []
        self.dy_values = []
        self.original_rows = 0
        self.original_cols = 0

    def parse_grid_data_from_content(self, file_content: str) -> list:
        """
        Parses grid data from a string of file content.
        Expected format: "element_name value" per line.
        """
        parsed_data_ordered = []
        lines = file_content.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        element_name = parts[0].strip()
                        value = int(parts[1].strip())
                        parsed_data_ordered.append((element_name, value))
                    else:
                        print(f"Warning: Skipping malformed line {line_num}: '{line}' - Expected 'name value' format.")
                except ValueError:
                    print(f"Warning: Could not convert value to integer for line {line_num}: '{line}' - Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred while parsing line {line_num}: {e}")

        self.parsed_data = parsed_data_ordered
        return parsed_data_ordered

    def extract_and_sort_grid_values(self, original_rows: int, original_cols: int) -> tuple[list, list]:
        """
        Extracts and sorts DX and DY values from parsed data based on their numerical index.
        """
        self.original_rows = original_rows
        self.original_cols = original_cols
        
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
        
        expected_elements = original_rows * original_cols
        if len(self.dx_values) < expected_elements or len(self.dy_values) < expected_elements:
            raise ValueError(f"Insufficient data for {original_rows}x{original_cols} grid. "
                             f"Found {len(self.dx_values)} DX and {len(self.dy_values)} DY elements, "
                             f"but expected {expected_elements} of each.")
        
        self.dx_values = self.dx_values[:expected_elements]
        self.dy_values = self.dy_values[:expected_elements]

        return self.dx_values, self.dy_values

    def reshape_to_2d_grid(self, values: list) -> np.ndarray:
        """
        Reshapes a 1D list of values into a 2D numpy array based on original dimensions.
        """
        expected_elements = self.original_rows * self.original_cols
        if len(values) != expected_elements:
            raise ValueError(f"Mismatch in data length ({len(values)}) and expected grid size ({expected_elements}).")
        return np.array(values).reshape(self.original_rows, self.original_cols)

    def convert_to_hex(self, value: int, bits: int = 16) -> str:
        """
        Converts a decimal value to hexadecimal format.
        Handles signed integers using two's complement for negative values.
        """
        if bits == 16:
            # For 16-bit signed integers (-32768 to 32767)
            if value < 0:
                # Convert negative to two's complement
                hex_value = hex(value & 0xFFFF)[2:].upper().zfill(4)
            else:
                hex_value = hex(value)[2:].upper().zfill(4)
        elif bits == 32:
            # For 32-bit signed integers
            if value < 0:
                hex_value = hex(value & 0xFFFFFFFF)[2:].upper().zfill(8)
            else:
                hex_value = hex(value)[2:].upper().zfill(8)
        else:
            # Default case
            if value < 0:
                hex_value = hex(value & 0xFFFF)[2:].upper().zfill(4)
            else:
                hex_value = hex(value)[2:].upper().zfill(4)
        
        return f"0x{hex_value}"

# --- File Creation Helpers ---

def create_csv_file_with_hex(grid_2d: np.ndarray, filename: str, grid_type: str, hex_bits: int = 16, output_dir: str = None) -> str:
    """Creates a CSV file from 2D grid data with decimal and hex values."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
    else:
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv',
            prefix=filename.replace('.csv', '_')
        )
        file_path = temp_file.name
        temp_file.close()
    
    processor = GDCGridProcessor()
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = []
        for col in range(grid_2d.shape[1]):
            header.extend([f'{grid_type}_Col{col}_Dec', f'{grid_type}_Col{col}_Hex'])
        writer.writerow(header)
        
        # Write data rows
        for row in grid_2d:
            csv_row = []
            for val in row:
                decimal_val = int(val)
                hex_val = processor.convert_to_hex(decimal_val, hex_bits)
                csv_row.extend([decimal_val, hex_val])
            writer.writerow(csv_row)
            
    return file_path

def create_combined_csv_file(dx_grid: np.ndarray, dy_grid: np.ndarray, filename: str, hex_bits: int = 16, output_dir: str = None) -> str:
    """Creates a combined CSV file with both DX and DY grids side by side."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
    else:
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv',
            prefix=filename.replace('.csv', '_')
        )
        file_path = temp_file.name
        temp_file.close()
    
    processor = GDCGridProcessor()
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = []
        # DX columns
        for col in range(dx_grid.shape[1]):
            header.extend([f'DX_Col{col}_Dec', f'DX_Col{col}_Hex'])
        # DY columns
        for col in range(dy_grid.shape[1]):
            header.extend([f'DY_Col{col}_Dec', f'DY_Col{col}_Hex'])
        writer.writerow(header)
        
        # Write data rows
        for i in range(dx_grid.shape[0]):
            csv_row = []
            # DX values for this row
            for val in dx_grid[i]:
                decimal_val = int(val)
                hex_val = processor.convert_to_hex(decimal_val, hex_bits)
                csv_row.extend([decimal_val, hex_val])
            # DY values for this row
            for val in dy_grid[i]:
                decimal_val = int(val)
                hex_val = processor.convert_to_hex(decimal_val, hex_bits)
                csv_row.extend([decimal_val, hex_val])
            writer.writerow(csv_row)
            
    return file_path

def create_txt_file(content: str, filename: str, output_dir: str = None) -> str:
    """Creates a TXT file with content."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
    else:
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt',
            prefix=filename.replace('.txt', '_')
        )
        file_path = temp_file.name
        temp_file.close()
    
    with open(file_path, 'w') as f:
        f.write(content)
            
    return file_path

def create_zip_file(file_paths: list[str], output_zip_path: str = None) -> str:
    """Creates a zip archive containing specified files."""
    if output_zip_path is None:
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_path = temp_zip.name
        temp_zip.close()
    else:
        zip_path = output_zip_path

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
                # Clean up temp files
                if "tmp" in os.path.basename(file_path) and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Warning: Could not remove temporary file {file_path}: {e}")
            else:
                print(f"Warning: File not found for zipping: {file_path}")

    return zip_path

def create_visualization(grid_2d: np.ndarray, title: str, colormap: str = DEFAULT_COLORMAP) -> str:
    """Creates a heatmap visualization of the grid data."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(grid_2d, annot=True, cmap=colormap, cbar=True, 
                     fmt="d", square=False, cbar_kws={'shrink': 0.8},
                     annot_kws={'size': 8})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Column Index', fontsize=12)
    plt.ylabel('Row Index', fontsize=12)
    
    # Add grid dimensions as subtitle
    plt.text(0.5, -0.1, f'Grid Size: {grid_2d.shape[0]} × {grid_2d.shape[1]}', 
             transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return temp_file.name

# --- Main Processing Function ---

def process_gdc_data(
    file_content_or_path: str,
    original_rows: int,
    original_cols: int,
    hex_bits: int,
    show_visualizations: bool,
    output_folder_path: str
):
    """Main processing function for the Gradio interface."""
    # Determine if input is raw content or a file path
    content = ""
    if os.path.exists(str(file_content_or_path)):
        with open(file_content_or_path, 'r') as f:
            content = f.read()
    else:
        content = file_content_or_path

    if not content.strip():
        return "Error: No data provided. Please paste content or upload a file.", \
               "", None, None, None, gr.update(visible=False)

    try:
        processor = GDCGridProcessor()

        # Parse the data
        parsed_data = processor.parse_grid_data_from_content(content)

        if not parsed_data:
            return "Error: No valid data found in the provided content.", \
                   "", None, None, None, gr.update(visible=False)

        # Extract and sort values
        dx_values, dy_values = processor.extract_and_sort_grid_values(original_rows, original_cols)

        # Reshape to 2D grids
        dx_grid_2d = processor.reshape_to_2d_grid(dx_values)
        dy_grid_2d = processor.reshape_to_2d_grid(dy_values)

        # Prepare output files
        all_output_files_for_zip = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_subdir = None
        if output_folder_path:
            final_output_subdir = os.path.join(output_folder_path, f"GDC_Hex_Conversion_{timestamp}")
            os.makedirs(final_output_subdir, exist_ok=True)
            print(f"Saving results to: {final_output_subdir}")
        
        # Create GDC format text files with hex values
        dx_gdc_with_hex = processor.values_to_gdc_format_with_hex(dx_values, 'dx', hex_bits)
        dy_gdc_with_hex = processor.values_to_gdc_format_with_hex(dy_values, 'dy', hex_bits)
        combined_gdc_with_hex = dx_gdc_with_hex + '\n' + dy_gdc_with_hex
        
        # Create text files
        dx_txt = create_txt_file(dx_gdc_with_hex, f"dx_grid_{original_rows}x{original_cols}_with_hex.txt", 
                                final_output_subdir)
        dy_txt = create_txt_file(dy_gdc_with_hex, f"dy_grid_{original_rows}x{original_cols}_with_hex.txt", 
                                final_output_subdir)
        combined_txt = create_txt_file(combined_gdc_with_hex, f"combined_grid_{original_rows}x{original_cols}_with_hex.txt", 
                                      final_output_subdir)
        
        all_output_files_for_zip.extend([dx_csv, dy_csv, combined_csv, dx_txt, dy_txt, combined_txt, original_content_txt])

        # Create visualizations
        dx_viz = None
        dy_viz = None

        if show_visualizations:
            dx_viz = create_visualization(dx_grid_2d, f"DX Grid Values ({original_rows}×{original_cols})")
            dy_viz = create_visualization(dy_grid_2d, f"DY Grid Values ({original_rows}×{original_cols})")

            # Copy visualization files to output directory if specified
            if final_output_subdir:
                for src_path in [dx_viz, dy_viz]:
                    if src_path and os.path.exists(src_path):
                        dest_path = os.path.join(final_output_subdir, os.path.basename(src_path))
                        shutil.copy(src_path, dest_path)
                        all_output_files_for_zip.append(dest_path)

        # Create zip file for download
        zip_for_download = create_zip_file(all_output_files_for_zip)

        # Create sample output for display
        sample_output_lines = []
        for i in range(min(10, len(dx_values))):  # Show first 10 lines as examples
            dx_dec = dx_values[i]
            dx_hex = processor.convert_to_hex(dx_dec, hex_bits)
            sample_output_lines.append(f"yuv_gdc_grid_dx_0_{i} {dx_dec} {dx_hex}")
        
        for i in range(min(5, len(dy_values))):  # Show first 5 DY lines as examples
            dy_dec = dy_values[i]
            dy_hex = processor.convert_to_hex(dy_dec, hex_bits)
            sample_output_lines.append(f"yuv_gdc_grid_dy_0_{i} {dy_dec} {dy_hex}")

        sample_output = "\n".join(sample_output_lines)

        # Create summary
        summary = f"""
✅ Successfully processed GDC grid data and converted to hex format!

📊 **Processing Results:**
- Parsed: {len(dx_values)} DX and {len(dy_values)} DY elements
- Grid shape: {dx_grid_2d.shape}
- Hex format: {hex_bits}-bit

📈 **Grid Statistics:**

**DX Grid:**
- Min: {np.min(dx_grid_2d)} | Max: {np.max(dx_grid_2d)}
- Mean: {np.mean(dx_grid_2d):.2f} | Std: {np.std(dx_grid_2d):.2f}

**DY Grid:**
- Min: {np.min(dy_grid_2d)} | Max: {np.max(dy_grid_2d)}
- Mean: {np.mean(dy_grid_2d):.2f} | Std: {np.std(dy_grid_2d):.2f}

🔢 **Sample Output Format:**
{sample_output}

📁 **Generated Files:**
- Individual TXT files: DX and DY grids in GDC format with hex values
- Combined TXT file: Both grids in one file with hex values
- CSV files: Alternative format with decimal and hex columns
- Original TXT file: Copy of input data
"""
        if show_visualizations:
            summary += """
- Grid visualizations: Heatmaps showing value distributions
"""
        if final_output_subdir:
            summary += f"\n💾 **Saved to:** {final_output_subdir}"

        return (summary, 
                sample_output, 
                zip_for_download, 
                dx_viz, dy_viz, 
                gr.update(visible=True))

    except ValueError as ve:
        return (f"❌ Input Error: {str(ve)}", "", None, None, None, 
                gr.update(visible=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"❌ Unexpected error: {str(e)}", "", None, None, None, 
                gr.update(visible=False))

# --- Gradio Interface Setup ---

def create_gradio_interface():
    """Configures and returns the Gradio Blocks interface."""
    with gr.Blocks(title="GDC Grid to Hex Converter", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔢 GDC Grid to Hex Converter
        
        **Convert your GDC (Geometric Distortion Correction) grid values from decimal to hexadecimal format**
        
        This tool parses GDC grid files and converts decimal values to hexadecimal format, outputting CSV files with both decimal and hex values side by side for easy reference.
        """)
        
        with gr.Row():
            # Left Column - Input and Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 📥 **Input Configuration**")
                
                # Input options
                text_input = gr.Textbox(
                    label="📋 Paste GDC Grid Content",
                    placeholder="Paste your GDC grid data here...\nFormat: yuv_gdc_grid_dx_0_0 1234",
                    lines=8,
                    show_copy_button=True
                )
                
                file_input = gr.File(
                    label="📁 Or Upload GDC Grid File (.txt)",
                    file_types=[".txt"],
                    interactive=True
                )
                
                # Grid dimensions
                with gr.Accordion("⚙️ Grid Configuration", open=True):
                    with gr.Row():
                        original_rows = gr.Number(
                            label="Grid Rows", value=9, precision=0, minimum=1, maximum=100
                        )
                        original_cols = gr.Number(
                            label="Grid Columns", value=7, precision=0, minimum=1, maximum=100
                        )
                    
                    hex_bits = gr.Radio(
                        label="Hex Format",
                        choices=[16, 32],
                        value=16,
                        info="Choose 16-bit (4 hex digits) or 32-bit (8 hex digits)"
                    )
                
                # Output settings
                output_folder_path = gr.Textbox(
                    label="💾 Local Output Folder Path (Optional)",
                    placeholder="/path/to/output/folder",
                    info="Specify a directory to save results on the server"
                )

                show_vis = gr.Checkbox(
                    label="🎨 Generate Visualizations",
                    value=True,
                    info="Create heatmap visualizations of the grids"
                )
                
                process_btn = gr.Button(
                    "🔄 Convert to Hex",
                    variant="primary",
                    size="lg"
                )
                
                # Summary output
                summary_output = gr.Textbox(
                    label="📊 Processing Summary",
                    lines=15,
                    show_copy_button=True,
                    interactive=False
                )
                
                # Download section
                download_output = gr.File(
                    label="📦 Download All Results (ZIP)",
                    visible=False,
                    file_count="single"
                )
            
            # Right Column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### ✨ **Conversion Results**")
                
                gr.Markdown("#### 📝 **Sample Output Format**")
                hex_output = gr.Textbox(
                    label="GDC Format with Hex Values (Sample)",
                    lines=10,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Output format examples will appear here..."
                )
        
        # Visualizations section
        gr.Markdown("---")
        gr.Markdown("### 📊 **Grid Visualizations**")
        
        with gr.Row():
            dx_viz = gr.Image(
                label="📈 DX Grid Values", 
                visible=False, 
                interactive=False,
                height=400
            )
            dy_viz = gr.Image(
                label="📉 DY Grid Values", 
                visible=False, 
                interactive=False,
                height=400
            )
        
        # Register the click event
        def complete_processing(text_content, file_obj, orig_rows, orig_cols, 
                              hex_format, show_vis, output_path):
            """Complete processing with proper visibility handling."""
            
            if text_content and text_content.strip():
                result = process_gdc_data(text_content, orig_rows, orig_cols, 
                                        hex_format, show_vis, output_path)
            elif file_obj:
                result = process_gdc_data(file_obj.name, orig_rows, orig_cols, 
                                        hex_format, show_vis, output_path)
            else:
                result = ("❌ No data provided. Please paste content or upload a file.", 
                         "", None, None, None, gr.update(visible=False))
            
            # Unpack result
            summary, hex_out, download, dx_viz_img, dy_viz_img, download_vis_update = result
            
            # Update visibility based on whether visualizations were generated
            vis_available = show_vis and dx_viz_img is not None
            
            return (
                summary, hex_out, download,
                gr.update(value=dx_viz_img, visible=vis_available),
                gr.update(value=dy_viz_img, visible=vis_available),
                download_vis_update
            )

        process_btn.click(
            fn=complete_processing,
            inputs=[
                text_input, file_input, original_rows, original_cols,
                hex_bits, show_vis, output_folder_path
            ],
            outputs=[
                summary_output, hex_output, download_output,
                dx_viz, dy_viz, download_output
            ]
        )
        
        # Information section
        gr.Markdown("""
        ---
        ### 📚 **Output Formats**
        
        **🔸 Individual TXT Files**: Separate files for DX and DY grids in GDC format with hex values  
        **🔸 Combined TXT File**: Both DX and DY grids in one file with hex values  
        **🔸 CSV Files**: Alternative format with decimal and hex columns for spreadsheet use  
        **🔸 Original TXT File**: Copy of your input data for reference  
        **🔸 Visualizations**: Heatmap representations of the grid values  
        
        ### 📝 **Output Format Example**
        ```
        yuv_gdc_grid_dx_0_0 1234 0x04D2
        yuv_gdc_grid_dx_0_1 1235 0x04D3
        yuv_gdc_grid_dx_0_2 1236 0x04D4
        ...
        yuv_gdc_grid_dy_0_0 5678 0x162E
        yuv_gdc_grid_dy_0_1 5679 0x162F
        ...
        ```
        
        ### 🧩 **Expected Input Format**
        ```
        yuv_gdc_grid_dx_0_0 1234
        yuv_gdc_grid_dx_0_1 1235
        yuv_gdc_grid_dx_0_2 1236
        ...
        yuv_gdc_grid_dy_0_0 5678
        yuv_gdc_grid_dy_0_1 5679
        ...
        ```
        
        **🔸 16-bit Hex**: Values from -32768 to 32767 → 4 hex digits (e.g., 0x04D2)  
        **🔸 32-bit Hex**: Extended range → 8 hex digits (e.g., 0x000004D2)  
        **🔸 Negative Values**: Automatically handled using two's complement representation  
        """)
            
    return interface

# --- Main Application Launch ---

if __name__ == "__main__":
    # Dependencies check
    try:
        import gradio as gr
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import shutil
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install: pip install gradio numpy matplotlib seaborn pandas")
        exit(1)
        
    interface = create_gradio_interface()
    interface.launch(
        server_name="localhost",
        server_port=7860,
        share=True,
        debug=True
    )