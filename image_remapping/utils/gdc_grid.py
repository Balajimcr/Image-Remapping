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
from datetime import datetime
import shutil # Added for file copying

# --- Constants ---
DEFAULT_COLORMAP = 'RdBu_r'

# --- Core Logic Classes ---

class GDCGridProcessor:
    """
    Handles parsing, extracting, and interpolating GDC grid data.
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

    def interpolate_grid_bicubic(self, grid_2d: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
        """
        Interpolates a 2D grid using bicubic interpolation.
        """
        original_rows, original_cols = grid_2d.shape

        x_orig = np.linspace(0, 1, original_cols)
        y_orig = np.linspace(0, 1, original_rows)
        x_target = np.linspace(0, 1, target_cols)
        y_target = np.linspace(0, 1, target_rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spline = RectBivariateSpline(y_orig, x_orig, grid_2d, kx=3, ky=3, s=0)
            interpolated_grid = spline(y_target, x_target)

        return interpolated_grid

    def grid_2d_to_gdc_format(self, grid_2d: np.ndarray, grid_type: str) -> str:
        """
        Converts a 2D grid back to GDC format text.
        
        Args:
            grid_2d (np.ndarray): The 2D grid data
            grid_type (str): Either 'dx' or 'dy'
        
        Returns:
            str: GDC formatted text
        """
        rows, cols = grid_2d.shape
        gdc_lines = []
        
        # Flatten the grid in row-major order and create GDC format
        flat_grid = grid_2d.flatten()
        
        for i, value in enumerate(flat_grid):
            element_name = f"yuv_gdc_grid_{grid_type}_0_{i}"
            # Convert to integer for GDC format
            int_value = int(round(value))
            gdc_lines.append(f"{element_name} {int_value}")
        
        return '\n'.join(gdc_lines)

# --- File and Visualization Helpers ---

def create_csv_file(grid_2d: np.ndarray, filename: str, output_dir: str = None) -> str:
    """Creates a CSV file from 2D grid data."""
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
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in grid_2d:
            rounded_row = [round(val, 6) for val in row]
            writer.writerow(rounded_row)
            
    return file_path

def create_txt_file(content: str, filename: str, output_dir: str = None) -> str:
    """Creates a TXT file with GDC format content."""
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
            # Ensure the file actually exists before trying to add it
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
                # Clean up individual temp files if they were created and added to zip
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
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with better formatting
    ax = sns.heatmap(grid_2d, annot=False, cmap=colormap, cbar=True, 
                     fmt=".0f", square=False, cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.xlabel('Column Index', fontsize=10)
    plt.ylabel('Row Index', fontsize=10)
    
    # Add grid dimensions as subtitle
    plt.text(0.5, -0.1, f'Grid Size: {grid_2d.shape[0]} × {grid_2d.shape[1]}', 
             transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return temp_file.name

def create_distortion_grid_visualization(dx_grid: np.ndarray, dy_grid: np.ndarray, 
                                       title: str = "Distortion Grid") -> str:
    """
    Creates a distortion grid visualization showing target points, distorted points, and vectors.
    Similar to the lens distortion visualization with blue target points, red distorted points,
    and green distortion vectors.
    """
    rows, cols = dx_grid.shape
    
    # Create target grid (regular grid points)
    target_x = np.zeros((rows, cols))
    target_y = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            target_x[r, c] = c
            target_y[r, c] = r
    
    # Apply distortion (target + displacement = distorted position)
    # Note: DX/DY values might need scaling depending on the data range
    # A common range for GDC values is -512 to 511. Scaling factor helps visualize.
    max_abs_val = max(np.max(np.abs(dx_grid)), np.max(np.abs(dy_grid)))
    # Prevent division by zero if all values are zero
    scale_factor = (cols / 10.0) / max_abs_val if max_abs_val > 0 else 0.001 
    
    distorted_x = target_x + dx_grid * scale_factor
    distorted_y = target_y + dy_grid * scale_factor
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set colors
    target_color = 'blue'
    distorted_color = 'red'
    vector_color = 'green'
    grid_color = 'lightgray'
    
    # Draw grid lines (target grid)
    for r in range(rows):
        ax.plot([target_x[r, 0], target_x[r, -1]], 
                [target_y[r, 0], target_y[r, -1]], 
                color=grid_color, alpha=0.7, linewidth=1)
    
    for c in range(cols):
        ax.plot([target_x[0, c], target_x[-1, c]], 
                [target_y[0, c], target_y[-1, c]], 
                color=grid_color, alpha=0.7, linewidth=1)
    
    # Draw distortion vectors (only if displacement is significant)
    min_vector_length_display = 0.02 * np.sqrt(cols**2 + rows**2) / 10 # Relative to grid size
    
    for r in range(rows):
        for c in range(cols):
            dx_vec = distorted_x[r, c] - target_x[r, c]
            dy_vec = distorted_y[r, c] - target_y[r, c]
            vector_length = np.sqrt(dx_vec*dx_vec + dy_vec*dy_vec)
            
            if vector_length > min_vector_length_display:
                ax.annotate('', xy=(distorted_x[r, c], distorted_y[r, c]),
                           xytext=(target_x[r, c], target_y[r, c]),
                           arrowprops=dict(arrowstyle='->', color=vector_color, 
                                         lw=1.5, alpha=0.8, mutation_scale=10)) # mutation_scale for arrow head size
    
    # Plot target points (blue dots)
    ax.scatter(target_x.flatten(), target_y.flatten(), 
              c=target_color, s=30, alpha=0.8, label='Target Grid', zorder=3)
    
    # Plot distorted points (red dots)
    ax.scatter(distorted_x.flatten(), distorted_y.flatten(), 
              c=distorted_color, s=25, alpha=0.8, label='Distorted Points', zorder=4)
    
    # Set equal aspect ratio and clean up the plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_title(f'{title} ({cols}x{rows})', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=target_color, 
                  markersize=8, label='Target Grid'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=distorted_color, 
                  markersize=8, label='Distorted Points'),
        plt.Line2D([0], [0], color=vector_color, linewidth=2, 
                  label='Distortion Vectors')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Set reasonable axis limits
    all_x = np.concatenate([target_x.flatten(), distorted_x.flatten()])
    all_y = np.concatenate([target_y.flatten(), distorted_y.flatten()])
    
    margin_x = (np.max(all_x) - np.min(all_x)) * 0.1
    margin_y = (np.max(all_y) - np.min(all_y)) * 0.1
    ax.set_xlim(np.min(all_x) - margin_x, np.max(all_x) + margin_x)
    ax.set_ylim(np.min(all_y) - margin_y, np.max(all_y) + margin_y)
    
    # Invert y-axis to match typical image coordinates (origin top-left)
    ax.invert_yaxis()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return temp_file.name

def create_side_by_side_comparison(original_grid: np.ndarray, interpolated_grid: np.ndarray, grid_type: str) -> str:
    """
    Creates a side-by-side heatmap comparison of original (small) and interpolated (large) grids.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use same colormap and scale for both plots
    vmin = min(np.min(original_grid), np.min(interpolated_grid))
    vmax = max(np.max(original_grid), np.max(interpolated_grid))
    
    # Original (smaller) grid
    sns.heatmap(original_grid, cmap=DEFAULT_COLORMAP, ax=ax1, 
                cbar=False, annot=False, fmt=".0f", 
                vmin=vmin, vmax=vmax, square=False) 
    ax1.set_title(f'Original {grid_type} Grid\n({original_grid.shape[0]} × {original_grid.shape[1]})', 
                  fontweight='bold', fontsize=11)
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')

    # Interpolated (bigger) grid
    sns.heatmap(interpolated_grid, cmap=DEFAULT_COLORMAP, ax=ax2, 
                cbar=False, annot=False, fmt=".0f", 
                vmin=vmin, vmax=vmax, square=False)
    ax2.set_title(f'Interpolated {grid_type} Grid\n({interpolated_grid.shape[0]} × {interpolated_grid.shape[1]})', 
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')

    # Create a ScalarMappable for the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=DEFAULT_COLORMAP)
    sm.set_array([])  # This is required for ScalarMappable
    
    # Add main title first
    fig.suptitle(f'{grid_type} Grid Comparison: Small → Large', fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout manually instead of using tight_layout
    plt.subplots_adjust(left=0.05, right=0.85, top=0.88, bottom=0.12, wspace=0.3)
    
    # Add shared colorbar using the ScalarMappable
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.65])  # Position [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)  # Use the ScalarMappable instead of the Axes object
    cbar.set_label(f'{grid_type} Values', rotation=270, labelpad=15)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return temp_file.name

def create_grid_overview(dx_orig: np.ndarray, dy_orig: np.ndarray, 
                        dx_interp: np.ndarray, dy_interp: np.ndarray) -> str:
    """Creates a 2x2 overview of all grids."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Determine common vmin/vmax for DX and DY independently or together
    # For a general overview, it's often better to scale individually or by grid type.
    # Here, let's scale DX grids together and DY grids together for better comparison.
    dx_vmin = min(np.min(dx_orig), np.min(dx_interp))
    dx_vmax = max(np.max(dx_orig), np.max(dx_interp))
    dy_vmin = min(np.min(dy_orig), np.min(dy_interp))
    dy_vmax = max(np.max(dy_orig), np.max(dy_interp))

    # DX Original
    im1 = sns.heatmap(dx_orig, cmap=DEFAULT_COLORMAP, ax=ax1, annot=False, fmt=".0f",
                      cbar=True, vmin=dx_vmin, vmax=dx_vmax, cbar_kws={'shrink': 0.7})
    ax1.set_title(f'Original DX\n({dx_orig.shape[0]}×{dx_orig.shape[1]})', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # DX Interpolated
    im2 = sns.heatmap(dx_interp, cmap=DEFAULT_COLORMAP, ax=ax2, annot=False, fmt=".0f",
                      cbar=True, vmin=dx_vmin, vmax=dx_vmax, cbar_kws={'shrink': 0.7})
    ax2.set_title(f'Interpolated DX\n({dx_interp.shape[0]}×{dx_interp.shape[1]})', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # DY Original
    im3 = sns.heatmap(dy_orig, cmap=DEFAULT_COLORMAP, ax=ax3, annot=False, fmt=".0f",
                      cbar=True, vmin=dy_vmin, vmax=dy_vmax, cbar_kws={'shrink': 0.7})
    ax3.set_title(f'Original DY\n({dy_orig.shape[0]}×{dy_orig.shape[1]})', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    
    # DY Interpolated
    im4 = sns.heatmap(dy_interp, cmap=DEFAULT_COLORMAP, ax=ax4, annot=False, fmt=".0f",
                      cbar=True, vmin=dy_vmin, vmax=dy_vmax, cbar_kws={'shrink': 0.7})
    ax4.set_title(f'Interpolated DY\n({dy_interp.shape[0]}×{dy_interp.shape[1]})', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    
    plt.suptitle('Complete Grid Overview: Original vs Interpolated', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92) # Adjust spacing
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return temp_file.name

# --- Main Gradio Processing Function ---

def process_gdc_data(
    file_content_or_path: str,
    original_rows: int,
    original_cols: int,
    target_rows: int,
    target_cols: int,
    show_visualizations: bool,
    output_folder_path: str
):
    """Main processing function for the Gradio interface."""
    # Determine if input is raw content or a file path
    content = ""
    if os.path.exists(str(file_content_or_path)): # Check if it's a temp file path from gr.File
        with open(file_content_or_path, 'r') as f:
            content = f.read()
    else: # Assume it's direct text input
        content = file_content_or_path

    if not content.strip():
        # Corrected return tuple to match expected outputs of complete_processing
        return "Error: No data provided. Please paste content or upload a file.", \
               "", "", None, None, None, None, None, gr.update(visible=False)

    try:
        processor = GDCGridProcessor()

        # Parse the data
        parsed_data = processor.parse_grid_data_from_content(content)

        if not parsed_data:
            # Corrected return tuple
            return "Error: No valid data found in the provided content.", \
                   "", "", None, None, None, None, None, gr.update(visible=False)

        # Extract and sort values
        dx_values, dy_values = processor.extract_and_sort_grid_values(original_rows, original_cols)

        # Reshape to 2D grids
        dx_grid_2d = processor.reshape_to_2d_grid(dx_values)
        dy_grid_2d = processor.reshape_to_2d_grid(dy_values)

        # Perform interpolation
        dx_interpolated = processor.interpolate_grid_bicubic(dx_grid_2d, target_rows, target_cols)
        dy_interpolated = processor.interpolate_grid_bicubic(dy_grid_2d, target_rows, target_cols)

        # Convert interpolated grids back to GDC format
        dx_gdc_output = processor.grid_2d_to_gdc_format(dx_interpolated, 'dx')
        dy_gdc_output = processor.grid_2d_to_gdc_format(dy_interpolated, 'dy')
        combined_gdc_output = dx_gdc_output + '\n' + dy_gdc_output

        # Prepare output files
        all_output_files_for_zip = [] # List to track all files that should go into the downloadable zip
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_subdir = None
        if output_folder_path:
            final_output_subdir = os.path.join(output_folder_path, f"GDC_Interpolation_{timestamp}")
            os.makedirs(final_output_subdir, exist_ok=True)
            print(f"Saving results to: {final_output_subdir}")
        
        # Create CSV files
        original_dx_csv = create_csv_file(dx_grid_2d, f"original_dx_{original_rows}x{original_cols}.csv", final_output_subdir)
        original_dy_csv = create_csv_file(dy_grid_2d, f"original_dy_{original_rows}x{original_cols}.csv", final_output_subdir)
        interp_dx_csv = create_csv_file(dx_interpolated, f"interpolated_dx_{target_rows}x{target_cols}.csv", final_output_subdir)
        interp_dy_csv = create_csv_file(dy_interpolated, f"interpolated_dy_{target_rows}x{target_cols}.csv", final_output_subdir)
        
        # Create TXT files in GDC format
        interp_dx_txt = create_txt_file(dx_gdc_output, f"interpolated_dx_{target_rows}x{target_cols}.txt", final_output_subdir)
        interp_dy_txt = create_txt_file(dy_gdc_output, f"interpolated_dy_{target_rows}x{target_cols}.txt", final_output_subdir)
        combined_txt = create_txt_file(combined_gdc_output, f"interpolated_combined_{target_rows}x{target_cols}.txt", final_output_subdir)
        
        # Collect all file paths for the downloadable zip
        all_output_files_for_zip.extend([
            original_dx_csv, original_dy_csv, interp_dx_csv, interp_dy_csv,
            interp_dx_txt, interp_dy_txt, combined_txt
        ])

        # Create visualizations
        dx_comparison = None
        dy_comparison = None
        grid_overview = None
        distortion_viz = None

        if show_visualizations:
            # Create side-by-side comparisons (heatmaps)
            dx_comparison = create_side_by_side_comparison(dx_grid_2d, dx_interpolated, "DX")
            dy_comparison = create_side_by_side_comparison(dy_grid_2d, dy_interpolated, "DY")
            
            # Create complete overview (heatmaps)
            grid_overview = create_grid_overview(dx_grid_2d, dy_grid_2d, dx_interpolated, dy_interpolated)
            
            # Create distortion grid visualization (vectors and points)
            distortion_viz = create_distortion_grid_visualization(
                dx_grid_2d, dy_grid_2d, "Original Distortion Grid") # Using original for the main distortion viz

            # Copy visualization temp files to final output directory if specified
            if final_output_subdir:
                for src_path in [dx_comparison, dy_comparison, grid_overview, distortion_viz]:
                    if src_path and os.path.exists(src_path): # Check if visualization was actually created
                        dest_path = os.path.join(final_output_subdir, os.path.basename(src_path))
                        shutil.copy(src_path, dest_path)
                        all_output_files_for_zip.append(dest_path) # Add to the list for the main zip

        # Create zip file for download. This zip will contain all temp CSVs, TXTs, and copied images.
        zip_for_download = create_zip_file(all_output_files_for_zip)

        # If a specific output folder was provided, create a *final* zip there
        if final_output_subdir:
            final_zip_path = os.path.join(final_output_subdir, f"GDC_Interpolation_Results_{timestamp}.zip")
            files_to_zip_from_final_dir = [os.path.join(final_output_subdir, f) for f in os.listdir(final_output_subdir) if f.endswith(('.csv', '.txt', '.png'))]
            create_zip_file(files_to_zip_from_final_dir, final_zip_path)

        # Create summary
        summary = f"""
✅ Successfully processed GDC grid data!

📊 **Processing Results:**
- Parsed: {len(dx_values)} DX and {len(dy_values)} DY elements
- Original grid shape: {dx_grid_2d.shape}
- Target grid shape: ({target_rows}, {target_cols})
- Interpolation method: Bicubic

📈 **Grid Statistics:**

**DX Grid (Original):**
- Min: {np.min(dx_grid_2d):.0f} | Max: {np.max(dx_grid_2d):.0f}
- Mean: {np.mean(dx_grid_2d):.2f} | Std: {np.std(dx_grid_2d):.2f}

**DY Grid (Original):**
- Min: {np.min(dy_grid_2d):.0f} | Max: {np.max(dy_grid_2d):.0f}
- Mean: {np.mean(dy_grid_2d):.2f} | Std: {np.std(dy_grid_2d):.2f}

**DX Grid (Interpolated):**
- Min: {np.min(dx_interpolated):.0f} | Max: {np.max(dx_interpolated):.0f}
- Mean: {np.mean(dx_interpolated):.2f} | Std: {np.std(dx_interpolated):.2f}

**DY Grid (Interpolated):**
- Min: {np.min(dy_interpolated):.0f} | Max: {np.max(dy_interpolated):.0f}
- Mean: {np.mean(dy_interpolated):.2f} | Std: {np.std(dy_interpolated):.2f}

📁 **Generated Files:**
- CSV format: Original and interpolated grids
- TXT format: GDC-compatible interpolated grids
- Combined TXT: Both DX and DY grids in one file
"""
        if show_visualizations:
            summary += """
- Heatmap comparisons: Small vs Large grids (DX, DY)
- Complete overview: All 4 grids in one view
- Distortion grid: Target points, distorted points, and vectors
"""
        if final_output_subdir:
            summary += f"\n💾 **Saved to:** {final_output_subdir}"

        return (summary, dx_gdc_output, combined_gdc_output, zip_for_download, 
                dx_comparison, dy_comparison, grid_overview, distortion_viz,
                gr.update(visible=True))

    except ValueError as ve:
        return (f"❌ Input Error: {str(ve)}", "", "", None, None, None, None, None, 
                gr.update(visible=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"❌ Unexpected error: {str(e)}", "", "", None, None, None, None, None, 
                gr.update(visible=False))

# --- Gradio Interface Setup ---

def create_gradio_interface():
    """Configures and returns the Gradio Blocks interface."""
    with gr.Blocks(title="GDC Grid Interpolation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🌐 GDC Grid Interpolation Tool
        
        **Transform your GDC (Geometric Distortion Correction) grids with high-quality bicubic interpolation**
        
        This tool parses GDC grid files and interpolates them from lower to higher resolution, outputting both CSV and original GDC .txt formats. Features comprehensive visualizations including distortion vector plots similar to lens distortion analysis tools.
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
                with gr.Accordion("⚙️ Grid Dimensions", open=True):
                    with gr.Row():
                        original_rows = gr.Number(
                            label="Original Rows", value=9, precision=0, minimum=1, maximum=100
                        )
                        original_cols = gr.Number(
                            label="Original Columns", value=7, precision=0, minimum=1, maximum=100
                        )
                    
                    with gr.Row():
                        target_rows = gr.Number(
                            label="Target Rows", value=33, precision=0, minimum=1, maximum=1000
                        )
                        target_cols = gr.Number(
                            label="Target Columns", value=33, precision=0, minimum=1, maximum=1000
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
                    info="Create heatmap visualizations"
                )
                
                process_btn = gr.Button(
                    "🚀 Process Grid",
                    variant="primary",
                    size="lg"
                )
                
                # Summary output
                summary_output = gr.Textbox(
                    label="📊 Processing Summary",
                    lines=12,
                    show_copy_button=True,
                    interactive=False
                )
                
                # Download section
                download_output = gr.File(
                    label="📦 Download All Results (ZIP)",
                    visible=False,
                    file_count="single"
                )
            
            # Right Column - Outputs and Visualizations
            with gr.Column(scale=1):
                gr.Markdown("### ✨ **Output Results**")
                
                # Output grids in GDC format
                gr.Markdown("#### 📋 **Interpolated DX Grid (GDC Format)**")
                dx_output = gr.Textbox(
                    label="DX Grid Output",
                    lines=8,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Processed DX grid data will appear here..."
                )
                
                gr.Markdown("#### 📋 **Combined DX + DY Grid (GDC Format)**")
                combined_output = gr.Textbox(
                    label="Combined Grid Output",
                    lines=8,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Combined grid data will appear here..."
                )
        
        # Visualizations section
        gr.Markdown("---")
        gr.Markdown("### 📊 **Grid Visualizations - Small vs Large Comparison**")
        
        with gr.Row():
            dx_comparison = gr.Image(
                label="📈 DX Grid: Original (Small) vs Interpolated (Large)", 
                visible=False, 
                interactive=False,
                height=350
            )
        
        with gr.Row():
            dy_comparison = gr.Image(
                label="📉 DY Grid: Original (Small) vs Interpolated (Large)", 
                visible=False, 
                interactive=False,
                height=350
            )
        
        with gr.Row():
            grid_overview = gr.Image(
                label="🔍 Complete Grid Overview (All 4 Grids)", 
                visible=False, 
                interactive=False,
                height=500
            )
        
        # Distortion Grid Visualization
        gr.Markdown("### 🎯 **Distortion Grid Visualization**")
        gr.Markdown("""
        **This visualization shows the actual geometric distortion pattern:**
        - 🔵 **Blue dots**: Target grid points (regular/ideal grid positions)
        - 🔴 **Red dots**: Distorted points (actual positions after applying DX/DY displacements)
        - 🟢 **Green arrows**: Distortion vectors showing the direction and magnitude of displacement
        
        Similar to lens distortion analysis, this helps visualize how the grid deforms from ideal to actual positions.
        """)
        with gr.Row():
            distortion_viz = gr.Image(
                label="🌐 Distortion Grid: Target Points → Distorted Points with Vectors", 
                visible=False, 
                interactive=False,
                height=450
            )
        
        # Register the click event
        def complete_processing(text_content, file_obj, orig_rows, orig_cols, 
                              targ_rows, targ_cols, show_vis, output_path):
            """Complete processing with proper visibility handling."""
            
            # Process the data
            if text_content and text_content.strip():
                result = process_gdc_data(text_content, orig_rows, orig_cols, 
                                        targ_rows, targ_cols, show_vis, output_path)
            elif file_obj:
                result = process_gdc_data(file_obj.name, orig_rows, orig_cols, 
                                        targ_rows, targ_cols, show_vis, output_path)
            else:
                result = ("❌ No data provided. Please paste content or upload a file.", 
                         "", "", None, None, None, None, None, gr.update(visible=False))
            
            # Unpack result
            summary, dx_out, combined_out, download, dx_comp, dy_comp, overview, distortion, download_vis_update = result
            
            # Update visibility based on whether visualizations were generated
            vis_available = show_vis and dx_comp is not None # Check if vis_enabled AND a plot path was returned
            
            return (
                summary, dx_out, combined_out, download,
                gr.update(value=dx_comp, visible=vis_available),      # dx_comparison
                gr.update(value=dy_comp, visible=vis_available),      # dy_comparison  
                gr.update(value=overview, visible=vis_available),     # grid_overview
                gr.update(value=distortion, visible=vis_available),   # distortion_viz
                download_vis_update # Pass the direct update object for download_output visibility
            )

        process_btn.click(
            fn=complete_processing,
            inputs=[
                text_input, file_input, original_rows, original_cols,
                target_rows, target_cols, show_vis, output_folder_path
            ],
            outputs=[
                summary_output, dx_output, combined_output, download_output,
                dx_comparison, dy_comparison, grid_overview, distortion_viz, download_output # Download output twice to update value and visibility
            ]
        )
        
        # Information section
        gr.Markdown("""
        ---
        ### 📚 **Output Formats**
        
        **🔸 CSV Files**: Numerical grid data in spreadsheet format for analysis  
        **🔸 TXT Files**: Original GDC format compatible with your processing pipeline  
        **🔸 Combined File**: Both DX and DY grids in a single GDC-format file  
        **🔸 Visualizations**: Multiple visualization types for comprehensive analysis  
        
        ### 📊 **Visualization Guide**
        
        **🔸 DX/DY Heatmap Comparisons**: Shows original (small) vs interpolated (large) grids as color maps  
        **🔸 Complete Grid Overview**: All 4 grids (original DX/DY + interpolated DX/DY) in one heatmap view  
        **🔸 Distortion Grid Visualization**: Shows actual geometric distortion with:
        - 🔵 **Blue dots**: Target grid points (regular grid)
        - 🔴 **Red dots**: Distorted points (after applying DX/DY displacements)  
        - 🟢 **Green arrows**: Distortion vectors showing displacement direction and magnitude
        **🔸 Color Scale**: Shared across comparisons for accurate visual comparison  
        
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
        
        The suffix `_0_N` represents the flattened index of the grid element.
        """)
            
    return interface

# --- Main Application Launch ---

if __name__ == "__main__":
    # Dependencies check
    try:
        import gradio as gr
        import numpy as np
        from scipy.interpolate import RectBivariateSpline
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import shutil # Ensure shutil is imported for file copying
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install: pip install gradio numpy scipy matplotlib seaborn pandas")
        exit(1)
        
    interface = create_gradio_interface()
    interface.launch(
        server_name="localhost", # Use "0.0.0.0" to make it accessible from other devices on the network
        server_port=7860,
        share=True,
        debug=True
    )