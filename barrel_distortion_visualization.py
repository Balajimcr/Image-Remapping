import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import math
from typing import Tuple, Optional

def visualize_distortion_grid(simulator, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create visualization of the distortion grid
    
    Returns:
        Visualization image as numpy array
    """
    source_coords, target_coords = simulator.generate_sparse_distortion_grid()
    
    # Create visualization canvas
    scale_factor = 0.5
    display_width = int(simulator.image_width * scale_factor)
    display_height = int(simulator.image_height * scale_factor)
    
    img = np.ones((display_height, display_width, 3), dtype=np.uint8) * 255
    
    # Colors
    target_color = (255, 0, 0)    # Blue for target points
    source_color = (0, 0, 255)    # Red for source points
    grid_color = (200, 200, 200)  # Light gray for grid lines
    vector_color = (0, 150, 0)    # Green for distortion vectors
    
    # Scale coordinates for display
    def scale_coord(coord):
        return (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
    
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
            if np.linalg.norm(np.array(source_pt) - np.array(target_pt)) > 2:
                cv2.arrowedLine(img, target_pt, source_pt, vector_color, 2, tipLength=0.1)
            
            # Draw target point (blue)
            cv2.circle(img, target_pt, 4, target_color, -1)
            
            # Draw source point (red)
            cv2.circle(img, source_pt, 3, source_color, -1)
    
    # Add legend
    legend_y = 30
    cv2.circle(img, (20, legend_y), 4, target_color, -1)
    cv2.putText(img, "Target Grid", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    legend_y += 25
    cv2.circle(img, (20, legend_y), 3, source_color, -1)
    cv2.putText(img, "Distorted Points", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    legend_y += 25
    cv2.arrowedLine(img, (15, legend_y), (25, legend_y), vector_color, 2, tipLength=0.3)
    cv2.putText(img, "Distortion Vectors", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add title
    title = f"Barrel Distortion Grid ({simulator.grid_cols}x{simulator.grid_rows})"
    cv2.putText(img, title, (display_width//2 - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, img)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_distortion_heatmap(simulator) -> np.ndarray:
    """
    Create heatmap showing distortion magnitude with simple JET colormap
    
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
    
    # Create matplotlib figure with JET colormap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap with JET colormap and no annotations
    im = ax.imshow(distortion_magnitudes, cmap='jet', interpolation='bilinear', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Distortion Magnitude (pixels)')
    
    # Clean appearance - no grid annotations or text
    ax.set_title(f'Distortion Magnitude Heatmap ({simulator.grid_cols}x{simulator.grid_rows} Grid)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    
    # Remove ticks for cleaner look
    ax.set_xticks(range(simulator.grid_cols))
    ax.set_yticks(range(simulator.grid_rows))
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    
    # Decode image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def export_grid_to_csv(simulator) -> str:
    """
    Export grid parameters to CSV format
    
    Returns:
        CSV formatted string with grid data
    """
    source_coords, target_coords = simulator.generate_sparse_distortion_grid()
    
    # Calculate displacement vectors
    displacement_x = source_coords[:, :, 0] - target_coords[:, :, 0]
    displacement_y = source_coords[:, :, 1] - target_coords[:, :, 1]
    
    csv_lines = []
    csv_lines.append("Array_Index_Row_Col,Value_Decimal,Value_Hex")
    
    # Export Grid X displacements
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"grid_x_{row}_{col}"
            value = displacement_x[row, col]
            # Convert to hex (handling negative values)
            hex_value = f"0x{int(value) & 0xFFFFFFFF:08X}" if value < 0 else f"0x{int(value):08X}"
            csv_lines.append(f"{index},{value:.6f},{hex_value}")
    
    # Export Grid Y displacements  
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"grid_y_{row}_{col}"
            value = displacement_y[row, col]
            # Convert to hex (handling negative values)
            hex_value = f"0x{int(value) & 0xFFFFFFFF:08X}" if value < 0 else f"0x{int(value):08X}"
            csv_lines.append(f"{index},{value:.6f},{hex_value}")
    
    return "\n".join(csv_lines)

def convert_barrel_distortion_to_gdc_grid(simulator, gdc_image_width=8192, gdc_image_height=6144) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert barrel distortion grid coordinates to GDC (Geometric Distortion Correction) format
    
    Args:
        gdc_image_width: Target GDC image width (default 8192)
        gdc_image_height: Target GDC image height (default 6144)
    
    Returns:
        tuple: (grid_distort_x, grid_distort_y) - Arrays in GDC format with int32 values
    """
    source_coords, target_coords = simulator.generate_sparse_distortion_grid()
    
    # Initialize output grids
    grid_distort_x = np.zeros((simulator.grid_rows, simulator.grid_cols), dtype=np.int32)
    grid_distort_y = np.zeros((simulator.grid_rows, simulator.grid_cols), dtype=np.int32)
    
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            # Get source and target coordinates
            src_x, src_y = source_coords[row, col]
            tgt_x, tgt_y = target_coords[row, col]
            
            # Calculate delta (Source - Target)
            delta_x = float(src_x - tgt_x)
            delta_y = float(src_y - tgt_y)
            
            # Apply the GDC conversion formula for X
            # grid_distort_x = (ceil(((Delta_x * round((8192 << 14) / image_width)) << 2) / (2 ^ 14))) << 7
            
            # Step 1: Calculate scale factor (integer operations)
            scale_factor_x = int(round((8192 * (2 ** 14)) / gdc_image_width))
            
            # Step 2: Apply scaling and bit shifting
            scaled_delta_x = delta_x * scale_factor_x
            shifted_delta_x = scaled_delta_x * 4  # << 2 equivalent
            
            # Step 3: Divide by 2^14 and ceiling
            intermediate_x = math.ceil(shifted_delta_x / (2 ** 14))
            
            # Step 4: Final bit shift (<< 7) - ensure integer
            grid_distort_x[row, col] = int(intermediate_x) << 7
            
            # Apply the GDC conversion formula for Y
            # grid_distort_y = (ceil(((Delta_y * round((6144 << 14) / image_height)) << 2) / (2 ^ 14))) << 7
            
            # Step 1: Calculate scale factor (integer operations)
            scale_factor_y = int(round((6144 * (2 ** 14)) / gdc_image_height))
            
            # Step 2: Apply scaling and bit shifting
            scaled_delta_y = delta_y * scale_factor_y
            shifted_delta_y = scaled_delta_y * 4  # << 2 equivalent
            
            # Step 3: Divide by 2^14 and ceiling
            intermediate_y = math.ceil(shifted_delta_y / (2 ** 14))
            
            # Step 4: Final bit shift (<< 7) - ensure integer
            grid_distort_y[row, col] = int(intermediate_y) << 7
    
    return grid_distort_x, grid_distort_y

def export_gdc_grid_to_csv(simulator, gdc_image_width=8192, gdc_image_height=6144) -> str:
    """
    Export GDC grid parameters to CSV format
    
    Args:
        gdc_image_width: Target GDC image width (default 8192)
        gdc_image_height: Target GDC image height (default 6144)
    
    Returns:
        CSV formatted string with GDC grid data
    """
    grid_distort_x, grid_distort_y = convert_barrel_distortion_to_gdc_grid(
        simulator, gdc_image_width, gdc_image_height)
    
    csv_lines = []
    csv_lines.append("Array_Index_Row_Col,Value_Decimal,Value_Hex")
    
    # Export GDC Grid X values
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"gdc_grid_x_{row}_{col}"
            value = grid_distort_x[row, col]
            hex_value = f"0x{value:08X}" if value >= 0 else f"0x{value & 0xFFFFFFFF:08X}"
            csv_lines.append(f"{index},{value},{hex_value}")
    
    # Export GDC Grid Y values
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"gdc_grid_y_{row}_{col}"
            value = grid_distort_y[row, col]
            hex_value = f"0x{value:08X}" if value >= 0 else f"0x{value & 0xFFFFFFFF:08X}"
            csv_lines.append(f"{index},{value},{hex_value}")
    
    return "\n".join(csv_lines)

def analyze_distortion_quality(original: np.ndarray, corrected: np.ndarray) -> dict:
    """
    Analyze the quality of distortion correction
    
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
    
    # Calculate SSIM (simplified version)
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