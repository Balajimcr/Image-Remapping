"""
Data export functions for various formats including CSV and GDC

This module handles exporting grid data, distortion parameters, and 
correction maps in different formats for analysis and hardware implementation.
"""

import numpy as np
import math
import csv
import json
from typing import Tuple, Optional, Dict, Any
from config.settings import CSV_DECIMAL_PRECISION, HEX_FORMAT_WIDTH, DEFAULT_GDC_WIDTH, DEFAULT_GDC_HEIGHT

def export_grid_to_csv(simulator) -> str:
    """
    Export grid displacement data to CSV format
    
    Args:
        simulator: Lens distortion simulator instance
        
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
            hex_value = format_hex_value(value)
            csv_lines.append(f"{index},{value:.{CSV_DECIMAL_PRECISION}f},{hex_value}")
    
    # Export Grid Y displacements  
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"grid_y_{row}_{col}"
            value = displacement_y[row, col]
            # Convert to hex (handling negative values)
            hex_value = format_hex_value(value)
            csv_lines.append(f"{index},{value:.{CSV_DECIMAL_PRECISION}f},{hex_value}")
    
    return "\n".join(csv_lines)

def export_dense_grid_to_csv(map_x: np.ndarray, map_y: np.ndarray, 
                           output_file: Optional[str] = None) -> str:
    """
    Export dense remapping grid to CSV format
    
    Args:
        map_x: X-coordinate mapping array
        map_y: Y-coordinate mapping array
        output_file: Optional output file path
        
    Returns:
        CSV formatted string
    """
    height, width = map_x.shape
    csv_lines = []
    csv_lines.append("Row,Col,Map_X,Map_Y,Displacement_X,Displacement_Y")
    
    for row in range(height):
        for col in range(width):
            map_x_val = map_x[row, col]
            map_y_val = map_y[row, col]
            disp_x = map_x_val - col
            disp_y = map_y_val - row
            
            csv_lines.append(
                f"{row},{col},{map_x_val:.{CSV_DECIMAL_PRECISION}f},"
                f"{map_y_val:.{CSV_DECIMAL_PRECISION}f},"
                f"{disp_x:.{CSV_DECIMAL_PRECISION}f},"
                f"{disp_y:.{CSV_DECIMAL_PRECISION}f}"
            )
    
    csv_data = "\n".join(csv_lines)
    
    if output_file:
        with open(output_file, 'w', newline='') as f:
            f.write(csv_data)
    
    return csv_data

def convert_barrel_distortion_to_gdc_grid(simulator, 
                                        gdc_image_width: int = DEFAULT_GDC_WIDTH, 
                                        gdc_image_height: int = DEFAULT_GDC_HEIGHT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert barrel distortion grid coordinates to GDC (Geometric Distortion Correction) format
    
    Args:
        simulator: Lens distortion simulator instance
        gdc_image_width: Target GDC image width
        gdc_image_height: Target GDC image height
    
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
            scale_factor_x = int(round((gdc_image_width * (2 ** 14)) / simulator.image_width))
            
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
            scale_factor_y = int(round((gdc_image_height * (2 ** 14)) / simulator.image_height))
            
            # Step 2: Apply scaling and bit shifting
            scaled_delta_y = delta_y * scale_factor_y
            shifted_delta_y = scaled_delta_y * 4  # << 2 equivalent
            
            # Step 3: Divide by 2^14 and ceiling
            intermediate_y = math.ceil(shifted_delta_y / (2 ** 14))
            
            # Step 4: Final bit shift (<< 7) - ensure integer
            grid_distort_y[row, col] = int(intermediate_y) << 7
    
    return grid_distort_x, grid_distort_y

def export_gdc_grid_to_csv(simulator, 
                          gdc_image_width: int = DEFAULT_GDC_WIDTH, 
                          gdc_image_height: int = DEFAULT_GDC_HEIGHT) -> str:
    """
    Export GDC grid parameters to CSV format
    
    Args:
        simulator: Lens distortion simulator instance
        gdc_image_width: Target GDC image width
        gdc_image_height: Target GDC image height
    
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
            hex_value = format_hex_value(value, signed=True)
            csv_lines.append(f"{index},{value},{hex_value}")
    
    # Export GDC Grid Y values
    for row in range(simulator.grid_rows):
        for col in range(simulator.grid_cols):
            index = f"gdc_grid_y_{row}_{col}"
            value = grid_distort_y[row, col]
            hex_value = format_hex_value(value, signed=True)
            csv_lines.append(f"{index},{value},{hex_value}")
    
    return "\n".join(csv_lines)

def export_distortion_parameters(simulator, format_type: str = "json") -> str:
    """
    Export distortion parameters in various formats
    
    Args:
        simulator: Lens distortion simulator instance
        format_type: Output format ("json", "xml", "yaml", "txt")
        
    Returns:
        Formatted parameter string
    """
    params = simulator.get_distortion_info()
    
    if format_type.lower() == "json":
        return json.dumps(params, indent=2)
    
    elif format_type.lower() == "xml":
        xml_lines = ["<?xml version='1.0' encoding='UTF-8'?>", "<distortion_parameters>"]
        
        for key, value in params.items():
            if isinstance(value, dict):
                xml_lines.append(f"  <{key}>")
                for sub_key, sub_value in value.items():
                    xml_lines.append(f"    <{sub_key}>{sub_value}</{sub_key}>")
                xml_lines.append(f"  </{key}>")
            elif isinstance(value, (list, tuple)):
                xml_lines.append(f"  <{key}>")
                for i, item in enumerate(value):
                    xml_lines.append(f"    <item_{i}>{item}</item_{i}>")
                xml_lines.append(f"  </{key}>")
            else:
                xml_lines.append(f"  <{key}>{value}</{key}>")
        
        xml_lines.append("</distortion_parameters>")
        return "\n".join(xml_lines)
    
    elif format_type.lower() == "yaml":
        yaml_lines = []
        
        def format_yaml_value(key, value, indent=0):
            prefix = "  " * indent
            if isinstance(value, dict):
                lines = [f"{prefix}{key}:"]
                for sub_key, sub_value in value.items():
                    lines.extend(format_yaml_value(sub_key, sub_value, indent + 1))
                return lines
            elif isinstance(value, (list, tuple)):
                lines = [f"{prefix}{key}:"]
                for item in value:
                    lines.append(f"{prefix}  - {item}")
                return lines
            else:
                return [f"{prefix}{key}: {value}"]
        
        for key, value in params.items():
            yaml_lines.extend(format_yaml_value(key, value))
        
        return "\n".join(yaml_lines)
    
    elif format_type.lower() == "txt":
        txt_lines = ["Distortion Parameters", "=" * 20, ""]
        
        for key, value in params.items():
            if isinstance(value, dict):
                txt_lines.append(f"{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    txt_lines.append(f"  {sub_key}: {sub_value}")
                txt_lines.append("")
            elif isinstance(value, (list, tuple)):
                txt_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            else:
                txt_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(txt_lines)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def export_correction_maps(map_x: np.ndarray, map_y: np.ndarray, 
                          output_prefix: str, format_type: str = "binary") -> list:
    """
    Export correction maps in various formats
    
    Args:
        map_x: X-coordinate mapping array
        map_y: Y-coordinate mapping array
        output_prefix: Prefix for output filenames
        format_type: Output format ("binary", "text", "csv")
        
    Returns:
        List of created filenames
    """
    created_files = []
    
    if format_type == "binary":
        # Save as numpy binary files
        x_file = f"{output_prefix}_map_x.npy"
        y_file = f"{output_prefix}_map_y.npy"
        
        np.save(x_file, map_x)
        np.save(y_file, map_y)
        
        created_files.extend([x_file, y_file])
    
    elif format_type == "text":
        # Save as text files
        x_file = f"{output_prefix}_map_x.txt"
        y_file = f"{output_prefix}_map_y.txt"
        
        np.savetxt(x_file, map_x, fmt=f'%.{CSV_DECIMAL_PRECISION}f')
        np.savetxt(y_file, map_y, fmt=f'%.{CSV_DECIMAL_PRECISION}f')
        
        created_files.extend([x_file, y_file])
    
    elif format_type == "csv":
        # Save as CSV with coordinates
        csv_file = f"{output_prefix}_maps.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Row', 'Col', 'Map_X', 'Map_Y'])
            
            height, width = map_x.shape
            for row in range(height):
                for col in range(width):
                    writer.writerow([row, col, 
                                   f"{map_x[row, col]:.{CSV_DECIMAL_PRECISION}f}",
                                   f"{map_y[row, col]:.{CSV_DECIMAL_PRECISION}f}"])
        
        created_files.append(csv_file)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    return created_files

def format_hex_value(value: float, signed: bool = True) -> str:
    """
    Format a numeric value as hexadecimal string
    
    Args:
        value: Numeric value to format
        signed: Whether to handle signed values
        
    Returns:
        Formatted hexadecimal string
    """
    int_value = int(round(value))
    
    if signed and int_value < 0:
        # Two's complement representation for negative values
        hex_value = f"0x{int_value & 0xFFFFFFFF:0{HEX_FORMAT_WIDTH}X}"
    else:
        hex_value = f"0x{int_value:0{HEX_FORMAT_WIDTH}X}"
    
    return hex_value

def create_export_metadata(simulator, export_type: str, **kwargs) -> Dict[str, Any]:
    """
    Create metadata for exported data
    
    Args:
        simulator: Lens distortion simulator instance
        export_type: Type of export performed
        **kwargs: Additional metadata
        
    Returns:
        Metadata dictionary
    """
    import datetime
    
    metadata = {
        'export_timestamp': datetime.datetime.now().isoformat(),
        'export_type': export_type,
        'software_version': '1.0.0',
        'distortion_parameters': simulator.get_distortion_info(),
        'image_dimensions': (simulator.image_width, simulator.image_height),
        'grid_dimensions': (simulator.grid_rows, simulator.grid_cols),
        'export_parameters': kwargs
    }
    
    return metadata

def export_calibration_data(simulator, corrector, output_dir: str = ".") -> Dict[str, str]:
    """
    Export comprehensive calibration data package
    
    Args:
        simulator: Lens distortion simulator instance
        corrector: Lens distortion corrector instance
        output_dir: Output directory path
        
    Returns:
        Dictionary mapping data types to filenames
    """
    import os
    from datetime import datetime
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_dir = os.path.join(output_dir, f"calibration_data_{timestamp}")
    os.makedirs(calib_dir, exist_ok=True)
    
    exported_files = {}
    
    # Export distortion parameters
    params_file = os.path.join(calib_dir, "distortion_parameters.json")
    with open(params_file, 'w') as f:
        f.write(export_distortion_parameters(simulator, "json"))
    exported_files['parameters'] = params_file
    
    # Export grid displacement data
    grid_csv_file = os.path.join(calib_dir, "grid_displacements.csv")
    with open(grid_csv_file, 'w') as f:
        f.write(export_grid_to_csv(simulator))
    exported_files['grid_csv'] = grid_csv_file
    
    # Export GDC format data
    gdc_csv_file = os.path.join(calib_dir, "gdc_grid_data.csv")
    with open(gdc_csv_file, 'w') as f:
        f.write(export_gdc_grid_to_csv(simulator))
    exported_files['gdc_csv'] = gdc_csv_file
    
    # Export correction maps
    if corrector.map_computed:
        map_files = export_correction_maps(
            corrector.map_x, corrector.map_y,
            os.path.join(calib_dir, "correction_maps"),
            "binary"
        )
        exported_files['correction_maps'] = map_files
    
    # Export metadata
    metadata_file = os.path.join(calib_dir, "metadata.json")
    metadata = create_export_metadata(simulator, "full_calibration_package")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    exported_files['metadata'] = metadata_file
    
    # Create summary report
    summary_file = os.path.join(calib_dir, "calibration_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(generate_calibration_summary(simulator, exported_files))
    exported_files['summary'] = summary_file
    
    return exported_files

def generate_calibration_summary(simulator, exported_files: Dict[str, str]) -> str:
    """
    Generate a human-readable calibration summary
    
    Args:
        simulator: Lens distortion simulator instance
        exported_files: Dictionary of exported files
        
    Returns:
        Summary text
    """
    from datetime import datetime
    
    info = simulator.get_distortion_info()
    
    summary_lines = [
        "Lens Distortion Calibration Summary",
        "=" * 40,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Image Configuration:",
        f"  Dimensions: {info['image_dimensions'][0]} x {info['image_dimensions'][1]}",
        f"  Principal Point: ({info['principal_point'][0]:.2f}, {info['principal_point'][1]:.2f})",
        "",
        "Grid Configuration:",
        f"  Grid Size: {info['grid_dimensions'][0]} x {info['grid_dimensions'][1]}",
        "",
        "Distortion Parameters:",
        f"  Type: {info['distortion_type'].replace('_', ' ').title()}",
        f"  Severity: {info['severity'].title()}",
        "",
        "Radial Distortion Coefficients:",
        f"  K1: {info['radial_coefficients']['k1']:.6f}",
        f"  K2: {info['radial_coefficients']['k2']:.6f}",
        f"  K3: {info['radial_coefficients']['k3']:.6f}",
        "",
        "Tangential Distortion Coefficients:",
        f"  P1: {info['tangential_coefficients']['p1']:.6f}",
        f"  P2: {info['tangential_coefficients']['p2']:.6f}",
        "",
        "Exported Files:",
        "=" * 20
    ]
    
    for data_type, filename in exported_files.items():
        if isinstance(filename, list):
            summary_lines.append(f"  {data_type.replace('_', ' ').title()}:")
            for f in filename:
                summary_lines.append(f"    - {os.path.basename(f)}")
        else:
            summary_lines.append(f"  {data_type.replace('_', ' ').title()}: {os.path.basename(filename)}")
    
    summary_lines.extend([
        "",
        "Usage Notes:",
        "- Parameters file contains complete distortion model",
        "- Grid CSV contains sparse displacement vectors",
        "- GDC CSV contains hardware-ready fixed-point values",
        "- Correction maps are dense pixel-level mappings",
        "- Metadata includes export settings and software version"
    ])
    
    return "\n".join(summary_lines)