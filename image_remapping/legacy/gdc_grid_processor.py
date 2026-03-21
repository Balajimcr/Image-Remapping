#!/usr/bin/env python3
"""
GDC Grid Processing Core Module

Handles parsing, extracting, and interpolating GDC grid data.
This module provides the fundamental grid processing capabilities
for the GDC Image Remapping Suite.

Author: Balaji R
License: MIT
"""

import numpy as np
import re
import warnings
from typing import Tuple, List
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom


class GDCGridProcessor:
    """
    Core processor for GDC grid data handling and interpolation.
    
    This class provides comprehensive functionality for:
    - Parsing GDC format data
    - Grid extraction and validation
    - Bicubic interpolation
    - Format conversion
    """
    
    def __init__(self):
        self.parsed_data: List[Tuple[str, int]] = []
        self.dx_values: List[int] = []
        self.dy_values: List[int] = []
        self.original_rows: int = 0
        self.original_cols: int = 0
        
    def parse_grid_data_from_content(self, file_content: str) -> List[Tuple[str, int]]:
        """
        Parse grid data from file content string.
        
        Args:
            file_content: Raw file content with GDC data
            
        Returns:
            List of (element_name, value) tuples
            
        Raises:
            ValueError: If no valid data found
        """
        parsed_data_ordered = []
        lines = file_content.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    parts = line.split()
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

        if not parsed_data_ordered:
            raise ValueError("No valid GDC data found in content")
            
        self.parsed_data = parsed_data_ordered
        return parsed_data_ordered

    def extract_and_sort_grid_values(self, original_rows: int, original_cols: int) -> Tuple[List[int], List[int]]:
        """
        Extract and sort DX and DY values from parsed data.
        
        Args:
            original_rows: Expected number of rows in grid
            original_cols: Expected number of columns in grid
            
        Returns:
            Tuple of (dx_values, dy_values) lists
            
        Raises:
            ValueError: If insufficient data found
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
            raise ValueError(
                f"Insufficient data for {original_rows}x{original_cols} grid. "
                f"Found {len(self.dx_values)} DX and {len(self.dy_values)} DY elements, "
                f"but expected {expected_elements} of each."
            )
        
        self.dx_values = self.dx_values[:expected_elements]
        self.dy_values = self.dy_values[:expected_elements]

        return self.dx_values, self.dy_values

    def reshape_to_2d_grid(self, values: List[int]) -> np.ndarray:
        """
        Reshape 1D list of values into 2D numpy array.
        
        Args:
            values: List of values to reshape
            
        Returns:
            2D numpy array
            
        Raises:
            ValueError: If data length doesn't match expected grid size
        """
        expected_elements = self.original_rows * self.original_cols
        if len(values) != expected_elements:
            raise ValueError(
                f"Mismatch in data length ({len(values)}) and expected grid size ({expected_elements})."
            )
        return np.array(values).reshape(self.original_rows, self.original_cols)

    def interpolate_grid_bicubic(self, grid_2d: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
        """
        Interpolate 2D grid using bicubic interpolation.
        
        Args:
            grid_2d: Input 2D grid
            target_rows: Target number of rows
            target_cols: Target number of columns
            
        Returns:
            Interpolated 2D grid
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
    
    def interpolate_grid_linear(self, grid_2d: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
        """
        Interpolate 2D grid using linear interpolation.
        
        Args:
            grid_2d: Input 2D grid
            target_rows: Target number of rows
            target_cols: Target number of columns
            
        Returns:
            Interpolated 2D grid
        """
        zoom_factor_y = target_rows / grid_2d.shape[0]
        zoom_factor_x = target_cols / grid_2d.shape[1]
        return zoom(grid_2d, (zoom_factor_y, zoom_factor_x), order=1)

    def grid_2d_to_gdc_format(self, grid_2d: np.ndarray, grid_type: str) -> str:
        """
        Convert 2D grid back to GDC format text.
        
        Args:
            grid_2d: 2D grid array
            grid_type: 'dx' or 'dy'
            
        Returns:
            GDC formatted string
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
    
    def validate_grid_data(self, expected_rows: int, expected_cols: int) -> dict:
        """
        Validate parsed grid data against expected dimensions.
        
        Args:
            expected_rows: Expected number of rows
            expected_cols: Expected number of columns
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not self.parsed_data:
            results['valid'] = False
            results['errors'].append("No parsed data available")
            return results
        
        # Count DX and DY elements
        dx_elements = {}
        dy_elements = {}
        index_pattern = re.compile(r"_(dx|dy)_0_(\d+)$")
        
        for name, value in self.parsed_data:
            match = index_pattern.search(name)
            if match:
                index = int(match.group(2))
                if 'dx' in name:
                    if index in dx_elements:
                        results['warnings'].append(f"Duplicate DX element at index {index}")
                    dx_elements[index] = value
                elif 'dy' in name:
                    if index in dy_elements:
                        results['warnings'].append(f"Duplicate DY element at index {index}")
                    dy_elements[index] = value
        
        dx_count = len(dx_elements)
        dy_count = len(dy_elements)
        expected_count = expected_rows * expected_cols
        
        results['statistics'] = {
            'dx_elements': dx_count,
            'dy_elements': dy_count,
            'expected_elements': expected_count,
            'total_elements': len(self.parsed_data)
        }
        
        # Check for missing elements
        for i in range(expected_count):
            if i not in dx_elements:
                results['errors'].append(f"Missing DX element at index {i}")
                results['valid'] = False
            if i not in dy_elements:
                results['errors'].append(f"Missing DY element at index {i}")
                results['valid'] = False
        
        if dx_count < expected_count:
            results['errors'].append(f"Insufficient DX elements: {dx_count}/{expected_count}")
            results['valid'] = False
            
        if dy_count < expected_count:
            results['errors'].append(f"Insufficient DY elements: {dy_count}/{expected_count}")
            results['valid'] = False
            
        if dx_count != dy_count:
            results['warnings'].append(f"Mismatched DX/DY counts: {dx_count} vs {dy_count}")
        
        return results
    
    def compute_grid_statistics(self, grid_2d: np.ndarray) -> dict:
        """
        Compute comprehensive statistics for a 2D grid.
        
        Args:
            grid_2d: Input 2D grid
            
        Returns:
            Dictionary with grid statistics
        """
        flat_grid = grid_2d.flatten()
        
        return {
            'min': float(np.min(flat_grid)),
            'max': float(np.max(flat_grid)),
            'mean': float(np.mean(flat_grid)),
            'median': float(np.median(flat_grid)),
            'std': float(np.std(flat_grid)),
            'var': float(np.var(flat_grid)),
            'range': float(np.max(flat_grid) - np.min(flat_grid)),
            'percentile_25': float(np.percentile(flat_grid, 25)),
            'percentile_75': float(np.percentile(flat_grid, 75)),
            'shape': grid_2d.shape,
            'total_elements': grid_2d.size
        }