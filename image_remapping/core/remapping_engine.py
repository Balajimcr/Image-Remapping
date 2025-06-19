"""
Generic image remapping engine for various geometric transformations

This module provides the core functionality for applying geometric transformations
to images using OpenCV's remap function with different interpolation methods.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from enum import Enum

from config.settings import INTERPOLATION_METHODS, DEFAULT_INTERPOLATION

class InterpolationMethod(Enum):
    """Enumeration of supported interpolation methods"""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4

class BorderMode(Enum):
    """Enumeration of supported border modes"""
    CONSTANT = cv2.BORDER_CONSTANT
    REFLECT = cv2.BORDER_REFLECT
    WRAP = cv2.BORDER_WRAP
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT_101 = cv2.BORDER_REFLECT_101

class RemappingEngine:
    """
    Core engine for applying image remapping transformations
    """
    
    def __init__(self):
        self.cache = {}  # Cache for computed mapping arrays
        self.cache_enabled = True
        
    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching of mapping arrays"""
        self.cache_enabled = enabled
        if not enabled:
            self.cache.clear()
    
    def clear_cache(self):
        """Clear the mapping cache"""
        self.cache.clear()
    
    def apply_remapping(self, 
                       image: np.ndarray,
                       map_x: np.ndarray,
                       map_y: np.ndarray,
                       interpolation: Union[str, InterpolationMethod] = DEFAULT_INTERPOLATION,
                       border_mode: BorderMode = BorderMode.CONSTANT,
                       border_value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Apply remapping to an image using provided mapping arrays
        
        Args:
            image: Input image to remap
            map_x: X-coordinate mapping array
            map_y: Y-coordinate mapping array
            interpolation: Interpolation method
            border_mode: Border handling mode
            border_value: Border color for CONSTANT mode
            
        Returns:
            Remapped image
        """
        # Convert interpolation method if string
        if isinstance(interpolation, str):
            if interpolation in INTERPOLATION_METHODS:
                interp_method = INTERPOLATION_METHODS[interpolation]
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation}")
        elif isinstance(interpolation, InterpolationMethod):
            interp_method = interpolation.value
        else:
            interp_method = interpolation
        
        # Ensure mapping arrays are float32
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Apply remapping
        remapped_image = cv2.remap(
            image, map_x, map_y,
            interpolation=interp_method,
            borderMode=border_mode.value,
            borderValue=border_value
        )
        
        return remapped_image
    
    def generate_identity_maps(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate identity mapping arrays (no transformation)
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Identity mapping arrays (map_x, map_y)
        """
        cache_key = f"identity_{height}_{width}"
        
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        
        if self.cache_enabled:
            self.cache[cache_key] = (x_coords, y_coords)
        
        return x_coords, y_coords
    
    def generate_mapping_from_function(self,
                                     height: int,
                                     width: int,
                                     transform_func) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mapping arrays from a transformation function
        
        Args:
            height: Image height
            width: Image width
            transform_func: Function that takes (x, y) and returns (x_new, y_new)
            
        Returns:
            Mapping arrays (map_x, map_y)
        """
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                x_new, y_new = transform_func(x, y)
                map_x[y, x] = x_new
                map_y[y, x] = y_new
        
        return map_x, map_y
    
    def generate_mapping_vectorized(self,
                                  height: int,
                                  width: int,
                                  transform_func_vectorized) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mapping arrays using vectorized transformation function
        
        Args:
            height: Image height
            width: Image width
            transform_func_vectorized: Vectorized function that operates on coordinate arrays
            
        Returns:
            Mapping arrays (map_x, map_y)
        """
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        
        # Apply vectorized transformation
        map_x, map_y = transform_func_vectorized(x_coords, y_coords)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def invert_mapping(self, 
                      map_x: np.ndarray, 
                      map_y: np.ndarray,
                      method: str = 'interpolation') -> Tuple[np.ndarray, np.ndarray]:
        """
        Invert a mapping (forward to inverse or vice versa)
        
        Args:
            map_x: Input X mapping array
            map_y: Input Y mapping array
            method: Inversion method ('interpolation' or 'iterative')
            
        Returns:
            Inverted mapping arrays
        """
        height, width = map_x.shape
        
        if method == 'interpolation':
            # Use interpolation to invert the mapping
            from scipy.interpolate import griddata
            
            # Create source and target coordinates
            src_coords = np.column_stack([map_x.ravel(), map_y.ravel()])
            target_x, target_y = np.meshgrid(np.arange(width), np.arange(height))
            target_coords = np.column_stack([target_x.ravel(), target_y.ravel()])
            
            # Interpolate inverse mapping
            inv_x = griddata(src_coords, target_coords[:, 0], 
                           (target_x, target_y), method='linear', fill_value=0)
            inv_y = griddata(src_coords, target_coords[:, 1], 
                           (target_x, target_y), method='linear', fill_value=0)
            
            return inv_x.astype(np.float32), inv_y.astype(np.float32)
        
        else:
            raise NotImplementedError(f"Inversion method '{method}' not implemented")
    
    def combine_mappings(self, 
                        map1_x: np.ndarray, map1_y: np.ndarray,
                        map2_x: np.ndarray, map2_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine two mappings (composition of transformations)
        
        Args:
            map1_x, map1_y: First mapping
            map2_x, map2_y: Second mapping
            
        Returns:
            Combined mapping arrays
        """
        # Use interpolation to sample map2 at map1 coordinates
        from scipy.interpolate import RegularGridInterpolator
        
        height, width = map1_x.shape
        
        # Create interpolators for map2
        y_coords = np.arange(height)
        x_coords = np.arange(width)
        
        interp_x = RegularGridInterpolator((y_coords, x_coords), map2_x,
                                         bounds_error=False, fill_value=0)
        interp_y = RegularGridInterpolator((y_coords, x_coords), map2_y,
                                         bounds_error=False, fill_value=0)
        
        # Sample map2 at map1 coordinates
        combined_x = np.zeros_like(map1_x)
        combined_y = np.zeros_like(map1_y)
        
        for y in range(height):
            for x in range(width):
                intermediate_x = map1_x[y, x]
                intermediate_y = map1_y[y, x]
                
                if (0 <= intermediate_x < width and 0 <= intermediate_y < height):
                    combined_x[y, x] = interp_x((intermediate_y, intermediate_x))
                    combined_y[y, x] = interp_y((intermediate_y, intermediate_x))
        
        return combined_x.astype(np.float32), combined_y.astype(np.float32)
    
    def validate_mapping(self, 
                        map_x: np.ndarray, 
                        map_y: np.ndarray,
                        image_width: int,
                        image_height: int) -> dict:
        """
        Validate mapping arrays for potential issues
        
        Args:
            map_x: X mapping array
            map_y: Y mapping array
            image_width: Target image width
            image_height: Target image height
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check for NaN or infinite values
        if np.any(np.isnan(map_x)) or np.any(np.isnan(map_y)):
            results['errors'].append("NaN values found in mapping arrays")
            results['valid'] = False
        
        if np.any(np.isinf(map_x)) or np.any(np.isinf(map_y)):
            results['errors'].append("Infinite values found in mapping arrays")
            results['valid'] = False
        
        # Check bounds
        out_of_bounds_x = np.sum((map_x < 0) | (map_x >= image_width))
        out_of_bounds_y = np.sum((map_y < 0) | (map_y >= image_height))
        
        total_pixels = map_x.size
        if out_of_bounds_x > 0:
            percentage = (out_of_bounds_x / total_pixels) * 100
            results['warnings'].append(f"{out_of_bounds_x} pixels ({percentage:.1f}%) map outside X bounds")
        
        if out_of_bounds_y > 0:
            percentage = (out_of_bounds_y / total_pixels) * 100
            results['warnings'].append(f"{out_of_bounds_y} pixels ({percentage:.1f}%) map outside Y bounds")
        
        # Statistics
        results['statistics'] = {
            'x_range': (float(np.min(map_x)), float(np.max(map_x))),
            'y_range': (float(np.min(map_y)), float(np.max(map_y))),
            'x_mean': float(np.mean(map_x)),
            'y_mean': float(np.mean(map_y)),
            'x_std': float(np.std(map_x)),
            'y_std': float(np.std(map_y))
        }
        
        return results