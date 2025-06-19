"""
Lens distortion simulator with support for various distortion models

This module handles the simulation of different types of lens distortions
including barrel, pincushion, and fisheye effects using the Brown-Conrady model.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

from core.transform_models import RadialDistortionTransform
from core.remapping_engine import RemappingEngine
from config.settings import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, 
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS,
    DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2,
    PATTERN_TYPES, DEFAULT_PATTERN_TYPE
)

class LensDistortionSimulator:
    """
    Comprehensive lens distortion simulator using Brown-Conrady model
    """
    
    def __init__(self):
        self.image_width = DEFAULT_IMAGE_WIDTH
        self.image_height = DEFAULT_IMAGE_HEIGHT
        self.grid_rows = DEFAULT_GRID_ROWS
        self.grid_cols = DEFAULT_GRID_COLS
        self.k1 = DEFAULT_K1
        self.k2 = DEFAULT_K2
        self.k3 = DEFAULT_K3
        self.p1 = DEFAULT_P1
        self.p2 = DEFAULT_P2
        self.cx = None  # Principal point x (auto-calculated if None)
        self.cy = None  # Principal point y (auto-calculated if None)
        
        self.remapping_engine = RemappingEngine()
        self._distortion_transform = None
        self._update_transform()
        
    def set_parameters(self, image_width: int, image_height: int, 
                      grid_rows: int, grid_cols: int,
                      k1: float = 0.0, k2: float = 0.0, k3: float = 0.0,
                      p1: float = 0.0, p2: float = 0.0,
                      cx: Optional[float] = None, cy: Optional[float] = None):
        """Set all distortion parameters and update internal state"""
        self.image_width = image_width
        self.image_height = image_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.cx = cx if cx is not None else image_width / 2.0
        self.cy = cy if cy is not None else image_height / 2.0
        
        self._update_transform()
    
    def _update_transform(self):
        """Update the internal distortion transform object"""
        self._distortion_transform = RadialDistortionTransform(
            k1=self.k1, k2=self.k2, k3=self.k3,
            p1=self.p1, p2=self.p2,
            cx=self.cx, cy=self.cy,
            image_width=self.image_width,
            image_height=self.image_height
        )
    
    def apply_barrel_distortion(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply barrel distortion to a single point
        
        Args:
            x, y: Undistorted coordinates
            
        Returns:
            Distorted coordinates (x_dist, y_dist)
        """
        x_arr = np.array([x])
        y_arr = np.array([y])
        x_dist, y_dist = self._distortion_transform.transform_points(x_arr, y_arr)
        return float(x_dist[0]), float(y_dist[0])
    
    def apply_inverse_barrel_distortion(self, x_dist: float, y_dist: float, 
                                       max_iterations: int = 10, tolerance: float = 1e-6) -> Tuple[float, float]:
        """
        Apply inverse barrel distortion using iterative method
        
        Args:
            x_dist, y_dist: Distorted coordinates
            max_iterations: Maximum number of iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Undistorted coordinates (x_undist, y_undist)
        """
        x_arr = np.array([x_dist])
        y_arr = np.array([y_dist])
        x_undist, y_undist = self._distortion_transform.inverse_transform_points(x_arr, y_arr)
        return float(x_undist[0]), float(y_undist[0])
    
    def generate_sparse_distortion_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sparse distortion grid with current parameters
        
        Returns:
            source_coords, target_coords: Arrays of shape (grid_rows, grid_cols, 2)
        """
        # Create target grid (undistorted)
        x_positions = np.linspace(0, self.image_width, self.grid_cols)
        y_positions = np.linspace(0, self.image_height, self.grid_rows)
        
        target_coords = np.zeros((self.grid_rows, self.grid_cols, 2))
        source_coords = np.zeros((self.grid_rows, self.grid_cols, 2))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Target coordinates (undistorted grid)
                target_x = x_positions[col]
                target_y = y_positions[row]
                target_coords[row, col] = [target_x, target_y]
                
                # Apply distortion to get source coordinates
                source_x, source_y = self.apply_barrel_distortion(target_x, target_y)
                source_coords[row, col] = [source_x, source_y]
        
        return source_coords, target_coords
    
    def generate_correction_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correction maps for cv2.remap to correct distortion
        
        Returns:
            map_x, map_y: Remapping arrays for cv2.remap
        """
        h, w = self.image_height, self.image_width
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        
        # Apply distortion transformation to get mapping
        map_x, map_y = self._distortion_transform.transform_points(x_coords, y_coords)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def generate_distortion_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate maps for cv2.remap to apply distortion to an undistorted image
        
        Returns:
            map_x, map_y: Remapping arrays for cv2.remap
        """
        h, w = self.image_height, self.image_width
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        
        # Apply inverse distortion transformation to get mapping
        map_x, map_y = self._distortion_transform.inverse_transform_points(x_coords, y_coords)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def create_sample_image(self, pattern_type: str = DEFAULT_PATTERN_TYPE) -> np.ndarray:
        """
        Create a sample test image for distortion demonstration
        
        Args:
            pattern_type: "checkerboard", "grid", "circles", or "text"
            
        Returns:
            RGB image array
        """
        if pattern_type not in PATTERN_TYPES:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
        
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        if pattern_type == "checkerboard":
            # Create checkerboard pattern
            square_size = 25
            for y in range(0, self.image_height, square_size):
                for x in range(0, self.image_width, square_size):
                    if ((x // square_size) + (y // square_size)) % 2 == 0:
                        img[y:y+square_size, x:x+square_size] = [255, 255, 255]
                        
        elif pattern_type == "grid":
            # Create grid lines
            img.fill(255)  # White background
            line_spacing = 10
            line_thickness = 2
            
            # Vertical lines
            for x in range(0, self.image_width, line_spacing):
                cv2.line(img, (x, 0), (x, self.image_height), (0, 0, 0), line_thickness)
            
            # Horizontal lines
            for y in range(0, self.image_height, line_spacing):
                cv2.line(img, (0, y), (self.image_width, y), (0, 0, 0), line_thickness)
                
        elif pattern_type == "circles":
            # Create concentric circles
            img.fill(255)  # White background
            center_x, center_y = self.image_width // 2, self.image_height // 2
            max_radius = min(center_x, center_y)
            
            for radius in range(10, max_radius, 75):
                cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), 3)
                
        elif pattern_type == "text":
            # Create text pattern
            img.fill(255)  # White background
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Add text at various positions
            positions = [
                (50, 50), (self.image_width//2, 50), (self.image_width-200, 50),
                (50, self.image_height//2), (self.image_width//2, self.image_height//2), (self.image_width-200, self.image_height//2),
                (50, self.image_height-50), (self.image_width//2, self.image_height-50), (self.image_width-200, self.image_height-50)
            ]
            
            for i, (x, y) in enumerate(positions):
                cv2.putText(img, f"TEXT{i+1}", (x, y), font, 1, (0, 0, 0), 2)
        
        return img
    
    def apply_distortion_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply barrel distortion to an input image
        
        Args:
            image: Input RGB image
            
        Returns:
            Distorted image
        """
        map_x, map_y = self.generate_distortion_maps()
        return self.remapping_engine.apply_remapping(image, map_x, map_y)
    
    def correct_distortion(self, distorted_image: np.ndarray) -> np.ndarray:
        """
        Correct barrel distortion in an image using the generated maps
        
        Args:
            distorted_image: Distorted input image
            
        Returns:
            Corrected image
        """
        map_x, map_y = self.generate_correction_maps()
        return self.remapping_engine.apply_remapping(distorted_image, map_x, map_y)
    
    def get_distortion_info(self) -> dict:
        """
        Get comprehensive information about current distortion parameters
        
        Returns:
            Dictionary with distortion information
        """
        return {
            'image_dimensions': (self.image_width, self.image_height),
            'grid_dimensions': (self.grid_rows, self.grid_cols),
            'principal_point': (self.cx, self.cy),
            'radial_coefficients': {'k1': self.k1, 'k2': self.k2, 'k3': self.k3},
            'tangential_coefficients': {'p1': self.p1, 'p2': self.p2},
            'distortion_type': self._classify_distortion_type(),
            'severity': self._estimate_distortion_severity()
        }
    
    def _classify_distortion_type(self) -> str:
        """Classify the type of distortion based on coefficients"""
        if abs(self.k1) < 1e-6 and abs(self.k2) < 1e-6 and abs(self.k3) < 1e-6:
            if abs(self.p1) < 1e-6 and abs(self.p2) < 1e-6:
                return "none"
            else:
                return "tangential_only"
        elif self.k1 < 0:
            return "barrel"
        elif self.k1 > 0:
            return "pincushion"
        else:
            return "complex"
    
    def _estimate_distortion_severity(self) -> str:
        """Estimate the severity of distortion"""
        radial_magnitude = abs(self.k1) + abs(self.k2) + abs(self.k3)
        tangential_magnitude = abs(self.p1) + abs(self.p2)
        total_magnitude = radial_magnitude + tangential_magnitude
        
        if total_magnitude < 0.05:
            return "minimal"
        elif total_magnitude < 0.2:
            return "moderate"
        elif total_magnitude < 0.5:
            return "significant"
        else:
            return "severe"
    
    def create_preset_distortion(self, preset_name: str):
        """
        Apply predefined distortion presets
        
        Args:
            preset_name: Name of the preset ('barrel', 'pincushion', 'fisheye', etc.)
        """
        presets = {
            'barrel': {'k1': -0.2, 'k2': 0.05, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0},
            'pincushion': {'k1': 0.15, 'k2': -0.03, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0},
            'fisheye': {'k1': -0.4, 'k2': 0.1, 'k3': -0.02, 'p1': 0.0, 'p2': 0.0},
            'mild_barrel': {'k1': -0.1, 'k2': 0.01, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0},
            'complex': {'k1': -0.3, 'k2': 0.08, 'k3': -0.01, 'p1': 0.02, 'p2': 0.01},
            'identity': {'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0}
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset = presets[preset_name]
        self.set_parameters(
            self.image_width, self.image_height,
            self.grid_rows, self.grid_cols,
            **preset
        )