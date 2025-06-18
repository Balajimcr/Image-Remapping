import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import math
from typing import Tuple, Optional

class BarrelDistortionSimulator:
    """
    Advanced barrel distortion simulator with configurable parameters
    """
    
    def __init__(self):
        self.image_width = 1920
        self.image_height = 1080
        self.grid_rows = 7
        self.grid_cols = 9
        self.k1 = 0.0  # Radial distortion coefficient 1
        self.k2 = 0.0  # Radial distortion coefficient 2
        self.k3 = 0.0  # Radial distortion coefficient 3
        self.p1 = 0.0  # Tangential distortion coefficient 1
        self.p2 = 0.0  # Tangential distortion coefficient 2
        self.cx = None  # Principal point x (auto-calculated if None)
        self.cy = None  # Principal point y (auto-calculated if None)
        
    def set_parameters(self, image_width: int, image_height: int, 
                      grid_rows: int, grid_cols: int,
                      k1: float = 0.0, k2: float = 0.0, k3: float = 0.0,
                      p1: float = 0.0, p2: float = 0.0,
                      cx: Optional[float] = None, cy: Optional[float] = None):
        """Set all distortion parameters"""
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
    
    def apply_barrel_distortion(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply barrel distortion to a point using Brown-Conrady model
        
        Args:
            x, y: Undistorted coordinates
            
        Returns:
            Distorted coordinates (x_dist, y_dist)
        """
        # Normalize coordinates relative to principal point
        x_norm = (x - self.cx) / self.image_width
        y_norm = (y - self.cy) / self.image_height
        
        # Calculate radial distance squared
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion factor
        radial_factor = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        # Tangential distortion
        tangential_x = 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm**2)
        tangential_y = self.p1 * (r2 + 2 * y_norm**2) + 2 * self.p2 * x_norm * y_norm
        
        # Apply distortion
        x_dist_norm = x_norm * radial_factor + tangential_x
        y_dist_norm = y_norm * radial_factor + tangential_y
        
        # Convert back to pixel coordinates
        x_dist = x_dist_norm * self.image_width + self.cx
        y_dist = y_dist_norm * self.image_height + self.cy
        
        return x_dist, y_dist
    
    def apply_inverse_barrel_distortion(self, x_dist: float, y_dist: float, 
                                       max_iterations: int = 10, tolerance: float = 1e-6) -> Tuple[float, float]:
        """
        Apply inverse barrel distortion using iterative Newton-Raphson method
        
        Args:
            x_dist, y_dist: Distorted coordinates
            max_iterations: Maximum number of iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Undistorted coordinates (x_undist, y_undist)
        """
        # Initial guess: use distorted coordinates as starting point
        x_undist = x_dist
        y_undist = y_dist
        
        for _ in range(max_iterations):
            # Apply forward distortion to current estimate
            x_forward, y_forward = self.apply_barrel_distortion(x_undist, y_undist)
            
            # Calculate error
            error_x = x_forward - x_dist
            error_y = y_forward - y_dist
            
            # Check convergence
            if abs(error_x) < tolerance and abs(error_y) < tolerance:
                break
            
            # Calculate numerical derivatives (Jacobian)
            delta = 0.01
            
            # Partial derivatives with respect to x
            x_plus, _ = self.apply_barrel_distortion(x_undist + delta, y_undist)
            x_minus, _ = self.apply_barrel_distortion(x_undist - delta, y_undist)
            dx_dx = (x_plus - x_minus) / (2 * delta)
            
            _, y_plus_x = self.apply_barrel_distortion(x_undist + delta, y_undist)
            _, y_minus_x = self.apply_barrel_distortion(x_undist - delta, y_undist)
            dy_dx = (y_plus_x - y_minus_x) / (2 * delta)
            
            # Partial derivatives with respect to y
            x_plus_y, _ = self.apply_barrel_distortion(x_undist, y_undist + delta)
            x_minus_y, _ = self.apply_barrel_distortion(x_undist, y_undist - delta)
            dx_dy = (x_plus_y - x_minus_y) / (2 * delta)
            
            _, y_plus = self.apply_barrel_distortion(x_undist, y_undist + delta)
            _, y_minus = self.apply_barrel_distortion(x_undist, y_undist - delta)
            dy_dy = (y_plus - y_minus) / (2 * delta)
            
            # Calculate determinant of Jacobian
            det = dx_dx * dy_dy - dx_dy * dy_dx
            
            if abs(det) < 1e-12:  # Avoid division by zero
                break
            
            # Newton-Raphson update
            delta_x = (dy_dy * error_x - dx_dy * error_y) / det
            delta_y = (-dy_dx * error_x + dx_dx * error_y) / det
            
            x_undist -= delta_x
            y_undist -= delta_y
        
        return x_undist, y_undist
    
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
        
        For cv2.remap, the maps tell us where to sample from in the source (distorted) image
        to fill each pixel in the destination (corrected) image.
        
        Returns:
            map_x, map_y: Remapping arrays for cv2.remap
        """
        h, w = self.image_height, self.image_width
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # For each pixel (x, y) in the corrected output image,
                # find where that undistorted point would be in the distorted input image
                # This is the forward distortion applied to the undistorted coordinates
                distorted_x, distorted_y = self.apply_barrel_distortion(x, y)
                map_x[y, x] = distorted_x
                map_y[y, x] = distorted_y
        
        return map_x, map_y
    
    def generate_distortion_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate maps for cv2.remap to apply distortion to an undistorted image
        
        For cv2.remap, the maps tell us where to sample from in the source (undistorted) image
        to fill each pixel in the destination (distorted) image.
        
        Returns:
            map_x, map_y: Remapping arrays for cv2.remap
        """
        h, w = self.image_height, self.image_width
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # For each pixel (x, y) in the distorted output image,
                # find where that distorted point came from in the undistorted input image
                # This requires the inverse distortion
                undistorted_x, undistorted_y = self.apply_inverse_barrel_distortion(x, y)
                map_x[y, x] = undistorted_x
                map_y[y, x] = undistorted_y
        
        return map_x, map_y
    
    def create_sample_image(self, pattern_type: str = "checkerboard") -> np.ndarray:
        """
        Create a sample test image for distortion demonstration
        
        Args:
            pattern_type: "checkerboard", "grid", "circles", or "text"
            
        Returns:
            RGB image array
        """
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        if pattern_type == "checkerboard":
            # Create checkerboard pattern
            square_size = 80
            for y in range(0, self.image_height, square_size):
                for x in range(0, self.image_width, square_size):
                    if ((x // square_size) + (y // square_size)) % 2 == 0:
                        img[y:y+square_size, x:x+square_size] = [255, 255, 255]
                        
        elif pattern_type == "grid":
            # Create grid lines
            img.fill(255)  # White background
            line_spacing = 100
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
            
            for radius in range(50, max_radius, 75):
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
        # Use the pre-computed distortion maps for better performance and consistency
        map_x, map_y = self.generate_distortion_maps()
        
        # Apply remapping
        distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return distorted_image
    
    def correct_distortion(self, distorted_image: np.ndarray) -> np.ndarray:
        """
        Correct barrel distortion in an image using the generated maps
        
        Args:
            distorted_image: Distorted input image
            
        Returns:
            Corrected image
        """
        map_x, map_y = self.generate_correction_maps()
        
        # Apply correction mapping
        corrected_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return corrected_image