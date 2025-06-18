import numpy as np
import cv2
import warnings
from typing import Tuple, Optional
from scipy.optimize import fsolve

class BarrelDistortionCorrector:
    """
    Inverse mapping corrector for barrel distortion using multiple approaches
    """
    
    def __init__(self, distortion_simulator):
        """
        Initialize corrector with the distortion simulator parameters
        
        Args:
            distortion_simulator: Instance of BarrelDistortionSimulator
        """
        self.simulator = distortion_simulator
        self.map_x = None
        self.map_y = None
        self.map_computed = False
        
    def iterative_inverse_mapping(self, x_dist: float, y_dist: float, 
                                 max_iterations: int = 10, tolerance: float = 1e-6) -> Tuple[float, float]:
        """
        Compute inverse mapping using iterative Newton-Raphson method
        
        Args:
            x_dist, y_dist: Distorted coordinates
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Undistorted coordinates (x_undist, y_undist)
        """
        # Initial guess - use distorted coordinates as starting point
        x_undist = x_dist
        y_undist = y_dist
        
        for _ in range(max_iterations):
            # Apply forward distortion to current guess
            x_forward, y_forward = self.simulator.apply_barrel_distortion(x_undist, y_undist)
            
            # Calculate error
            error_x = x_forward - x_dist
            error_y = y_forward - y_dist
            
            # Check convergence
            if abs(error_x) < tolerance and abs(error_y) < tolerance:
                break
                
            # Compute numerical jacobian for Newton-Raphson step
            epsilon = 1e-6
            
            # Partial derivatives
            x_forward_dx, _ = self.simulator.apply_barrel_distortion(x_undist + epsilon, y_undist)
            x_forward_dy, _ = self.simulator.apply_barrel_distortion(x_undist, y_undist + epsilon)
            _, y_forward_dx = self.simulator.apply_barrel_distortion(x_undist + epsilon, y_undist)
            _, y_forward_dy = self.simulator.apply_barrel_distortion(x_undist, y_undist + epsilon)
            
            # Jacobian matrix
            J11 = (x_forward_dx - x_forward) / epsilon
            J12 = (x_forward_dy - x_forward) / epsilon
            J21 = (y_forward_dx - y_forward) / epsilon
            J22 = (y_forward_dy - y_forward) / epsilon
            
            # Determinant
            det = J11 * J22 - J12 * J21
            
            if abs(det) < 1e-12:
                # Singular jacobian, use simple correction
                x_undist -= 0.1 * error_x
                y_undist -= 0.1 * error_y
            else:
                # Newton-Raphson update
                dx = (J22 * error_x - J12 * error_y) / det
                dy = (J11 * error_y - J21 * error_x) / det
                
                x_undist -= dx
                y_undist -= dy
        
        return x_undist, y_undist
    
    def analytical_inverse_mapping(self, x_dist: float, y_dist: float) -> Tuple[float, float]:
        """
        Analytical inverse mapping for simple radial distortion (k1 only)
        More accurate but limited to cases where tangential distortion is negligible
        """
        # Normalize coordinates
        x_norm_dist = (x_dist - self.simulator.cx) / self.simulator.image_width
        y_norm_dist = (y_dist - self.simulator.cy) / self.simulator.image_height
        
        # For pure radial distortion with k1 only
        if abs(self.simulator.k2) < 1e-10 and abs(self.simulator.k3) < 1e-10 and \
           abs(self.simulator.p1) < 1e-10 and abs(self.simulator.p2) < 1e-10:
            
            r_dist_sq = x_norm_dist**2 + y_norm_dist**2
            
            if abs(self.simulator.k1) < 1e-10:
                # No distortion
                return x_dist, y_dist
            
            # Solve: r_dist^2 = r_undist^2 * (1 + k1 * r_undist^2)^2
            # This is a cubic equation in r_undist^2
            def equation(r_undist_sq):
                return r_undist_sq * (1 + self.simulator.k1 * r_undist_sq)**2 - r_dist_sq
            
            try:
                # Initial guess
                r_undist_sq_guess = r_dist_sq / (1 + self.simulator.k1 * r_dist_sq)
                r_undist_sq = fsolve(equation, r_undist_sq_guess)[0]
                
                if r_undist_sq < 0:
                    r_undist_sq = r_dist_sq
                
                # Scale factor
                if r_dist_sq > 1e-10:
                    scale = np.sqrt(r_undist_sq / r_dist_sq)
                    x_norm_undist = x_norm_dist * scale
                    y_norm_undist = y_norm_dist * scale
                else:
                    x_norm_undist = x_norm_dist
                    y_norm_undist = y_norm_dist
                
            except:
                # Fallback to iterative method
                return self.iterative_inverse_mapping(x_dist, y_dist)
        else:
            # Complex distortion - use iterative method
            return self.iterative_inverse_mapping(x_dist, y_dist)
        
        # Convert back to pixel coordinates
        x_undist = x_norm_undist * self.simulator.image_width + self.simulator.cx
        y_undist = y_norm_undist * self.simulator.image_height + self.simulator.cy
        
        return x_undist, y_undist
    
    def compute_inverse_maps(self, method: str = "iterative") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute inverse mapping lookup tables for efficient image correction
        
        Args:
            method: "iterative" or "analytical"
            
        Returns:
            map_x, map_y: Mapping arrays for cv2.remap
        """
        height = self.simulator.image_height
        width = self.simulator.image_width
        
        # Create coordinate grids
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        # Choose inverse mapping method
        if method == "analytical":
            inverse_func = self.analytical_inverse_mapping
        else:
            inverse_func = self.iterative_inverse_mapping
        
        # Compute mapping for each pixel
        for y in range(height):
            for x in range(width):
                # For each pixel in the corrected image, find where to sample from distorted image
                # We want to correct distortion, so we need forward mapping for cv2.remap
                distorted_x, distorted_y = self.simulator.apply_barrel_distortion(x, y)
                map_x[y, x] = distorted_x
                map_y[y, x] = distorted_y
        
        self.map_x = map_x
        self.map_y = map_y
        self.map_computed = True
        
        return map_x, map_y
    
    def correct_image(self, distorted_image: np.ndarray, 
                     method: str = "iterative",
                     interpolation: int = cv2.INTER_LINEAR,
                     recompute_maps: bool = False) -> np.ndarray:
        """
        Correct barrel distortion in an image
        
        Args:
            distorted_image: Input distorted image
            method: "iterative" or "analytical"
            interpolation: OpenCV interpolation method
            recompute_maps: Force recomputation of mapping tables
            
        Returns:
            Corrected image
        """
        # Compute maps if not already computed or if recompute is requested
        if not self.map_computed or recompute_maps:
            self.compute_inverse_maps(method)
        
        # Apply inverse mapping using OpenCV remap
        corrected_image = cv2.remap(distorted_image, self.map_x, self.map_y, 
                                   interpolation, borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
        
        return corrected_image
    
    def validate_correction(self, original_image: np.ndarray, 
                           corrected_image: np.ndarray,
                           grid_points: Optional[np.ndarray] = None) -> dict:
        """
        Validate the correction quality
        
        Args:
            original_image: Original undistorted image
            corrected_image: Corrected image
            grid_points: Optional grid points for geometric validation
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {}
        
        # Image quality metrics
        if original_image.shape == corrected_image.shape:
            # PSNR
            mse = np.mean((original_image.astype(float) - corrected_image.astype(float))**2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                metrics['psnr'] = psnr
            
            # SSIM (simplified version)
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
            corr_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) if len(corrected_image.shape) == 3 else corrected_image
            
            # Basic correlation coefficient
            correlation = np.corrcoef(orig_gray.flatten(), corr_gray.flatten())[0, 1]
            metrics['correlation'] = correlation
        
        # Geometric validation with grid points
        if grid_points is not None:
            geometric_errors = []
            for point in grid_points:
                x_orig, y_orig = point
                # Apply distortion then correction
                x_dist, y_dist = self.simulator.apply_barrel_distortion(x_orig, y_orig)
                x_corr, y_corr = self.iterative_inverse_mapping(x_dist, y_dist)
                
                error = np.sqrt((x_orig - x_corr)**2 + (y_orig - y_corr)**2)
                geometric_errors.append(error)
            
            metrics['mean_geometric_error'] = np.mean(geometric_errors)
            metrics['max_geometric_error'] = np.max(geometric_errors)
            metrics['std_geometric_error'] = np.std(geometric_errors)
        
        return metrics