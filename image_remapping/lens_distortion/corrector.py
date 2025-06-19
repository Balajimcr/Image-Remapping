"""
Advanced lens distortion correction algorithms

This module provides multiple algorithms for correcting lens distortions
including iterative Newton-Raphson and analytical methods.
"""

import numpy as np
import cv2
import warnings
from typing import Tuple, Optional
from scipy.optimize import fsolve

from core.remapping_engine import RemappingEngine, InterpolationMethod
from config.settings import MAX_ITERATIONS, CONVERGENCE_TOLERANCE, NUMERICAL_DERIVATIVE_EPSILON

class LensDistortionCorrector:
    """
    Advanced lens distortion corrector with multiple correction algorithms
    """
    
    def __init__(self, distortion_simulator):
        """
        Initialize corrector with the distortion simulator parameters
        
        Args:
            distortion_simulator: Instance of LensDistortionSimulator
        """
        self.simulator = distortion_simulator
        self.remapping_engine = RemappingEngine()
        self.map_x = None
        self.map_y = None
        self.map_computed = False
        
    def iterative_inverse_mapping(self, x_dist: float, y_dist: float, 
                                 max_iterations: int = MAX_ITERATIONS, 
                                 tolerance: float = CONVERGENCE_TOLERANCE) -> Tuple[float, float]:
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
        
        for iteration in range(max_iterations):
            # Apply forward distortion to current guess
            x_forward, y_forward = self.simulator.apply_barrel_distortion(x_undist, y_undist)
            
            # Calculate error
            error_x = x_forward - x_dist
            error_y = y_forward - y_dist
            
            # Check convergence
            if abs(error_x) < tolerance and abs(error_y) < tolerance:
                break
                
            # Compute numerical jacobian for Newton-Raphson step
            epsilon = NUMERICAL_DERIVATIVE_EPSILON
            
            # Partial derivatives
            x_forward_dx, _ = self.simulator.apply_barrel_distortion(x_undist + epsilon, y_undist)
            x_forward_dy, _ = self.simulator.apply_barrel_distortion(x_undist, y_undist + epsilon)
            _, y_forward_dx = self.simulator.apply_barrel_distortion(x_undist + epsilon, y_undist)
            _, y_forward_dy = self.simulator.apply_barrel_distortion(x_undist, y_undist + epsilon)
            
            # Jacobian matrix elements
            J11 = (x_forward_dx - x_forward) / epsilon
            J12 = (x_forward_dy - x_forward) / epsilon
            J21 = (y_forward_dx - y_forward) / epsilon
            J22 = (y_forward_dy - y_forward) / epsilon
            
            # Determinant
            det = J11 * J22 - J12 * J21
            
            if abs(det) < 1e-12:
                # Singular jacobian, use simple correction with damping
                damping_factor = 0.1
                x_undist -= damping_factor * error_x
                y_undist -= damping_factor * error_y
            else:
                # Newton-Raphson update using Cramer's rule
                dx = (J22 * error_x - J12 * error_y) / det
                dy = (J11 * error_y - J21 * error_x) / det
                
                # Apply update with optional damping for stability
                damping_factor = 1.0
                if iteration > 5:  # Add damping for stability in later iterations
                    damping_factor = 0.5
                
                x_undist -= damping_factor * dx
                y_undist -= damping_factor * dy
        
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
        if (abs(self.simulator.k2) < 1e-10 and abs(self.simulator.k3) < 1e-10 and 
            abs(self.simulator.p1) < 1e-10 and abs(self.simulator.p2) < 1e-10):
            
            r_dist_sq = x_norm_dist**2 + y_norm_dist**2
            
            if abs(self.simulator.k1) < 1e-10:
                # No distortion
                return x_dist, y_dist
            
            # Solve: r_dist^2 = r_undist^2 * (1 + k1 * r_undist^2)^2
            # This is a cubic equation in r_undist^2
            def equation(r_undist_sq):
                return r_undist_sq * (1 + self.simulator.k1 * r_undist_sq)**2 - r_dist_sq
            
            try:
                # Initial guess using first-order approximation
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
    
    def polynomial_inverse_mapping(self, x_dist: float, y_dist: float, degree: int = 3) -> Tuple[float, float]:
        """
        Polynomial approximation for inverse mapping
        Faster than iterative but less accurate for high distortions
        
        Args:
            x_dist, y_dist: Distorted coordinates
            degree: Polynomial degree for approximation
            
        Returns:
            Undistorted coordinates
        """
        # Normalize coordinates
        x_norm = (x_dist - self.simulator.cx) / self.simulator.image_width
        y_norm = (y_dist - self.simulator.cy) / self.simulator.image_height
        
        r2 = x_norm**2 + y_norm**2
        
        # Polynomial approximation of inverse radial distortion
        # This is a simplified model - could be improved with proper fitting
        k1, k2, k3 = self.simulator.k1, self.simulator.k2, self.simulator.k3
        
        if degree >= 1:
            correction = -k1 * r2
        if degree >= 2:
            correction += -(k2 - 3*k1**2) * r2**2
        if degree >= 3:
            correction += -(k3 - 8*k1*k2 + 12*k1**3) * r2**3
        
        scale_factor = 1 + correction
        
        x_norm_undist = x_norm * scale_factor
        y_norm_undist = y_norm * scale_factor
        
        # Convert back to pixel coordinates
        x_undist = x_norm_undist * self.simulator.image_width + self.simulator.cx
        y_undist = y_norm_undist * self.simulator.image_height + self.simulator.cy
        
        return x_undist, y_undist
    
    def compute_inverse_maps(self, method: str = "iterative") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute inverse mapping lookup tables for efficient image correction
        
        Args:
            method: "iterative", "analytical", or "polynomial"
            
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
        elif method == "polynomial":
            inverse_func = lambda x, y: self.polynomial_inverse_mapping(x, y, degree=3)
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
    
    def compute_inverse_maps_vectorized(self, method: str = "iterative") -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized version of inverse mapping computation for better performance
        
        Args:
            method: "iterative", "analytical", or "polynomial"
            
        Returns:
            map_x, map_y: Mapping arrays for cv2.remap
        """
        height = self.simulator.image_height
        width = self.simulator.image_width
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        
        # Use the simulator's transformation directly for forward mapping
        map_x, map_y = self.simulator._distortion_transform.transform_points(x_coords, y_coords)
        
        self.map_x = map_x.astype(np.float32)
        self.map_y = map_y.astype(np.float32)
        self.map_computed = True
        
        return self.map_x, self.map_y
    
    def correct_image(self, distorted_image: np.ndarray, 
                     method: str = "iterative",
                     interpolation: str = "linear",
                     recompute_maps: bool = False,
                     use_vectorized: bool = True) -> np.ndarray:
        """
        Correct lens distortion in an image
        
        Args:
            distorted_image: Input distorted image
            method: "iterative", "analytical", or "polynomial"
            interpolation: Interpolation method for remapping
            recompute_maps: Force recomputation of mapping tables
            use_vectorized: Use vectorized computation for better performance
            
        Returns:
            Corrected image
        """
        # Compute maps if not already computed or if recompute is requested
        if not self.map_computed or recompute_maps:
            if use_vectorized:
                self.compute_inverse_maps_vectorized(method)
            else:
                self.compute_inverse_maps(method)
        
        # Apply inverse mapping using the remapping engine
        corrected_image = self.remapping_engine.apply_remapping(
            distorted_image, self.map_x, self.map_y, interpolation=interpolation
        )
        
        return corrected_image
    
    def validate_correction(self, original_image: np.ndarray, 
                           corrected_image: np.ndarray,
                           grid_points: Optional[np.ndarray] = None) -> dict:
        """
        Validate the correction quality using multiple metrics
        
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
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = np.mean((original_image.astype(float) - corrected_image.astype(float))**2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                metrics['psnr'] = psnr
            else:
                metrics['psnr'] = float('inf')
            
            # Structural Similarity (simplified version)
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
            corr_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) if len(corrected_image.shape) == 3 else corrected_image
            
            # Normalized cross-correlation
            correlation = np.corrcoef(orig_gray.flatten(), corr_gray.flatten())[0, 1]
            metrics['correlation'] = correlation
            
            # Mean Squared Error
            metrics['mse'] = mse
            
            # Mean Absolute Error
            mae = np.mean(np.abs(original_image.astype(float) - corrected_image.astype(float)))
            metrics['mae'] = mae
        
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
            metrics['median_geometric_error'] = np.median(geometric_errors)
        
        return metrics
    
    def compare_correction_methods(self, distorted_image: np.ndarray,
                                 original_image: Optional[np.ndarray] = None) -> dict:
        """
        Compare different correction methods and their performance
        
        Args:
            distorted_image: Input distorted image
            original_image: Optional original image for quality comparison
            
        Returns:
            Dictionary with comparison results
        """
        methods = ["iterative", "analytical", "polynomial"]
        results = {}
        
        for method in methods:
            try:
                # Measure correction time
                import time
                start_time = time.time()
                
                corrected = self.correct_image(
                    distorted_image, method=method, recompute_maps=True
                )
                
                correction_time = time.time() - start_time
                
                result = {
                    'corrected_image': corrected,
                    'correction_time': correction_time,
                    'method_available': True
                }
                
                # Quality metrics if original image is provided
                if original_image is not None:
                    metrics = self.validate_correction(original_image, corrected)
                    result.update(metrics)
                
                results[method] = result
                
            except Exception as e:
                results[method] = {
                    'method_available': False,
                    'error': str(e)
                }
        
        return results
    
    def get_optimal_method(self, distortion_info: dict) -> str:
        """
        Recommend optimal correction method based on distortion characteristics
        
        Args:
            distortion_info: Distortion information from simulator
            
        Returns:
            Recommended method name
        """
        distortion_type = distortion_info.get('distortion_type', 'complex')
        severity = distortion_info.get('severity', 'moderate')
        radial_coeffs = distortion_info.get('radial_coefficients', {})
        tangential_coeffs = distortion_info.get('tangential_coefficients', {})
        
        # Check if only K1 is significant (analytical method works best)
        k1_only = (abs(radial_coeffs.get('k2', 0)) < 1e-6 and 
                  abs(radial_coeffs.get('k3', 0)) < 1e-6 and
                  abs(tangential_coeffs.get('p1', 0)) < 1e-6 and
                  abs(tangential_coeffs.get('p2', 0)) < 1e-6)
        
        if k1_only and severity in ['minimal', 'moderate']:
            return "analytical"
        elif severity == 'minimal':
            return "polynomial"
        else:
            return "iterative"