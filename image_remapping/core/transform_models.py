"""
Base classes and implementations for different transformation models

This module provides a unified interface for various geometric transformation
models including affine, projective, radial distortion, and others.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from config.settings import MATH_CONSTANTS

class BaseTransform(ABC):
    """Abstract base class for all geometric transformations"""
    
    def __init__(self, **params):
        self.params = params
        self._matrix = None
        
    @abstractmethod
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinate arrays"""
        pass
    
    @abstractmethod
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse transformation to coordinate arrays"""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get transformation parameters"""
        return self.params.copy()
    
    def set_parameters(self, **params):
        """Update transformation parameters"""
        self.params.update(params)
        self._matrix = None  # Invalidate cached matrix

class AffineTransform(BaseTransform):
    """
    Affine transformation model
    Supports translation, rotation, scaling, and shearing
    """
    
    def __init__(self, 
                 tx: float = 0, ty: float = 0,           # Translation
                 rotation: float = 0,                     # Rotation in degrees
                 scale_x: float = 1, scale_y: float = 1,  # Scaling
                 shear_x: float = 0, shear_y: float = 0,  # Shearing
                 matrix: Optional[np.ndarray] = None):    # Direct matrix
        """
        Initialize affine transformation
        
        Args:
            tx, ty: Translation parameters
            rotation: Rotation angle in degrees
            scale_x, scale_y: Scaling factors
            shear_x, shear_y: Shearing parameters
            matrix: 3x3 transformation matrix (overrides other parameters)
        """
        super().__init__(
            tx=tx, ty=ty, rotation=rotation,
            scale_x=scale_x, scale_y=scale_y,
            shear_x=shear_x, shear_y=shear_y
        )
        
        if matrix is not None:
            self._matrix = matrix.copy()
    
    def _compute_matrix(self) -> np.ndarray:
        """Compute the 3x3 affine transformation matrix"""
        if self._matrix is not None:
            return self._matrix
        
        # Convert rotation to radians
        angle_rad = self.params['rotation'] * MATH_CONSTANTS['deg_to_rad']
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Build transformation matrix
        matrix = np.array([
            [self.params['scale_x'] * cos_a - self.params['shear_y'] * sin_a,
             -self.params['scale_x'] * sin_a - self.params['shear_y'] * cos_a,
             self.params['tx']],
            [self.params['scale_y'] * sin_a + self.params['shear_x'] * cos_a,
             self.params['scale_y'] * cos_a - self.params['shear_x'] * sin_a,
             self.params['ty']],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self._matrix = matrix
        return matrix
    
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply affine transformation to points"""
        matrix = self._compute_matrix()
        
        # Convert to homogeneous coordinates
        ones = np.ones_like(x)
        points = np.stack([x, y, ones], axis=0)
        
        # Apply transformation
        transformed = matrix @ points.reshape(3, -1)
        
        x_new = transformed[0].reshape(x.shape)
        y_new = transformed[1].reshape(y.shape)
        
        return x_new, y_new
    
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse affine transformation"""
        matrix = self._compute_matrix()
        inv_matrix = np.linalg.inv(matrix)
        
        # Convert to homogeneous coordinates
        ones = np.ones_like(x)
        points = np.stack([x, y, ones], axis=0)
        
        # Apply inverse transformation
        transformed = inv_matrix @ points.reshape(3, -1)
        
        x_new = transformed[0].reshape(x.shape)
        y_new = transformed[1].reshape(y.shape)
        
        return x_new, y_new

class ProjectiveTransform(BaseTransform):
    """
    Projective (homography) transformation model
    8 degrees of freedom
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Initialize projective transformation
        
        Args:
            matrix: 3x3 homography matrix
        """
        super().__init__()
        self._matrix = matrix.copy()
    
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply projective transformation"""
        # Convert to homogeneous coordinates
        ones = np.ones_like(x)
        points = np.stack([x, y, ones], axis=0)
        
        # Apply transformation
        transformed = self._matrix @ points.reshape(3, -1)
        
        # Convert back from homogeneous coordinates
        x_new = transformed[0] / transformed[2]
        y_new = transformed[1] / transformed[2]
        
        return x_new.reshape(x.shape), y_new.reshape(y.shape)
    
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse projective transformation"""
        inv_matrix = np.linalg.inv(self._matrix)
        
        # Convert to homogeneous coordinates
        ones = np.ones_like(x)
        points = np.stack([x, y, ones], axis=0)
        
        # Apply inverse transformation
        transformed = inv_matrix @ points.reshape(3, -1)
        
        # Convert back from homogeneous coordinates
        x_new = transformed[0] / transformed[2]
        y_new = transformed[1] / transformed[2]
        
        return x_new.reshape(x.shape), y_new.reshape(y.shape)

class RadialDistortionTransform(BaseTransform):
    """
    Radial distortion transformation (Brown-Conrady model)
    Used for lens distortion modeling
    """
    
    def __init__(self,
                 k1: float = 0, k2: float = 0, k3: float = 0,
                 p1: float = 0, p2: float = 0,
                 cx: float = 0, cy: float = 0,
                 image_width: int = 1280, image_height: int = 720):
        """
        Initialize radial distortion model
        
        Args:
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            cx, cy: Principal point coordinates
            image_width, image_height: Image dimensions for normalization
        """
        super().__init__(
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            cx=cx, cy=cy, image_width=image_width, image_height=image_height
        )
    
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply radial distortion (forward transformation)"""
        # Normalize coordinates relative to principal point
        x_norm = (x - self.params['cx']) / self.params['image_width']
        y_norm = (y - self.params['cy']) / self.params['image_height']
        
        # Calculate radial distance squared
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion factor
        k1, k2, k3 = self.params['k1'], self.params['k2'], self.params['k3']
        radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        # Tangential distortion
        p1, p2 = self.params['p1'], self.params['p2']
        tangential_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
        tangential_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
        
        # Apply distortion
        x_dist_norm = x_norm * radial_factor + tangential_x
        y_dist_norm = y_norm * radial_factor + tangential_y
        
        # Convert back to pixel coordinates
        x_dist = x_dist_norm * self.params['image_width'] + self.params['cx']
        y_dist = y_dist_norm * self.params['image_height'] + self.params['cy']
        
        return x_dist, y_dist
    
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse radial distortion (iterative Newton-Raphson)"""
        # This is computationally expensive and should use the corrector algorithms
        # For now, return a simple approximation
        return self._iterative_inverse_points(x, y)
    
    def _iterative_inverse_points(self, x_dist: np.ndarray, y_dist: np.ndarray,
                                 max_iterations: int = 10, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Iterative inverse transformation using Newton-Raphson"""
        # Initial guess
        x_undist = x_dist.copy()
        y_undist = y_dist.copy()
        
        for _ in range(max_iterations):
            # Apply forward distortion to current estimate
            x_forward, y_forward = self.transform_points(x_undist, y_undist)
            
            # Calculate error
            error_x = x_forward - x_dist
            error_y = y_forward - y_dist
            
            # Check convergence
            if np.all(np.abs(error_x) < tolerance) and np.all(np.abs(error_y) < tolerance):
                break
            
            # Simple correction (can be improved with proper Jacobian)
            x_undist -= 0.9 * error_x
            y_undist -= 0.9 * error_y
        
        return x_undist, y_undist

class PolynomialTransform(BaseTransform):
    """
    Polynomial transformation model
    Flexible model for arbitrary deformations
    """
    
    def __init__(self, degree: int = 2, 
                 coeffs_x: Optional[np.ndarray] = None,
                 coeffs_y: Optional[np.ndarray] = None):
        """
        Initialize polynomial transformation
        
        Args:
            degree: Polynomial degree
            coeffs_x: Coefficients for X transformation
            coeffs_y: Coefficients for Y transformation
        """
        super().__init__(degree=degree)
        
        # Number of coefficients for given degree
        num_coeffs = (degree + 1) * (degree + 2) // 2
        
        if coeffs_x is None:
            # Identity transformation
            coeffs_x = np.zeros(num_coeffs)
            coeffs_x[1] = 1  # x coefficient
        
        if coeffs_y is None:
            # Identity transformation
            coeffs_y = np.zeros(num_coeffs)
            coeffs_y[2] = 1  # y coefficient (assuming order: 1, x, y, x^2, xy, y^2, ...)
        
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
    
    def _generate_polynomial_terms(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate polynomial terms up to specified degree"""
        degree = self.params['degree']
        terms = []
        
        # Constant term
        terms.append(np.ones_like(x))
        
        # Generate terms for each degree
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                term = (x ** i) * (y ** j)
                terms.append(term)
        
        return np.stack(terms, axis=0)
    
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply polynomial transformation"""
        terms = self._generate_polynomial_terms(x, y)
        
        # Apply coefficients
        x_new = np.sum(self.coeffs_x[:, np.newaxis, np.newaxis] * terms, axis=0)
        y_new = np.sum(self.coeffs_y[:, np.newaxis, np.newaxis] * terms, axis=0)
        
        return x_new, y_new
    
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse polynomial transformation (not generally available in closed form)"""
        raise NotImplementedError("Inverse polynomial transformation requires iterative methods")

class ThinPlateSplineTransform(BaseTransform):
    """
    Thin Plate Spline transformation
    Used for smooth interpolation between control points
    """
    
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray):
        """
        Initialize TPS transformation
        
        Args:
            source_points: Control points in source image (N x 2)
            target_points: Corresponding points in target image (N x 2)
        """
        super().__init__()
        self.source_points = source_points
        self.target_points = target_points
        self._compute_tps_parameters()
    
    def _compute_tps_parameters(self):
        """Compute TPS transformation parameters"""
        n = len(self.source_points)
        
        # Build kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r2 = np.sum((self.source_points[i] - self.source_points[j])**2)
                    if r2 > 0:
                        K[i, j] = r2 * np.log(r2)
        
        # Build P matrix (affine part)
        P = np.column_stack([np.ones(n), self.source_points])
        
        # Build system matrix
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        
        # Build target vector for X coordinates
        b_x = np.zeros(n + 3)
        b_x[:n] = self.target_points[:, 0]
        
        # Build target vector for Y coordinates
        b_y = np.zeros(n + 3)
        b_y[:n] = self.target_points[:, 1]
        
        # Solve for parameters
        self.params_x = np.linalg.solve(A, b_x)
        self.params_y = np.linalg.solve(A, b_y)
    
    def _tps_kernel(self, r2: np.ndarray) -> np.ndarray:
        """TPS radial basis function"""
        result = np.zeros_like(r2)
        mask = r2 > 0
        result[mask] = r2[mask] * np.log(r2[mask])
        return result
    
    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply TPS transformation"""
        points = np.column_stack([x.ravel(), y.ravel()])
        n_control = len(self.source_points)
        
        # Compute distances to control points
        distances = np.zeros((len(points), n_control))
        for i, cp in enumerate(self.source_points):
            distances[:, i] = np.sum((points - cp)**2, axis=1)
        
        # Apply kernel function
        kernel_values = self._tps_kernel(distances)
        
        # Add affine terms
        affine_terms = np.column_stack([np.ones(len(points)), points])
        
        # Compute transformation
        x_new = np.dot(kernel_values, self.params_x[:n_control]) + \
                np.dot(affine_terms, self.params_x[n_control:])
        y_new = np.dot(kernel_values, self.params_y[:n_control]) + \
                np.dot(affine_terms, self.params_y[n_control:])
        
        return x_new.reshape(x.shape), y_new.reshape(y.shape)
    
    def inverse_transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse TPS transformation (requires iterative solution)"""
        raise NotImplementedError("Inverse TPS transformation requires iterative methods")

class TransformFactory:
    """Factory class for creating transformation objects"""
    
    @staticmethod
    def create_transform(transform_type: str, **params) -> BaseTransform:
        """
        Create a transformation object
        
        Args:
            transform_type: Type of transformation
            **params: Transformation parameters
            
        Returns:
            Transformation object
        """
        if transform_type.lower() == 'affine':
            return AffineTransform(**params)
        elif transform_type.lower() == 'projective':
            return ProjectiveTransform(**params)
        elif transform_type.lower() == 'radial':
            return RadialDistortionTransform(**params)
        elif transform_type.lower() == 'polynomial':
            return PolynomialTransform(**params)
        elif transform_type.lower() == 'tps':
            return ThinPlateSplineTransform(**params)
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    @staticmethod
    def get_available_transforms() -> list:
        """Get list of available transformation types"""
        return ['affine', 'projective', 'radial', 'polynomial', 'tps']