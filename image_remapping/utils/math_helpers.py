"""
Mathematical helper functions for image remapping operations

This module provides utility functions for mathematical operations
commonly used in geometric transformations and distortion modeling.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Union
from config.settings import MATH_CONSTANTS

def normalize_coordinates(x: np.ndarray, y: np.ndarray, 
                         width: int, height: int,
                         center_x: Optional[float] = None,
                         center_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize pixel coordinates to [-1, 1] range relative to image center
    
    Args:
        x, y: Pixel coordinates
        width, height: Image dimensions
        center_x, center_y: Principal point (default: image center)
        
    Returns:
        Normalized coordinates
    """
    if center_x is None:
        center_x = width / 2.0
    if center_y is None:
        center_y = height / 2.0
    
    # Normalize to [-1, 1] range
    x_norm = (x - center_x) / (width / 2.0)
    y_norm = (y - center_y) / (height / 2.0)
    
    return x_norm, y_norm

def denormalize_coordinates(x_norm: np.ndarray, y_norm: np.ndarray,
                           width: int, height: int,
                           center_x: Optional[float] = None,
                           center_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized coordinates back to pixel coordinates
    
    Args:
        x_norm, y_norm: Normalized coordinates
        width, height: Image dimensions
        center_x, center_y: Principal point (default: image center)
        
    Returns:
        Pixel coordinates
    """
    if center_x is None:
        center_x = width / 2.0
    if center_y is None:
        center_y = height / 2.0
    
    # Convert back to pixel coordinates
    x = x_norm * (width / 2.0) + center_x
    y = y_norm * (height / 2.0) + center_y
    
    return x, y

def calculate_radial_distance(x: np.ndarray, y: np.ndarray,
                             center_x: float, center_y: float) -> np.ndarray:
    """
    Calculate radial distance from center point
    
    Args:
        x, y: Coordinate arrays
        center_x, center_y: Center point
        
    Returns:
        Radial distance array
    """
    return np.sqrt((x - center_x)**2 + (y - center_y)**2)

def calculate_angle(x: np.ndarray, y: np.ndarray,
                   center_x: float, center_y: float) -> np.ndarray:
    """
    Calculate angle from center point in radians
    
    Args:
        x, y: Coordinate arrays
        center_x, center_y: Center point
        
    Returns:
        Angle array in radians
    """
    return np.arctan2(y - center_y, x - center_x)

def polar_to_cartesian(r: np.ndarray, theta: np.ndarray,
                      center_x: float, center_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates to Cartesian
    
    Args:
        r: Radial distance
        theta: Angle in radians
        center_x, center_y: Center point
        
    Returns:
        Cartesian coordinates (x, y)
    """
    x = r * np.cos(theta) + center_x
    y = r * np.sin(theta) + center_y
    return x, y

def cartesian_to_polar(x: np.ndarray, y: np.ndarray,
                      center_x: float, center_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar
    
    Args:
        x, y: Cartesian coordinates
        center_x, center_y: Center point
        
    Returns:
        Polar coordinates (r, theta)
    """
    r = calculate_radial_distance(x, y, center_x, center_y)
    theta = calculate_angle(x, y, center_x, center_y)
    return r, theta

def degrees_to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert degrees to radians"""
    return degrees * MATH_CONSTANTS['deg_to_rad']

def radians_to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert radians to degrees"""
    return radians * MATH_CONSTANTS['rad_to_deg']

def apply_rotation_matrix(x: np.ndarray, y: np.ndarray, 
                         angle: float, 
                         center_x: float = 0, center_y: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 2D rotation matrix to coordinates
    
    Args:
        x, y: Input coordinates
        angle: Rotation angle in radians
        center_x, center_y: Rotation center
        
    Returns:
        Rotated coordinates
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Translate to origin
    x_centered = x - center_x
    y_centered = y - center_y
    
    # Apply rotation
    x_rot = x_centered * cos_a - y_centered * sin_a
    y_rot = x_centered * sin_a + y_centered * cos_a
    
    # Translate back
    x_new = x_rot + center_x
    y_new = y_rot + center_y
    
    return x_new, y_new

def compute_jacobian_2d(func, x: float, y: float, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute 2D Jacobian matrix using finite differences
    
    Args:
        func: Function that takes (x, y) and returns (fx, fy)
        x, y: Point at which to compute Jacobian
        epsilon: Step size for finite differences
        
    Returns:
        2x2 Jacobian matrix
    """
    # Function value at (x, y)
    fx, fy = func(x, y)
    
    # Partial derivatives with respect to x
    fx_plus_x, fy_plus_x = func(x + epsilon, y)
    dfx_dx = (fx_plus_x - fx) / epsilon
    dfy_dx = (fy_plus_x - fy) / epsilon
    
    # Partial derivatives with respect to y
    fx_plus_y, fy_plus_y = func(x, y + epsilon)
    dfx_dy = (fx_plus_y - fx) / epsilon
    dfy_dy = (fy_plus_y - fy) / epsilon
    
    # Construct Jacobian matrix
    jacobian = np.array([
        [dfx_dx, dfx_dy],
        [dfy_dx, dfy_dy]
    ])
    
    return jacobian

def solve_2x2_system(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve 2x2 linear system Ax = b
    
    Args:
        A: 2x2 coefficient matrix
        b: 2x1 right-hand side vector
        
    Returns:
        Solution vector or None if singular
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    if abs(det) < 1e-12:
        return None  # Singular matrix
    
    # Cramer's rule
    x = np.array([
        (A[1, 1] * b[0] - A[0, 1] * b[1]) / det,
        (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
    ])
    
    return x

def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perform bilinear interpolation on image
    
    Args:
        image: Input image
        x, y: Interpolation coordinates (can be non-integer)
        
    Returns:
        Interpolated values
    """
    height, width = image.shape[:2]
    
    # Clamp coordinates to valid range
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    
    # Get integer and fractional parts
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    
    # Calculate weights
    wx = x - x0
    wy = y - y0
    
    # Get corner values
    if len(image.shape) == 2:  # Grayscale
        v00 = image[y0, x0]
        v01 = image[y1, x0]
        v10 = image[y0, x1]
        v11 = image[y1, x1]
        
        # Bilinear interpolation
        result = (v00 * (1 - wx) * (1 - wy) +
                 v10 * wx * (1 - wy) +
                 v01 * (1 - wx) * wy +
                 v11 * wx * wy)
    else:  # Color
        result = np.zeros(image.shape[2:] + x.shape, dtype=image.dtype)
        for c in range(image.shape[2]):
            v00 = image[y0, x0, c]
            v01 = image[y1, x0, c]
            v10 = image[y0, x1, c]
            v11 = image[y1, x1, c]
            
            result[c] = (v00 * (1 - wx) * (1 - wy) +
                        v10 * wx * (1 - wy) +
                        v01 * (1 - wx) * wy +
                        v11 * wx * wy)
        
        # Transpose to match expected output shape
        if len(x.shape) == 0:  # Single point
            result = result.squeeze()
        else:
            result = np.transpose(result, list(range(1, len(result.shape))) + [0])
    
    return result

def bicubic_interpolate_1d(values: np.ndarray, t: float) -> float:
    """
    Perform bicubic interpolation for 1D case
    
    Args:
        values: Array of 4 values [f(-1), f(0), f(1), f(2)]
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated value
    """
    # Catmull-Rom spline coefficients
    a = -0.5 * values[0] + 1.5 * values[1] - 1.5 * values[2] + 0.5 * values[3]
    b = values[0] - 2.5 * values[1] + 2.0 * values[2] - 0.5 * values[3]
    c = -0.5 * values[0] + 0.5 * values[2]
    d = values[1]
    
    return a * t**3 + b * t**2 + c * t + d

def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_value: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        image1, image2: Input images
        max_value: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((image1.astype(float) - image2.astype(float))**2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr

def calculate_rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between two images
    
    Args:
        image1, image2: Input images
        
    Returns:
        RMSE value
    """
    mse = np.mean((image1.astype(float) - image2.astype(float))**2)
    return np.sqrt(mse)

def calculate_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate normalized cross-correlation between two images
    
    Args:
        image1, image2: Input images
        
    Returns:
        Correlation coefficient
    """
    # Flatten images
    flat1 = image1.flatten().astype(float)
    flat2 = image2.flatten().astype(float)
    
    # Calculate correlation coefficient
    correlation_matrix = np.corrcoef(flat1, flat2)
    return correlation_matrix[0, 1]

def fit_polynomial_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit 2D polynomial to scattered data points
    
    Args:
        x, y: Input coordinates
        z: Output values
        degree: Polynomial degree
        
    Returns:
        Coefficient array
    """
    # Generate polynomial terms
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append((x ** i) * (y ** j))
    
    # Stack terms into design matrix
    A = np.stack(terms, axis=1)
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    
    return coeffs

def evaluate_polynomial_2d(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate 2D polynomial at given points
    
    Args:
        coeffs: Polynomial coefficients
        x, y: Evaluation coordinates
        degree: Polynomial degree
        
    Returns:
        Evaluated values
    """
    result = np.zeros_like(x, dtype=float)
    coeff_idx = 0
    
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            result += coeffs[coeff_idx] * (x ** i) * (y ** j)
            coeff_idx += 1
    
    return result

def compute_grid_statistics(grid: np.ndarray) -> dict:
    """
    Compute comprehensive statistics for a 2D grid
    
    Args:
        grid: 2D array
        
    Returns:
        Dictionary with statistics
    """
    flat_grid = grid.flatten()
    
    stats = {
        'min': float(np.min(flat_grid)),
        'max': float(np.max(flat_grid)),
        'mean': float(np.mean(flat_grid)),
        'median': float(np.median(flat_grid)),
        'std': float(np.std(flat_grid)),
        'var': float(np.var(flat_grid)),
        'range': float(np.max(flat_grid) - np.min(flat_grid)),
        'percentile_25': float(np.percentile(flat_grid, 25)),
        'percentile_75': float(np.percentile(flat_grid, 75)),
        'iqr': float(np.percentile(flat_grid, 75) - np.percentile(flat_grid, 25)),
        'skewness': float(compute_skewness(flat_grid)),
        'kurtosis': float(compute_kurtosis(flat_grid))
    }
    
    return stats

def compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of data distribution"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    normalized = (data - mean) / std
    skewness = np.mean(normalized ** 3)
    return skewness

def compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of data distribution"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    normalized = (data - mean) / std
    kurtosis = np.mean(normalized ** 4) - 3  # Excess kurtosis
    return kurtosis

def create_meshgrid(width: int, height: int, 
                   x_range: Optional[Tuple[float, float]] = None,
                   y_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create coordinate meshgrid with optional custom ranges
    
    Args:
        width, height: Grid dimensions
        x_range, y_range: Optional coordinate ranges
        
    Returns:
        Coordinate meshgrids (X, Y)
    """
    if x_range is None:
        x_coords = np.arange(width, dtype=np.float32)
    else:
        x_coords = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
    
    if y_range is None:
        y_coords = np.arange(height, dtype=np.float32)
    else:
        y_coords = np.linspace(y_range[0], y_range[1], height, dtype=np.float32)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    return X, Y

def robust_invert_2x2(matrix: np.ndarray, regularization: float = 1e-12) -> Optional[np.ndarray]:
    """
    Robustly invert 2x2 matrix with regularization
    
    Args:
        matrix: 2x2 input matrix
        regularization: Regularization parameter for near-singular matrices
        
    Returns:
        Inverted matrix or None if inversion fails
    """
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    if abs(det) < regularization:
        # Add regularization to diagonal
        regularized_matrix = matrix + regularization * np.eye(2)
        det = (regularized_matrix[0, 0] * regularized_matrix[1, 1] - 
               regularized_matrix[0, 1] * regularized_matrix[1, 0])
        
        if abs(det) < regularization:
            return None
        
        # Invert regularized matrix
        inv_matrix = np.array([
            [regularized_matrix[1, 1], -regularized_matrix[0, 1]],
            [-regularized_matrix[1, 0], regularized_matrix[0, 0]]
        ]) / det
    else:
        # Normal inversion
        inv_matrix = np.array([
            [matrix[1, 1], -matrix[0, 1]],
            [-matrix[1, 0], matrix[0, 0]]
        ]) / det
    
    return inv_matrix

def compute_transformation_error(points_source: np.ndarray, 
                               points_target: np.ndarray,
                               transform_func) -> dict:
    """
    Compute error metrics for a transformation
    
    Args:
        points_source: Source points (N x 2)
        points_target: Target points (N x 2)
        transform_func: Transformation function
        
    Returns:
        Error metrics dictionary
    """
    # Apply transformation to source points
    transformed_points = np.array([
        transform_func(pt[0], pt[1]) for pt in points_source
    ])
    
    # Calculate errors
    errors = np.linalg.norm(transformed_points - points_target, axis=1)
    
    metrics = {
        'mean_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'percentile_95': float(np.percentile(errors, 95))
    }
    
    return metrics

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
               default_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero
        
    Returns:
        Result of division with safe handling
    """
    result = np.full_like(numerator, default_value, dtype=float)
    non_zero_mask = np.abs(denominator) > 1e-12
    result[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]
    return result

def clamp_values(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clamp array values to specified range
    
    Args:
        array: Input array
        min_val, max_val: Clamping range
        
    Returns:
        Clamped array
    """
    return np.clip(array, min_val, max_val)