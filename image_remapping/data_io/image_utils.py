"""
Image loading, saving, and basic processing utilities

This module provides utilities for handling various image formats,
basic image operations, and validation functions.
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional, List, Union
from PIL import Image, ImageEnhance
from config.settings import SUPPORTED_IMAGE_FORMATS

def load_image(file_path: str, color_mode: str = 'BGR') -> np.ndarray:
    """
    Load an image from file with format validation
    
    Args:
        file_path: Path to image file
        color_mode: Color mode ('BGR', 'RGB', 'GRAY')
        
    Returns:
        Image array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {file_ext}")
    
    try:
        if color_mode.upper() == 'GRAY':
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if color_mode.upper() == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        return image
    
    except Exception as e:
        raise ValueError(f"Error loading image {file_path}: {str(e)}")

def save_image(image: np.ndarray, file_path: str, quality: int = 95) -> bool:
    """
    Save an image to file with quality control
    
    Args:
        image: Image array to save
        file_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Determine file format and parameters
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif file_ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            params = []
        
        # Save image
        success = cv2.imwrite(file_path, image, params)
        
        return success
    
    except Exception:
        return False

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                interpolation: str = 'linear') -> np.ndarray:
    """
    Resize an image with various interpolation methods
    
    Args:
        image: Input image
        target_size: (width, height) target dimensions
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    interp_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'area': cv2.INTER_AREA
    }
    
    if interpolation not in interp_methods:
        raise ValueError(f"Unknown interpolation method: {interpolation}")
    
    return cv2.resize(image, target_size, interpolation=interp_methods[interpolation])

def crop_image(image: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to specified bounding box
    
    Args:
        image: Input image
        crop_box: (x, y, width, height) bounding box
        
    Returns:
        Cropped image
    """
    x, y, width, height = crop_box
    
    # Validate crop box
    img_height, img_width = image.shape[:2]
    
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    return image[y:y+height, x:x+width]

def enhance_image(image: np.ndarray, 
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image with brightness, contrast, and saturation adjustments
    
    Args:
        image: Input image (BGR format)
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)
        
    Returns:
        Enhanced image
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)
    
    if saturation != 1.0 and len(image.shape) == 3:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)
    
    # Convert back to BGR for OpenCV
    enhanced = np.array(pil_image)
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    return enhanced

def validate_image(image: np.ndarray) -> dict:
    """
    Validate image properties and detect potential issues
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'properties': {}
    }
    
    # Basic shape validation
    if len(image.shape) not in [2, 3]:
        results['errors'].append("Invalid image dimensions")
        results['valid'] = False
        return results
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    results['properties'].update({
        'height': height,
        'width': width,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_mb': image.nbytes / (1024 * 1024)
    })
    
    # Size validation
    if height < 10 or width < 10:
        results['errors'].append("Image too small (minimum 10x10 pixels)")
        results['valid'] = False
    
    if height > 10000 or width > 10000:
        results['warnings'].append("Very large image - may cause memory issues")
    
    # Data type validation
    if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        results['warnings'].append(f"Unusual data type: {image.dtype}")
    
    # Value range validation
    if image.dtype == np.uint8:
        if np.min(image) < 0 or np.max(image) > 255:
            results['errors'].append("Invalid value range for uint8 image")
            results['valid'] = False
    elif image.dtype in [np.float32, np.float64]:
        if np.min(image) < 0 or np.max(image) > 1:
            results['warnings'].append("Float image values outside [0,1] range")
    
    # Channel validation
    if channels not in [1, 3, 4]:
        results['warnings'].append(f"Unusual number of channels: {channels}")
    
    # Statistical properties
    results['properties'].update({
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    })
    
    return results

def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """
    Convert image between different color spaces
    
    Args:
        image: Input image
        conversion: Conversion type (e.g., 'BGR2RGB', 'BGR2GRAY', 'RGB2HSV')
        
    Returns:
        Converted image
    """
    conversion_map = {
        'BGR2RGB': cv2.COLOR_BGR2RGB,
        'RGB2BGR': cv2.COLOR_RGB2BGR,
        'BGR2GRAY': cv2.COLOR_BGR2GRAY,
        'RGB2GRAY': cv2.COLOR_RGB2GRAY,
        'BGR2HSV': cv2.COLOR_BGR2HSV,
        'RGB2HSV': cv2.COLOR_RGB2HSV,
        'HSV2BGR': cv2.COLOR_HSV2BGR,
        'HSV2RGB': cv2.COLOR_HSV2RGB,
        'BGR2LAB': cv2.COLOR_BGR2LAB,
        'RGB2LAB': cv2.COLOR_RGB2LAB,
        'LAB2BGR': cv2.COLOR_LAB2BGR,
        'LAB2RGB': cv2.COLOR_LAB2RGB
    }
    
    if conversion not in conversion_map:
        raise ValueError(f"Unsupported conversion: {conversion}")
    
    return cv2.cvtColor(image, conversion_map[conversion])

def create_test_image(width: int, height: int, pattern: str = 'checkerboard') -> np.ndarray:
    """
    Create test images with various patterns
    
    Args:
        width: Image width
        height: Image height
        pattern: Pattern type
        
    Returns:
        Generated test image
    """
    if pattern == 'checkerboard':
        return create_checkerboard(width, height)
    elif pattern == 'grid':
        return create_grid_pattern(width, height)
    elif pattern == 'gradient':
        return create_gradient_pattern(width, height)
    elif pattern == 'circles':
        return create_circles_pattern(width, height)
    elif pattern == 'noise':
        return create_noise_pattern(width, height)
    else:
        raise ValueError(f"Unknown pattern type: {pattern}")

def create_checkerboard(width: int, height: int, square_size: int = 50) -> np.ndarray:
    """Create a checkerboard pattern"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                image[y:y+square_size, x:x+square_size] = [255, 255, 255]
    
    return image

def create_grid_pattern(width: int, height: int, spacing: int = 50) -> np.ndarray:
    """Create a grid line pattern"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Vertical lines
    for x in range(0, width, spacing):
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), 2)
    
    # Horizontal lines
    for y in range(0, height, spacing):
        cv2.line(image, (0, y), (width, y), (0, 0, 0), 2)
    
    return image

def create_gradient_pattern(width: int, height: int, direction: str = 'horizontal') -> np.ndarray:
    """Create a gradient pattern"""
    if direction == 'horizontal':
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        image = np.broadcast_to(gradient[np.newaxis, :], (height, width))
    else:  # vertical
        gradient = np.linspace(0, 255, height, dtype=np.uint8)
        image = np.broadcast_to(gradient[:, np.newaxis], (height, width))
    
    return np.stack([image, image, image], axis=-1)

def create_circles_pattern(width: int, height: int) -> np.ndarray:
    """Create concentric circles pattern"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    max_radius = min(center)
    
    for radius in range(25, max_radius, 50):
        cv2.circle(image, center, radius, (0, 0, 0), 3)
    
    return image

def create_noise_pattern(width: int, height: int, noise_type: str = 'gaussian') -> np.ndarray:
    """Create noise pattern for testing"""
    if noise_type == 'gaussian':
        noise = np.random.normal(128, 30, (height, width, 3))
    elif noise_type == 'uniform':
        noise = np.random.uniform(0, 255, (height, width, 3))
    else:
        noise = np.random.randint(0, 256, (height, width, 3))
    
    return np.clip(noise, 0, 255).astype(np.uint8)

def batch_process_images(input_dir: str, output_dir: str, 
                        process_func, **kwargs) -> List[str]:
    """
    Process all images in a directory
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        process_func: Processing function to apply
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of processed filenames
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in SUPPORTED_IMAGE_FORMATS:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Load image
                image = load_image(input_path)
                
                # Process image
                processed_image = process_func(image, **kwargs)
                
                # Save processed image
                if save_image(processed_image, output_path):
                    processed_files.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return processed_files

def get_image_info(image: np.ndarray) -> dict:
    """
    Get comprehensive information about an image
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes,
        'size_mb': image.nbytes / (1024 * 1024),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    }
    
    # Add channel information
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['channel_means'] = [float(np.mean(image[:, :, i])) for i in range(image.shape[2])]
    else:
        info['channels'] = 1
        info['channel_means'] = [info['mean_value']]
    
    return info