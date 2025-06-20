"""
Global application settings and defaults for Image Remapping Suite
Updated to include GDC Grid Processing configuration
"""

# Image dimensions
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720
MIN_IMAGE_WIDTH = 640
MAX_IMAGE_WIDTH = 7680
MIN_IMAGE_HEIGHT = 480
MAX_IMAGE_HEIGHT = 4320

# Grid parameters
DEFAULT_GRID_ROWS = 7
DEFAULT_GRID_COLS = 9
MIN_GRID_SIZE = 7
MAX_GRID_SIZE = 50

# Distortion coefficients (Brown-Conrady model)
DEFAULT_K1 = -0.2
DEFAULT_K2 = 0.05
DEFAULT_K3 = 0.0
DEFAULT_P1 = 0.0
DEFAULT_P2 = 0.0

# Correction algorithm parameters
MAX_ITERATIONS = 10
CONVERGENCE_TOLERANCE = 1e-6
NUMERICAL_DERIVATIVE_EPSILON = 1e-6

# GDC (Geometric Distortion Correction) parameters
DEFAULT_GDC_WIDTH = 8192
DEFAULT_GDC_HEIGHT = 6144

# GDC Grid Processing defaults
GRID_DEFAULTS = {
    'original_rows': 9,
    'original_cols': 7,
    'target_rows': 33,
    'target_cols': 33,
    'interpolation_method': 'bicubic'
}

# Application configuration for GDC interface
APP_CONFIG = {
    'server_name': "localhost",
    'port': 7860,
    'share': False,
    'debug': True
}

# File formats and export settings
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_GRID_FORMATS = ['.txt', '.csv', '.dat']
CSV_DECIMAL_PRECISION = 6
HEX_FORMAT_WIDTH = 8

# Visualization settings
VISUALIZATION_SCALE_FACTOR = 0.5
GRID_LINE_THICKNESS = 2
VECTOR_LINE_THICKNESS = 2
POINT_RADIUS = 4

# Color schemes (BGR format for OpenCV)
COLORS = {
    'target_grid': (255, 0, 0),      # Blue for target points
    'source_distorted': (0, 0, 255), # Red for source points
    'grid_lines': (200, 200, 200),   # Light gray for grid lines
    'distortion_vectors': (0, 150, 0), # Green for distortion vectors
    'background': (255, 255, 255),   # White background
    'text': (0, 0, 0)                # Black text
}

# Pattern types for test images
PATTERN_TYPES = ['checkerboard', 'grid', 'circles', 'text']
DEFAULT_PATTERN_TYPE = 'checkerboard'

# Interpolation methods
INTERPOLATION_METHODS = {
    'nearest': 0,    # cv2.INTER_NEAREST
    'linear': 1,     # cv2.INTER_LINEAR
    'cubic': 2,      # cv2.INTER_CUBIC
    'lanczos': 4     # cv2.INTER_LANCZOS4
}
DEFAULT_INTERPOLATION = 'linear'

# Quality assessment parameters
QUALITY_METRICS = {
    'psnr_threshold_good': 30.0,
    'psnr_threshold_excellent': 40.0,
    'correlation_threshold_good': 0.95,
    'correlation_threshold_excellent': 0.99,
    'geometric_error_threshold_excellent': 0.5,
    'geometric_error_threshold_good': 1.0,
    'geometric_error_threshold_acceptable': 2.0
}

# Gradio interface settings
GRADIO_SETTINGS = {
    'theme': 'soft',
    'default_port': 7860,
    'port_range': (7860, 7900),
    'share': True,
    'debug': False,
    'show_error': True
}

# Mathematical constants
MATH_CONSTANTS = {
    'pi': 3.14159265359,
    'deg_to_rad': 3.14159265359 / 180.0,
    'rad_to_deg': 180.0 / 3.14159265359
}

# Validation ranges
VALIDATION_RANGES = {
    'k1': (-0.5, 0.5),
    'k2': (-0.2, 0.2),
    'k3': (-0.1, 0.1),
    'p1': (-0.1, 0.1),
    'p2': (-0.1, 0.1)
}

# GDC Grid validation ranges
GDC_VALIDATION_RANGES = {
    'grid_rows': (3, 100),
    'grid_cols': (3, 100),
    'target_rows': (10, 1000),
    'target_cols': (10, 1000),
    'interpolation_factor': (1.1, 50.0)
}

# GDC file parsing patterns
GDC_PARSING_CONFIG = {
    'dx_pattern': r"yuv_gdc_grid_dx_0_(\d+)",
    'dy_pattern': r"yuv_gdc_grid_dy_0_(\d+)",
    'element_pattern': r"_(dx|dy)_0_(\d+)$",
    'line_format': r"^(\S+)\s+(-?\d+)$",
    'max_file_size_mb': 100,
    'encoding': 'utf-8'
}

# Export format configurations
EXPORT_FORMATS = {
    'csv': {
        'delimiter': ',',
        'decimal_places': 6,
        'include_headers': True
    },
    'gdc': {
        'naming_convention': 'yuv_gdc_grid_{type}_0_{index}',
        'value_format': 'integer',
        'include_combined': True
    },
    'json': {
        'indent': 2,
        'include_metadata': True
    }
}

# Visualization configurations
VISUALIZATION_CONFIG = {
    'default_colormap': 'viridis',
    'comparison_colormap': 'RdBu_r',
    'figure_dpi': 150,
    'figure_size': (10, 8),
    'comparison_figure_size': (15, 6),
    'save_format': 'png',
    'bbox_inches': 'tight'
}

# Performance settings
PERFORMANCE = {
    'use_multiprocessing': True,
    'chunk_size': 1000,
    'memory_limit_mb': 2048,
    'max_grid_size': 1000000,  # Maximum total grid elements
    'interpolation_cache_size': 100  # Number of interpolation results to cache
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': None  # Set to filename for file logging
}

# Temporary file management
TEMP_FILE_CONFIG = {
    'cleanup_on_exit': True,
    'max_temp_files': 1000,
    'temp_dir': None,  # Use system default
    'file_retention_hours': 24
}

# Error handling configuration
ERROR_HANDLING = {
    'max_retries': 3,
    'retry_delay_seconds': 1.0,
    'graceful_degradation': True,
    'fallback_interpolation': 'linear'
}

# Security settings for file uploads
SECURITY_CONFIG = {
    'max_upload_size_mb': 50,
    'allowed_extensions': ['.txt', '.csv', '.dat'],
    'scan_content': True,
    'max_lines_per_file': 1000000
}