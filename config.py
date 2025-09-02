"""
Configuration file for the Image Processing Tool
Author: itislynx
Date: August 31, 2025
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the image processing tool."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    SRC_DIR = PROJECT_ROOT / "src"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    DATASETS_DIR = PROJECT_ROOT / "datasets"
    
    # Create directories if they don't exist
    OUTPUTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # Output subdirectories
    PROCESSED_IMAGES_DIR = OUTPUTS_DIR / "processed_images"
    COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
    INSIGHTS_DIR = OUTPUTS_DIR / "insights"
    GRAPHS_DIR = OUTPUTS_DIR / "graphs"
    TABLES_DIR = OUTPUTS_DIR / "tables"
    
    # Create output subdirectories
    for dir_path in [PROCESSED_IMAGES_DIR, COMPARISONS_DIR, INSIGHTS_DIR, GRAPHS_DIR, TABLES_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Image processing parameters
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dicom', '.dcm']
    SUPPORTED_ARCHIVE_FORMATS = ['.zip', '.rar', '.7z', '.tar', '.gz']
    
    # GPU/CPU configuration
    USE_GPU = True
    GPU_MEMORY_FRACTION = 0.8
    
    # Dataset analysis parameters
    SAMPLE_SIZE_FOR_PROCESSING = 3  # Number of random images to process
    MAX_IMAGES_FOR_ANALYSIS = 10000  # Maximum number of images to analyze for insights
    
    # Image processing algorithms to compare
    PROCESSING_ALGORITHMS = [
        'gabor_filter',
        'local_directional_pattern',
        'gray_level_run_length_matrix',
        'gray_level_co_occurrence_matrix',
        'gray_level_size_zone_matrix',
        'wavelet_transform',
        'fast_fourier_transform',
        'segmentation_based_fractal_texture_analysis',
        'local_binary_pattern_glcm'
    ]
    
    # Visualization parameters
    FIGURE_SIZE = (15, 10)
    DPI = 300
    
    # Debug settings
    DEBUG = True
    VERBOSE = True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 50)
        print("IMAGE PROCESSING TOOL CONFIGURATION")
        print("=" * 50)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Datasets Directory: {cls.DATASETS_DIR}")
        print(f"Outputs Directory: {cls.OUTPUTS_DIR}")
        print(f"Use GPU: {cls.USE_GPU}")
        print(f"Sample Size: {cls.SAMPLE_SIZE_FOR_PROCESSING}")
        print(f"Processing Algorithms: {len(cls.PROCESSING_ALGORITHMS)}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 50)
