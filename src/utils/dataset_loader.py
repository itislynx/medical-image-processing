"""
Dataset Loader and Analyzer
Handles loading images from various sources including compressed files
"""

import os
import zipfile
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
import tempfile

from config import Config


class DatasetLoader:
    """
    Comprehensive dataset loader for medical image datasets.
    Supports various archive formats and provides detailed dataset insights.
    """
    
    def __init__(self, dataset_path: Union[str, Path], use_gpu: bool = True):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset (folder or archive file)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.dataset_path = Path(dataset_path)
        self.use_gpu = use_gpu
        self.temp_dir = None
        self.image_paths = []
        self.dataset_info = {}
        
        if Config.DEBUG:
            print(f"[DEBUG] Initializing DatasetLoader for: {self.dataset_path}")
    
    def load_dataset(self) -> List[Path]:
        """
        Load dataset from various sources (folder, zip, rar, etc.).
        
        Returns:
            List of image file paths
        """
        print(f"Loading dataset from: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        if self.dataset_path.is_file():
            # Handle archive files
            self.image_paths = self._extract_and_load_archive()
        else:
            # Handle directory
            self.image_paths = self._load_from_directory()
        
        print(f"Found {len(self.image_paths)} images in the dataset")
        return self.image_paths
    
    def _extract_and_load_archive(self) -> List[Path]:
        """Extract archive and load images."""
        archive_path = self.dataset_path
        file_extension = archive_path.suffix.lower()
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        print(f"Extracting {file_extension} archive to temporary directory...")
        
        try:
            if file_extension == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
            elif file_extension == '.rar':
                if not RARFILE_AVAILABLE:
                    raise ValueError("rarfile package not installed. Install with: pip install rarfile")
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(temp_path)
            elif file_extension in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_path)
            else:
                raise ValueError(f"Unsupported archive format: {file_extension}")
                
            return self._load_from_directory(temp_path)
            
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return []
    
    def _load_from_directory(self, directory: Optional[Path] = None) -> List[Path]:
        """Load images from directory recursively."""
        if directory is None:
            directory = self.dataset_path
        
        image_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in Config.SUPPORTED_IMAGE_FORMATS:
                    image_paths.append(file_path)
        
        return sorted(image_paths)
    
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis.
        
        Returns:
            Dictionary containing dataset insights
        """
        print("Analyzing dataset...")
        
        if not self.image_paths:
            print("No images found. Loading dataset first...")
            self.load_dataset()
        
        # Limit analysis for performance
        analysis_paths = self.image_paths[:Config.MAX_IMAGES_FOR_ANALYSIS]
        
        insights = {
            'total_images': len(self.image_paths),
            'analyzed_images': len(analysis_paths),
            'file_formats': defaultdict(int),
            'image_dimensions': [],
            'file_sizes': [],
            'pixel_statistics': {
                'mean_values': [],
                'std_values': [],
                'min_values': [],
                'max_values': []
            },
            'class_distribution': defaultdict(int),
            'color_channels': defaultdict(int)
        }
        
        print(f"Analyzing {len(analysis_paths)} images...")
        
        for img_path in tqdm(analysis_paths, desc="Analyzing images"):
            try:
                # File format and size
                insights['file_formats'][img_path.suffix.lower()] += 1
                insights['file_sizes'].append(img_path.stat().st_size)
                
                # Load and analyze image
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                
                # Image dimensions
                if len(img.shape) == 3:
                    h, w, c = img.shape
                    insights['color_channels'][c] += 1
                else:
                    h, w = img.shape
                    c = 1
                    insights['color_channels'][1] += 1
                
                insights['image_dimensions'].append((h, w, c))
                
                # Pixel statistics
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                
                insights['pixel_statistics']['mean_values'].append(np.mean(img_gray))
                insights['pixel_statistics']['std_values'].append(np.std(img_gray))
                insights['pixel_statistics']['min_values'].append(np.min(img_gray))
                insights['pixel_statistics']['max_values'].append(np.max(img_gray))
                
                # Class distribution (based on parent folder name)
                class_name = img_path.parent.name
                insights['class_distribution'][class_name] += 1
                
            except Exception as e:
                if Config.DEBUG:
                    print(f"[DEBUG] Error analyzing {img_path}: {e}")
                continue
        
        self.dataset_info = insights
        return insights
    
    def generate_insights_report(self) -> pd.DataFrame:
        """Generate detailed insights report and save visualizations."""
        if not self.dataset_info:
            self.analyze_dataset()
        
        insights = self.dataset_info
        
        # Create summary DataFrame
        summary_data = {
            'Metric': [
                'Total Images',
                'Analyzed Images',
                'Unique File Formats',
                'Unique Classes',
                'Average File Size (MB)',
                'Average Image Width',
                'Average Image Height',
                'Average Pixel Intensity',
                'Pixel Intensity Std Dev'
            ],
            'Value': [
                insights['total_images'],
                insights['analyzed_images'],
                len(insights['file_formats']),
                len(insights['class_distribution']),
                f"{np.mean(insights['file_sizes']) / (1024*1024):.2f}",
                f"{np.mean([d[1] for d in insights['image_dimensions']]):.0f}",
                f"{np.mean([d[0] for d in insights['image_dimensions']]):.0f}",
                f"{np.mean(insights['pixel_statistics']['mean_values']):.2f}",
                f"{np.mean(insights['pixel_statistics']['std_values']):.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = Config.INSIGHTS_DIR / 'dataset_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Dataset summary saved to: {summary_path}")
        
        # Generate visualizations
        self._create_visualizations()
        
        return summary_df
    
    def _create_visualizations(self):
        """Create and save dataset visualization plots."""
        insights = self.dataset_info
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. File format distribution
        if insights['file_formats']:
            plt.figure(figsize=(10, 6))
            formats = list(insights['file_formats'].keys())
            counts = list(insights['file_formats'].values())
            
            plt.pie(counts, labels=formats, autopct='%1.1f%%', startangle=90)
            plt.title('File Format Distribution')
            plt.tight_layout()
            plt.savefig(Config.GRAPHS_DIR / 'file_format_distribution.png', dpi=Config.DPI, bbox_inches='tight')
            plt.close()
        
        # 2. Class distribution
        if len(insights['class_distribution']) > 1:
            plt.figure(figsize=(12, 6))
            classes = list(insights['class_distribution'].keys())
            counts = list(insights['class_distribution'].values())
            
            plt.bar(classes, counts)
            plt.title('Class Distribution')
            plt.xlabel('Classes')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(Config.GRAPHS_DIR / 'class_distribution.png', dpi=Config.DPI, bbox_inches='tight')
            plt.close()
        
        # 3. Image dimensions distribution
        if insights['image_dimensions']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Width distribution
            widths = [d[1] for d in insights['image_dimensions']]
            axes[0].hist(widths, bins=30, alpha=0.7, color='skyblue')
            axes[0].set_title('Image Width Distribution')
            axes[0].set_xlabel('Width (pixels)')
            axes[0].set_ylabel('Frequency')
            
            # Height distribution
            heights = [d[0] for d in insights['image_dimensions']]
            axes[1].hist(heights, bins=30, alpha=0.7, color='lightcoral')
            axes[1].set_title('Image Height Distribution')
            axes[1].set_xlabel('Height (pixels)')
            axes[1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(Config.GRAPHS_DIR / 'image_dimensions_distribution.png', dpi=Config.DPI, bbox_inches='tight')
            plt.close()
        
        # 4. Pixel intensity statistics
        if insights['pixel_statistics']['mean_values']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            stats = insights['pixel_statistics']
            
            # Mean pixel intensity
            axes[0, 0].hist(stats['mean_values'], bins=30, alpha=0.7, color='green')
            axes[0, 0].set_title('Mean Pixel Intensity Distribution')
            axes[0, 0].set_xlabel('Mean Intensity')
            axes[0, 0].set_ylabel('Frequency')
            
            # Standard deviation
            axes[0, 1].hist(stats['std_values'], bins=30, alpha=0.7, color='orange')
            axes[0, 1].set_title('Pixel Intensity Std Dev Distribution')
            axes[0, 1].set_xlabel('Standard Deviation')
            axes[0, 1].set_ylabel('Frequency')
            
            # Min values
            axes[1, 0].hist(stats['min_values'], bins=30, alpha=0.7, color='red')
            axes[1, 0].set_title('Minimum Pixel Intensity Distribution')
            axes[1, 0].set_xlabel('Min Intensity')
            axes[1, 0].set_ylabel('Frequency')
            
            # Max values
            axes[1, 1].hist(stats['max_values'], bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title('Maximum Pixel Intensity Distribution')
            axes[1, 1].set_xlabel('Max Intensity')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(Config.GRAPHS_DIR / 'pixel_intensity_statistics.png', dpi=Config.DPI, bbox_inches='tight')
            plt.close()
        
        print(f"Dataset visualizations saved to: {Config.GRAPHS_DIR}")
    
    def get_random_samples(self, n_samples: int = 3) -> List[Path]:
        """
        Get random sample images from the dataset.
        
        Args:
            n_samples: Number of random samples to return
            
        Returns:
            List of randomly selected image paths
        """
        if not self.image_paths:
            self.load_dataset()
        
        if len(self.image_paths) < n_samples:
            print(f"Warning: Dataset has only {len(self.image_paths)} images, returning all")
            return self.image_paths
        
        np.random.seed(42)  # For reproducible results
        selected_indices = np.random.choice(len(self.image_paths), n_samples, replace=False)
        samples = [self.image_paths[i] for i in selected_indices]
        
        if Config.DEBUG:
            print(f"[DEBUG] Selected {n_samples} random samples:")
            for i, sample in enumerate(samples):
                print(f"  {i+1}. {sample}")
        
        return samples
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            if Config.DEBUG:
                print(f"[DEBUG] Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
