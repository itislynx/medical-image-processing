"""
Main Image Processing Application for Disease Recognition
Comprehensive tool for analyzing medical images using state-of-the-art algorithms
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Optional

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from src.utils.dataset_loader import DatasetLoader
from src.processors.image_processor import ImageProcessor
from src.utils.visualization import Visualizer


class MedicalImageProcessor:
    """
    Main application class for medical image processing and analysis.
    Integrates dataset loading, processing, and visualization.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the medical image processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self.dataset_loader = None
        self.image_processor = ImageProcessor(use_gpu=use_gpu)
        self.visualizer = Visualizer()
        self.timing_data = {}
        
        if Config.DEBUG:
            Config.print_config()
    
    def load_dataset(self, dataset_path: str) -> DatasetLoader:
        """
        Load and analyze the dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            DatasetLoader instance
        """
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)
        
        self.dataset_loader = DatasetLoader(dataset_path, use_gpu=self.use_gpu)
        
        # Load images
        image_paths = self.dataset_loader.load_dataset()
        
        if not image_paths:
            raise ValueError("No images found in the dataset!")
        
        # Analyze dataset
        dataset_info = self.dataset_loader.analyze_dataset()
        
        # Generate insights report
        summary_df = self.dataset_loader.generate_insights_report()
        
        print(f"Dataset loaded successfully!")
        print(f"Total images: {len(image_paths)}")
        print(f"Classes found: {len(dataset_info.get('class_distribution', {}))}")
        
        return self.dataset_loader
    
    def process_sample_images(self, n_samples: int = 3) -> Dict[str, Dict]:
        """
        Process random sample images with all algorithms.
        
        Args:
            n_samples: Number of random samples to process
            
        Returns:
            Dictionary containing all processing results
        """
        if not self.dataset_loader:
            raise ValueError("Dataset must be loaded first!")
        
        print("=" * 60)
        print("PROCESSING SAMPLE IMAGES")
        print("=" * 60)
        
        # Get random samples
        sample_paths = self.dataset_loader.get_random_samples(n_samples)
        
        all_results = {}
        
        for i, sample_path in enumerate(sample_paths, 1):
            print(f"\\nProcessing sample {i}/{len(sample_paths)}: {sample_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Warning: Could not load image {sample_path}")
                    continue
                
                sample_name = sample_path.stem
                
                # Process with all algorithms
                start_time = time.time()
                sample_results = self.image_processor.process_all_algorithms(image)
                processing_time = time.time() - start_time
                
                # Store timing information
                if sample_name not in self.timing_data:
                    self.timing_data[sample_name] = {}
                self.timing_data[sample_name]['total_time'] = processing_time
                
                # Store results
                all_results[sample_name] = sample_results
                
                # Create before/after comparison
                self.visualizer.create_before_after_comparison(
                    image, sample_results, sample_name
                )
                
                # Save individual processed images
                self.visualizer.save_individual_processed_images(
                    sample_results, sample_name
                )
                
                print(f"Sample {sample_name} processed successfully in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing {sample_path}: {str(e)}")
                if Config.DEBUG:
                    import traceback
                    traceback.print_exc()
                continue
        
        return all_results
    
    def create_comprehensive_analysis(self, all_results: Dict[str, Dict]) -> None:
        """
        Create comprehensive analysis and visualizations.
        
        Args:
            all_results: Dictionary of all processing results
        """
        if not all_results:
            print("No results to analyze!")
            return
        
        print("=" * 60)
        print("CREATING COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        sample_names = list(all_results.keys())
        
        # Create comparison table
        comparison_df = self.visualizer.create_algorithm_comparison_table(
            all_results, sample_names
        )
        
        # Create feature comparison plots
        self.visualizer.create_feature_comparison_plots(
            all_results, sample_names
        )
        
        # Calculate algorithm timing averages
        algorithm_times = {}
        for sample_name, sample_results in all_results.items():
            for algorithm_name in sample_results.keys():
                if algorithm_name not in algorithm_times:
                    algorithm_times[algorithm_name] = []
                
                # Estimate time per algorithm (total time / number of algorithms)
                total_time = self.timing_data.get(sample_name, {}).get('total_time', 0)
                estimated_time = total_time / len(sample_results)
                algorithm_times[algorithm_name].append(estimated_time)
        
        # Average timing data
        avg_algorithm_times = {
            alg: np.mean(times) for alg, times in algorithm_times.items()
        }
        
        # Create timing analysis
        self.visualizer.create_processing_time_analysis(avg_algorithm_times)
        
        # Create comprehensive report
        dataset_info = self.dataset_loader.dataset_info if self.dataset_loader else {}
        
        self.visualizer.create_comprehensive_report(
            dataset_info, all_results, sample_names, avg_algorithm_times
        )
        
        print("Comprehensive analysis completed!")
    
    def run_full_analysis(self, dataset_path: str, n_samples: int = 3) -> None:
        """
        Run the complete image processing analysis pipeline.
        
        Args:
            dataset_path: Path to the dataset
            n_samples: Number of random samples to process
        """
        print("=" * 80)
        print("MEDICAL IMAGE PROCESSING TOOL")
        print("Specialized for AI Disease Recognition Projects")
        print("=" * 80)
        
        try:
            # Load dataset
            self.load_dataset(dataset_path)
            
            # Process sample images
            all_results = self.process_sample_images(n_samples)
            
            # Create comprehensive analysis
            self.create_comprehensive_analysis(all_results)
            
            print("\\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Output directory: {Config.OUTPUTS_DIR}")
            print(f"Processed {len(all_results)} sample images")
            print(f"Applied {len(Config.PROCESSING_ALGORITHMS)} processing algorithms")
            
            # Print output file locations
            print("\\nOutput files generated:")
            print(f"  • Processed images: {Config.PROCESSED_IMAGES_DIR}")
            print(f"  • Comparisons: {Config.COMPARISONS_DIR}")
            print(f"  • Analysis tables: {Config.TABLES_DIR}")
            print(f"  • Visualization graphs: {Config.GRAPHS_DIR}")
            print(f"  • Dataset insights: {Config.INSIGHTS_DIR}")
            print(f"  • Comprehensive report: {Config.OUTPUTS_DIR / 'comprehensive_report.html'}")
            
        except Exception as e:
            print(f"\\nError during analysis: {str(e)}")
            if Config.DEBUG:
                import traceback
                traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            if self.dataset_loader:
                self.dataset_loader.cleanup()
    
    def get_algorithm_info(self) -> Dict[str, str]:
        """
        Get information about available processing algorithms.
        
        Returns:
            Dictionary with algorithm names and descriptions
        """
        algorithms_info = {
            'gabor_filter': 'Multi-orientation Gabor filters for texture analysis in medical images',
            'local_directional_pattern': 'LDP captures directional texture information for disease pattern recognition',
            'gray_level_run_length_matrix': 'GLRLM analyzes texture based on run-length statistics',
            'gray_level_co_occurrence_matrix': 'GLCM computes spatial relationships between pixel intensities',
            'gray_level_size_zone_matrix': 'GLSZM analyzes size zones of similar gray-level regions',
            'wavelet_transform': 'Multi-resolution analysis using wavelets for frequency decomposition',
            'fast_fourier_transform': 'FFT provides frequency domain analysis of medical images',
            'segmentation_based_fractal_texture_analysis': 'SFTA combines segmentation with fractal analysis',
            'local_binary_pattern_glcm': 'LBGLCM combines LBP with GLCM for enhanced texture analysis'
        }
        
        return algorithms_info


def main():
    """Main function to run the image processing tool."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Medical Image Processing Tool for Disease Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --dataset ./datasets/lung_xray --samples 5
  python main.py --dataset ./datasets/ct_scans.zip --samples 3 --no-gpu
  python main.py --dataset ./datasets/ --samples 10
        '''
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to dataset (folder or archive file)'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=3,
        help='Number of random samples to process (default: 3)'
    )
    
    parser.add_argument(
        '--no-gpu', 
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--algorithms', 
        action='store_true',
        help='List available processing algorithms'
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = MedicalImageProcessor(use_gpu=not args.no_gpu)
    
    if args.algorithms:
        # Show algorithm information
        algorithms = processor.get_algorithm_info()
        print("Available Processing Algorithms:")
        print("=" * 50)
        for name, description in algorithms.items():
            print(f"• {name.replace('_', ' ').title()}")
            print(f"  {description}")
            print()
        return
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    # Run analysis
    processor.run_full_analysis(str(dataset_path), args.samples)


if __name__ == "__main__":
    main()
