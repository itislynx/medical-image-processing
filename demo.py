"""
Demo Script for Medical Image Processing Tool
Demonstrates the usage of the tool with sample data
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from main import MedicalImageProcessor
from config import Config


def create_sample_images():
    """Create sample medical-like images for demonstration."""
    print("Creating sample images for demonstration...")
    
    sample_dir = Config.DATASETS_DIR / "demo_samples"
    sample_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different classes
    (sample_dir / "normal").mkdir(exist_ok=True)
    (sample_dir / "abnormal").mkdir(exist_ok=True)
    
    # Generate synthetic medical-like images
    for i in range(3):
        # Normal samples - smoother textures
        normal_img = create_medical_like_image(texture_intensity=0.3, noise_level=0.1)
        cv2.imwrite(str(sample_dir / "normal" / f"normal_{i+1}.png"), normal_img)
        
        # Abnormal samples - more textured/noisy
        abnormal_img = create_medical_like_image(texture_intensity=0.7, noise_level=0.3)
        cv2.imwrite(str(sample_dir / "abnormal" / f"abnormal_{i+1}.png"), abnormal_img)
    
    print(f"Sample images created in: {sample_dir}")
    return sample_dir


def create_medical_like_image(size=(256, 256), texture_intensity=0.5, noise_level=0.2):
    """
    Create a synthetic medical-like image with texture and noise.
    
    Args:
        size: Image dimensions
        texture_intensity: Intensity of texture patterns
        noise_level: Amount of noise to add
    
    Returns:
        Generated image as numpy array
    """
    h, w = size
    
    # Create base image with gradients (simulating organ structures)
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    # Base structure with circular and linear gradients
    base = 128 + 50 * np.sin(np.pi * x) * np.cos(np.pi * y)
    circular = 30 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
    
    # Add texture patterns
    texture = texture_intensity * 50 * np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)
    texture += texture_intensity * 30 * np.sin(20 * np.pi * x) * np.cos(15 * np.pi * y)
    
    # Combine components
    image = base + circular + texture
    
    # Add noise
    noise = noise_level * 50 * np.random.randn(h, w)
    image += noise
    
    # Normalize to 0-255 range
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def run_demo():
    """Run the complete demonstration."""
    print("=" * 80)
    print("MEDICAL IMAGE PROCESSING TOOL - DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create sample data
        sample_dir = create_sample_images()
        
        # Initialize processor
        processor = MedicalImageProcessor(use_gpu=True)
        
        # Show available algorithms
        print("\\n" + "=" * 60)
        print("AVAILABLE ALGORITHMS")
        print("=" * 60)
        algorithms = processor.get_algorithm_info()
        for i, (name, description) in enumerate(algorithms.items(), 1):
            print(f"{i:2d}. {name.replace('_', ' ').title()}")
            print(f"    {description}")
            print()
        
        # Run the analysis
        print("=" * 60)
        print("RUNNING ANALYSIS")
        print("=" * 60)
        
        processor.run_full_analysis(str(sample_dir), n_samples=3)
        
        print("\\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        # Show what was generated
        print("\\nGenerated Files:")
        print(f"• Sample dataset: {sample_dir}")
        print(f"• Output directory: {Config.OUTPUTS_DIR}")
        
        # List output files
        output_files = []
        for output_subdir in [Config.PROCESSED_IMAGES_DIR, Config.COMPARISONS_DIR, 
                             Config.GRAPHS_DIR, Config.TABLES_DIR, Config.INSIGHTS_DIR]:
            if output_subdir.exists():
                files = list(output_subdir.glob("*"))
                if files:
                    output_files.extend(files)
        
        print(f"\\nTotal output files generated: {len(output_files)}")
        
        # Show some specific files
        important_files = [
            Config.OUTPUTS_DIR / 'comprehensive_report.html',
            Config.TABLES_DIR / 'algorithm_comparison_table.csv',
            Config.GRAPHS_DIR / 'algorithm_performance_heatmap.png'
        ]
        
        print("\\nKey output files:")
        for file_path in important_files:
            if file_path.exists():
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path} (not generated)")
        
        print("\\n" + "=" * 80)
        print("To view results, open the comprehensive report:")
        print(f"file:///{Config.OUTPUTS_DIR / 'comprehensive_report.html'}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\\nDemo failed with error: {str(e)}")
        if Config.DEBUG:
            import traceback
            traceback.print_exc()


def show_usage_examples():
    """Show various usage examples."""
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        {
            'title': 'Basic Usage - Process 3 samples from a folder',
            'command': 'python main.py --dataset ./datasets/lung_xray --samples 3'
        },
        {
            'title': 'Process from ZIP archive',
            'command': 'python main.py --dataset ./datasets/medical_images.zip --samples 5'
        },
        {
            'title': 'Run without GPU acceleration',
            'command': 'python main.py --dataset ./datasets/ct_scans --samples 3 --no-gpu'
        },
        {
            'title': 'Process many samples',
            'command': 'python main.py --dataset ./datasets/xray_images/ --samples 10'
        },
        {
            'title': 'List available algorithms',
            'command': 'python main.py --algorithms'
        },
        {
            'title': 'Run demo with synthetic data',
            'command': 'python demo.py'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   {example['command']}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo script for Medical Image Processing Tool')
    parser.add_argument('--examples', action='store_true', help='Show usage examples')
    args = parser.parse_args()
    
    if args.examples:
        show_usage_examples()
    else:
        run_demo()
