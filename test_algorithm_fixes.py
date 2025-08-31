#!/usr/bin/env python3
"""
Test script to verify the algorithm fixes
"""

import numpy as np
import cv2
from src.processors.image_processor import ImageProcessor
from config import Config

def test_algorithm_fixes():
    """Test the fixed algorithms with a simple synthetic image."""
    
    # Create a simple test image
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    print("Testing algorithm fixes...")
    print("=" * 50)
    
    # Initialize processor
    processor = ImageProcessor(use_gpu=False)
    
    # Test the two fixed algorithms specifically
    algorithms_to_test = {
        'Local Directional Pattern': processor.local_directional_pattern,
        'Gray Level Size Zone Matrix': processor.gray_level_size_zone_matrix
    }
    
    for name, algorithm in algorithms_to_test.items():
        print(f"\nTesting: {name}")
        try:
            result = algorithm(test_image)
            print(f"✓ SUCCESS: {len(result['features'])} features extracted")
            print(f"  - Features shape: {result['features'].shape}")
            print(f"  - Processed image shape: {result['processed_image'].shape}")
            print(f"  - Description: {result['description']}")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    test_algorithm_fixes()
