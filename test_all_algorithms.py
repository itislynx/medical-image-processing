#!/usr/bin/env python3
"""
Comprehensive test of all algorithms after fixes
"""

import numpy as np
import cv2
from src.processors.image_processor import ImageProcessor
from config import Config

def test_all_algorithms():
    """Test all algorithms with a synthetic image."""
    
    # Create a more complex test image with various textures
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add some patterns to make it more realistic
    test_image[:50, :50] = 100  # Uniform region
    test_image[50:, 50:] = 200  # Another uniform region
    
    # Add some texture
    for i in range(0, 100, 10):
        test_image[i:i+5, :, :] = np.random.randint(50, 150, size=(5, 100, 3))
    
    print("Testing all algorithms after fixes...")
    print("=" * 60)
    
    # Initialize processor
    processor = ImageProcessor(use_gpu=False)
    
    try:
        # Test all algorithms at once
        results = processor.process_all_algorithms(test_image)
        
        print(f"\n✓ Successfully processed image with {len(results)} algorithms")
        print("\nAlgorithm Results Summary:")
        print("-" * 40)
        
        for name, result in results.items():
            if len(result['features']) > 0:
                print(f"✓ {name:<35} : {len(result['features']):>3} features")
            else:
                print(f"✗ {name:<35} : Error occurred")
        
        # Count successful vs failed algorithms
        successful = sum(1 for r in results.values() if len(r['features']) > 0)
        total = len(results)
        
        print(f"\nSummary: {successful}/{total} algorithms working successfully")
        
        if successful == total:
            print("🎉 ALL ALGORITHMS ARE WORKING CORRECTLY!")
        else:
            print(f"⚠️  {total - successful} algorithms still need attention")
            
    except Exception as e:
        print(f"✗ ERROR in comprehensive test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_algorithms()
