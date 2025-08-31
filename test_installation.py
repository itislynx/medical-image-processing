"""
Test script to verify the installation and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    # Test optional imports
    optional_imports = {
        'cv2': 'OpenCV',
        'scipy': 'SciPy',
        'skimage': 'scikit-image',
        'pywt': 'PyWavelets',
        'seaborn': 'Seaborn'
    }
    
    missing_optional = []
    for module, name in optional_imports.items():
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError:
            print(f"! {name} not available (optional)")
            missing_optional.append(name)
    
    if missing_optional:
        print(f"\\nMissing optional dependencies: {', '.join(missing_optional)}")
        print("Run: pip install -r requirements.txt")
    
    return len(missing_optional) < len(optional_imports)  # Allow some missing


def test_project_structure():
    """Test if the project structure is correct."""
    print("\\nTesting project structure...")
    
    required_files = [
        'main.py',
        'config.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/processors/__init__.py',
        'src/processors/image_processor.py',
        'src/utils/__init__.py',
        'src/utils/dataset_loader.py',
        'src/utils/visualization.py'
    ]
    
    required_dirs = [
        'src',
        'src/processors',
        'src/utils',
        'datasets',
        'outputs'
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\\nMissing files: {missing_files}")
        print(f"Missing directories: {missing_dirs}")
        return False
    
    return True


def test_configuration():
    """Test if configuration loads correctly."""
    print("\\nTesting configuration...")
    
    try:
        from config import Config
        Config.print_config()
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\\nTesting GPU availability...")
    
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ CUDA available with {device_count} device(s)")
        return True
    except ImportError:
        print("! CuPy not available - GPU acceleration disabled")
        return False
    except Exception as e:
        print(f"! GPU test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MEDICAL IMAGE PROCESSING TOOL - INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Project Structure Test", test_project_structure),
        ("Configuration Test", test_configuration),
        ("GPU Availability Test", test_gpu_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n{'-' * 40}")
        print(f"Running {test_name}...")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\\n🎉 All tests passed! Installation is complete.")
        print("\\nYou can now run:")
        print("  python demo.py          # Run demonstration")
        print("  python main.py --help   # See all options")
    elif passed >= len(results) - 1:  # Allow GPU test to fail
        print("\\n⚠️  Installation mostly complete.")
        print("Some optional features may not be available.")
        print("\\nYou can still run:")
        print("  python demo.py          # Run demonstration")
        print("  python main.py --help   # See all options")
    else:
        print("\\n❌ Installation incomplete.")
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    run_all_tests()
