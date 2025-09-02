# PyFeats Integration Guide

This guide explains how to use the new PyFeats integration for advanced medical image feature extraction.

## Overview

The project now supports two processing engines:
1. **Custom Algorithms** (default) - Original implementation with 9 advanced algorithms
2. **PyFeats Library** - Industry-standard feature extraction library with 12+ methods

## Installation

1. Install PyFeats and additional dependencies:
```bash
pip install pyfeats flask flask-cors
```

2. Or install all requirements:
```bash
pip install -r requirements.txt
```

## Command Line Usage

### Using Custom Algorithms (Default)
```bash
# Regular usage - uses custom algorithms
python main.py --dataset ./datasets/chest_xray --samples 5

# Explicitly specify no PyFeats
python main.py --dataset ./datasets/chest_xray --samples 3 --no-gpu
```

### Using PyFeats Library
```bash
# Use PyFeats instead of custom algorithms
python main.py --dataset ./datasets/chest_xray --samples 5 --pyfeats

# Use PyFeats with specific options
python main.py --dataset ./datasets/chest_xray --samples 3 --pyfeats --no-gpu
```

### List Available Algorithms
```bash
# List custom algorithms
python main.py --algorithms

# List PyFeats methods
python main.py --algorithms --pyfeats
```

## GUI Usage

### Starting the GUI Server
```bash
python pyfeats_server.py
```

Then open your browser and go to: `http://localhost:5000`

### GUI Features
- **Real-time Processing**: Upload images and see results immediately
- **12 PyFeats Methods**: Choose from various feature extraction techniques
- **Interactive Visualization**: Compare original and processed images
- **Feature Information**: See the number of extracted features
- **Download Results**: Save processed images locally

### Available PyFeats Methods in GUI

1. **First Order Statistics (FOS)**: Basic intensity distribution features
2. **Gray Level Co-occurrence Matrix (GLCM)**: Texture based on spatial relationships
3. **Gray Level Run Length Matrix (GLRLM)**: Features from pixel runs
4. **Gray Level Size Zone Matrix (GLSZM)**: Connected region analysis
5. **Gray Level Dependence Matrix (GLDM)**: Dependency-based features
6. **Neighborhood Gray Tone Difference Matrix (NGTDM)**: Neighborhood analysis
7. **Local Binary Pattern (LBP)**: Local texture patterns
8. **Multi-scale LBP (MLBP)**: LBP at multiple scales
9. **Fractal Dimension**: Fractal-based texture characterization
10. **Hu Moments**: Shape invariant moments
11. **Zernike Moments**: Orthogonal shape descriptors
12. **Histogram of Oriented Gradients (HOG)**: Gradient-based features

## PyFeats vs Custom Algorithms

| Aspect | Custom Algorithms | PyFeats Library |
|--------|------------------|-----------------|
| **Methods** | 9 specialized algorithms | 12+ standardized methods |
| **Performance** | Optimized for medical images | General-purpose, widely tested |
| **Customization** | Highly customizable | Standard implementations |
| **Maintenance** | Custom maintenance required | Community maintained |
| **Standards** | Research-focused | Industry standard |
| **Speed** | Optimized for specific use cases | General optimization |

## Examples

### Example 1: Using PyFeats from Command Line
```bash
# Process chest X-rays with PyFeats
python main.py --dataset ./datasets/chest_xray --samples 5 --pyfeats

# Compare with custom algorithms
python main.py --dataset ./datasets/chest_xray --samples 5
```

### Example 2: GUI Workflow
1. Start the server: `python pyfeats_server.py`
2. Open browser: `http://localhost:5000`
3. Upload a medical image (X-ray, CT scan, etc.)
4. Select a filter method (e.g., GLCM, LBP, etc.)
5. Click "Process Image"
6. View results and download if needed

### Example 3: Feature Extraction Comparison
```bash
# Extract features using both methods for comparison
python main.py --dataset ./datasets/lung_scans --samples 3
python main.py --dataset ./datasets/lung_scans --samples 3 --pyfeats

# Check the outputs/ directory for results
```

## File Structure

```
image-processing/
├── src/
│   └── processors/
│       ├── image_processor.py      # Original custom algorithms
│       └── pyfeats_processor.py    # New PyFeats integration
├── pyfeats_gui.html                # Web-based GUI
├── pyfeats_server.py               # Flask backend for GUI
├── main.py                         # Updated with PyFeats support
└── requirements.txt                # Updated dependencies
```

## Troubleshooting

### PyFeats Installation Issues
```bash
# If pyfeats installation fails, try:
pip install --upgrade pip setuptools wheel
pip install pyfeats

# Or install from conda-forge:
conda install -c conda-forge pyfeats
```

### GUI Server Issues
```bash
# Make sure Flask dependencies are installed:
pip install Flask Flask-CORS

# Check if port 5000 is available:
netstat -ano | findstr :5000
```

### Import Errors
```bash
# Ensure you're running from the project root:
cd /path/to/image-processing
python main.py --help

# Check Python path:
python -c "import sys; print(sys.path)"
```

## Advanced Configuration

### Custom Parameters for PyFeats Methods
You can modify the PyFeats processor to use custom parameters:

```python
# In pyfeats_processor.py, modify method calls:
def _extract_glcm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
    features, labels = pyfeats.glcm_features(
        gray, 
        ignore_zeros=kwargs.get('ignore_zeros', True),
        distances=kwargs.get('distances', [1]),
        angles=kwargs.get('angles', [0, 45, 90, 135])
    )
```

### GUI Customization
The GUI can be customized by modifying `pyfeats_gui.html`:
- Add new filter methods
- Change styling and layout
- Add parameter controls for each method
- Implement batch processing

## Performance Notes

1. **PyFeats Methods**: Generally faster for standard feature extraction
2. **Custom Algorithms**: Better optimized for specific medical imaging tasks
3. **GUI Processing**: Real-time for small-to-medium images (< 1MB)
4. **Batch Processing**: Use command line for large datasets

## Support

- For PyFeats-specific issues: Check [PyFeats documentation](https://pypi.org/project/pyfeats/)
- For custom algorithm issues: Check the original processor implementation
- For GUI issues: Check Flask/browser console for errors
