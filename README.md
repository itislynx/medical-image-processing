# Medical Image Processing Tool

A comprehensive Python tool for analyzing medical images using state-of-the-art image processing algorithms, specifically designed for AI disease recognition projects.

## Features

### 🔬 Advanced Image Processing Algorithms
- **Gabor Filter**: Multi-orientation texture analysis
- **Local Directional Pattern (LDP)**: Directional texture features
- **Gray Level Run Length Matrix (GLRLM)**: Run-length texture analysis
- **Gray Level Co-occurrence Matrix (GLCM)**: Spatial relationship analysis
- **Gray Level Size Zone Matrix (GLSZM)**: Size zone texture features
- **Wavelet Transform**: Multi-resolution frequency analysis
- **Fast Fourier Transform (FFT)**: Frequency domain analysis
- **Segmentation-based Fractal Texture Analysis (SFTA)**: Fractal texture features
- **Local Binary Pattern + GLCM (LBGLCM)**: Combined texture analysis

### 📊 Comprehensive Analysis
- Dataset insights and statistics
- Before/after image comparisons
- Feature extraction and comparison
- Performance analysis and timing
- Correlation analysis between algorithms
- Interactive HTML reports

### 🚀 Performance Features
- GPU acceleration support (CUDA)
- Modular and extensible architecture
- Batch processing capabilities
- Memory-efficient processing
- Debug and verbose modes

### 📁 Dataset Support
- Multiple image formats (JPG, PNG, TIFF, BMP, DICOM)
- Archive support (ZIP, RAR, TAR, 7Z)
- Automatic dataset structure recognition
- Class distribution analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd image-processing
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure GPU (Optional)
If you have a CUDA-compatible GPU, ensure you have the correct CuPy version:

For CUDA 11.x:
```bash
pip install cupy-cuda11x
```

For CUDA 12.x:
```bash
pip install cupy-cuda12x
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python main.py --dataset ./datasets/lung_xray --samples 3
```

#### Advanced Usage
```bash
# Process 5 random samples from a ZIP archive
python main.py --dataset ./datasets/Chest_X-Ray_Image.zip --samples 5

# Run without GPU acceleration
python main.py --dataset ./datasets/ct_scans --samples 3 --no-gpu

# Process 10 samples from a directory
python main.py --dataset ./datasets/xray_images/ --samples 10
```

#### List Available Algorithms
```bash
python main.py --algorithms
```

### Python API

```python
from main import MedicalImageProcessor

# Initialize processor
processor = MedicalImageProcessor(use_gpu=True)

# Run complete analysis
processor.run_full_analysis('./datasets/medical_images', n_samples=5)

# Or run step by step
dataset_loader = processor.load_dataset('./datasets/medical_images')
results = processor.process_sample_images(n_samples=3)
processor.create_comprehensive_analysis(results)
```

## Project Structure

```
image-processing/
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                 # This file
│
├── src/
│   ├── __init__.py
│   ├── processors/
│   │   ├── __init__.py
│   │   └── image_processor.py # Core processing algorithms
│   └── utils/
│       ├── __init__.py
│       ├── dataset_loader.py  # Dataset loading and analysis
│       └── visualization.py   # Visualization and reporting
│
├── datasets/                 # Place your datasets here
│   ├── sample_data/
│   └── README.txt
│
└── outputs/                  # Generated outputs
    ├── processed_images/     # Individual algorithm results
    ├── comparisons/          # Before/after comparisons
    ├── insights/             # Dataset analysis
    ├── graphs/               # Visualization plots
    ├── tables/               # Analysis tables
    └── comprehensive_report.html
```

## Dataset Structure

### Supported Formats
- **Images**: JPG, JPEG, PNG, TIFF, TIF, BMP, DICOM, DCM
- **Archives**: ZIP, RAR, 7Z, TAR, GZ

### Example Dataset Structures

#### Option 1: Organized by Classes
```
datasets/lung_xray/
├── normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── covid/
    ├── image1.jpg
    └── ...
```

#### Option 2: Mixed Directory
```
datasets/medical_images/
├── scan001.png
├── scan002.png
├── xray001.jpg
└── ...
```

#### Option 3: Archive File
```
datasets/
└── medical_data.zip (containing any of the above structures)
```

## Output Files

### 1. Processed Images (`outputs/processed_images/`)
Individual results from each processing algorithm for each sample image.

### 2. Comparison Images (`outputs/comparisons/`)
Side-by-side before/after comparisons showing original, processed, and difference images.

### 3. Analysis Tables (`outputs/tables/`)
- `algorithm_comparison_table.csv`: Detailed comparison of all algorithms
- `algorithm_summary_statistics.csv`: Statistical summary of algorithm performance

### 4. Visualization Graphs (`outputs/graphs/`)
- Feature distribution plots
- Algorithm performance heatmaps
- Processing time comparisons
- Correlation matrices

### 5. Dataset Insights (`outputs/insights/`)
- `dataset_summary.csv`: Comprehensive dataset statistics
- Various distribution plots and analyses

### 6. Comprehensive Report (`outputs/comprehensive_report.html`)
Interactive HTML report with all results and visualizations.

## Algorithm Details

### 1. Gabor Filter
Multi-orientation Gabor filters for capturing texture information at different scales and orientations, particularly useful for detecting periodic patterns in medical images.

### 2. Local Directional Pattern (LDP)
Encodes directional information in local neighborhoods, effective for capturing edge and texture patterns in medical images.

### 3. Gray Level Run Length Matrix (GLRLM)
Analyzes texture based on runs of consecutive pixels with the same gray level, useful for detecting fine and coarse textures.

### 4. Gray Level Co-occurrence Matrix (GLCM)
Computes spatial relationships between pixels, extracting Haralick features for texture characterization.

### 5. Gray Level Size Zone Matrix (GLSZM)
Analyzes connected regions of similar gray levels, useful for detecting homogeneous areas in medical images.

### 6. Wavelet Transform
Provides multi-resolution analysis by decomposing images into different frequency components.

### 7. Fast Fourier Transform (FFT)
Analyzes frequency domain characteristics of medical images, useful for detecting periodic patterns and noise.

### 8. Segmentation-based Fractal Texture Analysis (SFTA)
Combines image segmentation with fractal dimension analysis for complex texture characterization.

### 9. Local Binary Pattern + GLCM (LBGLCM)
Combines LBP texture features with GLCM spatial analysis for enhanced texture description.

## Configuration

### GPU Settings
Edit `config.py` to configure GPU usage:
```python
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8
```

### Processing Parameters
Adjust algorithm parameters in `config.py`:
```python
SAMPLE_SIZE_FOR_PROCESSING = 3
MAX_IMAGES_FOR_ANALYSIS = 10000
PROCESSING_ALGORITHMS = [...]
```

### Debug Mode
Enable debug mode for detailed logging:
```python
DEBUG = True
VERBOSE = True
```

## Performance Optimization

### GPU Acceleration
- Install appropriate CuPy version for your CUDA installation
- Ensure sufficient GPU memory (recommended: 8GB+)
- Monitor GPU usage during processing

### Memory Management
- Large datasets are processed in batches
- Temporary files are automatically cleaned up
- Memory usage is optimized for different image sizes

### Processing Tips
- Use smaller sample sizes for initial testing
- Process archives directly to save disk space
- Enable verbose mode to monitor progress

## Troubleshooting

### Common Issues

#### ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

#### CUDA/CuPy Issues
```bash
# Check CUDA version
nvidia-smi

# Install appropriate CuPy version
pip install cupy-cuda11x  # or cupy-cuda12x
```

#### Memory Issues
- Reduce `SAMPLE_SIZE_FOR_PROCESSING` in config.py
- Reduce `MAX_IMAGES_FOR_ANALYSIS` in config.py
- Process smaller batches

#### Slow Processing
- Enable GPU acceleration
- Reduce number of samples
- Use smaller image sizes

### Debug Mode
Run with debug mode for detailed error information:
```python
# In config.py
DEBUG = True
VERBOSE = True
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{medical_image_processing_tool,
  title={Medical Image Processing Tool for Disease Recognition},
  author={AI Assistant},
  year={2025},
  url={https://github.com/your-repo/image-processing}
}
```

## Support

For support, please:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

## Acknowledgments

This tool implements various state-of-the-art image processing algorithms commonly used in medical image analysis and disease recognition research. Special thanks to the scientific community for developing these foundational algorithms.

---

**Note**: This tool is designed for research and educational purposes. For clinical applications, please ensure proper validation and compliance with relevant medical device regulations.
