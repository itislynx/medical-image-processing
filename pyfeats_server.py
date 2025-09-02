"""
Flask Web Server for PyFeats GUI Integration
Provides REST API endpoints for real-time image processing
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
import json
import traceback
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.processors.pyfeats_processor import PyFeatsProcessor, PYFEATS_AVAILABLE
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this from the project root directory")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize PyFeats processor
if PYFEATS_AVAILABLE:
    processor = PyFeatsProcessor()
    print("✓ PyFeats processor initialized successfully")
else:
    print("✗ PyFeats not available. Please install it using: pip install pyfeats")
    sys.exit(1)


def decode_base64_image(image_data: str) -> np.ndarray:
    """
    Decode base64 image data to numpy array.
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        Image as numpy array
    """
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_np


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Serve the main GUI page."""
    gui_path = Path(__file__).parent / "pyfeats_gui.html"
    if gui_path.exists():
        with open(gui_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "GUI file not found. Please ensure pyfeats_gui.html exists in the project directory."


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Get available PyFeats methods."""
    try:
        methods = processor.get_method_info()
        return jsonify({
            'success': True,
            'methods': methods
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/process/<method>', methods=['POST'])
def process_image(method):
    """
    Process image with specified PyFeats method.
    
    Args:
        method: PyFeats method name
    """
    import time
    start_time = time.time()
    
    try:
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        print(f"[INFO] Processing with method: {method}")
        
        # Decode image
        try:
            image = decode_base64_image(data['image'])
            print(f"[INFO] Image decoded: {image.shape}")
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode image: {str(e)}'
            }), 400
        
        # Get additional parameters
        kwargs = data.get('parameters', {})
        
        # Process image
        try:
            print(f"[INFO] Starting feature extraction...")
            result = processor.extract_features(image, method, **kwargs)
            processing_time = time.time() - start_time
            print(f"[INFO] Processing completed in {processing_time:.2f} seconds")
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid method or parameters: {str(e)}'
            }), 400
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"[ERROR] Processing failed after {processing_time:.2f} seconds: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
        
        # Encode processed image
        processed_image_b64 = encode_image_to_base64(result['processed_image'])
        
        # Prepare response
        response = {
            'success': True,
            'method': method,
            'processed_image': processed_image_b64,
            'feature_count': len(result['features']),
            'features': result['features'].tolist() if len(result['features']) > 0 else [],
            'feature_names': result['feature_names'],
            'description': result['description'],
            'processing_time': round(processing_time, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[ERROR] Server error after {processing_time:.2f} seconds: {str(e)}")
        
        if Config.DEBUG:
            traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'processing_time': round(processing_time, 2)
        }), 500


@app.route('/api/process_all', methods=['POST'])
def process_all_methods():
    """Process image with all available PyFeats methods."""
    try:
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        # Decode image
        try:
            image = decode_base64_image(data['image'])
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode image: {str(e)}'
            }), 400
        
        # Process with all methods
        try:
            all_results = processor.process_all_methods(image)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
        
        # Prepare response
        response = {
            'success': True,
            'results': {}
        }
        
        for method_name, result in all_results.items():
            # Encode processed image
            processed_image_b64 = encode_image_to_base64(result['processed_image'])
            
            response['results'][method_name] = {
                'processed_image': processed_image_b64,
                'feature_count': len(result['features']),
                'features': result['features'].tolist() if len(result['features']) > 0 else [],
                'feature_names': result['feature_names'],
                'description': result['description']
            }
        
        return jsonify(response)
        
    except Exception as e:
        if Config.DEBUG:
            traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'pyfeats_available': PYFEATS_AVAILABLE,
        'methods_count': len(processor.available_methods) if PYFEATS_AVAILABLE else 0
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 PyFeats GUI Server Starting")
    print("=" * 60)
    print(f"✓ PyFeats Available: {PYFEATS_AVAILABLE}")
    
    if PYFEATS_AVAILABLE:
        methods = processor.get_method_info()
        print(f"✓ Available Methods: {len(methods)}")
        for method_name, description in methods.items():
            print(f"  • {method_name}: {description}")
    
    print("\n🌐 Server Configuration:")
    print("  • Host: localhost")
    print("  • Port: 5000")
    print("  • Debug: True")
    print("  • CORS: Enabled")
    
    print(f"\n🎯 Access the GUI at: http://localhost:5000")
    print("=" * 60)
    
    try:
        app.run(host='localhost', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        if Config.DEBUG:
            traceback.print_exc()
