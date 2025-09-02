"""
PyFeats Integration for Medical Image Analysis
Implements PyFeats library for comprehensive feature extraction
"""

import numpy as np
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfeats
try:
    import pyfeats
    PYFEATS_AVAILABLE = True
except ImportError:
    PYFEATS_AVAILABLE = False
    print("Warning: PyFeats not installed. Install with: pip install pyfeats")

from typing import Dict, List, Optional, Tuple
from config import Config


class PyFeatsProcessor:
    """
    PyFeats-based image processor implementing comprehensive feature extraction
    using the PyFeats library for texture analysis and feature extraction.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize PyFeats processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration (not applicable for PyFeats)
        """
        if not PYFEATS_AVAILABLE:
            raise ImportError("PyFeats library is not installed. Please install it using: pip install pyfeats")
        
        self.use_gpu = use_gpu
        
        # Available PyFeats methods
        self.available_methods = {
            # First Order Statistics (FOS)
            'fos': {
                'name': 'First Order Statistics',
                'function': self._extract_fos,
                'description': 'Statistical measures of pixel intensity distribution'
            },
            
            # Gray Level Co-occurrence Matrix (GLCM)
            'glcm': {
                'name': 'Gray Level Co-occurrence Matrix',
                'function': self._extract_glcm,
                'description': 'Texture analysis based on spatial relationships between pixels'
            },
            
            # Gray Level Run Length Matrix (GLRLM)
            'glrlm': {
                'name': 'Gray Level Run Length Matrix',
                'function': self._extract_glrlm,
                'description': 'Features based on gray level runs in different directions'
            },
            
            # Gray Level Size Zone Matrix (GLSZM)
            'glszm': {
                'name': 'Gray Level Size Zone Matrix',
                'function': self._extract_glszm,
                'description': 'Features based on connected regions of similar intensity'
            },
            
            # Gray Level Dependence Matrix (GLDM)
            'gldm': {
                'name': 'Gray Level Dependence Matrix',
                'function': self._extract_gldm,
                'description': 'Features based on gray level dependencies'
            },
            
            # Neighborhood Gray Tone Difference Matrix (NGTDM)
            'ngtdm': {
                'name': 'Neighborhood Gray Tone Difference Matrix',
                'function': self._extract_ngtdm,
                'description': 'Texture features based on neighborhood differences'
            },
            
            # Local Binary Pattern (LBP)
            'lbp': {
                'name': 'Local Binary Pattern',
                'function': self._extract_lbp,
                'description': 'Local texture patterns based on binary comparisons'
            },
            
            # Multi-scale Local Binary Pattern (MLBP)
            'mlbp': {
                'name': 'Multi-scale Local Binary Pattern',
                'function': self._extract_mlbp,
                'description': 'LBP features at multiple scales'
            },
            
            # Fractal Dimension
            'fractal': {
                'name': 'Fractal Dimension',
                'function': self._extract_fractal,
                'description': 'Fractal-based texture characterization'
            },
            
            # Hu Moments
            'hu_moments': {
                'name': 'Hu Moments',
                'function': self._extract_hu_moments,
                'description': 'Invariant moments for shape analysis'
            },
            
            # Zernike Moments
            'zernike': {
                'name': 'Zernike Moments',
                'function': self._extract_zernike,
                'description': 'Orthogonal moments for shape description'
            },
            
            # Histogram of Oriented Gradients (HOG)
            'hog': {
                'name': 'Histogram of Oriented Gradients',
                'function': self._extract_hog,
                'description': 'Gradient-based descriptors for object detection'
            }
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] PyFeats processor initialized with {len(self.available_methods)} methods")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for PyFeats processing.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-255
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return gray
    
    def _extract_fos(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract First Order Statistics features."""
        gray = self.preprocess_image(image)
        
        try:
            features, labels = pyfeats.fos(gray, gray)
            
            # Create visualization (enhanced histogram equalization)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            fos_enhanced = clahe.apply(gray)
            
            # Add some statistical visualization overlay
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Create intensity bands visualization
            low_intensity = gray < (mean_val - std_val)
            high_intensity = gray > (mean_val + std_val)
            
            # Combine into a colored representation
            fos_colored = fos_enhanced.copy()
            fos_colored[low_intensity] = fos_colored[low_intensity] * 0.7  # Darken low intensity
            fos_colored[high_intensity] = np.minimum(255, fos_colored[high_intensity] * 1.3)  # Brighten high intensity
            
            return {
                'processed_image': fos_colored.astype(np.uint8),
                'features': np.array(features),
                'feature_names': labels,
                'description': 'First Order Statistics with intensity distribution visualization'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] FOS extraction error: {e}")
            return self._create_error_result(gray, "FOS extraction failed")
    
    def _extract_glcm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract GLCM features."""
        gray = self.preprocess_image(image)
        
        try:
            # Create a binary mask (all pixels are valid)
            mask = np.ones_like(gray, dtype=np.uint8)
            
            # PyFeats GLCM features function signature
            features, labels = pyfeats.glcm_features(gray, mask)
            
            # Create a meaningful visualization - enhance texture patterns
            # Apply edge enhancement to show texture features visually
            from scipy import ndimage
            
            # Create a texture-enhanced visualization
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            texture_enhanced = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize and enhance contrast
            texture_enhanced = cv2.normalize(texture_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            texture_enhanced = cv2.equalizeHist(texture_enhanced)
            
            return {
                'processed_image': texture_enhanced,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'GLCM features with texture enhancement visualization'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] GLCM extraction error: {e}")
            return self._create_error_result(gray, "GLCM extraction failed")
    
    def _extract_glrlm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract GLRLM features."""
        gray = self.preprocess_image(image)
        
        try:
            # Create a binary mask (all pixels are valid)
            mask = np.ones_like(gray, dtype=np.uint8)
            
            # PyFeats GLRLM features function signature
            features, labels = pyfeats.glrlm_features(gray, mask)
            
            # Create run-length pattern visualization
            # Quantize image for run-length analysis
            quantized = (gray // 32) * 32  # 8 levels
            
            # Apply a run-length enhancement filter
            kernel = np.array([[-1, 2, -1], [2, 5, 2], [-1, 2, -1]], dtype=np.float32)
            run_enhanced = cv2.filter2D(quantized.astype(np.float32), -1, kernel)
            run_enhanced = np.clip(run_enhanced, 0, 255).astype(np.uint8)
            
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            run_enhanced = clahe.apply(run_enhanced)
            
            return {
                'processed_image': run_enhanced,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'GLRLM features with run-length pattern visualization'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] GLRLM extraction error: {e}")
            return self._create_error_result(gray, "GLRLM extraction failed")
    
    def _extract_glszm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract GLSZM features with performance optimization."""
        gray = self.preprocess_image(image)
        
        try:
            # Performance optimization: Resize large images
            original_shape = gray.shape
            max_size = 512  # Maximum image size for processing
            
            if max(gray.shape) > max_size:
                # Calculate resize ratio
                ratio = max_size / max(gray.shape)
                new_width = int(gray.shape[1] * ratio)
                new_height = int(gray.shape[0] * ratio)
                
                # Resize for processing
                gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
                if Config.DEBUG:
                    print(f"[DEBUG] GLSZM: Resized from {original_shape} to {gray_resized.shape} for performance")
            else:
                gray_resized = gray.copy()
            
            # Create a binary mask (all pixels are valid)
            mask = np.ones_like(gray_resized, dtype=np.uint8)
            
            # PyFeats GLSZM features function signature
            features, labels = pyfeats.glszm_features(gray_resized, mask)
            
            # Create zone-based visualization
            # Quantize image to fewer levels for zone visualization
            n_levels = 8
            quantized = (gray_resized // (256 // n_levels)) * (256 // n_levels)
            
            # Apply morphological operations to enhance zones
            kernel = np.ones((3,3), np.uint8)
            zones_enhanced = cv2.morphologyEx(quantized, cv2.MORPH_CLOSE, kernel)
            zones_enhanced = cv2.morphologyEx(zones_enhanced, cv2.MORPH_OPEN, kernel)
            
            # Apply color mapping for better visualization
            zones_colored = cv2.applyColorMap(zones_enhanced, cv2.COLORMAP_JET)
            zones_gray = cv2.cvtColor(zones_colored, cv2.COLOR_BGR2GRAY)
            
            # Resize back to original size if needed
            if max(original_shape) > max_size:
                zones_final = cv2.resize(zones_gray, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                zones_final = zones_gray
            
            return {
                'processed_image': zones_final,
                'features': np.array(features),
                'feature_names': labels,
                'description': f'GLSZM features with zone visualization (processed at {gray_resized.shape})'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] GLSZM extraction error: {e}")
            return self._create_error_result(gray, "GLSZM extraction failed")
    
    def _extract_gldm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract GLDM features."""
        gray = self.preprocess_image(image)
        
        try:
            # Check if GLDM is available in this version of PyFeats
            if hasattr(pyfeats, 'gldm_features'):
                # Create a binary mask (all pixels are valid)
                mask = np.ones_like(gray, dtype=np.uint8)
                
                features, labels = pyfeats.gldm_features(gray, mask)
                
                return {
                    'processed_image': gray,
                    'features': np.array(features),
                    'feature_names': labels,
                    'description': 'Gray Level Dependence Matrix (GLDM) features'
                }
            else:
                # Fallback to a simple statistical analysis
                features = [
                    np.mean(gray),
                    np.std(gray),
                    np.var(gray),
                    np.max(gray),
                    np.min(gray)
                ]
                labels = ['gldm_mean', 'gldm_std', 'gldm_var', 'gldm_max', 'gldm_min']
                
                return {
                    'processed_image': gray,
                    'features': np.array(features),
                    'feature_names': labels,
                    'description': 'GLDM features (fallback to basic statistics)'
                }
                
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] GLDM extraction error: {e}")
            return self._create_error_result(gray, "GLDM extraction failed")
    
    def _extract_ngtdm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract NGTDM features."""
        gray = self.preprocess_image(image)
        
        try:
            # Create a binary mask (all pixels are valid)
            mask = np.ones_like(gray, dtype=np.uint8)
            
            # PyFeats NGTDM features function signature
            features, labels = pyfeats.ngtdm_features(gray, mask)
            
            # Create neighborhood difference visualization
            # Apply Laplacian filter to highlight neighborhood differences
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            
            # Normalize and enhance
            ngtdm_vis = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply adaptive threshold to highlight differences
            adaptive_thresh = cv2.adaptiveThreshold(ngtdm_vis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Combine original with edge information
            combined = cv2.addWeighted(gray, 0.7, adaptive_thresh, 0.3, 0)
            
            return {
                'processed_image': combined,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'NGTDM features with neighborhood difference visualization'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] NGTDM extraction error: {e}")
            return self._create_error_result(gray, "NGTDM extraction failed")
    
    def _extract_lbp(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract LBP features."""
        gray = self.preprocess_image(image)
        
        try:
            # Create a binary mask (all pixels are valid)
            mask = np.ones_like(gray, dtype=np.uint8)
            
            # PyFeats LBP features function signature
            features, labels = pyfeats.lbp_features(gray, mask)
            
            # Create LBP visualization using PyFeats LBP function
            try:
                lbp_image = pyfeats.lbp(gray, 8, 1)
                lbp_image = cv2.normalize(lbp_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            except:
                # Fallback to original image if LBP visualization fails
                lbp_image = gray
            
            return {
                'processed_image': lbp_image,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'Local Binary Pattern (LBP) features'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] LBP extraction error: {e}")
            return self._create_error_result(gray, "LBP extraction failed")
    
    def _extract_mlbp(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract Multi-scale LBP features."""
        gray = self.preprocess_image(image)
        
        try:
            # Check if MLBP is available in this version of PyFeats
            if hasattr(pyfeats, 'mlbp_features'):
                # Create a binary mask (all pixels are valid)
                mask = np.ones_like(gray, dtype=np.uint8)
                
                features, labels = pyfeats.mlbp_features(gray, mask)
                
                # Create multi-scale LBP visualization
                try:
                    mlbp_image = pyfeats.mlbp(gray, [1, 2, 3], [8, 16, 24])
                    mlbp_image = cv2.normalize(mlbp_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                except:
                    mlbp_image = gray
                
                return {
                    'processed_image': mlbp_image,
                    'features': np.array(features),
                    'feature_names': labels,
                    'description': 'Multi-scale Local Binary Pattern (MLBP) features'
                }
            else:
                # Fallback to regular LBP
                from skimage.feature import local_binary_pattern
                
                # Multi-scale LBP using different radii
                radii = [1, 2, 3]
                n_points = [8, 16, 24]
                
                features = []
                labels = []
                combined_lbp = np.zeros_like(gray, dtype=np.float32)
                
                for i, (radius, points) in enumerate(zip(radii, n_points)):
                    lbp = local_binary_pattern(gray, points, radius, method='uniform')
                    combined_lbp += lbp / len(radii)
                    
                    # Extract histogram features
                    hist, _ = np.histogram(lbp, bins=points+2, range=(0, points+2))
                    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                    
                    features.extend(hist[:10])  # Use first 10 bins
                    labels.extend([f'mlbp_r{radius}_bin{j}' for j in range(10)])
                
                combined_lbp = cv2.normalize(combined_lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                return {
                    'processed_image': combined_lbp,
                    'features': np.array(features),
                    'feature_names': labels,
                    'description': 'Multi-scale LBP features (fallback implementation)'
                }
                
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] MLBP extraction error: {e}")
            return self._create_error_result(gray, "MLBP extraction failed")
    
    def _extract_fractal(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract Fractal Dimension features."""
        gray = self.preprocess_image(image)
        
        try:
            # Check if differential box counting is available in PyFeats
            if hasattr(pyfeats, 'dbc_fractal_dimension'):
                fd = pyfeats.dbc_fractal_dimension(gray)
            elif hasattr(pyfeats, 'fractal_dimension'):
                # Alternative function name
                fd = pyfeats.fractal_dimension(gray)
            else:
                # Fallback to custom implementation
                fd = self._custom_fractal_dimension(gray)
            
            features = [fd] if not isinstance(fd, (list, np.ndarray)) else fd
            if isinstance(fd, (list, np.ndarray)) and len(fd) > 1:
                labels = [f'fractal_dim_{i+1}' for i in range(len(features))]
            else:
                labels = ['fractal_dimension']
            
            return {
                'processed_image': gray,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'Fractal Dimension using Box Counting method'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] Fractal extraction error: {e}")
            return self._create_error_result(gray, "Fractal extraction failed")
    
    def _custom_fractal_dimension(self, image: np.ndarray) -> float:
        """Custom fractal dimension calculation using box-counting method."""
        try:
            # Convert to binary image
            threshold = np.mean(image)
            binary = (image > threshold).astype(np.uint8)
            
            # Box-counting method
            sizes = [2, 4, 8, 16, 32]
            counts = []
            
            for size in sizes:
                if size >= min(binary.shape):
                    break
                    
                count = 0
                for i in range(0, binary.shape[0], size):
                    for j in range(0, binary.shape[1], size):
                        box = binary[i:i+size, j:j+size]
                        if np.any(box):
                            count += 1
                
                if count > 0:
                    counts.append(count)
            
            if len(counts) < 2:
                return 1.5  # Default fractal dimension
            
            # Linear regression in log-log space
            log_sizes = np.log(sizes[:len(counts)])
            log_counts = np.log(counts)
            
            # Fractal dimension is negative slope
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -slope
            
            return max(1.0, min(3.0, fractal_dim))  # Clamp to reasonable range
            
        except:
            return 1.5  # Default value on error
    
    def _extract_hu_moments(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract Hu Moments features."""
        gray = self.preprocess_image(image)
        
        try:
            # Calculate Hu moments
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Apply log transform to handle large values
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            labels = [f'hu_moment_{i+1}' for i in range(len(hu_moments))]
            
            # Create shape-enhanced visualization
            # Apply edge detection and morphological operations to highlight shapes
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to make them more visible
            kernel = np.ones((2,2), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine original with enhanced edges
            shape_enhanced = cv2.addWeighted(gray, 0.8, edges_dilated, 0.4, 0)
            
            # Apply contrast enhancement
            shape_enhanced = cv2.equalizeHist(shape_enhanced)
            
            return {
                'processed_image': shape_enhanced,
                'features': hu_moments,
                'feature_names': labels,
                'description': 'Hu Moments with shape boundary enhancement'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] Hu Moments extraction error: {e}")
            return self._create_error_result(gray, "Hu Moments extraction failed")
    
    def _extract_zernike(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract Zernike Moments features."""
        gray = self.preprocess_image(image)
        
        try:
            # Check if Zernike moments are available in PyFeats
            if hasattr(pyfeats, 'zernike_moments'):
                radius = kwargs.get('radius', min(gray.shape) // 4)
                features, labels = pyfeats.zernike_moments(gray, radius)
            elif hasattr(pyfeats, 'zernike'):
                # Alternative function name
                radius = kwargs.get('radius', min(gray.shape) // 4)
                features, labels = pyfeats.zernike(gray, radius)
            else:
                # Fallback to custom implementation using cv2 moments
                features = self._custom_zernike_moments(gray)
                labels = [f'zernike_moment_{i+1}' for i in range(len(features))]
            
            # Create radial pattern visualization for Zernike moments
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            
            # Create radial coordinate system
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Create concentric circles visualization
            max_radius = min(center_x, center_y)
            normalized_r = (r / max_radius) * 255
            normalized_r = np.clip(normalized_r, 0, 255).astype(np.uint8)
            
            # Apply radial enhancement based on intensity
            radial_enhanced = gray.copy().astype(np.float32)
            radial_mask = r <= max_radius
            
            # Enhance based on radial distance
            radial_enhanced[radial_mask] = radial_enhanced[radial_mask] * (1 + 0.3 * np.sin(normalized_r[radial_mask] * np.pi / 128))
            radial_enhanced = np.clip(radial_enhanced, 0, 255).astype(np.uint8)
            
            # Apply circular gradient overlay
            circle_overlay = cv2.addWeighted(radial_enhanced, 0.8, normalized_r, 0.2, 0)
            
            return {
                'processed_image': circle_overlay,
                'features': np.array(features),
                'feature_names': labels,
                'description': 'Zernike Moments with radial pattern visualization'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] Zernike extraction error: {e}")
            return self._create_error_result(gray, "Zernike extraction failed")
    
    def _custom_zernike_moments(self, image: np.ndarray) -> List[float]:
        """Custom Zernike moments calculation using regular moments as approximation."""
        try:
            # Calculate regular image moments
            moments = cv2.moments(image)
            
            # Extract central moments (approximation of Zernike moments)
            features = []
            
            # Spatial moments
            if moments['m00'] != 0:
                # Normalized central moments
                mu20 = moments['mu20'] / (moments['m00'] ** (2 + 2/2))
                mu02 = moments['mu02'] / (moments['m00'] ** (2 + 2/2))
                mu11 = moments['mu11'] / (moments['m00'] ** (2 + 2/2))
                mu30 = moments['mu30'] / (moments['m00'] ** (3 + 3/2))
                mu03 = moments['mu03'] / (moments['m00'] ** (3 + 3/2))
                mu21 = moments['mu21'] / (moments['m00'] ** (3 + 3/2))
                mu12 = moments['mu12'] / (moments['m00'] ** (3 + 3/2))
                
                features = [mu20, mu02, mu11, mu30, mu03, mu21, mu12]
            else:
                features = [0.0] * 7
            
            return features
            
        except:
            return [0.0] * 7  # Return zeros on error
    
    def _extract_hog(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract HOG features."""
        gray = self.preprocess_image(image)
        
        try:
            from skimage.feature import hog
            
            # HOG parameters
            orientations = kwargs.get('orientations', 9)
            pixels_per_cell = kwargs.get('pixels_per_cell', (8, 8))
            cells_per_block = kwargs.get('cells_per_block', (2, 2))
            
            # Extract HOG features
            features, hog_image = hog(
                gray,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=True,
                block_norm='L2-Hys'
            )
            
            # Normalize HOG image for visualization
            hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            labels = [f'hog_feature_{i}' for i in range(len(features))]
            
            return {
                'processed_image': hog_image,
                'features': features,
                'feature_names': labels,
                'description': f'HOG features with {orientations} orientations'
            }
        except Exception as e:
            if Config.DEBUG:
                print(f"[DEBUG] HOG extraction error: {e}")
            return self._create_error_result(gray, "HOG extraction failed")
    
    def _create_error_result(self, image: np.ndarray, error_msg: str) -> Dict:
        """Create a default result when feature extraction fails."""
        return {
            'processed_image': image,
            'features': np.array([]),
            'feature_names': [],
            'description': error_msg
        }
    
    def extract_features(self, image: np.ndarray, method: str, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features using specified PyFeats method.
        
        Args:
            image: Input image
            method: Feature extraction method name
            **kwargs: Additional parameters for the method
            
        Returns:
            Dictionary containing extracted features and processed image
        """
        if method not in self.available_methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(self.available_methods.keys())}")
        
        method_info = self.available_methods[method]
        
        if Config.DEBUG:
            print(f"[DEBUG] Extracting features using {method_info['name']}")
        
        return method_info['function'](image, **kwargs)
    
    def process_all_methods(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Apply all available PyFeats methods to an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing results from all methods
        """
        results = {}
        
        print(f"Processing image with {len(self.available_methods)} PyFeats methods...")
        
        for method_name, method_info in self.available_methods.items():
            try:
                if Config.DEBUG:
                    print(f"[DEBUG] Processing: {method_info['name']}")
                
                result = self.extract_features(image, method_name)
                results[method_name] = result
                
                if Config.VERBOSE:
                    feature_count = len(result['features'])
                    print(f"✓ {method_info['name']}: {feature_count} features extracted")
                    
            except Exception as e:
                print(f"✗ Error in {method_info['name']}: {str(e)}")
                if Config.DEBUG:
                    import traceback
                    traceback.print_exc()
                
                # Create dummy result to maintain consistency
                results[method_name] = self._create_error_result(
                    self.preprocess_image(image),
                    f"Error in {method_info['name']}: {str(e)}"
                )
        
        return results
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about available PyFeats methods.
        
        Returns:
            Dictionary mapping method names to descriptions
        """
        return {name: info['description'] for name, info in self.available_methods.items()}
    
    def list_methods(self):
        """Print information about all available methods."""
        print("Available PyFeats Methods:")
        print("=" * 50)
        
        for method_name, method_info in self.available_methods.items():
            print(f"• {method_info['name']} ({method_name})")
            print(f"  {method_info['description']}")
            print()
