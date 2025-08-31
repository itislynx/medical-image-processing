"""
Image Processing Algorithms for Medical Image Analysis
Implements state-of-the-art algorithms for disease recognition projects
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.fft import fft2, fftshift
import pywt
from skimage import filters, feature, measure, morphology, segmentation
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
try:
    import mahotas
    MAHOTAS_AVAILABLE = True
except ImportError:
    MAHOTAS_AVAILABLE = False
from typing import Dict, Tuple, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

from config import Config


class ImageProcessor:
    """
    Comprehensive image processor implementing various state-of-the-art algorithms
    for medical image analysis and disease recognition.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the image processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        self.use_gpu = use_gpu
        
        # Try to import CuPy for GPU acceleration
        self.cupy_available = False
        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.cupy_available = True
                if Config.DEBUG:
                    print("[DEBUG] GPU acceleration enabled with CuPy")
            except ImportError:
                if Config.DEBUG:
                    print("[DEBUG] CuPy not available, using CPU only")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for consistent processing.
        
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
    
    def gabor_filter(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply Gabor filter bank for texture analysis.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing filtered images and features
        """
        gray = self.preprocess_image(image)
        
        # Gabor filter parameters
        frequencies = kwargs.get('frequencies', [0.05, 0.15, 0.25])
        orientations = kwargs.get('orientations', [0, 45, 90, 135])
        
        responses = []
        filtered_images = []
        
        for freq in frequencies:
            for angle in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                          2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                filtered_images.append(filtered)
                
                # Extract features
                responses.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.max(filtered),
                    np.min(filtered)
                ])
        
        # Combine responses for visualization
        combined = np.hstack([img for img in filtered_images[:4]])  # Show first 4
        
        result = {
            'processed_image': combined,
            'features': np.array(responses),
            'feature_names': [f'gabor_{i}' for i in range(len(responses))],
            'description': f'Gabor filter with {len(frequencies)} frequencies and {len(orientations)} orientations'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] Gabor filter: {len(filtered_images)} responses, {len(responses)} features")
        
        return result
    
    def local_directional_pattern(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute Local Directional Pattern (LDP) features.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing LDP features and processed image
        """
        gray = self.preprocess_image(image)
        
        # LDP parameters
        radius = kwargs.get('radius', 1)
        n_points = kwargs.get('n_points', 8)
        
        # Directional masks (8-directional)
        masks = [
            np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),      # 0°
            np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),      # 45°
            np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),      # 90°
            np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),      # 135°
            np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),      # 180°
            np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),      # 225°
            np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),      # 270°
            np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])       # 315°
        ]
        
        ldp_responses = []
        ldp_image = np.zeros_like(gray, dtype=np.float32)  # Use float32 to avoid overflow
        
        # Apply each directional mask
        for i, mask in enumerate(masks):
            response = cv2.filter2D(gray.astype(np.float32), -1, mask.astype(np.float32))
            ldp_responses.append(response)
            ldp_image += (response > 0).astype(np.float32) * (2 ** i)
        
        # Calculate histogram features
        ldp_image_uint8 = np.clip(ldp_image, 0, 255).astype(np.uint8)
        hist, _ = np.histogram(ldp_image_uint8, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        # Additional statistical features
        features = [
            np.mean(ldp_image_uint8),
            np.std(ldp_image_uint8),
            np.max(ldp_image_uint8),
            np.min(ldp_image_uint8)
        ]
        features.extend(hist[:50])  # Use first 50 histogram bins as features
        
        result = {
            'processed_image': ldp_image_uint8,
            'features': np.array(features),
            'feature_names': ['ldp_mean', 'ldp_std', 'ldp_max', 'ldp_min'] + 
                           [f'ldp_hist_{i}' for i in range(50)],
            'description': f'Local Directional Pattern with {len(masks)} directions'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] LDP: {len(features)} features extracted")
        
        return result
    
    def gray_level_run_length_matrix(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute Gray Level Run Length Matrix (GLRLM) features.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing GLRLM features
        """
        gray = self.preprocess_image(image)
        
        # Quantize image to reduce computational complexity
        levels = kwargs.get('levels', 16)
        quantized = (gray // (256 // levels)).astype(np.uint8)
        
        def compute_glrlm(image, angle=0):
            """Compute GLRLM for a specific angle."""
            h, w = image.shape
            max_run_length = max(h, w)
            glrlm = np.zeros((levels, max_run_length), dtype=np.int32)
            
            if angle == 0:  # Horizontal
                for i in range(h):
                    runs = self._get_runs(image[i, :])
                    for gray_level, run_length in runs:
                        if gray_level < levels and run_length <= max_run_length:
                            glrlm[gray_level, run_length-1] += 1
            
            elif angle == 90:  # Vertical
                for j in range(w):
                    runs = self._get_runs(image[:, j])
                    for gray_level, run_length in runs:
                        if gray_level < levels and run_length <= max_run_length:
                            glrlm[gray_level, run_length-1] += 1
            
            return glrlm
        
        # Compute GLRLM for different angles
        angles = kwargs.get('angles', [0, 90])
        glrlms = []
        
        for angle in angles:
            glrlm = compute_glrlm(quantized, angle)
            glrlms.append(glrlm)
        
        # Average GLRLM across angles
        avg_glrlm = np.mean(glrlms, axis=0)
        
        # Extract texture features
        features = self._extract_glrlm_features(avg_glrlm)
        
        # Create visualization
        visualization = cv2.resize(avg_glrlm, (gray.shape[1], gray.shape[0]))
        visualization = cv2.normalize(visualization, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        result = {
            'processed_image': visualization,
            'features': features,
            'feature_names': ['glrlm_sre', 'glrlm_lre', 'glrlm_gln', 'glrlm_rln', 
                            'glrlm_rp', 'glrlm_lgre', 'glrlm_hgre'],
            'description': f'GLRLM with {levels} gray levels and {len(angles)} angles'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] GLRLM: {len(features)} features extracted")
        
        return result
    
    def _get_runs(self, line):
        """Extract runs from a line."""
        runs = []
        if len(line) == 0:
            return runs
        
        current_value = line[0]
        current_length = 1
        
        for i in range(1, len(line)):
            if line[i] == current_value:
                current_length += 1
            else:
                runs.append((current_value, current_length))
                current_value = line[i]
                current_length = 1
        
        runs.append((current_value, current_length))
        return runs
    
    def _extract_glrlm_features(self, glrlm):
        """Extract statistical features from GLRLM."""
        if np.sum(glrlm) == 0:
            return np.zeros(7)
        
        # Normalize
        P = glrlm / np.sum(glrlm)
        
        # Marginal sums
        p_g = np.sum(P, axis=1)  # Gray level marginal
        p_r = np.sum(P, axis=0)  # Run length marginal
        
        # Features
        # Short Run Emphasis (SRE)
        r = np.arange(1, P.shape[1] + 1)
        sre = np.sum(p_r / (r ** 2))
        
        # Long Run Emphasis (LRE)
        lre = np.sum(p_r * (r ** 2))
        
        # Gray Level Non-uniformity (GLN)
        gln = np.sum(p_g ** 2)
        
        # Run Length Non-uniformity (RLN)
        rln = np.sum(p_r ** 2)
        
        # Run Percentage (RP)
        rp = np.sum(p_r) / np.sum(glrlm * np.arange(1, P.shape[1] + 1))
        
        # Low Gray-level Run Emphasis (LGRE)
        g = np.arange(1, P.shape[0] + 1)
        lgre = np.sum(p_g / (g ** 2))
        
        # High Gray-level Run Emphasis (HGRE)
        hgre = np.sum(p_g * (g ** 2))
        
        return np.array([sre, lre, gln, rln, rp, lgre, hgre])
    
    def gray_level_co_occurrence_matrix(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute Gray Level Co-occurrence Matrix (GLCM) features.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing GLCM features
        """
        gray = self.preprocess_image(image)
        
        # GLCM parameters
        distances = kwargs.get('distances', [1])
        angles = kwargs.get('angles', [0, 45, 90, 135])
        levels = kwargs.get('levels', 256)
        
        # Compute GLCM
        glcm = graycomatrix(
            gray, 
            distances=distances, 
            angles=np.radians(angles),
            levels=levels,
            symmetric=True, 
            normed=True
        )
        
        # Extract Haralick features
        features = []
        feature_names = []
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            # Average across distances and angles
            avg_value = np.mean(values)
            features.append(avg_value)
            feature_names.append(f'glcm_{prop}')
        
        # Create visualization (sum across distances and angles)
        glcm_vis = np.sum(glcm[:, :, 0, :], axis=2)  # Sum across angles for distance 0
        glcm_vis = cv2.normalize(glcm_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Resize for better visualization
        glcm_vis = cv2.resize(glcm_vis, (gray.shape[1], gray.shape[0]))
        
        result = {
            'processed_image': glcm_vis,
            'features': np.array(features),
            'feature_names': feature_names,
            'description': f'GLCM with {len(distances)} distances and {len(angles)} angles'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] GLCM: {len(features)} Haralick features extracted")
        
        return result
    
    def gray_level_size_zone_matrix(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute Gray Level Size Zone Matrix (GLSZM) features.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing GLSZM features
        """
        gray = self.preprocess_image(image)
        
        # Quantize image
        levels = kwargs.get('levels', 16)
        quantized = (gray // (256 // levels)).astype(np.uint8)
        
        # Connected components for each gray level
        max_size = gray.shape[0] * gray.shape[1]
        glszm = np.zeros((levels, max_size), dtype=np.int32)
        
        for level in range(levels):
            # Create binary image for current gray level
            binary = (quantized == level).astype(np.uint8)
            
            if np.sum(binary) > 0:
                # Find connected components
                labeled, num_features = measure.label(binary, connectivity=2, return_num=True)
                
                # Count size zones
                for region in measure.regionprops(labeled):
                    area = int(region.area)  # Convert to integer
                    if area < max_size and area > 0:  # Ensure positive area
                        glszm[level, area-1] += 1
        
        # Extract features
        features = self._extract_glszm_features(glszm)
        
        # Create visualization
        glszm_subset = glszm[:, :min(100, max_size)].astype(np.float32)
        if glszm_subset.shape[0] > 0 and glszm_subset.shape[1] > 0:
            visualization = cv2.resize(glszm_subset, (gray.shape[1], gray.shape[0]))
            visualization = cv2.normalize(visualization, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Fallback if glszm is empty
            visualization = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
        
        result = {
            'processed_image': visualization,
            'features': features,
            'feature_names': ['glszm_sre', 'glszm_lre', 'glszm_gln', 'glszm_zsn', 'glszm_zp', 'glszm_lgze', 'glszm_hgze'],
            'description': f'GLSZM with {levels} gray levels'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] GLSZM: {len(features)} features extracted")
        
        return result
    
    def _extract_glszm_features(self, glszm):
        """Extract statistical features from GLSZM."""
        if np.sum(glszm) == 0:
            return np.zeros(7)
        
        # Normalize
        P = glszm / np.sum(glszm)
        
        # Marginal sums
        p_g = np.sum(P, axis=1)  # Gray level marginal
        p_s = np.sum(P, axis=0)  # Size zone marginal
        
        # Features
        s = np.arange(1, P.shape[1] + 1)
        g = np.arange(1, P.shape[0] + 1)
        
        # Small Zone Emphasis (SRE)
        sre = np.sum(p_s / (s ** 2))
        
        # Large Zone Emphasis (LRE)
        lre = np.sum(p_s * (s ** 2))
        
        # Gray Level Non-uniformity (GLN)
        gln = np.sum(p_g ** 2)
        
        # Zone Size Non-uniformity (ZSN)
        zsn = np.sum(p_s ** 2)
        
        # Zone Percentage (ZP)
        total_zones = np.sum(glszm)
        total_pixels = np.sum(glszm * s)
        zp = total_zones / total_pixels if total_pixels > 0 else 0
        
        # Low Gray-level Zone Emphasis (LGZE)
        lgze = np.sum(p_g / (g ** 2))
        
        # High Gray-level Zone Emphasis (HGZE)
        hgze = np.sum(p_g * (g ** 2))
        
        return np.array([sre, lre, gln, zsn, zp, lgze, hgze])
    
    def wavelet_transform(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply Wavelet Transform for multi-resolution analysis.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing wavelet features and decomposition
        """
        gray = self.preprocess_image(image)
        
        # Wavelet parameters
        wavelet = kwargs.get('wavelet', 'db4')
        levels = kwargs.get('levels', 3)
        mode = kwargs.get('mode', 'symmetric')
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(gray, wavelet, level=levels, mode=mode)
        
        # Extract features from coefficients
        features = []
        feature_names = []
        
        # Approximation coefficients (low-frequency)
        cA = coeffs[0]
        features.extend([
            np.mean(cA),
            np.std(cA),
            np.var(cA),
            np.max(cA),
            np.min(cA)
        ])
        feature_names.extend(['wt_cA_mean', 'wt_cA_std', 'wt_cA_var', 'wt_cA_max', 'wt_cA_min'])
        
        # Detail coefficients (high-frequency)
        for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            for coeff, name in [(cH, 'cH'), (cV, 'cV'), (cD, 'cD')]:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.var(coeff),
                    np.max(coeff),
                    np.min(coeff)
                ])
                feature_names.extend([
                    f'wt_{name}{i}_mean', f'wt_{name}{i}_std', f'wt_{name}{i}_var',
                    f'wt_{name}{i}_max', f'wt_{name}{i}_min'
                ])
        
        # Reconstruct for visualization
        reconstructed = pywt.waverec2(coeffs, wavelet, mode=mode)
        reconstructed = np.uint8(np.clip(reconstructed, 0, 255))
        
        # Create multi-level visualization
        visualization = self._create_wavelet_visualization(coeffs)
        
        result = {
            'processed_image': visualization,
            'features': np.array(features),
            'feature_names': feature_names,
            'description': f'Wavelet Transform ({wavelet}) with {levels} levels'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] Wavelet Transform: {len(features)} features extracted")
        
        return result
    
    def _create_wavelet_visualization(self, coeffs):
        """Create visualization of wavelet decomposition."""
        # Get the approximation and first level details
        cA = coeffs[0]
        if len(coeffs) > 1:
            cH, cV, cD = coeffs[1]
            
            # Normalize coefficients for visualization
            cA_vis = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cH_vis = cv2.normalize(np.abs(cH), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cV_vis = cv2.normalize(np.abs(cV), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cD_vis = cv2.normalize(np.abs(cD), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Combine into single image
            top = np.hstack([cA_vis, cH_vis])
            bottom = np.hstack([cV_vis, cD_vis])
            combined = np.vstack([top, bottom])
            
            return combined
        else:
            return cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def fast_fourier_transform(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply Fast Fourier Transform for frequency domain analysis.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing FFT features and spectrum
        """
        gray = self.preprocess_image(image)
        
        # Apply FFT
        fft = fft2(gray)
        fft_shift = fftshift(fft)
        
        # Compute magnitude and phase spectra
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        # Extract features from frequency domain
        features = []
        feature_names = []
        
        # DC component (average intensity)
        dc_component = magnitude[magnitude.shape[0]//2, magnitude.shape[1]//2]
        features.append(dc_component)
        feature_names.append('fft_dc')
        
        # Energy in different frequency bands
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # Low frequency energy (center region)
        low_freq_mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 <= (min(h, w)//8)**2
        low_freq_mask[mask] = True
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        
        # High frequency energy (outer region)
        high_freq_energy = np.sum(magnitude) - low_freq_energy
        
        features.extend([low_freq_energy, high_freq_energy])
        feature_names.extend(['fft_low_freq', 'fft_high_freq'])
        
        # Statistical features of magnitude spectrum
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.min(magnitude)
        ])
        feature_names.extend(['fft_mag_mean', 'fft_mag_std', 'fft_mag_max', 'fft_mag_min'])
        
        # Create visualization (log of magnitude spectrum)
        magnitude_log = np.log1p(magnitude)
        magnitude_vis = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        result = {
            'processed_image': magnitude_vis,
            'features': np.array(features),
            'feature_names': feature_names,
            'description': 'FFT magnitude spectrum analysis'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] FFT: {len(features)} frequency domain features extracted")
        
        return result
    
    def segmentation_based_fractal_texture_analysis(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply Segmentation-based Fractal Texture Analysis (SFTA).
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing SFTA features
        """
        gray = self.preprocess_image(image)
        
        # SFTA parameters
        n_thresholds = kwargs.get('n_thresholds', 4)
        
        # Multi-level Otsu thresholding
        thresholds = filters.threshold_multiotsu(gray, classes=n_thresholds+1)
        
        # Create binary images for each threshold level
        binary_images = []
        for i in range(len(thresholds)):
            if i == 0:
                binary = gray <= thresholds[i]
            else:
                binary = (gray > thresholds[i-1]) & (gray <= thresholds[i])
            binary_images.append(binary.astype(np.uint8))
        
        # Add the last level
        binary_images.append((gray > thresholds[-1]).astype(np.uint8))
        
        # Extract features from each binary level
        features = []
        feature_names = []
        
        for i, binary in enumerate(binary_images):
            if np.sum(binary) == 0:
                # If no pixels in this level, add zero features
                level_features = [0] * 4
            else:
                # Fractal dimension using box-counting method
                fractal_dim = self._box_counting_dimension(binary)
                
                # Other texture features
                mean_val = np.mean(binary)
                area_ratio = np.sum(binary) / binary.size
                
                # Perimeter features
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total_perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours])
                
                level_features = [fractal_dim, mean_val, area_ratio, total_perimeter]
            
            features.extend(level_features)
            feature_names.extend([
                f'sfta_level{i}_fractal', f'sfta_level{i}_mean',
                f'sfta_level{i}_area', f'sfta_level{i}_perimeter'
            ])
        
        # Create visualization showing segmented levels
        colored_segmentation = np.zeros_like(gray)
        for i, binary in enumerate(binary_images):
            colored_segmentation += binary * (255 // (len(binary_images) - 1)) * i
        
        result = {
            'processed_image': colored_segmentation,
            'features': np.array(features),
            'feature_names': feature_names,
            'description': f'SFTA with {n_thresholds+1} segmentation levels'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] SFTA: {len(features)} fractal texture features extracted")
        
        return result
    
    def _box_counting_dimension(self, binary_image, max_box_size=None):
        """Compute fractal dimension using box-counting method."""
        if max_box_size is None:
            max_box_size = min(binary_image.shape) // 4
        
        sizes = []
        counts = []
        
        for size in range(2, max_box_size, 2):
            if size >= min(binary_image.shape):
                break
                
            count = 0
            for i in range(0, binary_image.shape[0], size):
                for j in range(0, binary_image.shape[1], size):
                    box = binary_image[i:i+size, j:j+size]
                    if np.any(box):
                        count += 1
            
            if count > 0:
                sizes.append(size)
                counts.append(count)
        
        if len(sizes) < 2:
            return 0
        
        # Linear regression in log-log space
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Fractal dimension is negative slope
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = -slope
        
        return fractal_dim
    
    def local_binary_pattern_glcm(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Combine Local Binary Pattern with GLCM (LBGLCM).
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing LBGLCM features
        """
        gray = self.preprocess_image(image)
        
        # LBP parameters
        radius = kwargs.get('radius', 3)
        n_points = kwargs.get('n_points', 24)
        method = kwargs.get('method', 'uniform')
        
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method=method)
        
        # Normalize LBP image
        lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Compute GLCM on LBP image
        distances = kwargs.get('distances', [1])
        angles = kwargs.get('angles', [0, 45, 90, 135])
        
        glcm = graycomatrix(
            lbp_norm,
            distances=distances,
            angles=np.radians(angles),
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract Haralick features from LBP-GLCM
        features = []
        feature_names = []
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            avg_value = np.mean(values)
            features.append(avg_value)
            feature_names.append(f'lbglcm_{prop}')
        
        # Additional LBP histogram features
        hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
        hist = hist / np.sum(hist)  # Normalize
        
        # Use first 10 histogram bins as features
        features.extend(hist[:10])
        feature_names.extend([f'lbglcm_hist_{i}' for i in range(10)])
        
        result = {
            'processed_image': lbp_norm,
            'features': np.array(features),
            'feature_names': feature_names,
            'description': f'LBP-GLCM with radius={radius}, points={n_points}'
        }
        
        if Config.DEBUG:
            print(f"[DEBUG] LBGLCM: {len(features)} combined texture features extracted")
        
        return result
    
    def process_all_algorithms(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Apply all available processing algorithms to an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing results from all algorithms
        """
        results = {}
        
        algorithms = {
            'gabor_filter': self.gabor_filter,
            'local_directional_pattern': self.local_directional_pattern,
            'gray_level_run_length_matrix': self.gray_level_run_length_matrix,
            'gray_level_co_occurrence_matrix': self.gray_level_co_occurrence_matrix,
            'gray_level_size_zone_matrix': self.gray_level_size_zone_matrix,
            'wavelet_transform': self.wavelet_transform,
            'fast_fourier_transform': self.fast_fourier_transform,
            'segmentation_based_fractal_texture_analysis': self.segmentation_based_fractal_texture_analysis,
            'local_binary_pattern_glcm': self.local_binary_pattern_glcm
        }
        
        print(f"Processing image with {len(algorithms)} algorithms...")
        
        for name, algorithm in algorithms.items():
            try:
                if Config.DEBUG:
                    print(f"[DEBUG] Processing: {name}")
                
                result = algorithm(image)
                results[name] = result
                
                if Config.VERBOSE:
                    print(f"✓ {name}: {len(result['features'])} features extracted")
                    
            except Exception as e:
                print(f"✗ Error in {name}: {str(e)}")
                if Config.DEBUG:
                    import traceback
                    traceback.print_exc()
                
                # Create dummy result to maintain consistency
                results[name] = {
                    'processed_image': self.preprocess_image(image),
                    'features': np.array([]),
                    'feature_names': [],
                    'description': f'Error in {name}: {str(e)}'
                }
        
        return results
