"""
Visualization utilities for image processing results
Handles creation             title_text = f'{algorithm_name.replace("_", " ").title()}\n{result["description"]}'
            ax_proc.set_title(title_text, fontsize=10) comparison plots, before/after images, and result tables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import cv2
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from config import Config


class Visualizer:
    """
    Comprehensive visualization toolkit for image processing results.
    Creates comparison plots, before/after images, and analysis tables.
    """
    
    def __init__(self):
        """Initialize the visualizer with styling."""
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directories if they don't exist
        for dir_path in [Config.PROCESSED_IMAGES_DIR, Config.COMPARISONS_DIR, 
                        Config.GRAPHS_DIR, Config.TABLES_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    def create_before_after_comparison(self, 
                                     original_image: np.ndarray,
                                     processed_results: Dict[str, Dict],
                                     image_name: str) -> None:
        """
        Create before/after comparison visualization for all algorithms.
        
        Args:
            original_image: Original input image
            processed_results: Dictionary of processing results from all algorithms
            image_name: Name identifier for the image
        """
        print(f"Creating before/after comparison for: {image_name}")
        
        # Convert original image to grayscale for consistent comparison
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image.copy()
        
        # Calculate grid dimensions
        n_algorithms = len(processed_results)
        n_cols = 3  # Original, Processed, Difference
        n_rows = n_algorithms
        
        # Create figure
        fig = plt.figure(figsize=(15, 4 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        for i, (algorithm_name, result) in enumerate(processed_results.items()):
            # Original image
            ax_orig = fig.add_subplot(gs[i, 0])
            ax_orig.imshow(original_gray, cmap='gray')
            ax_orig.set_title(f'Original\\n{image_name}', fontsize=10)
            ax_orig.axis('off')
            
            # Processed image
            ax_proc = fig.add_subplot(gs[i, 1])
            processed_img = result['processed_image']
            ax_proc.imshow(processed_img, cmap='gray')
            ax_proc.set_title(f'{algorithm_name.replace("_", " ").title()}\\n{result["description"]}', 
                            fontsize=10, wrap=True)
            ax_proc.axis('off')
            
            # Difference image (if possible)
            ax_diff = fig.add_subplot(gs[i, 2])
            try:
                # Resize processed image to match original
                if processed_img.shape != original_gray.shape:
                    processed_resized = cv2.resize(processed_img, 
                                                 (original_gray.shape[1], original_gray.shape[0]))
                else:
                    processed_resized = processed_img
                
                difference = cv2.absdiff(original_gray, processed_resized)
                ax_diff.imshow(difference, cmap='hot')
                ax_diff.set_title('Difference\\n(Original - Processed)', fontsize=10)
            except:
                # If difference calculation fails, show the processed image
                ax_diff.imshow(processed_img, cmap='gray')
                ax_diff.set_title('Processed Result', fontsize=10)
            
            ax_diff.axis('off')
        
        plt.suptitle(f'Image Processing Comparison - {image_name}', fontsize=16, y=0.98)
        
        # Save the comparison
        output_path = Config.COMPARISONS_DIR / f'{image_name}_comparison.png'
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Comparison saved to: {output_path}")
    
    def create_algorithm_comparison_table(self, 
                                        all_results: Dict[str, Dict[str, Dict]], 
                                        sample_names: List[str]) -> pd.DataFrame:
        """
        Create a comprehensive comparison table of all algorithms and samples.
        
        Args:
            all_results: Dictionary of {sample_name: {algorithm: results}}
            sample_names: List of sample image names
            
        Returns:
            DataFrame containing the comparison table
        """
        print("Creating algorithm comparison table...")
        
        # Collect data for the table
        table_data = []
        
        for sample_name in sample_names:
            if sample_name not in all_results:
                continue
                
            sample_results = all_results[sample_name]
            
            for algorithm_name, result in sample_results.items():
                features = result.get('features', np.array([]))
                
                row = {
                    'Sample': sample_name,
                    'Algorithm': algorithm_name.replace('_', ' ').title(),
                    'Description': result.get('description', ''),
                    'Feature_Count': len(features),
                    'Feature_Mean': np.mean(features) if len(features) > 0 else 0,
                    'Feature_Std': np.std(features) if len(features) > 0 else 0,
                    'Feature_Max': np.max(features) if len(features) > 0 else 0,
                    'Feature_Min': np.min(features) if len(features) > 0 else 0,
                    'Processing_Success': len(features) > 0
                }
                
                table_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(table_data)
        
        # Save to CSV
        table_path = Config.TABLES_DIR / 'algorithm_comparison_table.csv'
        comparison_df.to_csv(table_path, index=False)
        print(f"Comparison table saved to: {table_path}")
        
        # Create summary statistics table
        self._create_summary_statistics_table(comparison_df)
        
        return comparison_df
    
    def _create_summary_statistics_table(self, comparison_df: pd.DataFrame) -> None:
        """Create summary statistics table for algorithms."""
        # Group by algorithm and calculate statistics
        summary_stats = comparison_df.groupby('Algorithm').agg({
            'Feature_Count': ['mean', 'std', 'min', 'max'],
            'Feature_Mean': ['mean', 'std', 'min', 'max'],
            'Feature_Std': ['mean', 'std', 'min', 'max'],
            'Processing_Success': ['sum', 'count']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        # Calculate success rate
        summary_stats['Success_Rate'] = (
            summary_stats['Processing_Success_sum'] / 
            summary_stats['Processing_Success_count']
        ).round(4)
        
        # Save summary statistics
        summary_path = Config.TABLES_DIR / 'algorithm_summary_statistics.csv'
        summary_stats.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")
    
    def create_feature_comparison_plots(self, 
                                      all_results: Dict[str, Dict[str, Dict]], 
                                      sample_names: List[str]) -> None:
        """
        Create various plots comparing features across algorithms and samples.
        
        Args:
            all_results: Dictionary of {sample_name: {algorithm: results}}
            sample_names: List of sample image names
        """
        print("Creating feature comparison plots...")
        
        # 1. Feature count comparison
        self._plot_feature_counts(all_results, sample_names)
        
        # 2. Feature distribution comparison
        self._plot_feature_distributions(all_results, sample_names)
        
        # 3. Algorithm performance heatmap
        self._plot_algorithm_performance_heatmap(all_results, sample_names)
        
        # 4. Feature correlation matrix (if applicable)
        self._plot_feature_correlations(all_results, sample_names)
    
    def _plot_feature_counts(self, all_results: Dict, sample_names: List[str]) -> None:
        """Plot feature counts for each algorithm across samples."""
        # Collect feature counts
        algorithm_names = set()
        for sample_results in all_results.values():
            algorithm_names.update(sample_results.keys())
        
        algorithm_names = sorted(list(algorithm_names))
        
        # Create data matrix
        feature_counts = np.zeros((len(sample_names), len(algorithm_names)))
        
        for i, sample_name in enumerate(sample_names):
            for j, algorithm_name in enumerate(algorithm_names):
                if sample_name in all_results and algorithm_name in all_results[sample_name]:
                    features = all_results[sample_name][algorithm_name].get('features', np.array([]))
                    feature_counts[i, j] = len(features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Bar plot
        x = np.arange(len(algorithm_names))
        width = 0.25
        
        for i, sample_name in enumerate(sample_names):
            ax.bar(x + i * width, feature_counts[i], width, 
                  label=sample_name, alpha=0.8)
        
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Count Comparison Across Algorithms')
        ax.set_xticks(x + width * (len(sample_names) - 1) / 2)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in algorithm_names], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Config.GRAPHS_DIR / 'feature_counts_comparison.png', 
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(self, all_results: Dict, sample_names: List[str]) -> None:
        """Plot distribution of feature values for each algorithm."""
        algorithm_names = set()
        for sample_results in all_results.values():
            algorithm_names.update(sample_results.keys())
        
        algorithm_names = sorted(list(algorithm_names))[:6]  # Limit to first 6 for readability
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, algorithm_name in enumerate(algorithm_names):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Collect all feature values for this algorithm
            all_features = []
            colors = []
            labels = []
            
            for sample_name in sample_names:
                if (sample_name in all_results and 
                    algorithm_name in all_results[sample_name]):
                    
                    features = all_results[sample_name][algorithm_name].get('features', np.array([]))
                    if len(features) > 0:
                        all_features.extend(features)
                        colors.extend([sample_name] * len(features))
            
            if all_features:
                # Create histogram
                unique_colors = list(set(colors))
                for color in unique_colors:
                    color_features = [all_features[j] for j, c in enumerate(colors) if c == color]
                    ax.hist(color_features, bins=20, alpha=0.7, label=color, density=True)

                ax.set_title(f'{algorithm_name.replace("_", " ").title()}')
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No features extracted', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{algorithm_name.replace("_", " ").title()}')

        # Hide empty subplots
        for i in range(len(algorithm_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Value Distributions by Algorithm', fontsize=16)
        plt.tight_layout()
        plt.savefig(Config.GRAPHS_DIR / 'feature_distributions.png', 
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
    
    def _plot_algorithm_performance_heatmap(self, all_results: Dict, sample_names: List[str]) -> None:
        """Create a heatmap showing algorithm performance across samples."""
        algorithm_names = set()
        for sample_results in all_results.values():
            algorithm_names.update(sample_results.keys())
        
        algorithm_names = sorted(list(algorithm_names))
        
        # Create performance matrix (success/failure + feature count)
        performance_matrix = np.zeros((len(sample_names), len(algorithm_names)))
        
        for i, sample_name in enumerate(sample_names):
            for j, algorithm_name in enumerate(algorithm_names):
                if sample_name in all_results and algorithm_name in all_results[sample_name]:
                    features = all_results[sample_name][algorithm_name].get('features', np.array([]))
                    # Performance metric: log(feature_count + 1) for better visualization
                    performance_matrix[i, j] = np.log1p(len(features))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(performance_matrix, 
                   xticklabels=[name.replace('_', ' ').title() for name in algorithm_names],
                   yticklabels=sample_names,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   ax=ax)
        
        ax.set_title('Algorithm Performance Heatmap\\n(Log of Feature Count + 1)')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Samples')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Config.GRAPHS_DIR / 'algorithm_performance_heatmap.png', 
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlations(self, all_results: Dict, sample_names: List[str]) -> None:
        """Plot correlation matrix between different algorithms' features."""
        print("Computing feature correlations...")
        
        # Collect features from all algorithms for all samples
        feature_data = {}
        
        for sample_name in sample_names:
            if sample_name not in all_results:
                continue
                
            sample_results = all_results[sample_name]
            
            for algorithm_name, result in sample_results.items():
                features = result.get('features', np.array([]))
                
                if len(features) > 0:
                    # Use mean of features as a single representative value
                    if algorithm_name not in feature_data:
                        feature_data[algorithm_name] = []
                    feature_data[algorithm_name].append(np.mean(features))
        
        # Create DataFrame for correlation analysis
        max_length = max(len(values) for values in feature_data.values()) if feature_data else 0
        
        if max_length > 1:  # Need at least 2 samples for correlation
            # Pad shorter lists with NaN
            for algorithm in feature_data:
                while len(feature_data[algorithm]) < max_length:
                    feature_data[algorithm].append(np.nan)
            
            correlation_df = pd.DataFrame(feature_data)
            correlation_matrix = correlation_df.corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix,
                       mask=mask,
                       annot=True,
                       fmt='.3f',
                       cmap='coolwarm',
                       center=0,
                       square=True,
                       ax=ax)
            
            ax.set_title('Algorithm Feature Correlation Matrix')
            
            # Rotate labels
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(Config.GRAPHS_DIR / 'feature_correlations.png', 
                       dpi=Config.DPI, bbox_inches='tight')
            plt.close()
        else:
            print("Insufficient data for correlation analysis")
    
    def create_processing_time_analysis(self, timing_data: Dict[str, float]) -> None:
        """
        Create visualization of processing times for different algorithms.
        
        Args:
            timing_data: Dictionary of {algorithm_name: processing_time}
        """
        if not timing_data:
            return
        
        # Sort by processing time
        sorted_data = sorted(timing_data.items(), key=lambda x: x[1])
        algorithms, times = zip(*sorted_data)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(algorithms)), times, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{time:.3f}s', ha='left', va='center')
        
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels([alg.replace('_', ' ').title() for alg in algorithms])
        ax.set_xlabel('Processing Time (seconds)')
        ax.set_title('Algorithm Processing Time Comparison')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(Config.GRAPHS_DIR / 'processing_times.png', 
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Processing time analysis saved to: {Config.GRAPHS_DIR / 'processing_times.png'}")
    
    def save_individual_processed_images(self, 
                                       processed_results: Dict[str, Dict], 
                                       image_name: str) -> None:
        """
        Save individual processed images for each algorithm.
        
        Args:
            processed_results: Dictionary of processing results
            image_name: Name identifier for the image
        """
        print(f"Saving individual processed images for: {image_name}")
        
        for algorithm_name, result in processed_results.items():
            processed_img = result['processed_image']
            
            # Ensure the image is in proper format for saving
            if processed_img.dtype != np.uint8:
                processed_img = cv2.normalize(processed_img, None, 0, 255, 
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Create filename
            filename = f'{image_name}_{algorithm_name}.png'
            output_path = Config.PROCESSED_IMAGES_DIR / filename
            
            # Save image
            cv2.imwrite(str(output_path), processed_img)
            
            if Config.DEBUG:
                print(f"[DEBUG] Saved: {output_path}")
    
    def create_comprehensive_report(self, 
                                  dataset_info: Dict,
                                  all_results: Dict[str, Dict[str, Dict]], 
                                  sample_names: List[str],
                                  timing_data: Optional[Dict[str, float]] = None) -> None:
        """
        Create a comprehensive HTML report with all results.
        
        Args:
            dataset_info: Dataset analysis information
            all_results: All processing results
            sample_names: List of sample image names
            timing_data: Processing timing information
        """
        print("Creating comprehensive report...")
        
        # Create HTML report
        html_content = self._generate_html_report(dataset_info, all_results, 
                                                sample_names, timing_data)
        
        # Save HTML report
        report_path = Config.OUTPUTS_DIR / 'comprehensive_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def _generate_html_report(self, 
                            dataset_info: Dict,
                            all_results: Dict, 
                            sample_names: List[str],
                            timing_data: Optional[Dict] = None) -> str:
        """Generate HTML content for the comprehensive report."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Processing Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ text-align: center; color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; background-color: #f9f9f9; border-radius: 8px; }}
                .section h2 {{ color: #007acc; margin-top: 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: flex; justify-content: space-between; margin: 5px 0; }}
                .metric-label {{ font-weight: bold; }}
                .metric-value {{ color: #007acc; }}
                img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                .algorithm-list {{ list-style-type: none; padding: 0; }}
                .algorithm-list li {{ padding: 8px; margin: 5px 0; background: #e8f4f8; border-radius: 4px; }}
                .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Image Processing Analysis Report</h1>
                <p>Comprehensive analysis of medical image processing algorithms</p>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Dataset Statistics</h3>
                        <div class="metric">
                            <span class="metric-label">Total Images:</span>
                            <span class="metric-value">{dataset_info.get('total_images', 0)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Analyzed Images:</span>
                            <span class="metric-value">{dataset_info.get('analyzed_images', 0)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">File Formats:</span>
                            <span class="metric-value">{len(dataset_info.get('file_formats', {}))}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Classes Found:</span>
                            <span class="metric-value">{len(dataset_info.get('class_distribution', {}))}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Algorithms</h2>
                <div class="card">
                    <p>The following state-of-the-art image processing algorithms were applied:</p>
                    <ul class="algorithm-list">
                        <li><strong>Gabor Filter:</strong> Multi-orientation texture analysis</li>
                        <li><strong>Local Directional Pattern (LDP):</strong> Directional texture features</li>
                        <li><strong>Gray Level Run Length Matrix (GLRLM):</strong> Run-length texture analysis</li>
                        <li><strong>Gray Level Co-occurrence Matrix (GLCM):</strong> Spatial relationship analysis</li>
                        <li><strong>Gray Level Size Zone Matrix (GLSZM):</strong> Size zone texture features</li>
                        <li><strong>Wavelet Transform:</strong> Multi-resolution frequency analysis</li>
                        <li><strong>Fast Fourier Transform (FFT):</strong> Frequency domain analysis</li>
                        <li><strong>Segmentation-based Fractal Texture Analysis (SFTA):</strong> Fractal texture features</li>
                        <li><strong>Local Binary Pattern + GLCM (LBGLCM):</strong> Combined texture analysis</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Results</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Sample Images Processed</h3>
                        <ul>
        """
        
        # Add sample names to the report
        for sample_name in sample_names:
            html_template += f"<li>{sample_name}</li>"
        
        html_template += """
                        </ul>
                    </div>
        """
        
        # Add timing information if available
        if timing_data:
            html_template += """
                    <div class="card">
                        <h3>Processing Performance</h3>
            """
            for algorithm, time in timing_data.items():
                html_template += f"""
                        <div class="metric">
                            <span class="metric-label">{algorithm.replace('_', ' ').title()}:</span>
                            <span class="metric-value">{time:.3f}s</span>
                        </div>
                """
            html_template += "</div>"
        
        html_template += """
                </div>
            </div>
            
            <div class="section">
                <h2>Output Files</h2>
                <div class="card">
                    <p>The following output files have been generated:</p>
                    <ul>
                        <li><strong>Processed Images:</strong> Individual algorithm results</li>
                        <li><strong>Comparison Images:</strong> Before/after visualizations</li>
                        <li><strong>Analysis Tables:</strong> Feature comparison and statistics</li>
                        <li><strong>Visualization Graphs:</strong> Performance and correlation plots</li>
                        <li><strong>Dataset Insights:</strong> Comprehensive dataset analysis</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Medical Image Processing Tool</p>
                <p>Specialized for AI Disease Recognition Projects</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
