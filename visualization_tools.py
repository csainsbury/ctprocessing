#!/usr/bin/env python3
"""
Visualization Tools for CT Aortic Valve Level Detection
Provides quality control and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.gridspec as gridspec

class CTVisualizationTools:
    """Tools for visualizing CT processing results."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualization tools."""
        self.output_dir = Path(output_dir)
        
    def create_slice_montage(self, slice_paths: List[Path], output_path: Optional[Path] = None,
                           max_images: int = 16, figsize: tuple = (16, 12)) -> None:
        """
        Create a montage of extracted valve slices for visual inspection.
        
        Args:
            slice_paths: List of paths to slice files
            output_path: Path to save montage image
            max_images: Maximum number of images to include
            figsize: Figure size
        """
        # Limit number of images
        slice_paths = slice_paths[:max_images]
        
        # Calculate grid dimensions
        n_images = len(slice_paths)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(f'Aortic Valve Level Slices - {n_images} samples', fontsize=16)
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, slice_path in enumerate(slice_paths):
            try:
                # Load slice
                slice_data = np.load(slice_path)
                
                # Display slice
                axes[i].imshow(slice_data, cmap='gray')
                axes[i].set_title(f'{slice_path.stem}', fontsize=10)
                axes[i].axis('off')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{slice_path.name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Slice montage saved to: {output_path}")
        else:
            output_path = self.output_dir / 'slice_montage.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Slice montage saved to: {output_path}")
        
        plt.show()
    
    def create_quality_dashboard(self, batch_results: Dict, 
                               output_path: Optional[Path] = None) -> None:
        """
        Create a comprehensive quality control dashboard.
        
        Args:
            batch_results: Batch processing results
            output_path: Path to save dashboard
        """
        successful_results = [r for r in batch_results['results'] 
                            if r['status'] == 'completed']
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Extract data
        slice_positions = [r['valve_slice'] for r in successful_results]
        confidences = [r['confidence'] for r in successful_results]
        processing_times = [r.get('processing_time', 0) for r in successful_results]
        filenames = [r['filename'] for r in successful_results]
        
        # Create dashboard
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Title
        fig.suptitle('CT Aortic Valve Detection - Quality Dashboard', fontsize=16, y=0.95)
        
        # 1. Slice position distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(slice_positions, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Slice Position')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Slice Position Distribution')
        ax1.axvline(np.mean(slice_positions), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(slice_positions):.1f}')
        ax1.legend()
        
        # 2. Confidence distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax2.legend()
        
        # 3. Processing time distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if any(t > 0 for t in processing_times):
            ax3.hist(processing_times, bins=20, edgecolor='black', alpha=0.7, color='green')
            ax3.set_xlabel('Processing Time (s)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Processing Time Distribution')
            ax3.axvline(np.mean(processing_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(processing_times):.1f}s')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No timing data available', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Processing Time Distribution')
        
        # 4. Slice position vs confidence scatter
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(slice_positions, confidences, alpha=0.6, c=confidences, 
                            cmap='viridis')
        ax4.set_xlabel('Slice Position')
        ax4.set_ylabel('Confidence Score')
        ax4.set_title('Slice Position vs Confidence')
        plt.colorbar(scatter, ax=ax4, label='Confidence')
        
        # 5. Success/failure summary
        ax5 = fig.add_subplot(gs[1, 1])
        success_count = batch_results['successful']
        failed_count = batch_results['failed']
        labels = ['Successful', 'Failed']
        sizes = [success_count, failed_count]
        colors = ['lightgreen', 'lightcoral']
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Processing Success Rate')
        
        # 6. Quality categories
        ax6 = fig.add_subplot(gs[1, 2])
        high_conf = sum(1 for c in confidences if c > 0.5)
        med_conf = sum(1 for c in confidences if 0.2 <= c <= 0.5)
        low_conf = sum(1 for c in confidences if c < 0.2)
        
        quality_labels = ['High (>0.5)', 'Medium (0.2-0.5)', 'Low (<0.2)']
        quality_sizes = [high_conf, med_conf, low_conf]
        quality_colors = ['darkgreen', 'orange', 'red']
        ax6.pie(quality_sizes, labels=quality_labels, colors=quality_colors, 
                autopct='%1.1f%%', startangle=90)
        ax6.set_title('Confidence Quality Distribution')
        
        # 7. Top 10 highest confidence results
        ax7 = fig.add_subplot(gs[2, :2])
        top_indices = np.argsort(confidences)[-10:][::-1]
        top_files = [filenames[i][:20] + '...' if len(filenames[i]) > 20 else filenames[i] 
                    for i in top_indices]
        top_confidences = [confidences[i] for i in top_indices]
        
        bars = ax7.barh(range(len(top_files)), top_confidences, color='skyblue')
        ax7.set_yticks(range(len(top_files)))
        ax7.set_yticklabels(top_files, fontsize=8)
        ax7.set_xlabel('Confidence Score')
        ax7.set_title('Top 10 Highest Confidence Results')
        ax7.invert_yaxis()
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, top_confidences)):
            ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{conf:.3f}', va='center', fontsize=8)
        
        # 8. Statistics summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        stats_text = f"""
Statistics Summary:

Files Processed: {batch_results['total_files']}
Success Rate: {batch_results['success_rate']:.1%}

Slice Position:
  Mean: {np.mean(slice_positions):.1f}
  Std: {np.std(slice_positions):.1f}
  Range: [{min(slice_positions)}, {max(slice_positions)}]

Confidence:
  Mean: {np.mean(confidences):.3f}
  Std: {np.std(confidences):.3f}
  Range: [{min(confidences):.3f}, {max(confidences):.3f}]

Quality Assessment:
  High Confidence: {high_conf} ({high_conf/len(confidences)*100:.1f}%)
  Position Consistency: {'Good' if np.std(slice_positions) < 10 else 'Review'}
        """
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Quality dashboard saved to: {output_path}")
        else:
            output_path = self.output_dir / 'quality_dashboard.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Quality dashboard saved to: {output_path}")
        
        plt.show()
    
    def create_individual_report(self, ct_filename: str, metadata_path: Path,
                               slice_path: Path, output_path: Optional[Path] = None) -> None:
        """
        Create individual report for a single CT scan.
        
        Args:
            ct_filename: Original CT filename
            metadata_path: Path to metadata JSON
            slice_path: Path to extracted slice
            output_path: Path to save report
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load slice
        slice_data = np.load(slice_path)
        
        # Create report
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display slice
        ax1.imshow(slice_data, cmap='gray')
        ax1.set_title(f'Extracted Aortic Valve Slice\n{ct_filename}')
        ax1.axis('off')
        
        # Add crosshair at center
        center_y, center_x = slice_data.shape[0]//2, slice_data.shape[1]//2
        ax1.axhline(y=center_y, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=center_x, color='red', linestyle='--', alpha=0.5)
        
        # Metadata summary
        ax2.axis('off')
        summary_text = f"""
Processing Results:

Original File: {ct_filename}
Valve Slice Index: {metadata['valve_slice_index']}
Detection Confidence: {metadata['confidence']:.3f}
Detection Method: {metadata['method']}
Landmarks Used: {', '.join(metadata['landmarks_used'])}

Image Properties:
Original Shape: {metadata['original_shape']}
Final Shape: {metadata['final_shape']}
Voxel Spacing: {metadata['voxel_spacing']} mm

Slice Statistics:
Intensity Range: [{slice_data.min():.3f}, {slice_data.max():.3f}]
Mean Intensity: {slice_data.mean():.3f}
Std Intensity: {slice_data.std():.3f}

Quality Assessment:
Confidence Level: {'High' if metadata['confidence'] > 0.5 else 'Medium' if metadata['confidence'] > 0.2 else 'Low'}
Recommended Action: {'Accept' if metadata['confidence'] > 0.3 else 'Review'}
        """
        
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            output_path = self.output_dir / 'visualizations' / f'{Path(ct_filename).stem}_report.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f"Individual report saved to: {output_path}")
        plt.show()

def test_visualization():
    """Test visualization tools."""
    # This would typically be called after batch processing
    print("Visualization tools ready. Use after running batch processing.")
    
    # Example usage:
    print("""
    Example usage after batch processing:
    
    from visualization_tools import CTVisualizationTools
    import json
    from pathlib import Path
    
    # Initialize tools
    viz = CTVisualizationTools(Path('./output'))
    
    # Load batch results
    with open('./output/batch_summary.json', 'r') as f:
        results = json.load(f)
    
    # Create quality dashboard
    viz.create_quality_dashboard(results)
    
    # Create slice montage
    slice_paths = list(Path('./output/slices').glob('*.npy'))
    viz.create_slice_montage(slice_paths)
    """)

if __name__ == "__main__":
    test_visualization()