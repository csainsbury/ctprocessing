#!/usr/bin/env python3
"""
Improved CT Batch Processor with Enhanced Valve Detection
Uses the improved detection algorithm to prevent false positives.
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import traceback
from dataclasses import asdict
from tqdm import tqdm
import numpy as np

from image_loader import CTImageLoader
from landmark_detection import AnatomicalLandmarkDetector
from improved_valve_detection import ImprovedAorticValveLevelDetector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ImprovedCTBatchProcessor:
    """Enhanced batch processor with improved valve detection."""
    
    def __init__(self, output_dir: Path, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)):
        """
        Initialize improved batch processor.
        
        Args:
            output_dir: Directory to save results
            target_spacing: Target voxel spacing for processing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'slices').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Initialize processors with improved detection
        self.image_loader = CTImageLoader(target_spacing=target_spacing)
        self.landmark_detector = AnatomicalLandmarkDetector()
        self.valve_detector = ImprovedAorticValveLevelDetector()
        
    def process_single_ct(self, ct_path: Path) -> Dict:
        """
        Process a single CT scan with improved detection.
        
        Args:
            ct_path: Path to CT scan file
            
        Returns:
            Enhanced processing results dictionary
        """
        try:
            result = {
                'filename': ct_path.name,
                'status': 'processing',
                'error': None,
                'valve_slice': None,
                'confidence': None,
                'processing_time': None,
                'validation_scores': None,
                'anatomical_warnings': []
            }
            
            import time
            start_time = time.time()
            
            # Load and preprocess image
            processed_image, metadata = self.image_loader.process_image(ct_path)
            
            # Detect landmarks with progress info
            print(f"  Processing {ct_path.name}...")
            landmarks = self.landmark_detector.detect_all_landmarks(processed_image)
            
            # Use improved valve detection
            valve_candidate = self.valve_detector.detect_aortic_valve_level(processed_image, landmarks)
            
            # Enhanced quality assessment
            anatomical_warnings = self._assess_anatomical_quality(
                valve_candidate, processed_image.shape[2]
            )
            
            # Extract and save the selected slice
            valve_slice = processed_image[:, :, valve_candidate.slice_index]
            slice_path = self.output_dir / 'slices' / f"{ct_path.stem}_valve_slice.npy"
            np.save(slice_path, valve_slice)
            
            # Enhanced metadata with validation info
            result_metadata = {
                'original_file': str(ct_path),
                'valve_slice_index': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'method': valve_candidate.method,
                'landmarks_used': valve_candidate.landmarks_used,
                'validation_scores': valve_candidate.validation_scores,
                'anatomical_features': valve_candidate.anatomical_features,
                'quality_metrics': valve_candidate.quality_metrics,
                'anatomical_warnings': anatomical_warnings,
                'original_shape': metadata['original_shape'],
                'final_shape': metadata['final_shape'],
                'voxel_spacing': metadata['voxel_spacing'],
                'slice_path': str(slice_path),
                'relative_position': valve_candidate.slice_index / processed_image.shape[2]
            }
            
            metadata_path = self.output_dir / 'metadata' / f"{ct_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result_metadata, f, indent=2, default=str)
            
            # Update result with enhanced information
            result.update({
                'status': 'completed',
                'valve_slice': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'processing_time': time.time() - start_time,
                'validation_scores': valve_candidate.validation_scores,
                'anatomical_warnings': anatomical_warnings,
                'slice_path': str(slice_path),
                'metadata_path': str(metadata_path),
                'method': valve_candidate.method,
                'relative_position': valve_candidate.slice_index / processed_image.shape[2]
            })
            
            return result
            
        except Exception as e:
            return {
                'filename': ct_path.name,
                'status': 'failed',
                'error': str(e),
                'valve_slice': None,
                'confidence': None,
                'processing_time': None,
                'validation_scores': None,
                'anatomical_warnings': ['Processing failed']
            }
    
    def _assess_anatomical_quality(self, valve_candidate, total_slices: int) -> List[str]:
        """Assess anatomical quality and generate warnings."""
        warnings = []
        
        relative_pos = valve_candidate.slice_index / total_slices
        expected_range = (0.15, 0.55)
        
        # Position warnings
        if relative_pos < expected_range[0]:
            warnings.append(f"Slice position too superior ({relative_pos:.2f} < {expected_range[0]})")
        elif relative_pos > expected_range[1]:
            warnings.append(f"Slice position too inferior ({relative_pos:.2f} > {expected_range[1]})")
        
        # Confidence warnings
        if valve_candidate.confidence < 0.3:
            warnings.append("Low detection confidence")
        
        # Validation warnings
        if hasattr(valve_candidate, 'validation_scores'):
            avg_validation = np.mean(list(valve_candidate.validation_scores.values()))
            if avg_validation < 0.4:
                warnings.append("Poor anatomical validation")
        
        # Method warnings
        if valve_candidate.method == 'fallback':
            warnings.append("Used fallback detection method")
        
        return warnings
    
    def process_batch(self, ct_files: List[Path], max_workers: Optional[int] = None) -> Dict:
        """
        Process multiple CT scans with improved detection.
        
        Args:
            ct_files: List of CT file paths
            max_workers: Maximum number of parallel workers
            
        Returns:
            Enhanced batch processing results
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        print(f"Processing {len(ct_files)} CT scans with improved detection ({max_workers} workers)...")
        
        results = []
        failed_files = []
        successful_files = []
        warning_files = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_single_ct, ct_file): ct_file
                for ct_file in ct_files
            }
            
            # Process completed jobs with progress bar
            with tqdm(total=len(ct_files), desc="Processing CT scans") as pbar:
                for future in as_completed(future_to_file):
                    ct_file = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'completed':
                            successful_files.append(ct_file)
                            
                            # Check for anatomical warnings
                            if result.get('anatomical_warnings'):
                                warning_files.append((ct_file, result['anatomical_warnings']))
                        else:
                            failed_files.append((ct_file, result['error']))
                        
                        # Update progress bar with quality info
                        pbar.set_postfix({
                            'success': len(successful_files),
                            'warnings': len(warning_files),
                            'failed': len(failed_files)
                        })
                        
                    except Exception as e:
                        failed_files.append((ct_file, str(e)))
                        results.append({
                            'filename': ct_file.name,
                            'status': 'failed',
                            'error': str(e)
                        })
                    
                    pbar.update(1)
        
        # Enhanced batch summary with quality metrics
        batch_summary = {
            'total_files': len(ct_files),
            'successful': len(successful_files),
            'failed': len(failed_files),
            'warnings': len(warning_files),
            'success_rate': len(successful_files) / len(ct_files) if ct_files else 0,
            'results': results,
            'failed_files': [(str(f), e) for f, e in failed_files],
            'warning_files': [(str(f), w) for f, w in warning_files]
        }
        
        # Calculate quality statistics
        successful_results = [r for r in results if r['status'] == 'completed']
        if successful_results:
            confidences = [r['confidence'] for r in successful_results]
            positions = [r['relative_position'] for r in successful_results]
            
            batch_summary.update({
                'quality_stats': {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'mean_position': np.mean(positions),
                    'std_position': np.std(positions),
                    'positions_in_range': sum(1 for p in positions if 0.15 <= p <= 0.55),
                    'high_confidence_count': sum(1 for c in confidences if c > 0.5),
                    'low_confidence_count': sum(1 for c in confidences if c < 0.3)
                }
            })
        
        summary_path = self.output_dir / 'improved_batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\nImproved batch processing completed:")
        print(f"  Successful: {len(successful_files)}/{len(ct_files)}")
        print(f"  With warnings: {len(warning_files)}/{len(ct_files)}")
        print(f"  Failed: {len(failed_files)}/{len(ct_files)}")
        print(f"  Success rate: {batch_summary['success_rate']:.1%}")
        
        if 'quality_stats' in batch_summary:
            stats = batch_summary['quality_stats']
            print(f"  Quality metrics:")
            print(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            print(f"    Positions in anatomical range: {stats['positions_in_range']}/{len(successful_files)}")
            print(f"    High confidence cases: {stats['high_confidence_count']}/{len(successful_files)}")
        
        print(f"  Results saved to: {summary_path}")
        
        return batch_summary
    
    def create_enhanced_quality_report(self, batch_results: Dict) -> str:
        """Create enhanced quality control report."""
        successful_results = [r for r in batch_results['results'] if r['status'] == 'completed']
        
        if not successful_results:
            return "No successful results to analyze."
        
        # Enhanced analysis with validation scores
        slice_positions = [r['valve_slice'] for r in successful_results]
        confidences = [r['confidence'] for r in successful_results]
        relative_positions = [r.get('relative_position', 0) for r in successful_results]
        
        # Analyze validation scores
        validation_analysis = {}
        for result in successful_results:
            if result.get('validation_scores'):
                for key, value in result['validation_scores'].items():
                    if key not in validation_analysis:
                        validation_analysis[key] = []
                    validation_analysis[key].append(value)
        
        # Count warnings
        warning_count = len([r for r in successful_results if r.get('anatomical_warnings')])
        
        report = f"""
Enhanced CT Aortic Valve Level Detection - Quality Report
========================================================

Batch Summary:
- Total files processed: {batch_results['total_files']}
- Successful: {batch_results['successful']}
- With anatomical warnings: {batch_results.get('warnings', 0)}
- Failed: {batch_results['failed']}
- Success rate: {batch_results['success_rate']:.1%}

Slice Position Statistics:
- Mean slice position: {np.mean(slice_positions):.1f}
- Std deviation: {np.std(slice_positions):.1f}
- Range: {min(slice_positions)} - {max(slice_positions)}

Relative Position Analysis:
- Mean relative position: {np.mean(relative_positions):.3f}
- Std deviation: {np.std(relative_positions):.3f}
- Cases in anatomical range (0.15-0.55): {sum(1 for p in relative_positions if 0.15 <= p <= 0.55)} ({sum(1 for p in relative_positions if 0.15 <= p <= 0.55)/len(relative_positions)*100:.1f}%)

Confidence Statistics:
- Mean confidence: {np.mean(confidences):.3f}
- Std deviation: {np.std(confidences):.3f}
- Range: {min(confidences):.3f} - {max(confidences):.3f}

Quality Categories:
- High confidence (>0.5): {sum(1 for c in confidences if c > 0.5)} ({sum(1 for c in confidences if c > 0.5)/len(confidences)*100:.1f}%)
- Medium confidence (0.2-0.5): {sum(1 for c in confidences if 0.2 <= c <= 0.5)} ({sum(1 for c in confidences if 0.2 <= c <= 0.5)/len(confidences)*100:.1f}%)
- Low confidence (<0.2): {sum(1 for c in confidences if c < 0.2)} ({sum(1 for c in confidences if c < 0.2)/len(confidences)*100:.1f}%)

Validation Analysis:
"""
        
        for validation_type, scores in validation_analysis.items():
            if scores:
                report += f"- {validation_type}: mean {np.mean(scores):.3f} ± {np.std(scores):.3f}\n"
        
        report += f"""
Quality Issues:
- Cases with anatomical warnings: {warning_count} ({warning_count/len(successful_results)*100:.1f}%)

Improvements vs Original Method:
- Enhanced anatomical constraints prevent false positives
- Multi-level validation catches suspicious detections
- Detailed validation scores enable quality assessment
- Position-based filtering ensures anatomical consistency

Recommendations:
- Anatomical consistency: {'Good' if np.std(relative_positions) < 0.1 else 'Review needed'}
- Overall confidence: {'Good' if np.mean(confidences) > 0.4 else 'Review needed'}
- Warning rate: {'Acceptable' if warning_count/len(successful_results) < 0.3 else 'High - review parameters'}
"""
        
        report_path = self.output_dir / 'enhanced_quality_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Enhanced quality report saved to: {report_path}")
        return report

def main():
    """Main function for improved batch processing."""
    parser = argparse.ArgumentParser(
        description="Improved CT batch processor with enhanced valve detection"
    )
    parser.add_argument('input_dir', type=str, help='Directory containing CT scan files')
    parser.add_argument('--output_dir', type=str, default='./improved_output', 
                       help='Output directory for results')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                       help='File pattern to match (default: *.nii.gz)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 2.0],
                       help='Target voxel spacing (x y z)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Find CT files
    input_path = Path(args.input_dir)
    ct_files = list(input_path.glob(args.pattern))
    
    if not ct_files:
        print(f"No files found matching pattern '{args.pattern}' in {input_path}")
        return
    
    # Limit files if requested
    if args.max_files:
        ct_files = ct_files[:args.max_files]
    
    print(f"Found {len(ct_files)} CT files to process with improved detection")
    
    # Initialize improved processor
    processor = ImprovedCTBatchProcessor(
        output_dir=Path(args.output_dir),
        target_spacing=tuple(args.spacing)
    )
    
    # Process batch
    results = processor.process_batch(ct_files, max_workers=args.workers)
    
    # Create enhanced quality report
    processor.create_enhanced_quality_report(results)

if __name__ == "__main__":
    main()