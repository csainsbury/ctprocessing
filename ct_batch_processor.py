#!/usr/bin/env python3
"""
CT Batch Processor for Aortic Valve Level Detection
Efficiently processes multiple CT scans to extract consistent axial slices at aortic valve level.
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
from aortic_valve_detection import AorticValveLevelDetector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CTBatchProcessor:
    """Batch processor for CT aortic valve level detection."""
    
    def __init__(self, output_dir: Path, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)):
        """
        Initialize batch processor.
        
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
        
        # Initialize processors
        self.image_loader = CTImageLoader(target_spacing=target_spacing)
        self.landmark_detector = AnatomicalLandmarkDetector()
        self.valve_detector = AorticValveLevelDetector()
        
    def process_single_ct(self, ct_path: Path) -> Dict:
        """
        Process a single CT scan.
        
        Args:
            ct_path: Path to CT scan file
            
        Returns:
            Processing results dictionary
        """
        try:
            result = {
                'filename': ct_path.name,
                'status': 'processing',
                'error': None,
                'valve_slice': None,
                'confidence': None,
                'processing_time': None
            }
            
            import time
            start_time = time.time()
            
            # Load and preprocess image
            processed_image, metadata = self.image_loader.process_image(ct_path)
            
            # Detect landmarks (simplified for efficiency)
            landmarks = self._detect_landmarks_simplified(processed_image)
            
            # Detect aortic valve level (simplified)
            valve_candidate = self._detect_valve_simplified(processed_image, landmarks)
            
            # Extract and save the selected slice
            valve_slice = processed_image[:, :, valve_candidate.slice_index]
            slice_path = self.output_dir / 'slices' / f"{ct_path.stem}_valve_slice.npy"
            np.save(slice_path, valve_slice)
            
            # Save metadata
            result_metadata = {
                'original_file': str(ct_path),
                'valve_slice_index': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'method': valve_candidate.method,
                'landmarks_used': valve_candidate.landmarks_used,
                'original_shape': metadata['original_shape'],
                'final_shape': metadata['final_shape'],
                'voxel_spacing': metadata['voxel_spacing'],
                'slice_path': str(slice_path)
            }
            
            metadata_path = self.output_dir / 'metadata' / f"{ct_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result_metadata, f, indent=2)
            
            # Update result
            result.update({
                'status': 'completed',
                'valve_slice': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'processing_time': time.time() - start_time,
                'slice_path': str(slice_path),
                'metadata_path': str(metadata_path)
            })
            
            return result
            
        except Exception as e:
            return {
                'filename': ct_path.name,
                'status': 'failed',
                'error': str(e),
                'valve_slice': None,
                'confidence': None,
                'processing_time': None
            }
    
    def _detect_landmarks_simplified(self, image: np.ndarray) -> Dict:
        """Simplified landmark detection for efficiency."""
        landmarks = {'aorta': [], 'heart': []}
        
        # Focus on middle region for faster processing
        mid_start = int(image.shape[2] * 0.3)
        mid_end = int(image.shape[2] * 0.7)
        
        # Sample fewer slices
        for z in range(mid_start, mid_end, 3):  # Every 3rd slice
            slice_img = image[:, :, z]
            
            # Quick aorta detection using intensity thresholding
            high_intensity = slice_img > 0.6
            from skimage import measure, morphology
            
            high_intensity = morphology.remove_small_objects(high_intensity, min_size=20)
            labeled = measure.label(high_intensity)
            regions = measure.regionprops(labeled)
            
            for region in regions:
                if (50 < region.area < 500 and 
                    region.eccentricity < 0.8 and
                    region.centroid[1] < image.shape[1] * 0.6):  # Left side
                    
                    landmarks['aorta'].append({
                        'slice': z,
                        'center': region.centroid,
                        'radius': np.sqrt(region.area / np.pi),
                        'confidence': region.solidity
                    })
            
            # Quick heart detection
            med_intensity = (slice_img > 0.3) & (slice_img < 0.7)
            med_intensity = morphology.remove_small_objects(med_intensity, min_size=100)
            labeled = measure.label(med_intensity)
            
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                largest = max(regions, key=lambda x: x.area)
                
                if largest.area > 300:
                    landmarks['heart'].append({
                        'slice': z,
                        'centroid': largest.centroid,
                        'area': largest.area,
                        'confidence': largest.area / 1000.0
                    })
        
        return landmarks
    
    def _detect_valve_simplified(self, image: np.ndarray, landmarks: Dict):
        """Simplified valve detection for efficiency."""
        from aortic_valve_detection import AorticValveCandidate
        
        aorta_landmarks = landmarks.get('aorta', [])
        heart_landmarks = landmarks.get('heart', [])
        
        if not aorta_landmarks:
            # Fallback: use middle of image
            return AorticValveCandidate(
                slice_index=image.shape[2] // 2,
                confidence=0.1,
                method='fallback',
                landmarks_used=[],
                anatomical_features={},
                quality_metrics={}
            )
        
        # Simple approach: find aorta with best confidence in reasonable range
        valve_range = (int(image.shape[2] * 0.25), int(image.shape[2] * 0.65))
        valid_aorta = [a for a in aorta_landmarks 
                      if valve_range[0] <= a['slice'] <= valve_range[1]]
        
        if valid_aorta:
            best_aorta = max(valid_aorta, key=lambda x: x['confidence'])
            return AorticValveCandidate(
                slice_index=best_aorta['slice'],
                confidence=best_aorta['confidence'],
                method='simplified_aortic',
                landmarks_used=['aorta'],
                anatomical_features=best_aorta,
                quality_metrics={'aortic_confidence': best_aorta['confidence']}
            )
        else:
            # Use first reasonable aorta detection
            reasonable_aorta = [a for a in aorta_landmarks if a['confidence'] > 0.1]
            if reasonable_aorta:
                best_aorta = max(reasonable_aorta, key=lambda x: x['confidence'])
                return AorticValveCandidate(
                    slice_index=best_aorta['slice'],
                    confidence=best_aorta['confidence'] * 0.5,  # Reduced confidence
                    method='simplified_fallback',
                    landmarks_used=['aorta'],
                    anatomical_features=best_aorta,
                    quality_metrics={'aortic_confidence': best_aorta['confidence']}
                )
        
        # Final fallback
        return AorticValveCandidate(
            slice_index=image.shape[2] // 2,
            confidence=0.05,
            method='center_fallback',
            landmarks_used=[],
            anatomical_features={},
            quality_metrics={}
        )
    
    def process_batch(self, ct_files: List[Path], max_workers: Optional[int] = None) -> Dict:
        """
        Process multiple CT scans in parallel.
        
        Args:
            ct_files: List of CT file paths
            max_workers: Maximum number of parallel workers
            
        Returns:
            Batch processing results
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())  # Conservative default
        
        print(f"Processing {len(ct_files)} CT scans with {max_workers} workers...")
        
        results = []
        failed_files = []
        successful_files = []
        
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
                        else:
                            failed_files.append((ct_file, result['error']))
                        
                        pbar.set_postfix({
                            'success': len(successful_files),
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
        
        # Save batch results
        batch_summary = {
            'total_files': len(ct_files),
            'successful': len(successful_files),
            'failed': len(failed_files),
            'success_rate': len(successful_files) / len(ct_files) if ct_files else 0,
            'results': results,
            'failed_files': [(str(f), e) for f, e in failed_files]
        }
        
        summary_path = self.output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {len(successful_files)}/{len(ct_files)}")
        print(f"  Failed: {len(failed_files)}/{len(ct_files)}")
        print(f"  Success rate: {batch_summary['success_rate']:.1%}")
        print(f"  Results saved to: {summary_path}")
        
        return batch_summary
    
    def create_quality_report(self, batch_results: Dict) -> str:
        """Create a quality control report."""
        successful_results = [r for r in batch_results['results'] if r['status'] == 'completed']
        
        if not successful_results:
            return "No successful results to analyze."
        
        # Analyze slice positions and confidences
        slice_positions = [r['valve_slice'] for r in successful_results]
        confidences = [r['confidence'] for r in successful_results]
        
        report = f"""
CT Aortic Valve Level Detection - Quality Report
===============================================

Batch Summary:
- Total files processed: {batch_results['total_files']}
- Successful: {batch_results['successful']}
- Failed: {batch_results['failed']}
- Success rate: {batch_results['success_rate']:.1%}

Slice Position Statistics:
- Mean slice position: {np.mean(slice_positions):.1f}
- Std deviation: {np.std(slice_positions):.1f}
- Range: {min(slice_positions)} - {max(slice_positions)}

Confidence Statistics:
- Mean confidence: {np.mean(confidences):.3f}
- Std deviation: {np.std(confidences):.3f}
- Range: {min(confidences):.3f} - {max(confidences):.3f}

Quality Metrics:
- High confidence (>0.5): {sum(1 for c in confidences if c > 0.5)} ({sum(1 for c in confidences if c > 0.5)/len(confidences)*100:.1f}%)
- Medium confidence (0.2-0.5): {sum(1 for c in confidences if 0.2 <= c <= 0.5)} ({sum(1 for c in confidences if 0.2 <= c <= 0.5)/len(confidences)*100:.1f}%)
- Low confidence (<0.2): {sum(1 for c in confidences if c < 0.2)} ({sum(1 for c in confidences if c < 0.2)/len(confidences)*100:.1f}%)

Recommendations:
- Slice position consistency: {'Good' if np.std(slice_positions) < 10 else 'Review needed'}
- Overall confidence: {'Good' if np.mean(confidences) > 0.3 else 'Review needed'}
"""
        
        report_path = self.output_dir / 'quality_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Quality report saved to: {report_path}")
        return report

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch process CT scans for aortic valve level detection"
    )
    parser.add_argument('input_dir', type=str, help='Directory containing CT scan files')
    parser.add_argument('--output_dir', type=str, default='./output', 
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
    
    print(f"Found {len(ct_files)} CT files to process")
    
    # Initialize processor
    processor = CTBatchProcessor(
        output_dir=Path(args.output_dir),
        target_spacing=tuple(args.spacing)
    )
    
    # Process batch
    results = processor.process_batch(ct_files, max_workers=args.workers)
    
    # Create quality report
    processor.create_quality_report(results)

if __name__ == "__main__":
    main()