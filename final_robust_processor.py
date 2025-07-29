#!/usr/bin/env python3
"""
Final Robust CT Batch Processor
Handles orientation detection, sequence correction, and comprehensive validation.
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
from robust_valve_detection import RobustAorticValveLevelDetector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FinalRobustCTProcessor:
    """Final robust CT processor with comprehensive orientation and validation handling."""
    
    def __init__(self, output_dir: Path, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)):
        """Initialize final robust processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create comprehensive output structure
        (self.output_dir / 'slices').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'orientation_analysis').mkdir(exist_ok=True)
        (self.output_dir / 'quality_reports').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Initialize robust processing pipeline
        self.image_loader = CTImageLoader(target_spacing=target_spacing)
        self.landmark_detector = AnatomicalLandmarkDetector()
        self.valve_detector = RobustAorticValveLevelDetector()
        
    def process_single_ct_robust(self, ct_path: Path) -> Dict:
        """
        Process single CT with full robust pipeline including orientation correction.
        """
        try:
            result = {
                'filename': ct_path.name,
                'status': 'processing',
                'error': None,
                'valve_slice': None,
                'confidence': None,
                'quality_grade': None,
                'orientation_corrected': False,
                'processing_time': None,
                'warnings': []
            }
            
            import time
            start_time = time.time()
            
            print(f"🔍 Processing {ct_path.name} with robust pipeline...")
            
            # Step 1: Load and preprocess
            processed_image, metadata = self.image_loader.process_image(ct_path)
            
            # Step 2: Detect landmarks
            landmarks = self.landmark_detector.detect_all_landmarks(processed_image)
            
            # Step 3: Robust valve detection with orientation correction
            valve_candidate = self.valve_detector.detect_aortic_valve_level(processed_image, landmarks)
            
            # Extract valve slice from corrected image
            valve_slice = processed_image[:, :, valve_candidate.slice_index]
            slice_path = self.output_dir / 'slices' / f"{ct_path.stem}_valve_slice.npy"
            np.save(slice_path, valve_slice)
            
            # Comprehensive metadata
            comprehensive_metadata = {
                'original_file': str(ct_path),
                'valve_slice_index': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'quality_grade': self._determine_quality_grade(valve_candidate),
                'method': valve_candidate.method,
                'landmarks_used': valve_candidate.landmarks_used,
                'orientation_info': valve_candidate.orientation_info,
                'validation_scores': valve_candidate.validation_scores,
                'anatomical_features': valve_candidate.anatomical_features,
                'quality_metrics': valve_candidate.quality_metrics,
                'processing_metadata': {
                    'original_shape': metadata['original_shape'],
                    'final_shape': metadata['final_shape'],
                    'voxel_spacing': metadata['voxel_spacing'],
                    'orientation': metadata['orientation']
                },
                'file_paths': {
                    'slice_path': str(slice_path)
                },
                'relative_position': valve_candidate.slice_index / processed_image.shape[2],
                'processing_timestamp': time.time()
            }
            
            # Save comprehensive metadata
            metadata_path = self.output_dir / 'metadata' / f"{ct_path.stem}_comprehensive_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(comprehensive_metadata, f, indent=2, default=str)
            
            # Save orientation analysis separately
            orientation_path = self.output_dir / 'orientation_analysis' / f"{ct_path.stem}_orientation.json"
            with open(orientation_path, 'w') as f:
                json.dump(valve_candidate.orientation_info, f, indent=2, default=str)
            
            # Collect warnings
            warnings = []
            if 'warnings' in valve_candidate.orientation_info:
                warnings.extend(valve_candidate.orientation_info['warnings'])
            
            # Quality-based warnings
            if valve_candidate.confidence < 0.3:
                warnings.append("Low confidence detection")
            
            relative_pos = valve_candidate.slice_index / processed_image.shape[2]
            if not (0.2 <= relative_pos <= 0.5):
                warnings.append(f"Slice position outside typical range: {relative_pos:.2f}")
            
            # Update result
            result.update({
                'status': 'completed',
                'valve_slice': valve_candidate.slice_index,
                'confidence': valve_candidate.confidence,
                'quality_grade': comprehensive_metadata['quality_grade'],
                'orientation_corrected': valve_candidate.orientation_info.get('sequence_reversed', False),
                'processing_time': time.time() - start_time,
                'warnings': warnings,
                'method': valve_candidate.method,
                'relative_position': relative_pos,
                'slice_path': str(slice_path),
                'metadata_path': str(metadata_path),
                'orientation_confidence': valve_candidate.orientation_info.get('orientation_confidence', 0.5)
            })
            
            print(f"  ✅ Completed: slice {valve_candidate.slice_index}, "
                  f"confidence {valve_candidate.confidence:.3f}, "
                  f"grade {comprehensive_metadata['quality_grade']}")
            
            if valve_candidate.orientation_info.get('sequence_reversed'):
                print(f"  🔄 Orientation corrected")
            
            if warnings:
                print(f"  ⚠️  Warnings: {len(warnings)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            return {
                'filename': ct_path.name,
                'status': 'failed',
                'error': error_msg,
                'valve_slice': None,
                'confidence': None,
                'quality_grade': 'Failed',
                'orientation_corrected': False,
                'processing_time': None,
                'warnings': [error_msg]
            }
    
    def _determine_quality_grade(self, candidate) -> str:
        """Determine overall quality grade for the detection."""
        
        confidence = candidate.confidence
        avg_validation = np.mean(list(candidate.validation_scores.values()))
        orientation_conf = candidate.orientation_info.get('orientation_confidence', 0.5)
        
        # Comprehensive quality assessment
        overall_score = (confidence * 0.4 + avg_validation * 0.4 + orientation_conf * 0.2)
        
        if overall_score >= 0.7 and confidence >= 0.5:
            return 'Excellent'
        elif overall_score >= 0.5 and confidence >= 0.3:
            return 'Good'
        elif overall_score >= 0.3 and confidence >= 0.2:
            return 'Fair'
        else:
            return 'Poor'
    
    def process_batch_robust(self, ct_files: List[Path], max_workers: Optional[int] = None) -> Dict:
        """Process batch with comprehensive robust pipeline."""
        
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        print(f"🚀 Processing {len(ct_files)} CT scans with FINAL ROBUST pipeline ({max_workers} workers)")
        print("Features: Orientation detection, sequence correction, comprehensive validation")
        
        results = []
        failed_files = []
        successful_files = []
        orientation_corrected_files = []
        warning_files = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_single_ct_robust, ct_file): ct_file
                for ct_file in ct_files
            }
            
            # Process with comprehensive progress tracking
            with tqdm(total=len(ct_files), desc="Robust CT Processing") as pbar:
                for future in as_completed(future_to_file):
                    ct_file = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'completed':
                            successful_files.append(ct_file)
                            
                            if result['orientation_corrected']:
                                orientation_corrected_files.append(ct_file)
                            
                            if result['warnings']:
                                warning_files.append((ct_file, result['warnings']))
                        else:
                            failed_files.append((ct_file, result['error']))
                        
                        # Enhanced progress display
                        pbar.set_postfix({
                            'success': len(successful_files),
                            'corrected': len(orientation_corrected_files),
                            'warnings': len(warning_files),
                            'failed': len(failed_files)
                        })
                        
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        failed_files.append((ct_file, error_msg))
                        results.append({
                            'filename': ct_file.name,
                            'status': 'failed',
                            'error': error_msg
                        })
                    
                    pbar.update(1)
        
        # Comprehensive batch analysis
        batch_summary = self._create_comprehensive_batch_summary(
            ct_files, results, successful_files, failed_files, 
            orientation_corrected_files, warning_files
        )
        
        # Save results
        summary_path = self.output_dir / 'final_robust_batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        # Print comprehensive summary
        self._print_batch_summary(batch_summary)
        
        return batch_summary
    
    def _create_comprehensive_batch_summary(self, ct_files, results, successful_files, 
                                          failed_files, orientation_corrected_files, warning_files) -> Dict:
        """Create comprehensive batch processing summary."""
        
        successful_results = [r for r in results if r['status'] == 'completed']
        
        summary = {
            'processing_info': {
                'total_files': len(ct_files),
                'successful': len(successful_files),
                'failed': len(failed_files),
                'orientation_corrected': len(orientation_corrected_files),
                'with_warnings': len(warning_files),
                'success_rate': len(successful_files) / len(ct_files) if ct_files else 0
            },
            'results': results,
            'failed_files': [(str(f), e) for f, e in failed_files],
            'warning_files': [(str(f), w) for f, w in warning_files],
            'orientation_corrections': [str(f) for f in orientation_corrected_files]
        }
        
        if successful_results:
            # Quality analysis
            quality_grades = [r.get('quality_grade', 'Unknown') for r in successful_results]
            quality_distribution = {}
            for grade in quality_grades:
                quality_distribution[grade] = quality_distribution.get(grade, 0) + 1
            
            # Statistical analysis
            confidences = [r['confidence'] for r in successful_results]
            positions = [r['relative_position'] for r in successful_results]
            orientation_confidences = [r.get('orientation_confidence', 0.5) for r in successful_results]
            
            summary['quality_analysis'] = {
                'quality_distribution': quality_distribution,
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                },
                'position_stats': {
                    'mean': np.mean(positions),
                    'std': np.std(positions),
                    'in_expected_range': sum(1 for p in positions if 0.2 <= p <= 0.5),
                    'expected_range_percentage': sum(1 for p in positions if 0.2 <= p <= 0.5) / len(positions) * 100
                },
                'orientation_stats': {
                    'mean_confidence': np.mean(orientation_confidences),
                    'corrections_applied': len(orientation_corrected_files),
                    'correction_rate': len(orientation_corrected_files) / len(successful_files) * 100 if successful_files else 0
                }
            }
            
            # Method analysis
            methods_used = [r.get('method', 'unknown') for r in successful_results]
            method_distribution = {}
            for method in methods_used:
                method_distribution[method] = method_distribution.get(method, 0) + 1
            
            summary['method_analysis'] = {
                'method_distribution': method_distribution,
                'fallback_usage': method_distribution.get('robust_fallback', 0)
            }
        
        return summary
    
    def _print_batch_summary(self, summary: Dict):
        """Print comprehensive batch processing summary."""
        
        info = summary['processing_info']
        
        print(f"\n" + "="*80)
        print("FINAL ROBUST PROCESSING SUMMARY")
        print("="*80)
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"  Total files processed: {info['total_files']}")
        print(f"  ✅ Successful: {info['successful']} ({info['success_rate']:.1%})")
        print(f"  ❌ Failed: {info['failed']}")
        print(f"  🔄 Orientation corrected: {info['orientation_corrected']}")
        print(f"  ⚠️  With warnings: {info['with_warnings']}")
        
        if 'quality_analysis' in summary:
            qa = summary['quality_analysis']
            
            print(f"\n🎯 QUALITY ANALYSIS:")
            print(f"  Quality Distribution:")
            for grade, count in qa['quality_distribution'].items():
                percentage = count / info['successful'] * 100
                print(f"    {grade}: {count} ({percentage:.1f}%)")
            
            print(f"\n  Confidence Statistics:")
            cs = qa['confidence_stats']
            print(f"    Mean: {cs['mean']:.3f} ± {cs['std']:.3f}")
            print(f"    Range: {cs['min']:.3f} - {cs['max']:.3f}")
            
            print(f"\n  Position Analysis:")
            ps = qa['position_stats']
            print(f"    Mean relative position: {ps['mean']:.3f}")
            print(f"    Position consistency (std): {ps['std']:.3f}")
            print(f"    In expected range (0.2-0.5): {ps['in_expected_range']}/{info['successful']} ({ps['expected_range_percentage']:.1f}%)")
            
            print(f"\n🧭 ORIENTATION ANALYSIS:")
            os = qa['orientation_stats']
            print(f"    Mean orientation confidence: {os['mean_confidence']:.3f}")
            print(f"    Sequences corrected: {os['corrections_applied']} ({os['correction_rate']:.1f}%)")
        
        if summary.get('orientation_corrections'):
            print(f"\n🔄 ORIENTATION CORRECTIONS APPLIED:")
            for filename in summary['orientation_corrections'][:5]:  # Show first 5
                print(f"    • {Path(filename).name}")
            if len(summary['orientation_corrections']) > 5:
                print(f"    ... and {len(summary['orientation_corrections']) - 5} more")
        
        if summary.get('warning_files'):
            print(f"\n⚠️  FILES WITH WARNINGS:")
            for filename, warnings in summary['warning_files'][:3]:  # Show first 3
                print(f"    • {Path(filename).name}: {warnings[0]}")
            if len(summary['warning_files']) > 3:
                print(f"    ... and {len(summary['warning_files']) - 3} more")
        
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"    Results: {self.output_dir}")
        print(f"    Slices: {self.output_dir}/slices/")
        print(f"    Metadata: {self.output_dir}/metadata/")
        print(f"    Orientation analysis: {self.output_dir}/orientation_analysis/")
        
        print(f"\n🎉 ROBUST PROCESSING COMPLETE!")
        
        if info['orientation_corrected'] > 0:
            print(f"✨ Successfully corrected {info['orientation_corrected']} cases with sequence reversal!")

def main():
    """Main function for final robust processing."""
    parser = argparse.ArgumentParser(
        description="Final Robust CT Processor with Orientation Detection and Comprehensive Validation"
    )
    parser.add_argument('input_dir', type=str, help='Directory containing CT scan files')
    parser.add_argument('--output_dir', type=str, default='./final_robust_output', 
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
    
    print(f"🔍 Found {len(ct_files)} CT files for FINAL ROBUST processing")
    
    # Initialize final processor
    processor = FinalRobustCTProcessor(
        output_dir=Path(args.output_dir),
        target_spacing=tuple(args.spacing)
    )
    
    # Process with full robust pipeline
    results = processor.process_batch_robust(ct_files, max_workers=args.workers)

if __name__ == "__main__":
    main()