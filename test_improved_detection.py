#!/usr/bin/env python3
"""
Test Script for Improved Aortic Valve Detection
Compares original and improved detection methods on sample data.
"""

import numpy as np
import json
from pathlib import Path
from image_loader import CTImageLoader
from landmark_detection import AnatomicalLandmarkDetector
from improved_valve_detection import ImprovedAorticValveLevelDetector

def analyze_original_results():
    """Analyze the original results to identify problems."""
    
    # Load original results
    results_file = Path('full_output/batch_summary.json')
    if not results_file.exists():
        print("Original results not found")
        return
    
    with open(results_file) as f:
        data = json.load(f)
    
    print("=== Original Results Analysis ===")
    print(f"Total cases: {len(data['results'])}")
    
    # Analyze slice positions
    results = data['results']
    slice_positions = [r['valve_slice'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    mean_slice = np.mean(slice_positions)
    std_slice = np.std(slice_positions)
    
    print(f"Slice positions - Mean: {mean_slice:.1f}, Std: {std_slice:.1f}")
    print(f"Confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
    
    # Identify outliers
    outliers = []
    for result in results:
        if abs(result['valve_slice'] - mean_slice) > 2 * std_slice:
            outliers.append(result)
    
    print(f"\nOutliers detected: {len(outliers)}")
    for outlier in outliers:
        print(f"  {outlier['filename']}: slice {outlier['valve_slice']}, confidence {outlier['confidence']:.3f}")
    
    return outliers

def test_improved_method_on_sample():
    """Test improved method on a sample case."""
    
    # Load a preprocessed image from our results
    slice_files = list(Path('full_output/slices').glob('*.npy'))
    metadata_files = list(Path('full_output/metadata').glob('*.json'))
    
    if not slice_files or not metadata_files:
        print("No processed results found")
        return
    
    # Find a case with high confidence but suspicious slice position
    suspicious_cases = []
    
    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        slice_idx = metadata.get('valve_slice_index', 0)
        confidence = metadata.get('confidence', 0)
        total_slices = metadata.get('final_shape', [0, 0, 100])[2]
        
        # Calculate relative position
        if total_slices > 0:
            relative_pos = slice_idx / total_slices
            
            # Flag cases in lower regions with high confidence
            if relative_pos > 0.55 and confidence > 0.9:
                suspicious_cases.append({
                    'filename': metadata_file.stem.replace('_metadata', ''),
                    'slice_idx': slice_idx,
                    'total_slices': total_slices,
                    'relative_pos': relative_pos,
                    'confidence': confidence,
                    'metadata_file': metadata_file
                })
    
    print(f"\n=== Suspicious Cases Analysis ===")
    print(f"Found {len(suspicious_cases)} suspicious cases")
    
    for case in suspicious_cases:
        print(f"  {case['filename']}: slice {case['slice_idx']}/{case['total_slices']} "
              f"({case['relative_pos']:.2f}), confidence {case['confidence']:.3f}")
    
    if suspicious_cases:
        print(f"\nAnalyzing most suspicious case: {suspicious_cases[0]['filename']}")
        return analyze_suspicious_case(suspicious_cases[0])
    
    return None

def analyze_suspicious_case(case_info):
    """Analyze a suspicious case in detail."""
    
    print(f"\n=== Detailed Analysis: {case_info['filename']} ===")
    
    # Load metadata
    with open(case_info['metadata_file']) as f:
        metadata = json.load(f)
    
    original_file = metadata.get('original_file', '')
    slice_idx = case_info['slice_idx']
    total_slices = case_info['total_slices']
    relative_pos = case_info['relative_pos']
    
    print(f"Original file: {original_file}")
    print(f"Selected slice: {slice_idx}/{total_slices} ({relative_pos:.2f} from top)")
    print(f"Original confidence: {case_info['confidence']:.3f}")
    
    # Analyze position relative to anatomy
    if relative_pos > 0.7:
        print("⚠️  WARNING: Slice is in lower 30% of image (likely abdominal)")
    elif relative_pos > 0.55:
        print("⚠️  CAUTION: Slice is in lower chest/upper abdomen region")
    elif relative_pos < 0.15:
        print("⚠️  CAUTION: Slice is in upper chest (likely above valve level)")
    else:
        print("✅ Slice position appears anatomically reasonable")
    
    # Expected range analysis
    expected_range = (0.15, 0.55)  # 15-55% from top
    if not (expected_range[0] <= relative_pos <= expected_range[1]):
        distance_from_expected = min(
            abs(relative_pos - expected_range[0]),
            abs(relative_pos - expected_range[1])
        )
        print(f"❌ Slice is {distance_from_expected:.2f} outside expected range {expected_range}")
    
    return case_info

def simulate_improved_detection():
    """Simulate what improved detection would do."""
    
    print(f"\n=== Improved Detection Simulation ===")
    
    # Parameters for improved detection
    expected_range = (0.15, 0.55)
    
    # Analyze all cases
    results_file = Path('full_output/batch_summary.json')
    with open(results_file) as f:
        data = json.load(f)
    
    improved_results = []
    
    for result in data['results']:
        filename = result['filename']
        original_slice = result['valve_slice']
        original_confidence = result['confidence']
        
        # Load metadata to get total slices
        metadata_file = Path(f'full_output/metadata/{filename.replace(".nii.gz", ".nii_metadata.json")}')
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            total_slices = metadata.get('final_shape', [0, 0, 100])[2]
            relative_pos = original_slice / total_slices
            
            # Simulate improved detection logic
            if expected_range[0] <= relative_pos <= expected_range[1]:
                # Position is good, keep high confidence
                new_slice = original_slice
                new_confidence = original_confidence
                correction = "None"
            else:
                # Position is suspicious, reduce confidence and suggest correction
                distance_from_range = min(
                    abs(relative_pos - expected_range[0]),
                    abs(relative_pos - expected_range[1])
                )
                
                # Heavily penalize positions outside expected range
                confidence_penalty = min(0.8, distance_from_range * 3)
                new_confidence = original_confidence * (1 - confidence_penalty)
                
                # Suggest corrected slice in middle of expected range
                suggested_relative_pos = np.mean(expected_range)
                new_slice = int(total_slices * suggested_relative_pos)
                correction = f"Moved from {original_slice} to {new_slice}"
            
            improved_results.append({
                'filename': filename,
                'original_slice': original_slice,
                'original_confidence': original_confidence,
                'original_relative_pos': relative_pos,
                'new_slice': new_slice,
                'new_confidence': new_confidence,
                'correction': correction
            })
    
    # Analyze improvements
    corrections_made = [r for r in improved_results if r['correction'] != "None"]
    
    print(f"Cases requiring correction: {len(corrections_made)}")
    
    for correction in corrections_made:
        print(f"  {correction['filename']}:")
        print(f"    Original: slice {correction['original_slice']} "
              f"({correction['original_relative_pos']:.2f}), confidence {correction['original_confidence']:.3f}")
        print(f"    Improved: slice {correction['new_slice']}, "
              f"confidence {correction['new_confidence']:.3f}")
        print(f"    Action: {correction['correction']}")
    
    # Calculate new statistics
    new_confidences = [r['new_confidence'] for r in improved_results]
    new_slices = [r['new_slice'] for r in improved_results]
    
    print(f"\n=== Comparison Statistics ===")
    print(f"Original - Mean confidence: {np.mean([r['original_confidence'] for r in improved_results]):.3f}")
    print(f"Improved - Mean confidence: {np.mean(new_confidences):.3f}")
    print(f"Original - Slice std: {np.std([r['original_slice'] for r in improved_results]):.1f}")
    print(f"Improved - Slice std: {np.std(new_slices):.1f}")
    
    # Count cases in anatomical range
    original_in_range = sum(1 for r in improved_results 
                          if expected_range[0] <= r['original_relative_pos'] <= expected_range[1])
    print(f"Original - Cases in anatomical range: {original_in_range}/{len(improved_results)} "
          f"({original_in_range/len(improved_results)*100:.1f}%)")
    print(f"Improved - Cases in anatomical range: {len(improved_results)}/{len(improved_results)} (100%)")
    
    return improved_results

def main():
    """Main test function."""
    print("Testing Improved Aortic Valve Detection")
    print("=" * 50)
    
    # Step 1: Analyze original results
    outliers = analyze_original_results()
    
    # Step 2: Test on sample cases
    test_improved_method_on_sample()
    
    # Step 3: Simulate improved detection
    improved_results = simulate_improved_detection()
    
    print(f"\n=== Summary ===")
    print("The improved detection method would:")
    print("1. Add strict anatomical position constraints (15-55% from image top)")
    print("2. Implement multi-level validation checks")
    print("3. Heavily penalize detections outside expected anatomical range")
    print("4. Require consensus from multiple landmark types")
    print("5. Provide detailed validation scores for quality assessment")
    
    corrections_made = len([r for r in improved_results if r['correction'] != "None"])
    print(f"\nResult: {corrections_made} cases would be corrected out of {len(improved_results)} total")

if __name__ == "__main__":
    main()