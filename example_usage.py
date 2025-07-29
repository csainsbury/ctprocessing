#!/usr/bin/env python3
"""
Example Usage of CT Aortic Valve Level Detection Pipeline
Demonstrates different ways to use the pipeline components.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from image_loader import CTImageLoader
from landmark_detection import AnatomicalLandmarkDetector
from aortic_valve_detection import AorticValveLevelDetector
from ct_batch_processor import CTBatchProcessor
from visualization_tools import CTVisualizationTools

def example_single_scan_processing():
    """Example: Process a single CT scan step by step."""
    print("Example 1: Processing a single CT scan")
    print("-" * 40)
    
    # Find a CT file
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files found in current directory")
        return
    
    ct_file = ct_files[0]
    print(f"Processing: {ct_file.name}")
    
    # Step 1: Load and preprocess
    loader = CTImageLoader(target_spacing=(1.0, 1.0, 2.0))
    processed_image, metadata = loader.process_image(ct_file)
    print(f"Preprocessed shape: {processed_image.shape}")
    
    # Step 2: Detect landmarks
    landmark_detector = AnatomicalLandmarkDetector()
    landmarks = landmark_detector.detect_all_landmarks(processed_image)
    
    # Step 3: Detect aortic valve level
    valve_detector = AorticValveLevelDetector()
    valve_candidate = valve_detector.detect_aortic_valve_level(processed_image, landmarks)
    
    print(f"Selected valve slice: {valve_candidate.slice_index}")
    print(f"Confidence: {valve_candidate.confidence:.3f}")
    print(f"Method: {valve_candidate.method}")
    
    # Visualize result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show the selected slice
    valve_slice = processed_image[:, :, valve_candidate.slice_index]
    ax1.imshow(valve_slice, cmap='gray')
    ax1.set_title(f'Aortic Valve Level - Slice {valve_candidate.slice_index}')
    ax1.axis('off')
    
    # Show sagittal view with selected slice marked
    sagittal = processed_image[processed_image.shape[0]//2, :, :]
    ax2.imshow(sagittal, cmap='gray', aspect='auto')
    ax2.axvline(x=valve_candidate.slice_index, color='red', linewidth=2)
    ax2.set_title('Sagittal View (Red line = Valve Level)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_single_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return valve_candidate

def example_batch_processing():
    """Example: Batch process multiple CT scans."""
    print("\nExample 2: Batch processing")
    print("-" * 40)
    
    # Find CT files
    ct_files = list(Path('.').glob('*.nii.gz'))
    if len(ct_files) < 2:
        print("Need at least 2 CT files for batch processing example")
        return
    
    # Limit to first 3 files for example
    ct_files = ct_files[:3]
    print(f"Processing {len(ct_files)} files...")
    
    # Initialize batch processor
    output_dir = Path('./example_output')
    processor = CTBatchProcessor(output_dir=output_dir)
    
    # Process batch
    results = processor.process_batch(ct_files, max_workers=2)
    
    # Create quality report
    processor.create_quality_report(results)
    
    print(f"Batch processing completed:")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Results saved to: {output_dir}")
    
    return results

def example_visualization():
    """Example: Create visualizations from results."""
    print("\nExample 3: Creating visualizations")
    print("-" * 40)
    
    # Check if we have batch results
    results_file = Path('./example_output/batch_summary.json')
    if not results_file.exists():
        print("No batch results found. Run example_batch_processing() first.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize visualization tools
    viz_tools = CTVisualizationTools(Path('./example_output'))
    
    # Create quality dashboard
    viz_tools.create_quality_dashboard(results)
    
    # Create slice montage
    slice_dir = Path('./example_output/slices')
    if slice_dir.exists():
        slice_paths = list(slice_dir.glob('*.npy'))
        if slice_paths:
            viz_tools.create_slice_montage(slice_paths)
        else:
            print("No slice files found")
    else:
        print("Slice directory not found")

def example_quality_analysis():
    """Example: Analyze quality metrics from batch results."""
    print("\nExample 4: Quality analysis")
    print("-" * 40)
    
    # Load batch results
    results_file = Path('./example_output/batch_summary.json')
    if not results_file.exists():
        print("No batch results found. Run example_batch_processing() first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    successful_results = [r for r in results['results'] if r['status'] == 'completed']
    
    if not successful_results:
        print("No successful results to analyze")
        return
    
    # Extract metrics
    slice_positions = [r['valve_slice'] for r in successful_results]
    confidences = [r['confidence'] for r in successful_results]
    
    # Calculate statistics
    print(f"Quality Analysis Results:")
    print(f"  Files processed: {len(successful_results)}")
    print(f"  Slice position - Mean: {np.mean(slice_positions):.1f}, Std: {np.std(slice_positions):.1f}")
    print(f"  Confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
    
    # Quality categories
    high_conf = sum(1 for c in confidences if c > 0.5)
    med_conf = sum(1 for c in confidences if 0.2 <= c <= 0.5)
    low_conf = sum(1 for c in confidences if c < 0.2)
    
    print(f"  Quality distribution:")
    print(f"    High confidence (>0.5): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"    Medium confidence (0.2-0.5): {med_conf} ({med_conf/len(confidences)*100:.1f}%)")
    print(f"    Low confidence (<0.2): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
    
    # Recommendations
    print(f"  Recommendations:")
    if np.std(slice_positions) < 10:
        print(f"    ✓ Slice consistency is good (std < 10)")
    else:
        print(f"    ⚠ Slice consistency needs review (std >= 10)")
    
    if np.mean(confidences) > 0.3:
        print(f"    ✓ Overall confidence is acceptable")
    else:
        print(f"    ⚠ Overall confidence is low - review parameters")

def example_custom_configuration():
    """Example: Using custom configuration."""
    print("\nExample 5: Custom configuration")
    print("-" * 40)
    
    # Create custom configuration
    custom_config = {
        "processing": {
            "target_spacing": [0.5, 0.5, 1.0],  # Higher resolution
            "max_workers": 2
        },
        "detection": {
            "aorta_radius_range": [10, 30],  # Smaller expected radius
            "expected_valve_z_range": [30, 100]  # Different anatomical range
        },
        "quality_control": {
            "min_confidence_threshold": 0.2
        }
    }
    
    # Save custom config
    config_path = Path('./custom_config.json')
    with open(config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"Custom configuration saved to: {config_path}")
    print("Usage: python run_pipeline.py . --config custom_config.json")
    
    return custom_config

def main():
    """Run all examples."""
    print("CT Aortic Valve Level Detection - Examples")
    print("=" * 50)
    
    # Check if we have CT files
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files (*.nii.gz) found in current directory.")
        print("Please ensure CT scan files are available before running examples.")
        return
    
    print(f"Found {len(ct_files)} CT files to work with.")
    
    # Run examples
    try:
        # Example 1: Single scan processing
        valve_result = example_single_scan_processing()
        
        # Example 2: Batch processing
        batch_results = example_batch_processing()
        
        # Example 3: Visualization
        example_visualization()
        
        # Example 4: Quality analysis
        example_quality_analysis()
        
        # Example 5: Custom configuration
        example_custom_configuration()
        
        print(f"\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated files:")
        print("  - example_single_result.png")
        print("  - example_output/ directory")
        print("  - custom_config.json")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()