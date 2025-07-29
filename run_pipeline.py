#!/usr/bin/env python3
"""
Main Pipeline Runner for CT Aortic Valve Level Detection
Provides a simple interface to run the complete processing pipeline.
"""

import argparse
import json
from pathlib import Path
from ct_batch_processor import CTBatchProcessor
from visualization_tools import CTVisualizationTools

def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(
        description="CT Aortic Valve Level Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CT scans in current directory
  python run_pipeline.py .

  # Process with custom output directory
  python run_pipeline.py ./ct_data --output ./results

  # Process with specific number of workers
  python run_pipeline.py ./ct_data --workers 8

  # Process only first 5 files (for testing)
  python run_pipeline.py ./ct_data --max_files 5
        """
    )
    
    parser.add_argument('input_dir', type=str, 
                       help='Directory containing CT scan files (*.nii.gz)')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--config', type=str, default='./config.json',
                       help='Configuration file (default: ./config.json)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization creation')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Using configuration from: {config_path}")
    else:
        print(f"Configuration file not found: {config_path}")
        print("Using default settings")
        config = {
            "processing": {"target_spacing": [1.0, 1.0, 2.0]},
            "output": {"create_dashboard": True, "create_montage": True}
        }
    
    # Find CT files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return 1
    
    ct_files = list(input_path.glob('*.nii.gz'))
    if not ct_files:
        print(f"No *.nii.gz files found in: {input_path}")
        return 1
    
    # Limit files if requested
    if args.max_files:
        ct_files = ct_files[:args.max_files]
        print(f"Limited to first {args.max_files} files for testing")
    
    print(f"Found {len(ct_files)} CT files to process")
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    target_spacing = config['processing'].get('target_spacing', [1.0, 1.0, 2.0])
    processor = CTBatchProcessor(
        output_dir=output_dir,
        target_spacing=tuple(target_spacing)
    )
    
    # Process batch
    max_workers = args.workers or config['processing'].get('max_workers', None)
    print(f"\nStarting batch processing...")
    results = processor.process_batch(ct_files, max_workers=max_workers)
    
    # Create quality report
    processor.create_quality_report(results)
    
    # Create visualizations if requested
    if not args.no_viz and config['output'].get('create_dashboard', True):
        print(f"\nCreating visualizations...")
        viz_tools = CTVisualizationTools(output_dir)
        
        # Quality dashboard
        if results['successful'] > 0:
            viz_tools.create_quality_dashboard(results)
            
            # Slice montage
            if config['output'].get('create_montage', True):
                slice_dir = output_dir / 'slices'
                if slice_dir.exists():
                    slice_paths = list(slice_dir.glob('*.npy'))
                    if slice_paths:
                        viz_tools.create_slice_montage(slice_paths[:16])  # Max 16 images
        else:
            print("No successful results to visualize")
    
    # Print final summary
    print(f"\n" + "="*60)
    print(f"PIPELINE COMPLETED")
    print(f"="*60)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Files processed: {results['total_files']}")
    print(f"Successful: {results['successful']} ({results['success_rate']:.1%})")
    print(f"Failed: {results['failed']}")
    
    if results['successful'] > 0:
        print(f"\nOutput files:")
        print(f"  - Extracted slices: {output_dir}/slices/")
        print(f"  - Metadata: {output_dir}/metadata/")
        print(f"  - Batch summary: {output_dir}/batch_summary.json")
        print(f"  - Quality report: {output_dir}/quality_report.txt")
        if not args.no_viz:
            print(f"  - Visualizations: {output_dir}/")
    
    if results['failed'] > 0:
        print(f"\nFailed files:")
        for filename, error in results['failed_files'][:5]:  # Show first 5 failures
            print(f"  - {filename}: {error}")
        if len(results['failed_files']) > 5:
            print(f"  ... and {len(results['failed_files']) - 5} more")
    
    return 0 if results['successful'] > 0 else 1

if __name__ == "__main__":
    exit(main())