#!/usr/bin/env python3
"""
Convert extracted CT slices from .npy to .jpg format
Provides multiple conversion options with quality control.
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

class SliceConverter:
    """Convert CT slices to various image formats."""
    
    def __init__(self, output_quality=95):
        """Initialize converter with quality settings."""
        self.output_quality = output_quality
    
    def convert_simple(self, npy_path: Path, output_path: Path, 
                      normalize: bool = True) -> bool:
        """
        Simple conversion using PIL.
        
        Args:
            npy_path: Input .npy file
            output_path: Output image path
            normalize: Whether to normalize intensity
        """
        try:
            # Load numpy array
            slice_data = np.load(npy_path)
            
            # Handle normalization
            if normalize:
                # Data should already be 0-1 from preprocessing
                if slice_data.max() <= 1.0:
                    img_data = (slice_data * 255).astype(np.uint8)
                else:
                    # If not normalized, do it now
                    img_data = ((slice_data - slice_data.min()) / 
                              (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            else:
                img_data = slice_data.astype(np.uint8)
            
            # Convert and save
            img = Image.fromarray(img_data, mode='L')
            
            # Determine format from extension
            ext = output_path.suffix.lower()
            if ext == '.jpg' or ext == '.jpeg':
                img.save(output_path, 'JPEG', quality=self.output_quality)
            elif ext == '.png':
                img.save(output_path, 'PNG')
            elif ext == '.tiff' or ext == '.tif':
                img.save(output_path, 'TIFF')
            else:
                # Default to JPEG
                img.save(output_path, 'JPEG', quality=self.output_quality)
            
            return True
            
        except Exception as e:
            print(f"Error converting {npy_path.name}: {e}")
            return False
    
    def convert_with_contrast(self, npy_path: Path, output_path: Path,
                            window_center: float = 0.5, window_width: float = 1.0,
                            figsize: tuple = (8, 8), dpi: int = 150) -> bool:
        """
        Convert with contrast windowing using matplotlib.
        
        Args:
            npy_path: Input .npy file
            output_path: Output image path
            window_center: Center of intensity window
            window_width: Width of intensity window
            figsize: Figure size
            dpi: Output resolution
        """
        try:
            # Load data
            slice_data = np.load(npy_path)
            
            # Calculate window limits
            vmin = window_center - window_width/2
            vmax = window_center + window_width/2
            
            # Create figure
            plt.figure(figsize=figsize, facecolor='black')
            plt.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Save
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       pad_inches=0, facecolor='black')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error converting {npy_path.name}: {e}")
            return False
    
    def convert_with_annotations(self, npy_path: Path, output_path: Path,
                               metadata_path: Path = None) -> bool:
        """
        Convert with annotations (crosshairs, metadata overlay).
        
        Args:
            npy_path: Input .npy file
            output_path: Output image path
            metadata_path: Optional metadata file for annotations
        """
        try:
            # Load data
            slice_data = np.load(npy_path)
            
            # Load metadata if available
            metadata = {}
            if metadata_path and metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create figure
            plt.figure(figsize=(10, 8), facecolor='white')
            plt.imshow(slice_data, cmap='gray')
            
            # Add crosshairs at center
            center_y, center_x = slice_data.shape[0]//2, slice_data.shape[1]//2
            plt.axhline(y=center_y, color='red', linestyle='--', alpha=0.7, linewidth=1)
            plt.axvline(x=center_x, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add metadata text if available
            if metadata:
                info_text = f"Slice: {metadata.get('valve_slice_index', 'N/A')}\n"
                info_text += f"Confidence: {metadata.get('confidence', 'N/A'):.3f}\n"
                info_text += f"Method: {metadata.get('method', 'N/A')}"
                
                plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add title
            plt.title(f'Aortic Valve Level - {npy_path.stem}', pad=20)
            plt.axis('off')
            
            # Save
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error converting {npy_path.name}: {e}")
            return False
    
    def batch_convert(self, input_dir: Path, output_dir: Path, 
                     conversion_type: str = 'simple', 
                     output_format: str = 'jpg',
                     metadata_dir: Path = None) -> dict:
        """
        Batch convert all .npy files in directory.
        
        Args:
            input_dir: Directory containing .npy files
            output_dir: Output directory
            conversion_type: 'simple', 'contrast', or 'annotated'
            output_format: 'jpg', 'png', or 'tiff'
            metadata_dir: Directory with metadata files (for annotated mode)
        """
        # Find all .npy files
        npy_files = list(input_dir.glob('*.npy'))
        if not npy_files:
            print(f"No .npy files found in {input_dir}")
            return {'success': 0, 'failed': 0, 'files': []}
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Convert files
        successful = 0
        failed = 0
        results = []
        
        print(f"Converting {len(npy_files)} files to {output_format.upper()}...")
        
        for npy_file in tqdm(npy_files, desc="Converting"):
            # Determine output path
            output_file = output_dir / f"{npy_file.stem}.{output_format}"
            
            # Select conversion method
            success = False
            if conversion_type == 'simple':
                success = self.convert_simple(npy_file, output_file)
            elif conversion_type == 'contrast':
                success = self.convert_with_contrast(npy_file, output_file)
            elif conversion_type == 'annotated':
                metadata_file = None
                if metadata_dir:
                    metadata_file = metadata_dir / f"{npy_file.stem.replace('_valve_slice', '_metadata')}.json"
                success = self.convert_with_annotations(npy_file, output_file, metadata_file)
            
            # Track results
            if success:
                successful += 1
                results.append({'file': npy_file.name, 'status': 'success', 'output': str(output_file)})
            else:
                failed += 1
                results.append({'file': npy_file.name, 'status': 'failed', 'output': None})
        
        summary = {
            'total': len(npy_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(npy_files) if npy_files else 0,
            'files': results
        }
        
        print(f"\nConversion completed:")
        print(f"  Successful: {successful}/{len(npy_files)}")
        print(f"  Failed: {failed}/{len(npy_files)}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Output directory: {output_dir}")
        
        return summary

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Convert CT slice .npy files to image formats")
    parser.add_argument('input_dir', type=str, help='Directory containing .npy files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: input_dir/converted)')
    parser.add_argument('--format', choices=['jpg', 'png', 'tiff'], default='jpg',
                       help='Output format (default: jpg)')
    parser.add_argument('--type', choices=['simple', 'contrast', 'annotated'], default='simple',
                       help='Conversion type (default: simple)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality 1-100 (default: 95)')
    parser.add_argument('--metadata_dir', type=str, default=None,
                       help='Metadata directory (for annotated type)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_converted_{args.format}"
    
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else None
    
    # Initialize converter
    converter = SliceConverter(output_quality=args.quality)
    
    # Convert files
    results = converter.batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        conversion_type=args.type,
        output_format=args.format,
        metadata_dir=metadata_dir
    )
    
    # Save results summary
    summary_file = output_dir / 'conversion_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Conversion summary saved to: {summary_file}")
    return 0 if results['failed'] == 0 else 1

if __name__ == "__main__":
    exit(main())