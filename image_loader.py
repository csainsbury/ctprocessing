#!/usr/bin/env python3
"""
CT Image Loading and Preprocessing Module
Handles loading, orientation standardization, and preprocessing of CT chest scans.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

class CTImageLoader:
    """Handles loading and preprocessing of CT chest scans."""
    
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)):
        """
        Initialize CT image loader.
        
        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
        """
        self.target_spacing = target_spacing
        
    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load NIfTI file and extract metadata.
        
        Args:
            filepath: Path to NIfTI file
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        try:
            # Load with nibabel for better header handling
            nib_img = nib.load(str(filepath))
            image_data = nib_img.get_fdata()
            
            # Extract metadata
            header = nib_img.header
            affine = nib_img.affine
            
            metadata = {
                'original_shape': image_data.shape,
                'voxel_spacing': tuple(header.get_zooms()),
                'affine_matrix': affine,
                'orientation': nib.orientations.aff2axcodes(affine),
                'data_type': header.get_data_dtype(),
                'filepath': filepath
            }
            
            return image_data, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {str(e)}")
    
    def standardize_orientation(self, image: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Standardize image orientation to RAS+ (Right-Anterior-Superior).
        
        Args:
            image: Input image array
            metadata: Image metadata
            
        Returns:
            Tuple of (reoriented_image, updated_metadata)
        """
        current_orientation = metadata['orientation']
        
        # If already in RAS orientation, return as-is
        if current_orientation == ('R', 'A', 'S'):
            return image, metadata
        
        # Use SimpleITK for robust reorientation
        sitk_image = sitk.GetImageFromArray(image.transpose(2, 1, 0))
        sitk_image.SetSpacing([float(s) for s in metadata['voxel_spacing']])
        
        # Reorient to RAS
        reoriented = sitk.DICOMOrient(sitk_image, 'RAS')
        reoriented_array = sitk.GetArrayFromImage(reoriented).transpose(2, 1, 0)
        
        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata['orientation'] = ('R', 'A', 'S')
        updated_metadata['reoriented_shape'] = reoriented_array.shape
        updated_metadata['voxel_spacing'] = reoriented.GetSpacing()
        
        return reoriented_array, updated_metadata
    
    def resample_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resample image to target spacing.
        
        Args:
            image: Input image array
            metadata: Image metadata
            
        Returns:
            Tuple of (resampled_image, updated_metadata)
        """
        current_spacing = metadata['voxel_spacing']
        
        # Skip resampling if already at target spacing
        if np.allclose(current_spacing, self.target_spacing, atol=0.1):
            return image, metadata
        
        # Convert to SimpleITK for resampling
        sitk_image = sitk.GetImageFromArray(image.transpose(2, 1, 0))
        sitk_image.SetSpacing([float(s) for s in current_spacing])
        
        # Calculate new size
        original_size = sitk_image.GetSize()
        scale_factors = [curr/target for curr, target in zip(current_spacing, self.target_spacing)]
        new_size = [int(round(size * scale)) for size, scale in zip(original_size, scale_factors)]
        
        # Resample using linear interpolation
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.min())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled = resampler.Execute(sitk_image)
        resampled_array = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)
        
        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata['resampled_shape'] = resampled_array.shape
        updated_metadata['voxel_spacing'] = self.target_spacing
        updated_metadata['resampling_performed'] = True
        
        return resampled_array, updated_metadata
    
    def preprocess_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intensity preprocessing for CT images.
        
        Args:
            image: Input image array
            
        Returns:
            Intensity-normalized image
        """
        # Clip to reasonable CT range (-1000 to 3000 HU)
        clipped = np.clip(image, -1000, 3000)
        
        # Apply windowing for soft tissue (common for chest CT)
        # Window: center=40 HU, width=400 HU
        window_min = 40 - 200  # -160 HU
        window_max = 40 + 200  # 240 HU
        
        windowed = np.clip(clipped, window_min, window_max)
        
        # Normalize to [0, 1] range
        normalized = (windowed - window_min) / (window_max - window_min)
        
        return normalized.astype(np.float32)
    
    def process_image(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing pipeline for a single CT image.
        
        Args:
            filepath: Path to CT image file
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        print(f"Processing: {filepath.name}")
        
        # Load image
        image, metadata = self.load_nifti(filepath)
        print(f"  Original shape: {image.shape}, spacing: {metadata['voxel_spacing']}")
        
        # Standardize orientation
        image, metadata = self.standardize_orientation(image, metadata)
        print(f"  Orientation: {metadata['orientation']}")
        
        # Resample to target spacing
        image, metadata = self.resample_image(image, metadata)
        if metadata.get('resampling_performed', False):
            print(f"  Resampled to: {image.shape}, spacing: {metadata['voxel_spacing']}")
        
        # Intensity preprocessing
        processed_image = self.preprocess_intensity(image)
        print(f"  Intensity range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        metadata['preprocessing_completed'] = True
        metadata['final_shape'] = processed_image.shape
        metadata['intensity_normalized'] = True
        
        return processed_image, metadata

def test_loader():
    """Test the image loader with sample data."""
    loader = CTImageLoader()
    
    # Find first available CT scan
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files found in current directory")
        return
    
    # Test with first file
    test_file = ct_files[0]
    try:
        processed_image, metadata = loader.process_image(test_file)
        print(f"\nSuccessfully processed {test_file.name}")
        print(f"Final shape: {processed_image.shape}")
        print(f"Intensity range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        return processed_image, metadata
        
    except Exception as e:
        print(f"Error processing {test_file.name}: {e}")
        return None, None

if __name__ == "__main__":
    test_loader()