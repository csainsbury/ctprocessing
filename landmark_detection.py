#!/usr/bin/env python3
"""
Anatomical Landmark Detection Module
Identifies key anatomical structures in CT chest scans for consistent slice selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters, feature
from skimage.transform import hough_circle, hough_circle_peaks
from typing import Tuple, List, Dict, Optional
import warnings

class AnatomicalLandmarkDetector:
    """Detects anatomical landmarks in CT chest scans."""
    
    def __init__(self):
        """Initialize landmark detector with default parameters."""
        self.aorta_radius_range = (15, 35)  # Expected aorta radius in mm (at target spacing)
        self.vertebra_area_range = (200, 800)  # Expected vertebral body area
        self.heart_intensity_threshold = 0.3  # Threshold for cardiac structures
        
    def detect_vertebral_bodies(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vertebral bodies as reference landmarks.
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            List of detected vertebral body information
        """
        vertebrae = []
        
        # Process each axial slice
        for z in range(image.shape[2]):
            slice_img = image[:, :, z]
            
            # Threshold for bone (high intensity in normalized image)
            bone_mask = slice_img > 0.8
            
            # Remove small objects and fill holes
            bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
            bone_mask = ndimage.binary_fill_holes(bone_mask)
            
            # Find connected components
            labeled = measure.label(bone_mask)
            regions = measure.regionprops(labeled, intensity_image=slice_img)
            
            # Filter for vertebral body candidates
            for region in regions:
                area = region.area
                eccentricity = region.eccentricity
                solidity = region.solidity
                
                # Vertebral bodies are roughly circular, solid, and of appropriate size
                if (self.vertebra_area_range[0] < area < self.vertebra_area_range[1] and
                    eccentricity < 0.7 and solidity > 0.7):
                    
                    centroid = region.centroid
                    # Check if it's in posterior region (vertebrae are posterior)
                    if centroid[0] > image.shape[0] * 0.6:  # Posterior half
                        vertebrae.append({
                            'slice': z,
                            'centroid': centroid,
                            'area': area,
                            'eccentricity': eccentricity,
                            'bbox': region.bbox,
                            'confidence': solidity * (1 - eccentricity)
                        })
        
        return sorted(vertebrae, key=lambda x: x['confidence'], reverse=True)
    
    def detect_aortic_arch(self, image: np.ndarray) -> List[Dict]:
        """
        Detect aortic arch using circular Hough transform.
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            List of detected aortic arch candidates
        """
        aorta_candidates = []
        
        # Focus on upper chest region where aortic arch is located
        upper_bound = int(image.shape[2] * 0.3)  # Upper 30% of volume
        lower_bound = int(image.shape[2] * 0.7)  # Down to 70%
        
        for z in range(upper_bound, lower_bound):
            slice_img = image[:, :, z]
            
            # Apply Gaussian smoothing
            smoothed = filters.gaussian(slice_img, sigma=1.0)
            
            # Edge detection
            edges = feature.canny(smoothed, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
            
            # Hough circle detection
            hough_radii = np.arange(self.aorta_radius_range[0], self.aorta_radius_range[1], 2)
            hough_res = hough_circle(edges, hough_radii)
            
            # Find circle peaks
            accums, cx, cy, radii = hough_circle_peaks(
                hough_res, hough_radii, total_num_peaks=3, min_xdistance=20, min_ydistance=20
            )
            
            for i, (acc, x, y, r) in enumerate(zip(accums, cx, cy, radii)):
                # Check if circle is in reasonable location for aorta
                # Aorta is typically in left-anterior region of chest
                if (x < image.shape[1] * 0.7 and  # Not too far right
                    y < image.shape[0] * 0.7 and  # Not too posterior
                    acc > 0.3):  # Good circle fit
                    
                    # Calculate intensity statistics within circle
                    y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
                    mask = (x_coords - x)**2 + (y_coords - y)**2 <= r**2
                    
                    if np.sum(mask) > 0:
                        mean_intensity = np.mean(slice_img[mask])
                        std_intensity = np.std(slice_img[mask])
                        
                        aorta_candidates.append({
                            'slice': z,
                            'center': (y, x),
                            'radius': r,
                            'accumulator': acc,
                            'mean_intensity': mean_intensity,
                            'intensity_std': std_intensity,
                            'confidence': acc * (1 + mean_intensity)
                        })
        
        return sorted(aorta_candidates, key=lambda x: x['confidence'], reverse=True)
    
    def detect_cardiac_structures(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cardiac structures using intensity-based segmentation.
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            List of detected cardiac structure information
        """
        cardiac_structures = []
        
        # Focus on middle chest region
        upper_bound = int(image.shape[2] * 0.4)
        lower_bound = int(image.shape[2] * 0.8)
        
        for z in range(upper_bound, lower_bound):
            slice_img = image[:, :, z]
            
            # Threshold for cardiac structures (moderate intensity)
            cardiac_mask = (slice_img > self.heart_intensity_threshold) & (slice_img < 0.7)
            
            # Morphological operations to clean up mask
            cardiac_mask = morphology.remove_small_objects(cardiac_mask, min_size=100)
            cardiac_mask = morphology.closing(cardiac_mask, morphology.disk(3))
            
            # Find largest connected component (likely heart)
            labeled = measure.label(cardiac_mask)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                largest_region = max(regions, key=lambda x: x.area)
                
                if largest_region.area > 500:  # Reasonable size for cardiac structure
                    centroid = largest_region.centroid
                    
                    # Heart should be in anterior-left region
                    if (centroid[0] < image.shape[0] * 0.6 and  # Anterior
                        centroid[1] < image.shape[1] * 0.6):    # Left side
                        
                        cardiac_structures.append({
                            'slice': z,
                            'centroid': centroid,
                            'area': largest_region.area,
                            'bbox': largest_region.bbox,
                            'eccentricity': largest_region.eccentricity,
                            'confidence': largest_region.area / 1000.0
                        })
        
        return sorted(cardiac_structures, key=lambda x: x['confidence'], reverse=True)
    
    def detect_trachea_carina(self, image: np.ndarray) -> List[Dict]:
        """
        Detect trachea and carina (tracheal bifurcation).
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            List of trachea/carina candidates
        """
        trachea_candidates = []
        
        # Focus on upper chest region for trachea
        upper_bound = int(image.shape[2] * 0.2)
        lower_bound = int(image.shape[2] * 0.6)
        
        for z in range(upper_bound, lower_bound):
            slice_img = image[:, :, z]
            
            # Trachea appears as low-intensity (air) circular structure
            air_mask = slice_img < 0.1
            
            # Remove small objects
            air_mask = morphology.remove_small_objects(air_mask, min_size=50)
            
            # Find circular air-filled structures
            labeled = measure.label(air_mask)
            regions = measure.regionprops(labeled)
            
            for region in regions:
                area = region.area
                eccentricity = region.eccentricity
                centroid = region.centroid
                
                # Trachea characteristics: circular, appropriate size, central location
                if (100 < area < 500 and  # Reasonable size
                    eccentricity < 0.6 and  # Roughly circular
                    abs(centroid[1] - image.shape[1]/2) < image.shape[1]*0.2):  # Central
                    
                    trachea_candidates.append({
                        'slice': z,
                        'centroid': centroid,
                        'area': area,
                        'eccentricity': eccentricity,
                        'bbox': region.bbox,
                        'confidence': (1 - eccentricity) * (area / 300.0)
                    })
        
        return sorted(trachea_candidates, key=lambda x: x['confidence'], reverse=True)
    
    def detect_all_landmarks(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect all anatomical landmarks in the image.
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            Dictionary containing all detected landmarks
        """
        print("Detecting anatomical landmarks...")
        
        landmarks = {}
        
        print("  - Detecting vertebral bodies...")
        landmarks['vertebrae'] = self.detect_vertebral_bodies(image)
        print(f"    Found {len(landmarks['vertebrae'])} vertebral body candidates")
        
        print("  - Detecting aortic arch...")
        landmarks['aorta'] = self.detect_aortic_arch(image)
        print(f"    Found {len(landmarks['aorta'])} aortic arch candidates")
        
        print("  - Detecting cardiac structures...")
        landmarks['heart'] = self.detect_cardiac_structures(image)
        print(f"    Found {len(landmarks['heart'])} cardiac structure candidates")
        
        print("  - Detecting trachea/carina...")
        landmarks['trachea'] = self.detect_trachea_carina(image)
        print(f"    Found {len(landmarks['trachea'])} trachea candidates")
        
        return landmarks
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: Dict[str, List[Dict]], 
                          slice_idx: int, save_path: Optional[str] = None):
        """
        Visualize detected landmarks on a specific slice.
        
        Args:
            image: Preprocessed CT image
            landmarks: Detected landmarks
            slice_idx: Slice index to visualize
            save_path: Optional path to save visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display slice
        ax.imshow(image[:, :, slice_idx], cmap='gray')
        ax.set_title(f'Anatomical Landmarks - Slice {slice_idx}')
        
        # Plot vertebrae
        for vertebra in landmarks.get('vertebrae', []):
            if vertebra['slice'] == slice_idx:
                y, x = vertebra['centroid']
                ax.plot(x, y, 'ro', markersize=8, label='Vertebra')
                ax.text(x+5, y, f"V{vertebra['confidence']:.2f}", color='red')
        
        # Plot aorta
        for aorta in landmarks.get('aorta', []):
            if aorta['slice'] == slice_idx:
                y, x = aorta['center']
                r = aorta['radius']
                circle = plt.Circle((x, y), r, fill=False, color='blue', linewidth=2)
                ax.add_patch(circle)
                ax.plot(x, y, 'bo', markersize=6, label='Aorta')
                ax.text(x+r+5, y, f"A{aorta['confidence']:.2f}", color='blue')
        
        # Plot cardiac structures
        for heart in landmarks.get('heart', []):
            if heart['slice'] == slice_idx:
                y, x = heart['centroid']
                ax.plot(x, y, 'go', markersize=8, label='Heart')
                ax.text(x+5, y, f"H{heart['confidence']:.2f}", color='green')
        
        # Plot trachea
        for trachea in landmarks.get('trachea', []):
            if trachea['slice'] == slice_idx:
                y, x = trachea['centroid']
                ax.plot(x, y, 'yo', markersize=8, label='Trachea')
                ax.text(x+5, y, f"T{trachea['confidence']:.2f}", color='yellow')
        
        ax.legend()
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def test_landmark_detection():
    """Test landmark detection on sample data."""
    from image_loader import CTImageLoader
    from pathlib import Path
    
    # Load and preprocess image
    loader = CTImageLoader()
    ct_files = list(Path('.').glob('*.nii.gz'))
    
    if not ct_files:
        print("No CT files found")
        return
    
    processed_image, metadata = loader.process_image(ct_files[0])
    
    # Detect landmarks
    detector = AnatomicalLandmarkDetector()
    landmarks = detector.detect_all_landmarks(processed_image)
    
    # Visualize results on middle slice
    middle_slice = processed_image.shape[2] // 2
    detector.visualize_landmarks(processed_image, landmarks, middle_slice, 'landmarks_test.png')
    
    print(f"\nLandmark detection completed. Results saved to landmarks_test.png")
    return landmarks

if __name__ == "__main__":
    test_landmark_detection()