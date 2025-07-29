#!/usr/bin/env python3
"""
CT Image Orientation and Sequence Detection Module
Detects cranio-caudal vs caudo-cranial sequence orientation and anatomical consistency.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology, filters
from typing import Dict, List, Tuple, Optional
import warnings

class CTOrientationDetector:
    """Detects CT image sequence orientation and anatomical consistency."""
    
    def __init__(self):
        """Initialize orientation detector with anatomical parameters."""
        # Anatomical intensity thresholds (in normalized 0-1 range)
        self.air_threshold = 0.1       # Lungs, trachea
        self.soft_tissue_threshold = 0.4  # Heart, liver, muscle  
        self.bone_threshold = 0.8      # Ribs, vertebrae
        
        # Expected anatomical progressions
        self.lung_volume_expected_trend = 'decreasing'  # Superior→inferior
        self.liver_appearance_expected = 'inferior'     # Should appear in lower slices
        
    def analyze_image_sequence(self, image: np.ndarray) -> Dict:
        """
        Comprehensive analysis of CT image sequence orientation.
        
        Args:
            image: Preprocessed CT image (3D array)
            
        Returns:
            Dictionary with orientation analysis results
        """
        print("  Analyzing image sequence orientation...")
        
        analysis = {
            'sequence_orientation': 'unknown',
            'confidence': 0.0,
            'anatomical_evidence': {},
            'reversal_needed': False,
            'warnings': []
        }
        
        # Analyze multiple anatomical indicators
        lung_analysis = self._analyze_lung_progression(image)
        liver_analysis = self._analyze_liver_distribution(image)
        diaphragm_analysis = self._analyze_diaphragm_position(image)
        trachea_analysis = self._analyze_trachea_trajectory(image)
        vertebrae_analysis = self._analyze_vertebral_progression(image)
        
        # Combine evidence
        evidence = {
            'lung_progression': lung_analysis,
            'liver_distribution': liver_analysis,
            'diaphragm_position': diaphragm_analysis,
            'trachea_trajectory': trachea_analysis,
            'vertebral_progression': vertebrae_analysis
        }
        
        analysis['anatomical_evidence'] = evidence
        
        # Determine overall orientation
        orientation_votes = []
        confidence_weights = []
        
        for indicator, result in evidence.items():
            if result['orientation_vote'] != 'unknown':
                orientation_votes.append(result['orientation_vote'])
                confidence_weights.append(result['confidence'])
        
        if orientation_votes:
            # Weighted voting
            cranio_caudal_weight = sum(w for vote, w in zip(orientation_votes, confidence_weights) 
                                     if vote == 'cranio_caudal')
            caudo_cranial_weight = sum(w for vote, w in zip(orientation_votes, confidence_weights) 
                                     if vote == 'caudo_cranial')
            
            if cranio_caudal_weight > caudo_cranial_weight:
                analysis['sequence_orientation'] = 'cranio_caudal'  # Normal: head→feet
                analysis['confidence'] = cranio_caudal_weight / (cranio_caudal_weight + caudo_cranial_weight)
                analysis['reversal_needed'] = False
            else:
                analysis['sequence_orientation'] = 'caudo_cranial'  # Reversed: feet→head
                analysis['confidence'] = caudo_cranial_weight / (cranio_caudal_weight + caudo_cranial_weight)
                analysis['reversal_needed'] = True
                analysis['warnings'].append("Image sequence appears reversed - automatic correction applied")
        
        print(f"    Sequence orientation: {analysis['sequence_orientation']} "
              f"(confidence: {analysis['confidence']:.2f})")
        
        if analysis['reversal_needed']:
            print(f"    ⚠️  SEQUENCE REVERSAL DETECTED - will correct slice indexing")
        
        return analysis
    
    def _analyze_lung_progression(self, image: np.ndarray) -> Dict:
        """Analyze lung volume progression through slices."""
        lung_volumes = []
        
        # Sample slices throughout the volume
        sample_slices = np.linspace(0, image.shape[2]-1, min(20, image.shape[2]), dtype=int)
        
        for z in sample_slices:
            slice_img = image[:, :, z]
            
            # Detect lung regions (air-filled)
            lung_mask = slice_img < self.air_threshold
            lung_mask = morphology.remove_small_objects(lung_mask, min_size=100)
            
            # Find largest air-filled regions (likely lungs)
            labeled = measure.label(lung_mask)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                # Sum areas of large air regions (lungs)
                lung_area = sum(r.area for r in regions if r.area > 200)
                lung_volumes.append(lung_area)
            else:
                lung_volumes.append(0)
        
        if len(lung_volumes) < 5:
            return {'orientation_vote': 'unknown', 'confidence': 0.0, 'data': lung_volumes}
        
        # Analyze trend: lungs should be largest in upper chest, decrease toward abdomen
        correlation = np.corrcoef(range(len(lung_volumes)), lung_volumes)[0, 1]
        
        if correlation < -0.3:  # Strong negative correlation = decreasing lung volume
            # This suggests cranio-caudal orientation (normal)
            return {
                'orientation_vote': 'cranio_caudal',
                'confidence': min(abs(correlation), 0.8),
                'data': lung_volumes,
                'trend': 'decreasing'
            }
        elif correlation > 0.3:  # Strong positive correlation = increasing lung volume
            # This suggests caudo-cranial orientation (reversed)
            return {
                'orientation_vote': 'caudo_cranial', 
                'confidence': min(abs(correlation), 0.8),
                'data': lung_volumes,
                'trend': 'increasing'
            }
        else:
            return {
                'orientation_vote': 'unknown',
                'confidence': 0.0,
                'data': lung_volumes,
                'trend': 'unclear'
            }
    
    def _analyze_liver_distribution(self, image: np.ndarray) -> Dict:
        """Analyze liver tissue distribution (should be in inferior slices)."""
        # Look for liver-like tissue: moderate-high intensity, large regions
        liver_scores = []
        
        sample_slices = np.linspace(0, image.shape[2]-1, min(15, image.shape[2]), dtype=int)
        
        for z in sample_slices:
            slice_img = image[:, :, z]
            
            # Liver tissue: moderate to high intensity
            liver_mask = (slice_img > 0.3) & (slice_img < 0.8)
            liver_mask = morphology.remove_small_objects(liver_mask, min_size=200)
            
            # Look for large, solid regions in right side of body
            labeled = measure.label(liver_mask)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                # Score based on size and position (liver is typically right side, large)
                liver_score = 0
                for region in regions:
                    if region.area > 500:  # Large region
                        centroid_x = region.centroid[1]
                        # Check if in right side of image (liver location)
                        if centroid_x > image.shape[1] * 0.5:  # Right side
                            liver_score += region.area
                
                liver_scores.append(liver_score)
            else:
                liver_scores.append(0)
        
        if len(liver_scores) < 5:
            return {'orientation_vote': 'unknown', 'confidence': 0.0, 'data': liver_scores}
        
        # Liver should appear more in inferior (later) slices in cranio-caudal sequence
        max_liver_idx = np.argmax(liver_scores)
        relative_position = max_liver_idx / len(liver_scores)
        
        if relative_position > 0.6:  # Liver appears in later slices
            return {
                'orientation_vote': 'cranio_caudal',
                'confidence': min((relative_position - 0.5) * 2, 0.7),
                'data': liver_scores,
                'max_position': relative_position
            }
        elif relative_position < 0.4:  # Liver appears in earlier slices (suggests reversal)
            return {
                'orientation_vote': 'caudo_cranial',
                'confidence': min((0.5 - relative_position) * 2, 0.7),
                'data': liver_scores,
                'max_position': relative_position
            }
        else:
            return {
                'orientation_vote': 'unknown',
                'confidence': 0.0,
                'data': liver_scores,
                'max_position': relative_position
            }
    
    def _analyze_diaphragm_position(self, image: np.ndarray) -> Dict:
        """Analyze diaphragm position to determine orientation."""
        # Look for diaphragm: transition from lung (air) to liver/abdomen (tissue)
        transition_scores = []
        
        sample_slices = np.linspace(0, image.shape[2]-1, min(20, image.shape[2]), dtype=int)
        
        for i, z in enumerate(sample_slices[:-1]):
            current_slice = image[:, :, z]
            next_slice = image[:, :, sample_slices[i+1]]
            
            # Calculate transition from air to tissue
            current_air = np.sum(current_slice < self.air_threshold)
            next_air = np.sum(next_slice < self.air_threshold)
            
            # Strong decrease in air content suggests diaphragm crossing
            air_decrease = current_air - next_air
            transition_scores.append(air_decrease)
        
        if len(transition_scores) < 5:
            return {'orientation_vote': 'unknown', 'confidence': 0.0, 'data': transition_scores}
        
        # Find strongest air→tissue transition
        max_transition_idx = np.argmax(transition_scores)
        relative_position = max_transition_idx / len(transition_scores)
        
        # Diaphragm should appear around middle to lower chest in cranio-caudal sequence
        if 0.3 <= relative_position <= 0.7:
            return {
                'orientation_vote': 'cranio_caudal',
                'confidence': 0.5,
                'data': transition_scores,
                'diaphragm_position': relative_position
            }
        else:
            return {
                'orientation_vote': 'unknown',
                'confidence': 0.0,
                'data': transition_scores,
                'diaphragm_position': relative_position
            }
    
    def _analyze_trachea_trajectory(self, image: np.ndarray) -> Dict:
        """Analyze trachea trajectory from neck to carina."""
        trachea_positions = []
        
        # Look for trachea: central, circular, air-filled structure
        sample_slices = np.linspace(0, image.shape[2]-1, min(15, image.shape[2]), dtype=int)
        
        for z in sample_slices:
            slice_img = image[:, :, z]
            
            # Air-filled regions
            air_mask = slice_img < self.air_threshold
            air_mask = morphology.remove_small_objects(air_mask, min_size=30)
            
            labeled = measure.label(air_mask)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                
                # Look for trachea: circular, central, appropriate size
                best_trachea = None
                best_score = 0
                
                for region in regions:
                    if 50 < region.area < 400:  # Appropriate size
                        eccentricity = region.eccentricity
                        centroid = region.centroid
                        
                        # Check if central
                        center_distance = abs(centroid[1] - image.shape[1]/2)
                        centrality = max(0, 1 - center_distance / (image.shape[1] * 0.3))
                        
                        # Check if circular
                        circularity = 1 - eccentricity
                        
                        score = centrality * circularity
                        if score > best_score and score > 0.3:
                            best_score = score
                            best_trachea = {
                                'position': centroid,
                                'score': score,
                                'slice': z
                            }
                
                if best_trachea:
                    trachea_positions.append(best_trachea)
        
        if len(trachea_positions) < 3:
            return {'orientation_vote': 'unknown', 'confidence': 0.0, 'data': trachea_positions}
        
        # Analyze trajectory: trachea should be more anterior (smaller Y) in superior slices
        y_positions = [t['position'][0] for t in trachea_positions]
        z_positions = [t['slice'] for t in trachea_positions]
        
        if len(y_positions) >= 3:
            correlation = np.corrcoef(z_positions, y_positions)[0, 1]
            
            if correlation > 0.3:  # Trachea moves posterior as we go inferior
                return {
                    'orientation_vote': 'cranio_caudal',
                    'confidence': min(abs(correlation), 0.6),
                    'data': trachea_positions,
                    'trajectory': 'anterior_to_posterior'
                }
            elif correlation < -0.3:  # Trachea moves anterior as we go through slices
                return {
                    'orientation_vote': 'caudo_cranial',
                    'confidence': min(abs(correlation), 0.6),
                    'data': trachea_positions,  
                    'trajectory': 'posterior_to_anterior'
                }
        
        return {
            'orientation_vote': 'unknown',
            'confidence': 0.0,
            'data': trachea_positions,
            'trajectory': 'unclear'
        }
    
    def _analyze_vertebral_progression(self, image: np.ndarray) -> Dict:
        """Analyze vertebral body size progression."""
        vertebra_sizes = []
        
        sample_slices = np.linspace(0, image.shape[2]-1, min(15, image.shape[2]), dtype=int)
        
        for z in sample_slices:
            slice_img = image[:, :, z]
            
            # Bone tissue detection
            bone_mask = slice_img > self.bone_threshold
            bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
            
            labeled = measure.label(bone_mask)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                
                # Look for vertebral bodies: posterior, roughly circular, appropriate size
                vertebra_areas = []
                for region in regions:
                    if 100 < region.area < 600:  # Vertebra size range
                        centroid = region.centroid
                        # Check if posterior
                        if centroid[0] > image.shape[0] * 0.6:  # Posterior region
                            if region.eccentricity < 0.7:  # Roughly circular
                                vertebra_areas.append(region.area)
                
                if vertebra_areas:
                    vertebra_sizes.append(max(vertebra_areas))  # Largest vertebra in slice
                else:
                    vertebra_sizes.append(0)
            else:
                vertebra_sizes.append(0)
        
        if len(vertebra_sizes) < 5:
            return {'orientation_vote': 'unknown', 'confidence': 0.0, 'data': vertebra_sizes}
        
        # Vertebrae typically get larger from cervical to lumbar
        # In cranio-caudal sequence, should see increasing size trend
        non_zero_sizes = [s for s in vertebra_sizes if s > 0]
        if len(non_zero_sizes) >= 3:
            correlation = np.corrcoef(range(len(vertebra_sizes)), vertebra_sizes)[0, 1]
            
            if correlation > 0.2:  # Increasing vertebra size
                return {
                    'orientation_vote': 'cranio_caudal',
                    'confidence': min(abs(correlation), 0.5),
                    'data': vertebra_sizes,
                    'trend': 'increasing'
                }
            elif correlation < -0.2:  # Decreasing vertebra size
                return {
                    'orientation_vote': 'caudo_cranial',
                    'confidence': min(abs(correlation), 0.5),
                    'data': vertebra_sizes,
                    'trend': 'decreasing'
                }
        
        return {
            'orientation_vote': 'unknown',
            'confidence': 0.0,
            'data': vertebra_sizes,
            'trend': 'unclear'
        }
    
    def correct_sequence_if_needed(self, image: np.ndarray, 
                                 orientation_analysis: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Reverse image sequence if needed based on orientation analysis.
        
        Args:
            image: Original image array
            orientation_analysis: Results from analyze_image_sequence
            
        Returns:
            Tuple of (corrected_image, correction_info)
        """
        correction_info = {
            'sequence_reversed': False,
            'original_orientation': orientation_analysis['sequence_orientation'],
            'confidence': orientation_analysis['confidence']
        }
        
        if orientation_analysis['reversal_needed'] and orientation_analysis['confidence'] > 0.4:
            print("  🔄 Reversing image sequence to correct orientation...")
            
            # Reverse the slice order
            corrected_image = image[:, :, ::-1]
            correction_info['sequence_reversed'] = True
            
            print("  ✅ Image sequence corrected from caudo-cranial to cranio-caudal")
            
            return corrected_image, correction_info
        else:
            return image, correction_info

def test_orientation_detection():
    """Test orientation detection on sample data."""
    from image_loader import CTImageLoader
    from pathlib import Path
    
    # Load sample CT
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files found")
        return
    
    loader = CTImageLoader()
    processed_image, metadata = loader.process_image(ct_files[0])
    
    # Test orientation detection
    detector = CTOrientationDetector()
    orientation_analysis = detector.analyze_image_sequence(processed_image)
    
    print("\nOrientation Analysis Results:")
    print(f"Sequence orientation: {orientation_analysis['sequence_orientation']}")
    print(f"Confidence: {orientation_analysis['confidence']:.3f}")
    print(f"Reversal needed: {orientation_analysis['reversal_needed']}")
    
    if orientation_analysis['warnings']:
        print("Warnings:")
        for warning in orientation_analysis['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Test correction
    corrected_image, correction_info = detector.correct_sequence_if_needed(
        processed_image, orientation_analysis
    )
    
    print(f"\nCorrection applied: {correction_info['sequence_reversed']}")
    
    return orientation_analysis, correction_info

if __name__ == "__main__":
    test_orientation_detection()