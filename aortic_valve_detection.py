#!/usr/bin/env python3
"""
Aortic Valve Level Detection Module
Identifies consistent axial slice at aortic valve level using multiple anatomical landmarks.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AorticValveCandidate:
    """Data class for aortic valve level candidates."""
    slice_index: int
    confidence: float
    method: str
    landmarks_used: List[str]
    anatomical_features: Dict
    quality_metrics: Dict

class AorticValveLevelDetector:
    """Detects aortic valve level using multiple anatomical approaches."""
    
    def __init__(self):
        """Initialize aortic valve detector with default parameters."""
        self.methods = [
            'aortic_root_intensity',
            'aortic_arch_geometry',
            'cardiac_base_position',
            'multi_landmark_consensus'
        ]
        
        # Anatomical constraints (in mm, assuming 2mm slice thickness)
        self.expected_valve_z_range = (40, 120)  # Typical range from top of image
        self.aorta_diameter_range = (20, 40)     # Expected aortic root diameter
        
    def method_aortic_root_intensity(self, image: np.ndarray, landmarks: Dict) -> List[AorticValveCandidate]:
        """
        Method 1: Detect aortic valve level based on aortic root intensity characteristics.
        
        The aortic root typically shows a transition from the circular aortic arch to 
        the more complex geometry of the valve and left ventricle outflow.
        """
        candidates = []
        aorta_landmarks = landmarks.get('aorta', [])
        
        if not aorta_landmarks:
            return candidates
        
        # Group aortic detections by slice
        aorta_by_slice = {}
        for aorta in aorta_landmarks[:20]:  # Top 20 candidates
            slice_idx = aorta['slice']
            if slice_idx not in aorta_by_slice:
                aorta_by_slice[slice_idx] = []
            aorta_by_slice[slice_idx].append(aorta)
        
        # Analyze each slice with aortic structures
        for slice_idx, aorta_list in aorta_by_slice.items():
            if not (self.expected_valve_z_range[0] <= slice_idx <= self.expected_valve_z_range[1]):
                continue
                
            # Find best aorta candidate in this slice
            best_aorta = max(aorta_list, key=lambda x: x['confidence'])
            
            # Check for aortic root characteristics
            center_y, center_x = best_aorta['center']
            radius = best_aorta['radius']
            
            # Extract intensity profile around aortic structure
            slice_img = image[:, :, slice_idx]
            y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
            
            # Create masks for different radial zones
            inner_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius * 0.7)**2
            outer_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius * 1.3)**2) & \
                        ((x_coords - center_x)**2 + (y_coords - center_y)**2 > (radius * 0.7)**2)
            
            if np.sum(inner_mask) > 0 and np.sum(outer_mask) > 0:
                inner_intensity = np.mean(slice_img[inner_mask])
                outer_intensity = np.mean(slice_img[outer_mask])
                intensity_contrast = outer_intensity - inner_intensity
                
                # Look for adjacent slices to assess geometry changes
                geometry_score = self._assess_aortic_geometry_change(image, slice_idx, center_y, center_x)
                
                # Combine intensity and geometry features
                confidence = (
                    best_aorta['confidence'] * 0.4 +
                    min(intensity_contrast * 2, 1.0) * 0.3 +
                    geometry_score * 0.3
                )
                
                candidates.append(AorticValveCandidate(
                    slice_index=slice_idx,
                    confidence=confidence,
                    method='aortic_root_intensity',
                    landmarks_used=['aorta'],
                    anatomical_features={
                        'aortic_center': (center_y, center_x),
                        'aortic_radius': radius,
                        'intensity_contrast': intensity_contrast,
                        'geometry_score': geometry_score
                    },
                    quality_metrics={
                        'aortic_confidence': best_aorta['confidence'],
                        'intensity_snr': intensity_contrast / (np.std(slice_img[outer_mask]) + 1e-6)
                    }
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def method_aortic_arch_geometry(self, image: np.ndarray, landmarks: Dict) -> List[AorticValveCandidate]:
        """
        Method 2: Use aortic arch curvature analysis to find valve level.
        
        The aortic valve is typically located where the ascending aorta transitions
        from the arch configuration.
        """
        candidates = []
        aorta_landmarks = landmarks.get('aorta', [])
        
        if len(aorta_landmarks) < 3:
            return candidates
        
        # Analyze aortic arch trajectory
        arch_points = []
        for aorta in aorta_landmarks[:30]:
            slice_idx = aorta['slice']
            center_y, center_x = aorta['center']
            arch_points.append((slice_idx, center_x, center_y, aorta['confidence']))
        
        # Sort by slice index
        arch_points.sort(key=lambda x: x[0])
        
        # Find transition points in aortic trajectory
        for i in range(2, len(arch_points) - 2):
            slice_idx = arch_points[i][0]
            
            if not (self.expected_valve_z_range[0] <= slice_idx <= self.expected_valve_z_range[1]):
                continue
            
            # Analyze local curvature and trajectory changes
            local_points = arch_points[i-2:i+3]
            curvature_score = self._calculate_aortic_curvature(local_points)
            
            # Look for transition from arch to ascending aorta
            anterior_trend = self._assess_anterior_posterior_trend(local_points)
            
            confidence = curvature_score * 0.6 + anterior_trend * 0.4
            
            if confidence > 0.3:
                candidates.append(AorticValveCandidate(
                    slice_index=slice_idx,
                    confidence=confidence,
                    method='aortic_arch_geometry',
                    landmarks_used=['aorta'],
                    anatomical_features={
                        'curvature_score': curvature_score,
                        'anterior_trend': anterior_trend,
                        'arch_trajectory': local_points
                    },
                    quality_metrics={
                        'trajectory_smoothness': self._calculate_trajectory_smoothness(local_points)
                    }
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def method_cardiac_base_position(self, image: np.ndarray, landmarks: Dict) -> List[AorticValveCandidate]:
        """
        Method 3: Use cardiac structures to estimate valve level.
        
        The aortic valve is at the base of the heart, where the left ventricle
        outflow tract meets the aortic root.
        """
        candidates = []
        heart_landmarks = landmarks.get('heart', [])
        aorta_landmarks = landmarks.get('aorta', [])
        
        if not heart_landmarks or not aorta_landmarks:
            return candidates
        
        # Find superior extent of cardiac structures
        cardiac_slices = [h['slice'] for h in heart_landmarks]
        if not cardiac_slices:
            return candidates
        
        superior_cardiac_slice = min(cardiac_slices)
        
        # Look for aortic structures near superior cardiac boundary
        nearby_aorta = [a for a in aorta_landmarks 
                       if abs(a['slice'] - superior_cardiac_slice) <= 10]
        
        for aorta in nearby_aorta[:10]:
            slice_idx = aorta['slice']
            
            if not (self.expected_valve_z_range[0] <= slice_idx <= self.expected_valve_z_range[1]):
                continue
            
            # Find nearest cardiac structure
            nearest_heart = min(heart_landmarks, 
                              key=lambda h: abs(h['slice'] - slice_idx))
            
            # Calculate spatial relationship
            aorta_center = aorta['center']
            heart_center = nearest_heart['centroid']
            
            # Aortic valve should be anterior-right to main cardiac mass
            spatial_relationship = self._assess_aorta_heart_relationship(aorta_center, heart_center)
            
            # Consider slice position relative to cardiac base
            slice_position_score = max(0, 1 - abs(slice_idx - superior_cardiac_slice) / 15.0)
            
            confidence = (
                aorta['confidence'] * 0.3 +
                spatial_relationship * 0.4 +
                slice_position_score * 0.3
            )
            
            candidates.append(AorticValveCandidate(
                slice_index=slice_idx,
                confidence=confidence,
                method='cardiac_base_position',
                landmarks_used=['aorta', 'heart'],
                anatomical_features={
                    'aorta_center': aorta_center,
                    'heart_center': heart_center,
                    'spatial_relationship': spatial_relationship,
                    'cardiac_base_distance': abs(slice_idx - superior_cardiac_slice)
                },
                quality_metrics={
                    'cardiac_confidence': nearest_heart['confidence'],
                    'aortic_confidence': aorta['confidence']
                }
            ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def method_multi_landmark_consensus(self, image: np.ndarray, landmarks: Dict) -> List[AorticValveCandidate]:
        """
        Method 4: Consensus approach using multiple landmark types.
        
        Combines information from vertebrae, aorta, heart, and trachea to 
        identify the most anatomically consistent valve level.
        """
        candidates = []
        
        # Get landmarks
        aorta_landmarks = landmarks.get('aorta', [])
        heart_landmarks = landmarks.get('heart', [])
        vertebra_landmarks = landmarks.get('vertebrae', [])
        
        if not (aorta_landmarks and heart_landmarks):
            return candidates
        
        # For each potential slice, assess multi-landmark consistency
        for slice_idx in range(self.expected_valve_z_range[0], self.expected_valve_z_range[1]):
            
            # Find landmarks near this slice
            nearby_aorta = [a for a in aorta_landmarks if abs(a['slice'] - slice_idx) <= 3]
            nearby_heart = [h for h in heart_landmarks if abs(h['slice'] - slice_idx) <= 5]
            nearby_vertebra = [v for v in vertebra_landmarks if abs(v['slice'] - slice_idx) <= 3]
            
            if not (nearby_aorta and nearby_heart):
                continue
            
            # Select best landmark of each type
            best_aorta = max(nearby_aorta, key=lambda x: x['confidence'])
            best_heart = max(nearby_heart, key=lambda x: x['confidence'])
            
            # Assess anatomical consistency
            anatomical_scores = {
                'aorta_quality': best_aorta['confidence'],
                'heart_quality': best_heart['confidence'],
                'spatial_consistency': self._assess_multi_landmark_spatial_consistency(
                    best_aorta, best_heart, nearby_vertebra
                ),
                'slice_appropriateness': self._assess_slice_anatomical_appropriateness(
                    image, slice_idx, best_aorta, best_heart
                )
            }
            
            # Combine scores
            confidence = (
                anatomical_scores['aorta_quality'] * 0.25 +
                anatomical_scores['heart_quality'] * 0.25 +
                anatomical_scores['spatial_consistency'] * 0.3 +
                anatomical_scores['slice_appropriateness'] * 0.2
            )
            
            landmarks_used = ['aorta', 'heart']
            if nearby_vertebra:
                landmarks_used.append('vertebrae')
                confidence += 0.1 * max([v['confidence'] for v in nearby_vertebra])
            
            candidates.append(AorticValveCandidate(
                slice_index=slice_idx,
                confidence=confidence,
                method='multi_landmark_consensus',
                landmarks_used=landmarks_used,
                anatomical_features={
                    'best_aorta': best_aorta,
                    'best_heart': best_heart,
                    'vertebrae_present': len(nearby_vertebra) > 0
                },
                quality_metrics=anatomical_scores
            ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def detect_aortic_valve_level(self, image: np.ndarray, landmarks: Dict) -> AorticValveCandidate:
        """
        Main method to detect aortic valve level using all available methods.
        
        Args:
            image: Preprocessed CT image
            landmarks: Detected anatomical landmarks
            
        Returns:
            Best aortic valve level candidate
        """
        print("Detecting aortic valve level...")
        
        all_candidates = []
        
        # Apply each detection method
        for method_name in self.methods:
            method_func = getattr(self, f'method_{method_name}')
            method_candidates = method_func(image, landmarks)
            all_candidates.extend(method_candidates)
            print(f"  {method_name}: {len(method_candidates)} candidates")
        
        if not all_candidates:
            raise ValueError("No aortic valve level candidates found")
        
        # Combine results using weighted voting
        final_candidate = self._combine_candidates(all_candidates)
        
        print(f"Selected slice {final_candidate.slice_index} with confidence {final_candidate.confidence:.3f}")
        
        return final_candidate
    
    def _assess_aortic_geometry_change(self, image: np.ndarray, slice_idx: int, 
                                     center_y: float, center_x: float) -> float:
        """Assess geometry changes in adjacent slices."""
        if slice_idx <= 2 or slice_idx >= image.shape[2] - 3:
            return 0.0
        
        # Compare circular fits in adjacent slices
        scores = []
        for offset in [-2, -1, 1, 2]:
            adj_slice = slice_idx + offset
            if 0 <= adj_slice < image.shape[2]:
                adj_img = image[:, :, adj_slice]
                
                # Simple circularity assessment around the same center
                y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # Check intensity variation at different radii
                for r in [15, 20, 25]:
                    mask = (distances >= r-2) & (distances <= r+2)
                    if np.sum(mask) > 10:
                        intensity_var = np.std(adj_img[mask])
                        scores.append(1.0 / (1.0 + intensity_var))
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_aortic_curvature(self, points: List[Tuple]) -> float:
        """Calculate curvature of aortic trajectory."""
        if len(points) < 3:
            return 0.0
        
        # Extract coordinates
        z_coords = [p[0] for p in points]
        x_coords = [p[1] for p in points]
        y_coords = [p[2] for p in points]
        
        # Simple curvature approximation
        curvatures = []
        for i in range(1, len(points) - 1):
            # Vector from previous to current point
            v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
            # Vector from current to next point
            v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
            
            # Angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _assess_anterior_posterior_trend(self, points: List[Tuple]) -> float:
        """Assess anterior-posterior movement trend."""
        if len(points) < 3:
            return 0.0
        
        y_coords = [p[2] for p in points]  # Anterior-posterior coordinate
        z_coords = [p[0] for p in points]  # Superior-inferior coordinate
        
        # Look for anterior movement as we go superior (toward valve)
        if len(y_coords) >= 3:
            trend = np.polyfit(z_coords, y_coords, 1)[0]  # Slope
            return max(0, -trend)  # Anterior movement (decreasing y) is positive
        
        return 0.0
    
    def _calculate_trajectory_smoothness(self, points: List[Tuple]) -> float:
        """Calculate smoothness of trajectory."""
        if len(points) < 3:
            return 0.0
        
        # Calculate path variations
        distances = []
        for i in range(1, len(points)):
            dx = points[i][1] - points[i-1][1]
            dy = points[i][2] - points[i-1][2]
            distances.append(np.sqrt(dx**2 + dy**2))
        
        return 1.0 / (1.0 + np.std(distances)) if distances else 0.0
    
    def _assess_aorta_heart_relationship(self, aorta_center: Tuple, heart_center: Tuple) -> float:
        """Assess spatial relationship between aorta and heart."""
        # Aorta should be anterior and slightly right to heart center
        dy = aorta_center[0] - heart_center[0]  # Anterior-posterior difference
        dx = aorta_center[1] - heart_center[1]  # Left-right difference
        
        # Ideal relationship: aorta anterior (negative dy) and slightly right (positive dx)
        anterior_score = max(0, 1 - abs(dy) / 50.0) if dy < 10 else 0
        lateral_score = max(0, 1 - abs(dx - 20) / 30.0)  # Slightly right of heart
        
        return (anterior_score + lateral_score) / 2.0
    
    def _assess_multi_landmark_spatial_consistency(self, aorta: Dict, heart: Dict, 
                                                 vertebrae: List[Dict]) -> float:
        """Assess spatial consistency of multiple landmarks."""
        # Check if landmarks are in expected anatomical positions
        aorta_pos = aorta['center']
        heart_pos = heart['centroid']
        
        # Basic spatial relationship
        spatial_score = self._assess_aorta_heart_relationship(aorta_pos, heart_pos)
        
        # If vertebrae present, check posterior positioning
        if vertebrae:
            best_vertebra = max(vertebrae, key=lambda x: x['confidence'])
            vertebra_pos = best_vertebra['centroid']
            
            # Vertebra should be posterior to both aorta and heart
            if (vertebra_pos[0] > aorta_pos[0] and vertebra_pos[0] > heart_pos[0]):
                spatial_score += 0.2
        
        return min(spatial_score, 1.0)
    
    def _assess_slice_anatomical_appropriateness(self, image: np.ndarray, slice_idx: int,
                                               aorta: Dict, heart: Dict) -> float:
        """Assess if slice shows appropriate anatomy for valve level."""
        slice_img = image[:, :, slice_idx]
        
        # Check intensity distributions and structures
        aorta_center = aorta['center']
        heart_center = heart['centroid']
        
        # Sample intensities around structures
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        
        # Aortic region sampling
        aorta_mask = (x_coords - aorta_center[1])**2 + (y_coords - aorta_center[0])**2 <= 400
        aorta_intensities = slice_img[aorta_mask] if np.sum(aorta_mask) > 0 else []
        
        # Cardiac region sampling
        heart_mask = (x_coords - heart_center[1])**2 + (y_coords - heart_center[0])**2 <= 900
        heart_intensities = slice_img[heart_mask] if np.sum(heart_mask) > 0 else []
        
        # Assess intensity characteristics typical of valve level
        appropriateness_score = 0.5  # Baseline
        
        if len(aorta_intensities) > 10 and len(heart_intensities) > 10:
            # Aortic region should show contrast enhancement
            aorta_mean = np.mean(aorta_intensities)
            heart_mean = np.mean(heart_intensities)
            
            if aorta_mean > 0.4:  # Good contrast in aorta
                appropriateness_score += 0.3
            if heart_mean > 0.2 and heart_mean < 0.6:  # Moderate cardiac intensity
                appropriateness_score += 0.2
        
        return min(appropriateness_score, 1.0)
    
    def _combine_candidates(self, candidates: List[AorticValveCandidate]) -> AorticValveCandidate:
        """Combine candidates from multiple methods."""
        if not candidates:
            raise ValueError("No candidates to combine")
        
        # Group candidates by slice
        slice_votes = {}
        for candidate in candidates:
            slice_idx = candidate.slice_index
            if slice_idx not in slice_votes:
                slice_votes[slice_idx] = []
            slice_votes[slice_idx].append(candidate)
        
        # Calculate weighted scores for each slice
        slice_scores = {}
        for slice_idx, slice_candidates in slice_votes.items():
            # Weight different methods
            method_weights = {
                'aortic_root_intensity': 0.3,
                'aortic_arch_geometry': 0.2,
                'cardiac_base_position': 0.25,
                'multi_landmark_consensus': 0.25
            }
            
            total_score = 0.0
            total_weight = 0.0
            best_candidate = None
            
            for candidate in slice_candidates:
                weight = method_weights.get(candidate.method, 0.1)
                total_score += candidate.confidence * weight
                total_weight += weight
                
                if best_candidate is None or candidate.confidence > best_candidate.confidence:
                    best_candidate = candidate
            
            if total_weight > 0:
                slice_scores[slice_idx] = (total_score / total_weight, best_candidate)
        
        # Select best slice
        best_slice = max(slice_scores.keys(), key=lambda x: slice_scores[x][0])
        final_score, base_candidate = slice_scores[best_slice]
        
        # Create final candidate
        return AorticValveCandidate(
            slice_index=best_slice,
            confidence=final_score,
            method='combined',
            landmarks_used=list(set(sum([c.landmarks_used for c in candidates], []))),
            anatomical_features={'combined_from': [c.method for c in slice_votes[best_slice]]},
            quality_metrics={'method_agreement': len(slice_votes[best_slice])}
        )

def test_valve_detection():
    """Test aortic valve detection on sample data."""
    from image_loader import CTImageLoader
    from landmark_detection import AnatomicalLandmarkDetector
    from pathlib import Path
    
    # Load and preprocess image
    loader = CTImageLoader()
    ct_files = list(Path('.').glob('*.nii.gz'))
    
    if not ct_files:
        print("No CT files found")
        return
    
    processed_image, metadata = loader.process_image(ct_files[0])
    
    # Detect landmarks
    landmark_detector = AnatomicalLandmarkDetector()
    landmarks = landmark_detector.detect_all_landmarks(processed_image)
    
    # Detect aortic valve level
    valve_detector = AorticValveLevelDetector()
    valve_candidate = valve_detector.detect_aortic_valve_level(processed_image, landmarks)
    
    print(f"\nAortic valve detection results:")
    print(f"Selected slice: {valve_candidate.slice_index}")
    print(f"Confidence: {valve_candidate.confidence:.3f}")
    print(f"Method: {valve_candidate.method}")
    print(f"Landmarks used: {valve_candidate.landmarks_used}")
    
    # Visualize result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Show selected slice
    ax1.imshow(processed_image[:, :, valve_candidate.slice_index], cmap='gray')
    ax1.set_title(f'Selected Aortic Valve Level - Slice {valve_candidate.slice_index}')
    ax1.axis('off')
    
    # Show sagittal view with selected slice marked
    sagittal_view = processed_image[processed_image.shape[0]//2, :, :]
    ax2.imshow(sagittal_view, cmap='gray', aspect='auto')
    ax2.axvline(x=valve_candidate.slice_index, color='red', linewidth=2, label='Valve Level')
    ax2.set_title('Sagittal View with Valve Level')
    ax2.legend()
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('valve_detection_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return valve_candidate

if __name__ == "__main__":
    test_valve_detection()