#!/usr/bin/env python3
"""
Improved Aortic Valve Level Detection Module
Enhanced version with better anatomical constraints and validation to prevent false positives.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import measure, morphology, filters
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ImprovedAorticValveCandidate:
    """Enhanced data class for aortic valve level candidates with validation."""
    slice_index: int
    confidence: float
    method: str
    landmarks_used: List[str]
    anatomical_features: Dict
    quality_metrics: Dict
    validation_scores: Dict  # New: anatomical validation scores

class ImprovedAorticValveLevelDetector:
    """Enhanced aortic valve detector with improved anatomical constraints."""
    
    def __init__(self):
        """Initialize with enhanced parameters and validation rules."""
        self.methods = [
            'improved_aortic_analysis',
            'anatomical_context_validation',
            'multi_level_consistency',
            'enhanced_consensus'
        ]
        
        # Enhanced anatomical constraints
        self.valve_z_percentile_range = (0.15, 0.55)  # 15-55% from top of image
        self.aorta_diameter_range = (20, 45)  # mm
        self.min_cardiac_structures = 2  # Require multiple cardiac landmarks
        
        # Validation thresholds
        self.min_trachea_distance = 30  # Min distance from trachea (mm)
        self.max_abdominal_intensity = 0.4  # Max mean intensity for abdominal region
        self.vertebra_consistency_threshold = 0.7  # Require vertebra near valve level
        
    def validate_anatomical_context(self, image: np.ndarray, slice_idx: int, 
                                  landmarks: Dict) -> Dict[str, float]:
        """
        Comprehensive anatomical validation to prevent false positives.
        
        Args:
            image: Preprocessed CT image
            slice_idx: Candidate slice index
            landmarks: Detected anatomical landmarks
            
        Returns:
            Dictionary of validation scores (0-1, higher is better)
        """
        validation_scores = {}
        
        # 1. Position validation: Check if slice is in reasonable anatomical range
        image_height = image.shape[2]
        relative_position = slice_idx / image_height
        
        if self.valve_z_percentile_range[0] <= relative_position <= self.valve_z_percentile_range[1]:
            validation_scores['position_valid'] = 1.0
        else:
            # Heavily penalize positions outside expected range
            distance_from_range = min(
                abs(relative_position - self.valve_z_percentile_range[0]),
                abs(relative_position - self.valve_z_percentile_range[1])
            )
            validation_scores['position_valid'] = max(0, 1.0 - distance_from_range * 5)
        
        # 2. Cardiac context validation: Look for cardiac structures nearby
        cardiac_landmarks = landmarks.get('heart', [])
        cardiac_nearby = [h for h in cardiac_landmarks if abs(h['slice'] - slice_idx) <= 15]
        validation_scores['cardiac_context'] = min(1.0, len(cardiac_nearby) / 2.0)
        
        # 3. Vertebral consistency: Check for vertebral structures
        vertebra_landmarks = landmarks.get('vertebrae', [])
        vertebra_nearby = [v for v in vertebra_landmarks if abs(v['slice'] - slice_idx) <= 10]
        validation_scores['vertebral_consistency'] = min(1.0, len(vertebra_nearby) / 1.0)
        
        # 4. Tracheal distance validation: Valve should not be too close to trachea
        trachea_landmarks = landmarks.get('trachea', [])
        if trachea_landmarks:
            min_trachea_distance = min([abs(t['slice'] - slice_idx) for t in trachea_landmarks])
            validation_scores['tracheal_distance'] = min(1.0, min_trachea_distance / 20.0)
        else:
            validation_scores['tracheal_distance'] = 0.5  # Neutral if no trachea detected
        
        # 5. Intensity distribution validation: Check for abdominal characteristics
        slice_img = image[:, :, slice_idx]
        mean_intensity = np.mean(slice_img)
        
        # Abdominal slices typically have lower mean intensity due to soft tissue/bowel gas
        if mean_intensity < self.max_abdominal_intensity:
            validation_scores['intensity_distribution'] = 0.3  # Suspicious
        else:
            validation_scores['intensity_distribution'] = 1.0
        
        # 6. Multi-slice consistency: Check intensity pattern in nearby slices
        consistency_scores = []
        for offset in [-5, -3, -1, 1, 3, 5]:
            adj_idx = slice_idx + offset
            if 0 <= adj_idx < image.shape[2]:
                adj_slice = image[:, :, adj_idx]
                # Calculate correlation between slices (should be high for chest region)
                correlation = np.corrcoef(slice_img.flatten(), adj_slice.flatten())[0, 1]
                if not np.isnan(correlation):
                    consistency_scores.append(correlation)
        
        if consistency_scores:
            validation_scores['multi_slice_consistency'] = max(0, np.mean(consistency_scores))
        else:
            validation_scores['multi_slice_consistency'] = 0.5
        
        return validation_scores
    
    def improved_aortic_analysis(self, image: np.ndarray, landmarks: Dict) -> List[ImprovedAorticValveCandidate]:
        """Enhanced aortic root analysis with better filtering."""
        candidates = []
        aorta_landmarks = landmarks.get('aorta', [])
        
        if not aorta_landmarks:
            return candidates
        
        # Filter aortic detections by anatomical position first
        image_height = image.shape[2]
        valid_aorta = []
        
        for aorta in aorta_landmarks:
            slice_idx = aorta['slice']
            relative_pos = slice_idx / image_height
            
            # Only consider aorta in upper-middle chest region
            if self.valve_z_percentile_range[0] <= relative_pos <= self.valve_z_percentile_range[1]:
                # Additional filtering by radius (prevent detection of other circular structures)
                if self.aorta_diameter_range[0]/2 <= aorta['radius'] <= self.aorta_diameter_range[1]/2:
                    valid_aorta.append(aorta)
        
        print(f"  Filtered aortic detections: {len(valid_aorta)} valid out of {len(aorta_landmarks)} total")
        
        for aorta in valid_aorta[:10]:  # Process top 10 candidates
            slice_idx = aorta['slice']
            
            # Enhanced intensity analysis
            center_y, center_x = aorta['center']
            radius = aorta['radius']
            slice_img = image[:, :, slice_idx]
            
            # Create more sophisticated masks
            y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
            center_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius * 0.6)**2
            ring_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius * 1.2)**2) & \
                       ((x_coords - center_x)**2 + (y_coords - center_y)**2 > (radius * 0.8)**2)
            
            if np.sum(center_mask) > 10 and np.sum(ring_mask) > 10:
                center_intensity = np.mean(slice_img[center_mask])
                ring_intensity = np.mean(slice_img[ring_mask])
                
                # Enhanced contrast calculation
                contrast = ring_intensity - center_intensity
                contrast_normalized = max(0, min(1, contrast * 2))
                
                # Anatomical validation
                validation_scores = self.validate_anatomical_context(image, slice_idx, landmarks)
                avg_validation = np.mean(list(validation_scores.values()))
                
                # Enhanced confidence calculation with heavy validation weighting
                base_confidence = aorta['confidence'] * 0.3 + contrast_normalized * 0.2
                validated_confidence = base_confidence * avg_validation
                
                candidates.append(ImprovedAorticValveCandidate(
                    slice_index=slice_idx,
                    confidence=validated_confidence,
                    method='improved_aortic_analysis',
                    landmarks_used=['aorta'],
                    anatomical_features={
                        'aortic_center': (center_y, center_x),
                        'aortic_radius': radius,
                        'contrast': contrast,
                        'center_intensity': center_intensity,
                        'ring_intensity': ring_intensity
                    },
                    quality_metrics={
                        'aortic_confidence': aorta['confidence'],
                        'contrast_score': contrast_normalized
                    },
                    validation_scores=validation_scores
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def anatomical_context_validation(self, image: np.ndarray, landmarks: Dict) -> List[ImprovedAorticValveCandidate]:
        """Method focused on anatomical context and multi-landmark integration."""
        candidates = []
        
        # Require both aortic and cardiac structures
        aorta_landmarks = landmarks.get('aorta', [])
        heart_landmarks = landmarks.get('heart', [])
        
        if not (aorta_landmarks and heart_landmarks):
            return candidates
        
        # Find regions where aorta and heart coexist (typical for valve level)
        image_height = image.shape[2]
        
        for slice_idx in range(int(image_height * self.valve_z_percentile_range[0]), 
                              int(image_height * self.valve_z_percentile_range[1]), 2):
            
            # Find nearby landmarks
            nearby_aorta = [a for a in aorta_landmarks if abs(a['slice'] - slice_idx) <= 3]
            nearby_heart = [h for h in heart_landmarks if abs(h['slice'] - slice_idx) <= 5]
            
            if nearby_aorta and nearby_heart:
                best_aorta = max(nearby_aorta, key=lambda x: x['confidence'])
                best_heart = max(nearby_heart, key=lambda x: x['confidence'])
                
                # Anatomical relationship validation
                aorta_pos = best_aorta['center']
                heart_pos = best_heart['centroid']
                
                # Calculate spatial relationship score
                spatial_score = self._calculate_aorta_heart_relationship(aorta_pos, heart_pos)
                
                # Comprehensive validation
                validation_scores = self.validate_anatomical_context(image, slice_idx, landmarks)
                avg_validation = np.mean(list(validation_scores.values()))
                
                # Require high validation score for this method
                if avg_validation > 0.6:
                    confidence = (best_aorta['confidence'] * 0.3 + 
                                best_heart['confidence'] * 0.3 + 
                                spatial_score * 0.2 + 
                                avg_validation * 0.2)
                    
                    candidates.append(ImprovedAorticValveCandidate(
                        slice_index=slice_idx,
                        confidence=confidence,
                        method='anatomical_context_validation',
                        landmarks_used=['aorta', 'heart'],
                        anatomical_features={
                            'aorta_center': aorta_pos,
                            'heart_center': heart_pos,
                            'spatial_relationship': spatial_score
                        },
                        quality_metrics={
                            'aortic_confidence': best_aorta['confidence'],
                            'cardiac_confidence': best_heart['confidence']
                        },
                        validation_scores=validation_scores
                    ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def multi_level_consistency(self, image: np.ndarray, landmarks: Dict) -> List[ImprovedAorticValveCandidate]:
        """Method that checks consistency across multiple anatomical levels."""
        candidates = []
        
        # Analyze consistency of detections across the expected range
        image_height = image.shape[2]
        start_slice = int(image_height * self.valve_z_percentile_range[0])
        end_slice = int(image_height * self.valve_z_percentile_range[1])
        
        # Group landmarks by slice
        aorta_by_slice = {}
        for aorta in landmarks.get('aorta', []):
            slice_idx = aorta['slice']
            if start_slice <= slice_idx <= end_slice:
                if slice_idx not in aorta_by_slice:
                    aorta_by_slice[slice_idx] = []
                aorta_by_slice[slice_idx].append(aorta)
        
        # Look for consistent patterns
        for slice_idx, aorta_list in aorta_by_slice.items():
            if len(aorta_list) > 0:
                best_aorta = max(aorta_list, key=lambda x: x['confidence'])
                
                # Check consistency with nearby slices
                consistency_score = self._calculate_multi_slice_consistency(
                    image, slice_idx, best_aorta, aorta_by_slice
                )
                
                # Validation
                validation_scores = self.validate_anatomical_context(image, slice_idx, landmarks)
                avg_validation = np.mean(list(validation_scores.values()))
                
                # Require both consistency and validation
                if consistency_score > 0.5 and avg_validation > 0.5:
                    confidence = (best_aorta['confidence'] * 0.4 + 
                                consistency_score * 0.3 + 
                                avg_validation * 0.3)
                    
                    candidates.append(ImprovedAorticValveCandidate(
                        slice_index=slice_idx,
                        confidence=confidence,
                        method='multi_level_consistency',
                        landmarks_used=['aorta'],
                        anatomical_features={
                            'aortic_center': best_aorta['center'],
                            'consistency_score': consistency_score
                        },
                        quality_metrics={
                            'aortic_confidence': best_aorta['confidence'],
                            'consistency': consistency_score
                        },
                        validation_scores=validation_scores
                    ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def enhanced_consensus(self, image: np.ndarray, landmarks: Dict) -> List[ImprovedAorticValveCandidate]:
        """Enhanced consensus method with strict validation."""
        candidates = []
        
        # Only consider slices in anatomically appropriate range
        image_height = image.shape[2]
        start_slice = int(image_height * self.valve_z_percentile_range[0])
        end_slice = int(image_height * self.valve_z_percentile_range[1])
        
        for slice_idx in range(start_slice, end_slice, 3):  # Sample every 3rd slice
            
            # Comprehensive landmark assessment
            validation_scores = self.validate_anatomical_context(image, slice_idx, landmarks)
            avg_validation = np.mean(list(validation_scores.values()))
            
            # Only proceed if basic validation passes
            if avg_validation < 0.4:
                continue
            
            # Find nearby landmarks
            nearby_aorta = [a for a in landmarks.get('aorta', []) if abs(a['slice'] - slice_idx) <= 3]
            nearby_heart = [h for h in landmarks.get('heart', []) if abs(h['slice'] - slice_idx) <= 5]
            nearby_vertebra = [v for v in landmarks.get('vertebrae', []) if abs(v['slice'] - slice_idx) <= 5]
            
            # Require multiple landmark types for consensus
            landmark_types = sum([
                len(nearby_aorta) > 0,
                len(nearby_heart) > 0,
                len(nearby_vertebra) > 0
            ])
            
            if landmark_types >= 2:  # Require at least 2 landmark types
                scores = []
                
                if nearby_aorta:
                    best_aorta = max(nearby_aorta, key=lambda x: x['confidence'])
                    scores.append(best_aorta['confidence'])
                
                if nearby_heart:
                    best_heart = max(nearby_heart, key=lambda x: x['confidence'])
                    scores.append(best_heart['confidence'])
                
                if nearby_vertebra:
                    best_vertebra = max(nearby_vertebra, key=lambda x: x['confidence'])
                    scores.append(best_vertebra['confidence'])
                
                # Weighted consensus with heavy validation emphasis
                landmark_score = np.mean(scores)
                final_confidence = landmark_score * 0.4 + avg_validation * 0.6
                
                candidates.append(ImprovedAorticValveCandidate(
                    slice_index=slice_idx,
                    confidence=final_confidence,
                    method='enhanced_consensus',
                    landmarks_used=['aorta', 'heart', 'vertebrae'][:landmark_types],
                    anatomical_features={
                        'landmark_types': landmark_types,
                        'landmark_score': landmark_score
                    },
                    quality_metrics={
                        'consensus_strength': landmark_types / 3.0,
                        'validation_strength': avg_validation
                    },
                    validation_scores=validation_scores
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def detect_aortic_valve_level(self, image: np.ndarray, landmarks: Dict) -> ImprovedAorticValveCandidate:
        """Enhanced valve detection with strict validation."""
        print("Enhanced aortic valve level detection...")
        
        all_candidates = []
        
        # Apply each enhanced method
        for method_name in self.methods:
            method_func = getattr(self, method_name)
            method_candidates = method_func(image, landmarks)
            all_candidates.extend(method_candidates)
            print(f"  {method_name}: {len(method_candidates)} candidates")
        
        if not all_candidates:
            # Fallback with warning
            print("  WARNING: No candidates found with enhanced methods, using fallback")
            return self._create_fallback_candidate(image)
        
        # Enhanced candidate selection with validation filtering
        valid_candidates = []
        for candidate in all_candidates:
            avg_validation = np.mean(list(candidate.validation_scores.values()))
            if avg_validation > 0.4:  # Strict validation threshold
                valid_candidates.append(candidate)
        
        if not valid_candidates:
            print("  WARNING: No candidates passed validation, using best available")
            valid_candidates = all_candidates
        
        # Weighted voting with validation emphasis
        final_candidate = self._enhanced_candidate_selection(valid_candidates)
        
        print(f"Selected slice {final_candidate.slice_index} with confidence {final_candidate.confidence:.3f}")
        print(f"Validation scores: {final_candidate.validation_scores}")
        
        return final_candidate
    
    def _calculate_aorta_heart_relationship(self, aorta_center: Tuple, heart_center: Tuple) -> float:
        """Calculate spatial relationship score between aorta and heart."""
        dy = aorta_center[0] - heart_center[0]  # Anterior-posterior difference
        dx = aorta_center[1] - heart_center[1]  # Left-right difference
        
        # Aorta should be anterior and slightly right to heart center
        anterior_score = max(0, 1 - abs(dy) / 30.0) if dy < 5 else 0
        lateral_score = max(0, 1 - abs(dx - 15) / 25.0)  # Slightly right of heart
        
        return (anterior_score + lateral_score) / 2.0
    
    def _calculate_multi_slice_consistency(self, image: np.ndarray, slice_idx: int, 
                                         aorta: Dict, aorta_by_slice: Dict) -> float:
        """Calculate consistency across multiple slices."""
        consistency_scores = []
        
        for offset in [-6, -3, 3, 6]:
            nearby_slice = slice_idx + offset
            if nearby_slice in aorta_by_slice:
                nearby_aorta = max(aorta_by_slice[nearby_slice], key=lambda x: x['confidence'])
                
                # Compare positions and sizes
                pos_diff = np.sqrt((aorta['center'][0] - nearby_aorta['center'][0])**2 + 
                                 (aorta['center'][1] - nearby_aorta['center'][1])**2)
                size_diff = abs(aorta['radius'] - nearby_aorta['radius'])
                
                # Consistency decreases with differences
                pos_consistency = max(0, 1 - pos_diff / 20.0)
                size_consistency = max(0, 1 - size_diff / 5.0)
                
                consistency_scores.append((pos_consistency + size_consistency) / 2.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _enhanced_candidate_selection(self, candidates: List[ImprovedAorticValveCandidate]) -> ImprovedAorticValveCandidate:
        """Enhanced candidate selection with validation weighting."""
        # Group by slice
        slice_votes = {}
        for candidate in candidates:
            slice_idx = candidate.slice_index
            if slice_idx not in slice_votes:
                slice_votes[slice_idx] = []
            slice_votes[slice_idx].append(candidate)
        
        # Enhanced scoring
        slice_scores = {}
        for slice_idx, slice_candidates in slice_votes.items():
            method_weights = {
                'improved_aortic_analysis': 0.3,
                'anatomical_context_validation': 0.3,
                'multi_level_consistency': 0.2,
                'enhanced_consensus': 0.2
            }
            
            total_score = 0.0
            total_weight = 0.0
            best_candidate = None
            
            for candidate in slice_candidates:
                base_weight = method_weights.get(candidate.method, 0.1)
                validation_weight = np.mean(list(candidate.validation_scores.values()))
                
                # Weight by both method importance and validation strength
                effective_weight = base_weight * (0.5 + validation_weight * 0.5)
                
                total_score += candidate.confidence * effective_weight
                total_weight += effective_weight
                
                if best_candidate is None or candidate.confidence > best_candidate.confidence:
                    best_candidate = candidate
            
            if total_weight > 0:
                slice_scores[slice_idx] = (total_score / total_weight, best_candidate)
        
        # Select best slice
        best_slice = max(slice_scores.keys(), key=lambda x: slice_scores[x][0])
        final_score, base_candidate = slice_scores[best_slice]
        
        # Create enhanced final candidate
        return ImprovedAorticValveCandidate(
            slice_index=best_slice,
            confidence=final_score,
            method='enhanced_combined',
            landmarks_used=list(set(sum([c.landmarks_used for c in candidates], []))),
            anatomical_features={'combined_methods': len(slice_votes[best_slice])},
            quality_metrics={'method_agreement': len(slice_votes[best_slice])},
            validation_scores=base_candidate.validation_scores
        )
    
    def _create_fallback_candidate(self, image: np.ndarray) -> ImprovedAorticValveCandidate:
        """Create fallback candidate when no valid detections found."""
        # Use middle of anatomically appropriate range
        image_height = image.shape[2]
        fallback_slice = int(image_height * np.mean(self.valve_z_percentile_range))
        
        return ImprovedAorticValveCandidate(
            slice_index=fallback_slice,
            confidence=0.1,  # Low confidence for fallback
            method='fallback',
            landmarks_used=[],
            anatomical_features={},
            quality_metrics={},
            validation_scores={'fallback': 0.1}
        )

def test_improved_detection():
    """Test the improved detection on a problematic case."""
    from image_loader import CTImageLoader
    from landmark_detection import AnatomicalLandmarkDetector
    from pathlib import Path
    
    # Find available CT files
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files found")
        return
    
    # Use first available file for testing
    ct_file = ct_files[0]
    print(f"Testing improved detection on: {ct_file.name}")
    
    # Process with original and improved methods
    loader = CTImageLoader()
    processed_image, metadata = loader.process_image(ct_file)
    
    landmark_detector = AnatomicalLandmarkDetector()
    landmarks = landmark_detector.detect_all_landmarks(processed_image)
    
    # Test improved detector
    improved_detector = ImprovedAorticValveLevelDetector()
    improved_candidate = improved_detector.detect_aortic_valve_level(processed_image, landmarks)
    
    print(f"\nImproved Detection Results:")
    print(f"Selected slice: {improved_candidate.slice_index} (vs original: 137)")
    print(f"Confidence: {improved_candidate.confidence:.3f}")
    print(f"Method: {improved_candidate.method}")
    print(f"Validation scores: {improved_candidate.validation_scores}")
    
    # Compare with anatomical range
    image_height = processed_image.shape[2]
    relative_pos = improved_candidate.slice_index / image_height
    print(f"Relative position: {relative_pos:.2f} (expected: 0.15-0.55)")
    
    return improved_candidate

if __name__ == "__main__":
    test_improved_detection()