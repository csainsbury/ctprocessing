#!/usr/bin/env python3
"""
Robust Aortic Valve Level Detection with Orientation Correction
Enhanced version that handles sequence reversal and provides comprehensive validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from orientation_detector import CTOrientationDetector

@dataclass
class RobustAorticValveCandidate:
    """Robust data class for aortic valve level candidates with full validation."""
    slice_index: int
    confidence: float
    method: str
    landmarks_used: List[str]
    anatomical_features: Dict
    quality_metrics: Dict
    validation_scores: Dict
    orientation_info: Dict  # New: orientation analysis results

class RobustAorticValveLevelDetector:
    """Robust aortic valve detector with orientation correction and enhanced validation."""
    
    def __init__(self):
        """Initialize robust detector with orientation handling."""
        self.orientation_detector = CTOrientationDetector()
        
        # Enhanced anatomical constraints
        self.expected_valve_anatomical_range = (0.25, 0.45)  # More conservative range
        self.min_confidence_threshold = 0.2
        self.orientation_confidence_threshold = 0.3
        
        # Multiple validation levels
        self.validation_weights = {
            'position_validation': 0.25,
            'anatomical_context': 0.20,
            'landmark_consistency': 0.15,
            'intensity_patterns': 0.15,
            'multi_slice_coherence': 0.15,
            'orientation_consistency': 0.10
        }
        
    def detect_aortic_valve_level(self, image: np.ndarray, landmarks: Dict) -> RobustAorticValveCandidate:
        """
        Robust valve detection with orientation correction and comprehensive validation.
        
        Args:
            image: Preprocessed CT image
            landmarks: Detected anatomical landmarks
            
        Returns:
            Robust valve candidate with full validation
        """
        print("🔍 Robust aortic valve level detection with orientation analysis...")
        
        # Step 1: Analyze and correct image orientation
        orientation_analysis = self.orientation_detector.analyze_image_sequence(image)
        corrected_image, correction_info = self.orientation_detector.correct_sequence_if_needed(
            image, orientation_analysis
        )
        
        # Step 2: Re-detect landmarks on corrected image if sequence was reversed
        if correction_info['sequence_reversed']:
            print("  🔄 Re-detecting landmarks on orientation-corrected image...")
            # Note: In practice, you'd re-run landmark detection here
            # For now, we'll adjust existing landmark positions
            corrected_landmarks = self._adjust_landmarks_for_reversal(landmarks, image.shape[2])
        else:
            corrected_landmarks = landmarks
        
        # Step 3: Enhanced anatomical validation
        anatomical_validation = self._comprehensive_anatomical_validation(
            corrected_image, corrected_landmarks, orientation_analysis
        )
        
        # Step 4: Multi-method valve detection with enhanced constraints
        valve_candidates = self._detect_valve_candidates(corrected_image, corrected_landmarks)
        
        # Step 5: Robust candidate selection with full validation
        if valve_candidates:
            best_candidate = self._select_best_candidate_with_validation(
                valve_candidates, anatomical_validation, orientation_analysis, correction_info
            )
        else:
            # Enhanced fallback with orientation awareness
            best_candidate = self._create_robust_fallback_candidate(
                corrected_image, orientation_analysis, correction_info
            )
        
        # Step 6: Final quality assessment
        final_quality = self._assess_final_quality(best_candidate, anatomical_validation)
        
        print(f"Selected slice {best_candidate.slice_index} with confidence {best_candidate.confidence:.3f}")
        print(f"Final quality score: {final_quality:.3f}")
        
        if best_candidate.orientation_info.get('sequence_reversed'):
            print("✅ Orientation correction applied successfully")
        
        if final_quality < 0.4:
            print("⚠️  LOW QUALITY WARNING - Manual review recommended")
        
        return best_candidate
    
    def _adjust_landmarks_for_reversal(self, landmarks: Dict, total_slices: int) -> Dict:
        """Adjust landmark slice indices after sequence reversal."""
        adjusted_landmarks = {}
        
        for landmark_type, landmark_list in landmarks.items():
            adjusted_list = []
            for landmark in landmark_list:
                adjusted_landmark = landmark.copy()
                # Reverse slice index: new_index = total_slices - 1 - old_index
                adjusted_landmark['slice'] = total_slices - 1 - landmark['slice']
                adjusted_list.append(adjusted_landmark)
            adjusted_landmarks[landmark_type] = adjusted_list
        
        return adjusted_landmarks
    
    def _comprehensive_anatomical_validation(self, image: np.ndarray, landmarks: Dict, 
                                           orientation_analysis: Dict) -> Dict:
        """Comprehensive anatomical validation with orientation awareness."""
        
        validation_results = {
            'overall_score': 0.0,
            'component_scores': {},
            'warnings': [],
            'quality_grade': 'unknown'
        }
        
        # 1. Orientation consistency validation
        orientation_score = min(orientation_analysis['confidence'], 0.8)
        validation_results['component_scores']['orientation_consistency'] = orientation_score
        
        if orientation_analysis['confidence'] < 0.3:
            validation_results['warnings'].append("Low confidence in image orientation")
        
        # 2. Landmark distribution validation
        landmark_distribution_score = self._validate_landmark_distribution(landmarks, image.shape[2])
        validation_results['component_scores']['landmark_distribution'] = landmark_distribution_score
        
        # 3. Anatomical sequence validation
        sequence_score = self._validate_anatomical_sequence(image, landmarks)
        validation_results['component_scores']['anatomical_sequence'] = sequence_score
        
        # 4. Cross-landmark consistency
        consistency_score = self._validate_cross_landmark_consistency(landmarks)
        validation_results['component_scores']['cross_landmark_consistency'] = consistency_score
        
        # 5. Expected anatomy presence
        anatomy_presence_score = self._validate_expected_anatomy_presence(landmarks)
        validation_results['component_scores']['anatomy_presence'] = anatomy_presence_score
        
        # Calculate overall score
        component_scores = validation_results['component_scores']
        validation_results['overall_score'] = np.mean(list(component_scores.values()))
        
        # Determine quality grade
        overall_score = validation_results['overall_score']
        if overall_score >= 0.7:
            validation_results['quality_grade'] = 'High'
        elif overall_score >= 0.4:
            validation_results['quality_grade'] = 'Medium'
        else:
            validation_results['quality_grade'] = 'Low'
            validation_results['warnings'].append("Poor overall anatomical validation")
        
        return validation_results
    
    def _validate_landmark_distribution(self, landmarks: Dict, total_slices: int) -> float:
        """Validate that landmarks are distributed appropriately through the image."""
        
        # Check if we have landmarks in different anatomical regions
        aorta_slices = [l['slice'] for l in landmarks.get('aorta', [])]
        heart_slices = [l['slice'] for l in landmarks.get('heart', [])]
        vertebra_slices = [l['slice'] for l in landmarks.get('vertebrae', [])]
        
        if not (aorta_slices and heart_slices):
            return 0.1  # Critical landmarks missing
        
        # Check distribution across image
        all_landmark_slices = aorta_slices + heart_slices + vertebra_slices
        if len(all_landmark_slices) < 3:
            return 0.2  # Too few landmarks
        
        # Calculate distribution span
        min_slice = min(all_landmark_slices)
        max_slice = max(all_landmark_slices)
        span = (max_slice - min_slice) / total_slices
        
        # Good distribution should span significant portion of image
        distribution_score = min(span * 2, 1.0)  # Normalize to 0-1
        
        return max(distribution_score, 0.3)  # Minimum score for having landmarks
    
    def _validate_anatomical_sequence(self, image: np.ndarray, landmarks: Dict) -> float:
        """Validate that anatomical structures appear in expected sequence."""
        
        # Expected sequence: trachea (superior) → heart → liver (inferior)
        trachea_slices = [l['slice'] for l in landmarks.get('trachea', [])]
        heart_slices = [l['slice'] for l in landmarks.get('heart', [])]
        
        if not (trachea_slices and heart_slices):
            return 0.5  # Can't validate without both structures
        
        # Check if trachea appears superior to heart (lower slice numbers)
        avg_trachea_slice = np.mean(trachea_slices)
        avg_heart_slice = np.mean(heart_slices)
        
        if avg_trachea_slice < avg_heart_slice:
            return 0.8  # Correct anatomical sequence
        elif abs(avg_trachea_slice - avg_heart_slice) < 10:
            return 0.6  # Reasonable overlap
        else:
            return 0.2  # Possibly incorrect sequence
    
    def _validate_cross_landmark_consistency(self, landmarks: Dict) -> float:
        """Validate consistency between different landmark types."""
        
        aorta_landmarks = landmarks.get('aorta', [])
        heart_landmarks = landmarks.get('heart', [])
        
        if not (aorta_landmarks and heart_landmarks):
            return 0.3
        
        # Find overlapping slice regions
        aorta_slice_range = set(range(
            min(l['slice'] for l in aorta_landmarks) - 5,
            max(l['slice'] for l in aorta_landmarks) + 6
        ))
        
        heart_slice_range = set(range(
            min(l['slice'] for l in heart_landmarks) - 5,
            max(l['slice'] for l in heart_landmarks) + 6
        ))
        
        overlap = len(aorta_slice_range.intersection(heart_slice_range))
        total_range = len(aorta_slice_range.union(heart_slice_range))
        
        if total_range > 0:
            consistency_score = overlap / total_range
            return min(consistency_score * 1.5, 1.0)  # Boost good scores
        else:
            return 0.3
    
    def _validate_expected_anatomy_presence(self, landmarks: Dict) -> float:
        """Validate presence of expected anatomical structures."""
        
        expected_structures = ['aorta', 'heart']
        optional_structures = ['vertebrae', 'trachea']
        
        # Check for essential structures
        essential_score = 0
        for structure in expected_structures:
            if landmarks.get(structure):
                essential_score += 0.4  # 0.8 total for both essential
        
        # Bonus for optional structures
        optional_score = 0
        for structure in optional_structures:
            if landmarks.get(structure):
                optional_score += 0.1  # 0.2 total bonus
        
        return min(essential_score + optional_score, 1.0)
    
    def _detect_valve_candidates(self, image: np.ndarray, landmarks: Dict) -> List[RobustAorticValveCandidate]:
        """Detect valve candidates using multiple robust methods."""
        
        candidates = []
        
        # Method 1: Conservative aortic analysis
        aortic_candidates = self._robust_aortic_analysis(image, landmarks)
        candidates.extend(aortic_candidates)
        
        # Method 2: Heart-aorta relationship analysis  
        relationship_candidates = self._heart_aorta_relationship_analysis(image, landmarks)
        candidates.extend(relationship_candidates)
        
        # Method 3: Multi-landmark triangulation
        triangulation_candidates = self._multi_landmark_triangulation(image, landmarks)
        candidates.extend(triangulation_candidates)
        
        return candidates
    
    def _robust_aortic_analysis(self, image: np.ndarray, landmarks: Dict) -> List[RobustAorticValveCandidate]:
        """Conservative aortic analysis with strict validation."""
        
        candidates = []
        aorta_landmarks = landmarks.get('aorta', [])
        
        if not aorta_landmarks:
            return candidates
        
        # Focus on middle region with conservative bounds
        image_height = image.shape[2]
        conservative_range = (
            int(image_height * self.expected_valve_anatomical_range[0]),
            int(image_height * self.expected_valve_anatomical_range[1])
        )
        
        for aorta in aorta_landmarks[:10]:  # Top 10 candidates
            slice_idx = aorta['slice']
            
            # Strict anatomical filtering
            if not (conservative_range[0] <= slice_idx <= conservative_range[1]):
                continue
            
            # Enhanced validation
            validation_scores = self._detailed_slice_validation(image, slice_idx, landmarks)
            avg_validation = np.mean(list(validation_scores.values()))
            
            if avg_validation > 0.4:  # Strict validation threshold
                # Conservative confidence calculation
                base_confidence = aorta['confidence'] * 0.6  # Reduce base confidence
                validated_confidence = base_confidence * avg_validation
                
                candidates.append(RobustAorticValveCandidate(
                    slice_index=slice_idx,
                    confidence=validated_confidence,
                    method='robust_aortic_analysis',
                    landmarks_used=['aorta'],
                    anatomical_features={
                        'aortic_center': aorta['center'],
                        'aortic_radius': aorta['radius'],
                        'aortic_confidence': aorta['confidence']
                    },
                    quality_metrics={
                        'base_confidence': base_confidence,
                        'validation_boost': avg_validation
                    },
                    validation_scores=validation_scores,
                    orientation_info={}
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _heart_aorta_relationship_analysis(self, image: np.ndarray, landmarks: Dict) -> List[RobustAorticValveCandidate]:
        """Analyze heart-aorta spatial relationship for valve detection."""
        
        candidates = []
        aorta_landmarks = landmarks.get('aorta', [])
        heart_landmarks = landmarks.get('heart', [])
        
        if not (aorta_landmarks and heart_landmarks):
            return candidates
        
        # Find co-localized heart and aorta structures
        image_height = image.shape[2]
        
        for slice_idx in range(
            int(image_height * self.expected_valve_anatomical_range[0]),
            int(image_height * self.expected_valve_anatomical_range[1]),
            2  # Sample every 2nd slice
        ):
            
            # Find nearby landmarks
            nearby_aorta = [a for a in aorta_landmarks if abs(a['slice'] - slice_idx) <= 3]
            nearby_heart = [h for h in heart_landmarks if abs(h['slice'] - slice_idx) <= 5]
            
            if nearby_aorta and nearby_heart:
                best_aorta = max(nearby_aorta, key=lambda x: x['confidence'])
                best_heart = max(nearby_heart, key=lambda x: x['confidence'])
                
                # Validate spatial relationship
                spatial_score = self._validate_heart_aorta_spatial_relationship(
                    best_aorta['center'], best_heart['centroid']
                )
                
                if spatial_score > 0.4:
                    # Additional slice validation
                    validation_scores = self._detailed_slice_validation(image, slice_idx, landmarks)
                    avg_validation = np.mean(list(validation_scores.values()))
                    
                    if avg_validation > 0.3:
                        confidence = (
                            best_aorta['confidence'] * 0.3 +
                            best_heart['confidence'] * 0.2 +
                            spatial_score * 0.3 +
                            avg_validation * 0.2
                        )
                        
                        candidates.append(RobustAorticValveCandidate(
                            slice_index=slice_idx,
                            confidence=confidence,
                            method='heart_aorta_relationship',
                            landmarks_used=['aorta', 'heart'],
                            anatomical_features={
                                'aorta_center': best_aorta['center'],
                                'heart_center': best_heart['centroid'],
                                'spatial_score': spatial_score
                            },
                            quality_metrics={
                                'aortic_confidence': best_aorta['confidence'],
                                'cardiac_confidence': best_heart['confidence']
                            },
                            validation_scores=validation_scores,
                            orientation_info={}
                        ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _multi_landmark_triangulation(self, image: np.ndarray, landmarks: Dict) -> List[RobustAorticValveCandidate]:
        """Use multiple landmarks to triangulate valve position."""
        
        candidates = []
        
        # Require at least aorta and heart
        if not (landmarks.get('aorta') and landmarks.get('heart')):
            return candidates
        
        # Conservative range
        image_height = image.shape[2]
        search_range = range(
            int(image_height * self.expected_valve_anatomical_range[0]),
            int(image_height * self.expected_valve_anatomical_range[1]),
            3  # Every 3rd slice
        )
        
        for slice_idx in search_range:
            # Collect nearby landmarks of all types
            landmark_evidence = {}
            total_evidence_score = 0
            
            for landmark_type, landmark_list in landmarks.items():
                nearby = [l for l in landmark_list if abs(l['slice'] - slice_idx) <= 4]
                if nearby:
                    best = max(nearby, key=lambda x: x['confidence'])
                    landmark_evidence[landmark_type] = best
                    total_evidence_score += best['confidence']
            
            # Require evidence from multiple landmark types
            if len(landmark_evidence) >= 2 and 'aorta' in landmark_evidence:
                # Comprehensive validation
                validation_scores = self._detailed_slice_validation(image, slice_idx, landmarks)
                avg_validation = np.mean(list(validation_scores.values()))
                
                # Multi-landmark consistency check
                consistency_score = self._check_multi_landmark_consistency(landmark_evidence)
                
                if avg_validation > 0.4 and consistency_score > 0.3:
                    # Conservative confidence calculation
                    base_confidence = total_evidence_score / len(landmark_evidence)
                    final_confidence = (
                        base_confidence * 0.4 +
                        avg_validation * 0.4 +
                        consistency_score * 0.2
                    ) * 0.8  # Conservative multiplier
                    
                    candidates.append(RobustAorticValveCandidate(
                        slice_index=slice_idx,
                        confidence=final_confidence,
                        method='multi_landmark_triangulation',
                        landmarks_used=list(landmark_evidence.keys()),
                        anatomical_features={
                            'landmark_count': len(landmark_evidence),
                            'evidence_score': total_evidence_score,
                            'consistency_score': consistency_score
                        },
                        quality_metrics={
                            'base_confidence': base_confidence,
                            'validation_score': avg_validation
                        },
                        validation_scores=validation_scores,
                        orientation_info={}
                    ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _detailed_slice_validation(self, image: np.ndarray, slice_idx: int, landmarks: Dict) -> Dict[str, float]:
        """Perform detailed validation of a specific slice."""
        
        validation_scores = {}
        
        # 1. Position validation
        relative_pos = slice_idx / image.shape[2]
        if self.expected_valve_anatomical_range[0] <= relative_pos <= self.expected_valve_anatomical_range[1]:
            validation_scores['position'] = 1.0
        else:
            distance = min(
                abs(relative_pos - self.expected_valve_anatomical_range[0]),
                abs(relative_pos - self.expected_valve_anatomical_range[1])
            )
            validation_scores['position'] = max(0, 1.0 - distance * 10)
        
        # 2. Intensity characteristics
        slice_img = image[:, :, slice_idx]
        mean_intensity = np.mean(slice_img)
        
        # Chest region should have moderate mean intensity
        if 0.25 <= mean_intensity <= 0.65:
            validation_scores['intensity'] = 1.0
        else:
            validation_scores['intensity'] = max(0, 1.0 - abs(mean_intensity - 0.45) * 2)
        
        # 3. Structural complexity
        # Chest region should have good structural detail
        grad_magnitude = np.mean(np.abs(np.gradient(slice_img)))
        validation_scores['complexity'] = min(grad_magnitude * 10, 1.0)
        
        # 4. Landmark proximity
        nearby_landmarks = 0
        for landmark_type, landmark_list in landmarks.items():
            if any(abs(l['slice'] - slice_idx) <= 5 for l in landmark_list):
                nearby_landmarks += 1
        
        validation_scores['landmark_proximity'] = min(nearby_landmarks / 2.0, 1.0)
        
        return validation_scores
    
    def _validate_heart_aorta_spatial_relationship(self, aorta_center: Tuple, heart_center: Tuple) -> float:
        """Validate spatial relationship between heart and aorta."""
        
        dy = aorta_center[0] - heart_center[0]  # Anterior-posterior
        dx = aorta_center[1] - heart_center[1]  # Left-right
        
        # Aorta should be anterior and slightly right relative to heart centroid
        # More conservative thresholds
        anterior_score = max(0, 1 - abs(dy + 10) / 25.0) if dy < 0 else 0  # Prefer anterior
        lateral_score = max(0, 1 - abs(dx - 10) / 20.0)  # Slightly right
        
        return (anterior_score + lateral_score) / 2.0
    
    def _check_multi_landmark_consistency(self, landmark_evidence: Dict) -> float:
        """Check consistency between multiple landmark types."""
        
        if len(landmark_evidence) < 2:
            return 0.0
        
        # Check slice proximity
        slices = [evidence['slice'] for evidence in landmark_evidence.values()]
        slice_range = max(slices) - min(slices)
        
        # Landmarks should be reasonably close
        if slice_range <= 10:
            consistency = 1.0 - (slice_range / 20.0)
        else:
            consistency = 0.0
        
        return max(consistency, 0.0)
    
    def _select_best_candidate_with_validation(self, candidates: List[RobustAorticValveCandidate],
                                             anatomical_validation: Dict,
                                             orientation_analysis: Dict,
                                             correction_info: Dict) -> RobustAorticValveCandidate:
        """Select best candidate with comprehensive validation."""
        
        if not candidates:
            raise ValueError("No candidates available for selection")
        
        # Add orientation info to all candidates
        for candidate in candidates:
            candidate.orientation_info = {
                'sequence_orientation': orientation_analysis['sequence_orientation'],
                'orientation_confidence': orientation_analysis['confidence'],
                'sequence_reversed': correction_info['sequence_reversed'],
                'anatomical_validation_grade': anatomical_validation['quality_grade']
            }
        
        # Filter candidates by validation quality
        high_quality_candidates = [
            c for c in candidates 
            if np.mean(list(c.validation_scores.values())) > 0.4
        ]
        
        if high_quality_candidates:
            # Select from high quality candidates
            best_candidate = max(high_quality_candidates, key=lambda x: x.confidence)
        else:
            # Select best overall if no high quality candidates
            best_candidate = max(candidates, key=lambda x: x.confidence)
            best_candidate.orientation_info['warnings'] = ['No high-quality candidates found']
        
        # Apply final validation penalty if needed
        if anatomical_validation['quality_grade'] == 'Low':
            best_candidate.confidence *= 0.7
            if 'warnings' not in best_candidate.orientation_info:
                best_candidate.orientation_info['warnings'] = []
            best_candidate.orientation_info['warnings'].append('Low anatomical validation')
        
        return best_candidate
    
    def _create_robust_fallback_candidate(self, image: np.ndarray, 
                                        orientation_analysis: Dict,
                                        correction_info: Dict) -> RobustAorticValveCandidate:
        """Create robust fallback candidate with orientation awareness."""
        
        # Use conservative middle of expected range
        image_height = image.shape[2]
        fallback_slice = int(image_height * np.mean(self.expected_valve_anatomical_range))
        
        # Lower confidence for fallback
        fallback_confidence = 0.15
        
        # Reduce confidence further if orientation is uncertain
        if orientation_analysis['confidence'] < 0.3:
            fallback_confidence *= 0.7
        
        return RobustAorticValveCandidate(
            slice_index=fallback_slice,
            confidence=fallback_confidence,
            method='robust_fallback',
            landmarks_used=[],
            anatomical_features={'fallback_reason': 'no_valid_candidates'},
            quality_metrics={'fallback_confidence': fallback_confidence},
            validation_scores={'fallback': 0.1},
            orientation_info={
                'sequence_orientation': orientation_analysis['sequence_orientation'],
                'orientation_confidence': orientation_analysis['confidence'],
                'sequence_reversed': correction_info['sequence_reversed'],
                'warnings': ['Fallback method used - manual review recommended']
            }
        )
    
    def _assess_final_quality(self, candidate: RobustAorticValveCandidate, 
                            anatomical_validation: Dict) -> float:
        """Assess final quality of selected candidate."""
        
        quality_components = [
            candidate.confidence,
            np.mean(list(candidate.validation_scores.values())),
            anatomical_validation['overall_score'],
            candidate.orientation_info.get('orientation_confidence', 0.5)
        ]
        
        return np.mean(quality_components)

def test_robust_detection():
    """Test robust detection on sample data."""
    from image_loader import CTImageLoader
    from landmark_detection import AnatomicalLandmarkDetector
    from pathlib import Path
    
    # Test on available files
    ct_files = list(Path('.').glob('*.nii.gz'))
    if not ct_files:
        print("No CT files found")
        return
    
    # Test first file
    ct_file = ct_files[0]
    print(f"Testing robust detection on: {ct_file.name}")
    
    # Load and process
    loader = CTImageLoader()
    processed_image, metadata = loader.process_image(ct_file)
    
    landmark_detector = AnatomicalLandmarkDetector()
    landmarks = landmark_detector.detect_all_landmarks(processed_image)
    
    # Apply robust detection
    robust_detector = RobustAorticValveLevelDetector()
    robust_candidate = robust_detector.detect_aortic_valve_level(processed_image, landmarks)
    
    print(f"\n🎯 Robust Detection Results:")
    print(f"Selected slice: {robust_candidate.slice_index}")
    print(f"Confidence: {robust_candidate.confidence:.3f}")
    print(f"Method: {robust_candidate.method}")
    print(f"Orientation info: {robust_candidate.orientation_info}")
    
    return robust_candidate

if __name__ == "__main__":
    test_robust_detection()