# CT Aortic Valve Level Detection Pipeline - Complete Technical Specification

## Project Overview

Create an automated system to process chest CT scans (.nii.gz format) and identify consistent axial slices at the aortic valve level using multi-modal anatomical landmark detection. The system must handle large datasets (20k+ scans) efficiently with parallel processing and provide comprehensive quality control.

## System Architecture

```
Input: CT Scan (.nii.gz) → Preprocessing → Landmark Detection → Valve Detection → Output (slice + metadata)
```

### Core Components Required:
1. **Image Loader & Preprocessor** (`image_loader.py`)
2. **Anatomical Landmark Detector** (`landmark_detection.py`)
3. **Aortic Valve Level Detector** (`aortic_valve_detection.py`)
4. **Batch Processor** (`ct_batch_processor.py`)
5. **Visualization Tools** (`visualization_tools.py`)
6. **Format Converter** (`convert_slices_to_jpg.py`)
7. **Main Pipeline Runner** (`run_pipeline.py`)

## Environment Setup

### Required Libraries:
```python
# requirements.txt content:
nibabel>=5.0.0          # NIfTI file handling
numpy>=1.21.0           # Numerical operations
scipy>=1.7.0            # Scientific computing
matplotlib>=3.5.0       # Plotting and visualization
scikit-image>=0.19.0    # Image processing algorithms
SimpleITK>=2.2.0        # Advanced medical image processing
tqdm>=4.64.0            # Progress bars
argparse                # Command line parsing
pathlib                 # Path handling
```

### Configuration File Structure:
```json
{
  "processing": {
    "target_spacing": [1.0, 1.0, 2.0],
    "max_workers": 4,
    "batch_size": 10
  },
  "detection": {
    "aorta_radius_range": [15, 35],
    "vertebra_area_range": [200, 800],
    "heart_intensity_threshold": 0.3,
    "expected_valve_z_range": [40, 120]
  },
  "quality_control": {
    "min_confidence_threshold": 0.1,
    "slice_consistency_threshold": 10.0,
    "create_visualizations": true,
    "save_individual_reports": false
  },
  "output": {
    "save_slices": true,
    "save_metadata": true,
    "create_montage": true,
    "create_dashboard": true,
    "save_jpg": true,
    "jpg_quality": 95,
    "jpg_with_annotations": false
  }
}
```

## Detailed Component Specifications

### 1. Image Loader & Preprocessor (`image_loader.py`)

**Class**: `CTImageLoader`

**Constructor Parameters**:
- `target_spacing`: tuple (x, y, z) in mm, default (1.0, 1.0, 2.0)

**Key Methods**:

#### `load_nifti(filepath: Path) -> Tuple[np.ndarray, Dict]`
- Load NIfTI file using `nibabel.load()`
- Extract image data with `get_fdata()`
- Build metadata dictionary:
  ```python
  metadata = {
      'original_shape': image_data.shape,
      'voxel_spacing': tuple(header.get_zooms()),
      'affine_matrix': nib_img.affine,
      'orientation': nib.orientations.aff2axcodes(affine),
      'data_type': header.get_data_dtype(),
      'filepath': filepath
  }
  ```

#### `standardize_orientation(image, metadata) -> Tuple[np.ndarray, Dict]`
- Check current orientation from metadata
- If not RAS+ ('R', 'A', 'S'), reorient using SimpleITK:
  ```python
  sitk_image = sitk.GetImageFromArray(image.transpose(2, 1, 0))
  sitk_image.SetSpacing([float(s) for s in metadata['voxel_spacing']])
  reoriented = sitk.DICOMOrient(sitk_image, 'RAS')
  reoriented_array = sitk.GetArrayFromImage(reoriented).transpose(2, 1, 0)
  ```

#### `resample_image(image, metadata) -> Tuple[np.ndarray, Dict]`
- Skip if already at target spacing (tolerance: 0.1mm)
- Use SimpleITK ResampleImageFilter:
  ```python
  resampler = sitk.ResampleImageFilter()
  resampler.SetOutputSpacing(self.target_spacing)
  resampler.SetSize(new_size) # calculated from scale factors
  resampler.SetInterpolator(sitk.sitkLinear)
  resampler.SetDefaultPixelValue(image.min())
  ```

#### `preprocess_intensity(image) -> np.ndarray`
- Clip to CT range: `np.clip(image, -1000, 3000)`
- Apply soft tissue windowing: center=40 HU, width=400 HU
- Window range: [-160, 240] HU
- Normalize to [0,1]: `(windowed - window_min) / (window_max - window_min)`
- Return as float32

#### `process_image(filepath) -> Tuple[np.ndarray, Dict]`
- Sequential pipeline: load → orient → resample → intensity normalize
- Print progress information
- Return processed image and complete metadata

### 2. Anatomical Landmark Detector (`landmark_detection.py`)

**Class**: `AnatomicalLandmarkDetector`

**Constructor Parameters**:
- `aorta_radius_range`: (15, 35) mm
- `vertebra_area_range`: (200, 800) voxels
- `heart_intensity_threshold`: 0.3

**Key Methods**:

#### `detect_vertebral_bodies(image) -> List[Dict]`
For each axial slice:
1. **Bone thresholding**: `bone_mask = slice_img > 0.8`
2. **Morphological cleanup**:
   ```python
   bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
   bone_mask = ndimage.binary_fill_holes(bone_mask)
   ```
3. **Connected component analysis**: `measure.label()`, `measure.regionprops()`
4. **Vertebra filtering criteria**:
   - Area in range [200, 800] voxels
   - Eccentricity < 0.7 (roughly circular)
   - Solidity > 0.7 (solid structure)
   - Posterior position: centroid[0] > image.shape[0] * 0.6
5. **Return format**:
   ```python
   {
       'slice': z,
       'centroid': region.centroid,
       'area': region.area,
       'eccentricity': region.eccentricity,
       'bbox': region.bbox,
       'confidence': solidity * (1 - eccentricity)
   }
   ```

#### `detect_aortic_arch(image) -> List[Dict]`
Focus on upper chest (30-70% of volume):
1. **Image smoothing**: `filters.gaussian(slice_img, sigma=1.0)`
2. **Edge detection**: `feature.canny(smoothed, sigma=1.0, low=0.1, high=0.3)`
3. **Hough circle detection**:
   ```python
   hough_radii = np.arange(15, 35, 2)  # Expected aorta radius range
   hough_res = hough_circle(edges, hough_radii)
   accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                             total_num_peaks=3, 
                                             min_xdistance=20, min_ydistance=20)
   ```
4. **Aorta filtering criteria**:
   - Accumulator value > 0.3 (good circle fit)
   - Left-anterior position: x < image.shape[1] * 0.7, y < image.shape[0] * 0.7
   - Mean intensity within circle > threshold
5. **Calculate intensity statistics within detected circles**

#### `detect_cardiac_structures(image) -> List[Dict]`
Focus on middle chest (40-80% of volume):
1. **Intensity thresholding**: `(slice_img > 0.3) & (slice_img < 0.7)`
2. **Morphological operations**:
   ```python
   cardiac_mask = morphology.remove_small_objects(cardiac_mask, min_size=100)
   cardiac_mask = morphology.closing(cardiac_mask, morphology.disk(3))
   ```
3. **Find largest component**: Likely to be main cardiac mass
4. **Heart filtering criteria**:
   - Area > 500 voxels
   - Anterior-left position: centroid[0] < 0.6 * height, centroid[1] < 0.6 * width

#### `detect_trachea_carina(image) -> List[Dict]`
Focus on upper chest (20-60% of volume):
1. **Air detection**: `air_mask = slice_img < 0.1`
2. **Size filtering**: Remove objects < 50 voxels
3. **Trachea criteria**:
   - Area in [100, 500] voxels
   - Eccentricity < 0.6 (roughly circular)
   - Central location: |centroid[1] - width/2| < 0.2 * width

#### `detect_all_landmarks(image) -> Dict[str, List[Dict]]`
- Call all detection methods
- Return dictionary with keys: 'vertebrae', 'aorta', 'heart', 'trachea'
- Print progress and counts

### 3. Aortic Valve Level Detector (`aortic_valve_detection.py`)

**Data Class**: `AorticValveCandidate`
```python
@dataclass
class AorticValveCandidate:
    slice_index: int
    confidence: float
    method: str
    landmarks_used: List[str]
    anatomical_features: Dict
    quality_metrics: Dict
```

**Class**: `AorticValveLevelDetector`

**Constructor Parameters**:
- `expected_valve_z_range`: (40, 120) slice indices
- `aorta_diameter_range`: (20, 40) mm

**Detection Methods**:

#### `method_aortic_root_intensity(image, landmarks) -> List[AorticValveCandidate]`
1. **Group aortic detections by slice**
2. **For each slice with aorta**:
   - Create intensity masks at different radii around aortic center
   - Calculate inner vs outer intensity contrast
   - Assess geometry changes in adjacent slices
   - **Confidence formula**:
     ```python
     confidence = (aorta_confidence * 0.4 + 
                  intensity_contrast * 0.3 + 
                  geometry_score * 0.3)
     ```

#### `method_aortic_arch_geometry(image, landmarks) -> List[AorticValveCandidate]`
1. **Extract aortic trajectory**: Sort aorta points by slice index
2. **Calculate local curvature**: Analyze angle changes between consecutive vectors
3. **Assess anterior-posterior trend**: Look for anterior movement going superior
4. **Confidence formula**:
   ```python
   confidence = curvature_score * 0.6 + anterior_trend * 0.4
   ```

#### `method_cardiac_base_position(image, landmarks) -> List[AorticValveCandidate]`
1. **Find superior cardiac extent**: Minimum slice with cardiac structures
2. **Locate nearby aortic structures**: Within ±10 slices of cardiac base
3. **Assess spatial relationship**: Aorta should be anterior-right to heart
4. **Confidence formula**:
   ```python
   confidence = (aorta_confidence * 0.3 + 
                spatial_relationship * 0.4 + 
                slice_position_score * 0.3)
   ```

#### `method_multi_landmark_consensus(image, landmarks) -> List[AorticValveCandidate]`
1. **For each potential slice in valve range**:
   - Find nearby landmarks of each type
   - Select best landmark of each type
   - Assess spatial consistency between landmark types
   - Evaluate anatomical appropriateness of slice
2. **Confidence formula**:
   ```python
   confidence = (aorta_quality * 0.25 + 
                heart_quality * 0.25 + 
                spatial_consistency * 0.3 + 
                slice_appropriateness * 0.2)
   ```

#### `detect_aortic_valve_level(image, landmarks) -> AorticValveCandidate`
1. **Run all detection methods**
2. **Combine using weighted voting**:
   ```python
   method_weights = {
       'aortic_root_intensity': 0.3,
       'aortic_arch_geometry': 0.2,
       'cardiac_base_position': 0.25,
       'multi_landmark_consensus': 0.25
   }
   ```
3. **Select slice with highest combined score**

### 4. Batch Processor (`ct_batch_processor.py`)

**Class**: `CTBatchProcessor`

**Constructor Parameters**:
- `output_dir`: Path for results
- `target_spacing`: Voxel spacing tuple

**Key Methods**:

#### `process_single_ct(ct_path) -> Dict`
1. **Load and preprocess** using CTImageLoader
2. **Detect landmarks** using AnatomicalLandmarkDetector
3. **Detect valve level** using AorticValveLevelDetector
4. **Save outputs**:
   - Extract valve slice: `processed_image[:, :, valve_slice_index]`
   - Save as numpy: `np.save(slice_path, valve_slice)`
   - Save metadata as JSON
5. **Return result dictionary** with status, confidence, paths, timing

#### `process_batch(ct_files, max_workers) -> Dict`
1. **Use ProcessPoolExecutor** for parallel processing
2. **Progress tracking** with tqdm
3. **Error handling** for individual failures
4. **Aggregate results** and save batch summary
5. **Generate quality report**

#### `create_quality_report(batch_results) -> str`
- Calculate statistics: slice positions, confidences, success rates
- Generate quality categories (high/medium/low confidence)
- Provide recommendations based on metrics
- Save as text report

### 5. Visualization Tools (`visualization_tools.py`)

**Class**: `CTVisualizationTools`

#### `create_slice_montage(slice_paths, max_images=16)`
- Load .npy files
- Create grid layout (sqrt(n_images))
- Display with matplotlib subplots
- Add filenames as titles

#### `create_quality_dashboard(batch_results)`
- Multi-panel dashboard with:
  - Slice position histogram
  - Confidence distribution
  - Processing time distribution
  - Scatter plot: slice vs confidence
  - Success/failure pie chart
  - Quality category breakdown
  - Top performers list
  - Statistics summary table

#### `create_individual_report(ct_filename, metadata_path, slice_path)`
- Display extracted slice with crosshairs
- Overlay metadata information
- Show processing statistics

### 6. Format Converter (`convert_slices_to_jpg.py`)

**Class**: `SliceConverter`

#### `convert_simple(npy_path, output_path, normalize=True)`
- Load numpy array
- Normalize to 0-255 range: `(data * 255).astype(np.uint8)`
- Save using PIL: `Image.fromarray(img_data).save(output_path, 'JPEG', quality=95)`

#### `convert_with_contrast(npy_path, output_path, window_center=0.5, window_width=1.0)`
- Use matplotlib for contrast control
- Apply windowing: vmin/vmax based on center/width
- Save with specified DPI

#### `convert_with_annotations(npy_path, output_path, metadata_path)`
- Add crosshairs at image center
- Overlay metadata text (slice index, confidence, method)
- Include title and formatting

### 7. Main Pipeline Runner (`run_pipeline.py`)

#### `main()` Function Flow:
1. **Parse command line arguments**:
   - input_dir (required)
   - --output (default: ./output)
   - --config (default: ./config.json)
   - --workers (default: auto)
   - --max_files (for testing)
   - --no_viz (skip visualizations)

2. **Load configuration** from JSON file

3. **Find CT files**: `input_path.glob('*.nii.gz')`

4. **Initialize processor** with config parameters

5. **Process batch** with parallel execution

6. **Create visualizations** if requested:
   - Quality dashboard
   - Slice montage

7. **Print summary** with file counts and paths

## Implementation Details

### Error Handling Strategy:
- **File level**: Catch and log individual CT processing errors
- **Batch level**: Continue processing remaining files if some fail
- **Graceful degradation**: Fallback methods if landmark detection fails

### Memory Management:
- **Process images one at a time** to avoid memory issues
- **Use generators** for large file lists
- **Explicit cleanup** of large arrays after processing

### Performance Optimizations:
- **Parallel processing** with ProcessPoolExecutor
- **Simplified detection** for batch mode (reduced algorithm complexity)
- **Efficient resampling** with SimpleITK
- **Progress monitoring** with tqdm

### Quality Assurance:
- **Confidence scoring** for all detections
- **Multi-method validation** with consensus voting
- **Comprehensive logging** of processing steps
- **Visual verification** tools for manual review

## Expected Performance Metrics:
- **Success rate**: >95% on typical chest CT datasets
- **Processing time**: 2-15 seconds per scan (depends on image size)
- **Mean confidence**: >0.8 for high-quality datasets
- **Memory usage**: <1GB per worker process

## Testing Strategy:

### Unit Tests:
- Test each detection method independently
- Validate preprocessing steps
- Check output format consistency

### Integration Tests:
- End-to-end pipeline validation
- Batch processing with known datasets
- Error handling with corrupted files

### Performance Tests:
- Large dataset processing (100+ scans)
- Memory usage monitoring
- Processing time benchmarks

## Deployment Considerations:

### Hardware Requirements:
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended for large batches
- **Storage**: Fast SSD for large datasets

### Software Dependencies:
- **Python 3.9+**
- **Conda environment** recommended for isolation
- **All packages** specified in requirements.txt

### Usage Examples:
```bash
# Basic usage
python run_pipeline.py /path/to/ct/scans

# Advanced usage
python run_pipeline.py /path/to/ct/scans \
    --output /path/to/results \
    --workers 8 \
    --config custom_config.json

# Convert to images
python convert_slices_to_jpg.py /path/to/results/slices \
    --format jpg \
    --type annotated \
    --metadata_dir /path/to/results/metadata
```

This specification provides complete implementation details for recreating the entire CT aortic valve level detection pipeline. All algorithms, parameter values, class structures, and integration details are included to enable faithful reproduction of the system.