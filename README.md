# Full Medical Image Analysis Pipeline

Complete pipeline for chest X-ray analysis from DICOM to final diagnosis with lesion localization.

## Pipeline Overview

```
DICOM File
    ↓
[Stage 1] Convert DICOM → PNG
    ↓
[Stage 2] Binary Classification (No Finding / Abnormal)
    ↓
    ├─→ No Finding → Return "No Finding" (Pipeline ends)
    │
    └─→ Abnormal
           ↓
           ┌──────────────────────────────────────────────────┐
           │  ⚡ PARALLEL EXECUTION (Optimized)              │
           │                                                  │
           │  [Stage 3] Multi-label Classification            │
           │  → Tuberculosis, Pneumonia, etc. (28 classes)   │
           │                                                  │
           │  [Stage 4] Object Detection                      │
           │  → Lesion localization (18 classes)              │
           └──────────────────────────────────────────────────┘
                          ↓
              Results + Visualization
```

## Features

- **Stage 1: DICOM Conversion** - Convert DICOM files to PNG with proper preprocessing
- **Stage 2: Binary Classification** - Fast initial screening (No Finding vs Abnormal)
- **⚡ Parallel Execution** - Stages 3 & 4 run simultaneously when abnormal detected
  - **1.3-1.5x faster** than sequential execution
  - Optimized GPU utilization with ThreadPoolExecutor
- **Stage 3: Multi-label Classification** - Identify specific diseases (28 classes)
  - Priority detection for Tuberculosis
  - Optimized F2-score thresholds per class
- **Stage 4: Object Detection** - Localize lesions with bounding boxes (18 classes)
  - Always saves visualization when abnormal detected
  - Automatic bounding box drawing on detected lesions
- **GPU Acceleration** - Optimized ONNX Runtime with CUDA support
- **Comprehensive Results** - JSON output with all stages' results + performance metrics

## Requirements

```bash
pip install onnxruntime-gpu  # or onnxruntime for CPU only
pip install numpy opencv-python pillow torchvision pydicom
```

## Directory Structure

```
full_stream/
├── main.py                          # Main pipeline script
├── run_pipeline.sh                  # Shell script for easy execution
├── dcm2img2.py                      # DICOM to PNG converter
├── cls_normal_abnormal_onnx.py     # Binary classifier (Normal/Abnormal)
├── cls_multilabel_onnx.py          # Multi-label classifier (28 diseases)
├── detection_onnx.py               # Object detector (18 lesion types)
├── weight/                          # Model weights
│   ├── cls_2class/
│   │   └── model_2class_simplified.onnx
│   ├── cls_multilabel/
│   │   └── convnext_base_384_dynamic_simplified.onnx
│   └── detection/
│       └── detection.onnx
├── img_convert/                     # Converted PNG images
├── img_test/                        # Test DICOM files
└── outputs/                         # Results (JSON + visualizations)
```

## Usage

### Method 1: Using Shell Script (Recommended)

The script has a pre-configured DICOM path. You can edit the path in `run_pipeline.sh`:

```bash
cd full_stream

# Edit the DICOM_FILE path in run_pipeline.sh if needed
# DICOM_FILE="/path/to/your/file.dcm"

chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Method 2: Direct Python Command

```bash
python main.py \
    --dicom img_test/your_file.dcm \
    --model_2class weight/cls_2class/model_2class_simplified.onnx \
    --model_multilabel weight/cls_multilabel/convnext_base_384_dynamic_simplified.onnx \
    --model_detection weight/detection/detection.onnx
```

### Optional Arguments

```bash
python main.py \
    --dicom <path_to_dicom> \
    --model_2class <path_to_2class_model> \
    --model_multilabel <path_to_multilabel_model> \
    --model_detection <path_to_detection_model> \
    --img_convert_dir <directory_for_converted_images> \
    --output_dir <directory_for_results> \
    --output_json <path_to_save_json_results> \
    --no_save_detection  # Don't save detection visualization
```

## Output

### Console Output Example (With Abnormal Detection)

```
================================================================================
FULL MEDICAL IMAGE ANALYSIS PIPELINE
================================================================================

┌─ STAGE 1: Converting DICOM to PNG
└─ ✓ Converted to: img_convert/sample.png (45.2ms)

┌─ STAGE 2: Binary Classification (No Finding / Abnormal)
│  Result: Abnormal
│  Confidence: 0.9234
│  Probabilities:
│    - No Finding: 0.0766
│    - Abnormal:   0.9234
└─ ✓ Completed (123.4ms)

┌─ DECISION: Abnormal detected
└─ Proceeding to multi-label classification and detection (PARALLEL)

┌─ STAGE 3 & 4: Running Classification and Detection in parallel...
└─ ✓ Both tasks completed in parallel (189.3ms)

┌─ STAGE 3 Results: Multi-label Classification
│  Positive findings: 2
│  ⚠️  TUBERCULOSIS DETECTED (Priority)
│  Detected diseases:
│    - Tuberculosis                              0.8765
│    - Opacity                                   0.6543
└─ ✓ Completed (156.7ms)

┌─ STAGE 4 Results: Object Detection (Lesion Localization)
│  Found 3 lesion(s)
│  Detected lesions:
│    1. Opacity                     | Conf: 0.876 | Box: [234, 456, 567, 789]
│    2. Nodule_Mass                 | Conf: 0.654 | Box: [123, 234, 345, 456]
│    3. Consolidation               | Conf: 0.543 | Box: [345, 456, 567, 678]
└─ ✓ Completed (189.3ms)

⚡ Performance gain: Sequential would take ~346.0ms, parallel took 189.3ms (saved ~156.7ms)

✓ Detection visualization saved to: outputs/sample_detection_result.png

================================================================================
PIPELINE SUMMARY
================================================================================
Input DICOM: img_test/sample.dcm

Final Result: ABNORMAL
  → Diseases detected: 2
     ⚠️  TUBERCULOSIS DETECTED
     - Tuberculosis: 0.8765
     - Opacity: 0.6543
  → Lesions detected: 3
     1. Opacity: 0.876
     2. Nodule_Mass: 0.654
     3. Consolidation: 0.543

Total Pipeline Time: 357.9ms
================================================================================
```

### Console Output Example (With No Finding)

```
================================================================================
FULL MEDICAL IMAGE ANALYSIS PIPELINE
================================================================================

┌─ STAGE 1: Converting DICOM to PNG
└─ ✓ Converted to: img_convert/sample.png (45.2ms)

┌─ STAGE 2: Binary Classification (No Finding / Abnormal)
│  Result: No Finding
│  Confidence: 0.7016
│  Probabilities:
│    - No Finding: 0.7016
│    - Abnormal:   0.2984
└─ ✓ Completed (123.4ms)

┌─ DECISION: No Finding detected
└─ Pipeline terminated (no further analysis needed)

================================================================================
PIPELINE SUMMARY
================================================================================
Input DICOM: img_test/sample.dcm

Final Result: NO_FINDING
  → No abnormalities detected

Total Pipeline Time: 168.6ms
================================================================================
```

### JSON Output Structure (With Abnormal Detection)

```json
{
  "dicom_path": "img_test/sample.dcm",
  "pipeline_stages": {
    "stage1_conversion": {
      "status": "success",
      "png_path": "img_convert/sample.png",
      "time_ms": 45.2
    },
    "stage2_binary_classification": {
      "status": "success",
      "result": {
        "predicted_class": 1,
        "class_name": "Abnormal",
        "confidence": 0.9234,
        "probabilities": {
          "No Finding": 0.0766,
          "Abnormal": 0.9234
        }
      },
      "time_ms": 123.4
    },
    "stage3_multilabel_classification": {
      "status": "success",
      "result": {
        "positive_classes": [
          {
            "class": "Tuberculosis",
            "probability": 0.8765,
            "threshold": 0.45
          }
        ],
        "tuberculosis_detected": true,
        "all_probabilities": {...}
      },
      "time_ms": 156.7
    },
    "stage4_detection": {
      "status": "success",
      "detections": [
        {
          "bbox": [234.0, 456.0, 567.0, 789.0],
          "confidence": 0.876,
          "class_id": 11,
          "class_name": "Opacity"
        }
      ],
      "num_detections": 3,
      "inference_time_ms": 189.3,
      "total_time_ms": 189.3,
      "visualization_path": "outputs/sample_detection_result.png"
    },
    "parallel_execution": {
      "total_parallel_time_ms": 189.3,
      "sequential_time_estimate_ms": 346.0,
      "time_saved_ms": 156.7,
      "speedup_ratio": 1.83
    }
  },
  "final_result": {
    "status": "abnormal",
    "binary_classification": {...},
    "diseases": {...},
    "lesion_localization": {
      "num_lesions": 3,
      "detections": [...]
    }
  },
  "total_pipeline_time_ms": 357.9
}
```

### JSON Output Structure (With No Finding)

```json
{
  "dicom_path": "img_test/sample.dcm",
  "pipeline_stages": {
    "stage1_conversion": {
      "status": "success",
      "png_path": "img_convert/sample.png",
      "time_ms": 45.2
    },
    "stage2_binary_classification": {
      "status": "success",
      "result": {
        "predicted_class": 0,
        "class_name": "No Finding",
        "confidence": 0.7016,
        "probabilities": {
          "No Finding": 0.7016,
          "Abnormal": 0.2984
        }
      },
      "time_ms": 123.4
    }
  },
  "final_result": {
    "status": "no_finding",
    "message": "No abnormalities detected",
    "binary_classification": {...}
  },
  "total_pipeline_time_ms": 168.6
}
```

## Disease Classes (28 classes)

- Aortic enlargement
- Atelectasis
- Calcification
- Cardiomegaly
- Clavicle fracture
- Consolidation
- Edema
- Emphysema
- Enlarged PA
- Infiltration
- Interstitial lung disease (ILD)
- Lung cavity
- Lung cyst
- Mediastinal shift
- Nodule/Mass
- Opacity
- Other lesion
- Pleural effusion
- Pleural thickening
- Pneumothorax
- Pulmonary fibrosis
- Rib fracture
- COPD
- Cardiovascular disease
- Lung tumor
- Other
- Pneumonia
- **Tuberculosis** (Priority detection)

## Detection Classes (18 classes)

- Aortic_enlargement
- Atelectasis
- Calcification
- Cardiomegaly
- Clavicle_fracture
- Consolidation
- Emphysema
- Enlarged_PA
- Infiltration
- Interstitial_lung_disease
- Nodule_Mass
- Opacity
- Other_lesion
- Pleural_effusion
- Pleural_thickening
- Pneumothorax
- Pulmonary_fibrosis
- Rib_fracture

## Performance

### Typical Inference Times on NVIDIA GPU

**Sequential Execution (Old):**
- DICOM Conversion: ~50ms
- Binary Classification: ~120ms
- Multi-label Classification: ~150ms
- Object Detection: ~230ms
- **Total: ~550ms**

**With Parallel Optimization (New):**
- DICOM Conversion: ~50ms
- Binary Classification: ~120ms
- **Parallel Stage (max of both):**
  - Multi-label Classification: ~150ms
  - Object Detection: ~230ms
  - **Both complete in: ~230ms** (runs simultaneously)
- **Total: ~400ms** ⚡

### Performance Improvement

| Scenario | Sequential | Parallel | Speedup |
|----------|-----------|----------|---------|
| No Finding | ~170ms | ~170ms | 1.0x (no optimization needed) |
| Abnormal (with detection) | ~550ms | ~400ms | **1.38x faster** ⚡ |
| Time Saved | - | ~150ms | **27% reduction** |

### Key Optimizations

1. **Parallel Execution**: Stage 3 and 4 run simultaneously using ThreadPoolExecutor
2. **GPU Utilization**: Both models can use GPU concurrently (different CUDA streams)
3. **No Overhead**: Direct function calls without serialization
4. **Automatic Speedup**: Works out-of-the-box, no configuration needed

## Troubleshooting

### GPU not detected
```bash
# Check ONNX Runtime providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should see: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Model loading error
- Ensure all model files exist in `weight/` directory
- Check ONNX model compatibility with your ONNX Runtime version

### DICOM conversion error
- Verify DICOM file is valid
- Check if pydicom is installed correctly
- Ensure the DICOM file contains pixel data

## Notes

### Pipeline Behavior
- The pipeline automatically terminates after Stage 2 if "No Finding" is detected (no wasted computation)
- **Parallel execution** is automatically enabled when "Abnormal" is detected (1.3-1.5x faster)
- Tuberculosis detection has priority - if detected, it will be highlighted in output
- Detection visualization is **always saved** when abnormal is detected (even if no lesions found)
- All results are saved to JSON for further processing

### Technical Details
- GPU acceleration is automatically used if available (CUDA/TensorRT)
- Thread-safe parallel execution using Python's `concurrent.futures`
- Real-time performance metrics included in results
- Optimized for both speed and accuracy

### Output Files
When abnormal is detected, you'll get:
1. **JSON file**: Complete results with all metrics (`{filename}_results.json`)
2. **Visualization**: Annotated image with bounding boxes (`{filename}_detection_result.png`)
3. **Console output**: Real-time progress and summary

### Pre-configured Script
The `run_pipeline.sh` script has a pre-configured DICOM path:
```bash
DICOM_FILE="/home/vbdi/Documents/convnext-chexpert-attention/full_stream/dicom_test/DICOM_HOANG VAN MINH_1760674525219.dcm"
```
Edit this line to change the input file.

## License

See LICENSE file in the root directory.

