"""
Full pipeline: DICOM → Image → Classification (Normal/Abnormal) → Multi-label + Detection
Pipeline flow:
1. Convert DICOM to PNG
2. Classify: No Finding vs Abnormal
3. If No Finding → return "No Finding"
4. If Abnormal → run in parallel:
   - Multi-label classification (Tuberculosis, Pneumonia, etc.)
   - Detection (lesion localization)
"""

import os
import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import config
from src.config import (
    ModelPaths,
    DirectoryPaths,
    PipelineConfig,
    DetectionClasses,
    ImageConfig
)

# Import functions from existing modules
from src.utils.dicom import dicom2img
from src.models.binary import (
    load_onnx_model as load_2class_model,
    get_transforms as get_2class_transforms,
    predict_single_image as predict_2class
)
from src.models.detection import FastONNXDetector

def run_full_pipeline(image, binary_session, binary_input_name, binary_output_name, binary_transform,
                     multilabel_session, detection_model):
    """
    Full pipeline for chest X-ray analysis
    
    Args:
        image: Input image as numpy array (BGR format)
        binary_session: Pre-loaded binary model ONNX session
        binary_input_name: Binary model input name
        binary_output_name: Binary model output name
        binary_transform: Binary model transform
        multilabel_session: Pre-loaded multilabel model ONNX session
        detection_model: Pre-loaded detection model instance (FastONNXDetector)
    
    Returns:
        dict: Complete results from all stages
    """
    results = {
        'input_type': 'numpy_array',
        'input_shape': image.shape,
        'pipeline_stages': {},
        'final_result': None
    }
    
    print(f"\n{'='*80}")
    print("FULL MEDICAL IMAGE ANALYSIS PIPELINE")
    print(f"{'='*80}\n")
    
    # ====================
    # STAGE 1: Validate Input Image
    # ====================
    print("┌─ STAGE 1: Validating input image")
    start_time = time.time()
    
    # Validate image is numpy array
    if not isinstance(image, np.ndarray):
        print(f"└─ ✗ Invalid input: expected numpy array, got {type(image)}\n")
        results['pipeline_stages']['stage1_validation'] = {
            'status': 'failed',
            'error': f'Invalid input type: {type(image)}'
        }
        return results
    
    # Validate image dimensions
    if len(image.shape) not in [2, 3]:
        print(f"└─ ✗ Invalid image dimensions: {image.shape}\n")
        results['pipeline_stages']['stage1_validation'] = {
            'status': 'failed',
            'error': f'Invalid image dimensions: {image.shape}'
        }
        return results
    
    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    stage1_time = time.time() - start_time
    
    print(f"│  Image shape: {image.shape}")
    print(f"│  Image dtype: {image.dtype}")
    print(f"└─ ✓ Input validated ({stage1_time*1000:.1f}ms)\n")
    
    results['pipeline_stages']['stage1_validation'] = {
        'status': 'success',
        'input_type': 'numpy_array',
        'image_shape': image.shape,
        'image_dtype': str(image.dtype),
        'time_ms': stage1_time * 1000
    }
    
    # =======================================
    # STAGE 2 & 3: Run Binary Classification and Detection in PARALLEL (ALWAYS)
    # =======================================
    print("┌─ STAGE 2 & 3: Running Binary Classification and Detection in parallel...")
    print("│  (Detection always runs regardless of classification result)")
    print("│  (All processing done in memory - no temp files)")
    
    # Convert BGR to RGB for classification models (detection uses BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define parallel tasks
    def run_binary_classification():
        """Task 1: Binary classification (No Finding / Abnormal)"""
        start_time = time.time()
        # Use pre-loaded session
        result = predict_2class(binary_session, binary_input_name, binary_output_name, image_rgb, binary_transform)
        time_ms = (time.time() - start_time) * 1000
        return ('binary', result, time_ms)
    
    def run_detection():
        """Task 2: Object detection (always runs, uses BGR)"""
        start_time = time.time()
        # Use pre-loaded detector
        detections, inference_time_ms = detection_model.predict(image)
        total_time_ms = (time.time() - start_time) * 1000
        return ('detection', detections, inference_time_ms, total_time_ms)
    
    # Run both tasks in parallel
    parallel_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_binary = executor.submit(run_binary_classification)
        future_detection = executor.submit(run_detection)
        
        # Wait for both to complete
        binary_task_type, binary_result, stage2_time = future_binary.result()
        detection_task_type, detections, detection_inference_time, stage3_time = future_detection.result()
    
    parallel_time_stage2_3 = (time.time() - parallel_start_time) * 1000
    
    print(f"└─ ✓ Both tasks completed in parallel ({parallel_time_stage2_3:.1f}ms)\n")
    
    # =======================================
    # Display STAGE 2 Results: Binary Classification
    # =======================================
    print("┌─ STAGE 2 Results: Binary Classification")
    print(f"│  Result: {binary_result['class_name']}")
    print(f"│  Confidence: {binary_result['confidence']:.4f}")
    print(f"│  Probabilities:")
    print(f"│    - No Finding: {binary_result['probabilities']['No Finding']:.4f}")
    print(f"│    - Abnormal:   {binary_result['probabilities']['Abnormal']:.4f}")
    print(f"└─ ✓ Completed ({stage2_time:.1f}ms)\n")
    
    results['pipeline_stages']['stage2_binary_classification'] = {
        'status': 'success',
        'result': binary_result,
        'time_ms': stage2_time
    }
    
    # =======================================
    # Display STAGE 3 Results: Object Detection (ALWAYS)
    # =======================================
    print("┌─ STAGE 3 Results: Object Detection (Lesion Localization)")
    print(f"│  Found {len(detections)} lesion(s)")
    if detections:
        print(f"│  Detected lesions:")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            cls_name = det.get('class_name', f"Class {det['class_id']}")
            print(f"│    {i}. {cls_name:30s} | Conf: {det['confidence']:.3f} | Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    print(f"└─ ✓ Completed ({stage3_time:.1f}ms)\n")
    
    results['pipeline_stages']['stage3_detection'] = {
        'status': 'success',
        'detections': detections,
        'num_detections': len(detections),
        'inference_time_ms': detection_inference_time,
        'total_time_ms': stage3_time
    }
    
    print(f"⚡ Stage 2&3 Performance: Sequential would take ~{stage2_time + stage3_time:.1f}ms, "
          f"parallel took {parallel_time_stage2_3:.1f}ms "
          f"(saved ~{stage2_time + stage3_time - parallel_time_stage2_3:.1f}ms)\n")
    
    
    # =======================================
    # DECISION POINT: No Finding vs Abnormal
    # =======================================
    if binary_result['class_name'] == 'No Finding':
        print("┌─ DECISION: No Finding detected")
        print("└─ Skip multi-label classification (detection results available)\n")
        
        results['final_result'] = {
            'status': 'no_finding',
            'message': 'No abnormalities detected, but detection was performed',
            'binary_classification': binary_result,
            'lesion_localization': {
                'num_lesions': len(detections),
                'detections': detections
            }
        }
    else:
        print("┌─ DECISION: Abnormal detected")
        print("└─ Proceeding to multi-label classification for detailed diagnosis\n")
        
        # =======================================
        # STAGE 4: Multi-label Classification (only if Abnormal)
        # =======================================
        print("┌─ STAGE 4: Multi-label Classification (Disease Identification)")
        start_time = time.time()
        
        # Use pre-loaded session
        from src.models.multilabel import predict_with_session
        multilabel_result = predict_with_session(multilabel_session, image_rgb)
        stage4_time = (time.time() - start_time) * 1000
        
        print(f"│  Positive findings: {len(multilabel_result['positive_classes'])}")
        if multilabel_result['tuberculosis_detected']:
            print(f"│  ⚠️  TUBERCULOSIS DETECTED (Priority)")
        
        if len(multilabel_result['positive_classes']) > 0:
            print(f"│  Detected diseases:")
            for pred in multilabel_result['positive_classes']:
                print(f"│    - {pred['class']:40s} {pred['probability']:.4f}")
        else:
            print(f"│  No specific diseases detected above threshold")
        
        print(f"└─ ✓ Completed ({stage4_time:.1f}ms)\n")
        
        results['pipeline_stages']['stage4_multilabel_classification'] = {
            'status': 'success',
            'result': multilabel_result,
            'time_ms': stage4_time
        }
        
        # =======================================
        # FINAL RESULT (Abnormal with diseases)
        # =======================================
        results['final_result'] = {
            'status': 'abnormal',
            'binary_classification': binary_result,
            'diseases': multilabel_result,
            'lesion_localization': {
                'num_lesions': len(detections),
                'detections': detections
            }
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Full Medical Image Analysis Pipeline: Image/DICOM → Classification → Detection (uses config.py for defaults)'
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input file (DICOM .dcm or Image .png/.jpg/.jpeg/.bmp/.tiff, default: from config)')
    parser.add_argument('--model_2class', type=str, default=None,
                       help=f'Path to 2-class ONNX model (default: config)')
    parser.add_argument('--model_multilabel', type=str, default=None,
                       help=f'Path to multi-label ONNX model (default: config)')
    parser.add_argument('--model_detection', type=str, default=None,
                       help=f'Path to detection ONNX model (default: config)')
    parser.add_argument('--img_convert_dir', type=str, default=None,
                       help=f'Directory for temporary DICOM conversion (default: config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help=f'Directory to save output results (default: config)')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Optional: Save results to JSON file')
    parser.add_argument('--no_save_detection', action='store_true',
                       help='Do not save detection visualization image')
    
    args = parser.parse_args()
    
    # Use config defaults if arguments not provided
    input_path = args.input if args.input else DirectoryPaths.DEFAULT_INPUT
    model_2class = args.model_2class if args.model_2class else ModelPaths.BINARY_CLASSIFIER
    model_multilabel = args.model_multilabel if args.model_multilabel else ModelPaths.MULTILABEL_CLASSIFIER
    model_detection = args.model_detection if args.model_detection else ModelPaths.DETECTION
    img_convert_dir = args.img_convert_dir if args.img_convert_dir else DirectoryPaths.IMG_CONVERT
    output_dir = args.output_dir if args.output_dir else DirectoryPaths.OUTPUT
    
    # =======================================
    # Load image from file (DICOM or image file)
    # =======================================
    print(f"\n{'='*80}")
    print("LOADING INPUT IMAGE")
    print(f"{'='*80}\n")
    
    file_extension = Path(input_path).suffix.lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    dicom_extensions = ['.dcm', '.dicom']
    
    if file_extension in image_extensions:
        # Load image file directly
        print(f"Loading image file: {input_path}")
        if not os.path.exists(input_path):
            print(f"✗ Image file not found: {input_path}")
            return
        image = cv2.imread(input_path)
        if image is None:
            print(f"✗ Failed to read image: {input_path}")
            return
        print(f"✓ Image loaded: {image.shape}\n")
        input_name = Path(input_path).stem
        
    elif file_extension in dicom_extensions or file_extension == '':
        # Convert DICOM to image first
        print(f"Converting DICOM file: {input_path}")
        png_path = dicom2img(input_path, img_convert_dir)
        if png_path is None:
            print(f"✗ Failed to convert DICOM file")
            return
        image = cv2.imread(png_path)
        if image is None:
            print(f"✗ Failed to read converted image: {png_path}")
            return
        print(f"✓ DICOM converted and loaded: {image.shape}\n")
        input_name = Path(input_path).stem
        
    else:
        print(f"✗ Unsupported file type: {file_extension}")
        print(f"   Supported: DICOM (.dcm, .dicom) or Images (.png, .jpg, .jpeg, .bmp, .tiff)")
        return
    
    # Load models
    print(f"\n{'='*80}")
    print("LOADING MODELS")
    print(f"{'='*80}\n")
    
    import onnxruntime as ort
    from src.config import ONNXConfig
    
    # Load binary model
    binary_session, binary_input_name, binary_output_name = load_2class_model(model_2class)
    binary_transform = get_2class_transforms()
    print("✓ Binary classifier loaded")
    
    # Load multilabel model
    multilabel_session = ort.InferenceSession(model_multilabel, providers=ONNXConfig.PROVIDERS)
    print("✓ Multilabel classifier loaded")
    
    # Load detection model
    detection_model_instance = FastONNXDetector(model_detection, img_size=ImageConfig.DETECTION_SIZE)
    print("✓ Detection model loaded\n")
    
    # Run full pipeline with pre-loaded models
    total_start_time = time.time()
    results = run_full_pipeline(
        image=image,
        binary_session=binary_session,
        binary_input_name=binary_input_name,
        binary_output_name=binary_output_name,
        binary_transform=binary_transform,
        multilabel_session=multilabel_session,
        detection_model=detection_model_instance
    )
    total_time = time.time() - total_start_time
    
    results['total_pipeline_time_ms'] = total_time * 1000
    results['input_file'] = input_path
    
    # =======================================
    # SUMMARY
    # =======================================
    print(f"{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Input file: {input_path}")
    print(f"Input type: {results.get('input_type', 'unknown')}")
    print(f"Input shape: {results.get('input_shape', 'unknown')}")
    print(f"\nConfiguration:")
    print(f"  - Parallel execution: {PipelineConfig.ENABLE_PARALLEL} (workers: {PipelineConfig.MAX_WORKERS})")
    print(f"  - Detection threshold: conf={DetectionClasses.CONFIDENCE_THRESHOLD}, iou={DetectionClasses.IOU_THRESHOLD}")
    print(f"\nFinal Result: {results['final_result']['status'].upper()}")
    
    # Display results based on status
    lesions = results['final_result']['lesion_localization']
    print(f"  → Lesions detected: {lesions['num_lesions']}")
    if lesions['num_lesions'] > 0:
        for i, det in enumerate(lesions['detections'][:5], 1):  # Show top 5
            print(f"     {i}. {det['class_name']}: {det['confidence']:.3f}")
        if lesions['num_lesions'] > 5:
            print(f"     ... and {lesions['num_lesions'] - 5} more")
    
    if results['final_result']['status'] == 'no_finding':
        print("  → Diseases: No Finding (multi-label classification skipped)")
    else:
        multilabel = results['final_result']['diseases']
        
        print(f"  → Diseases detected: {len(multilabel['positive_classes'])}")
        if multilabel['tuberculosis_detected']:
            print(f"     ⚠️  TUBERCULOSIS DETECTED")
        if multilabel['positive_classes']:
            for pred in multilabel['positive_classes'][:5]:  # Show top 5
                print(f"     - {pred['class']}: {pred['probability']:.4f}")
            if len(multilabel['positive_classes']) > 5:
                print(f"     ... and {len(multilabel['positive_classes']) - 5} more")
    
    print(f"\nTotal Pipeline Time: {total_time*1000:.1f}ms")
    print(f"{'='*80}\n")
    
    # Save results to JSON in organized folder structure
    result_folder = os.path.join(output_dir, input_name)
    os.makedirs(result_folder, exist_ok=True)
    
    if args.output_json:
        # User specified custom path
        output_json_path = args.output_json
    else:
        # Save in result folder with simple name
        output_json_path = os.path.join(result_folder, "results.json")
    
    # Add result folder info to results
    results['result_folder'] = result_folder
    results['output_files'] = {
        'json': output_json_path,
        'visualization': results.get('pipeline_stages', {}).get('stage3_detection', {}).get('visualization_path', None)
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to folder: {result_folder}")
    print(f"  - JSON: {os.path.basename(output_json_path)}")
    if results['output_files']['visualization']:
        print(f"  - Visualization: {os.path.basename(results['output_files']['visualization'])}")
    print()
    
    print('✓ Pipeline completed successfully!')


if __name__ == '__main__':
    main()

