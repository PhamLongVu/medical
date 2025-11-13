"""
ONNX Runtime inference script for 2-class chest X-ray classification
Uses GPU acceleration with CUDA execution provider
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import json
import time
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from src.config import ImageConfig, BinaryClassifierConfig, ONNXConfig

def get_transforms():
    """Get transforms for inference - uses config"""
    IMAGE_SIZE = ImageConfig.BINARY_CLASSIFIER_SIZE
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=ImageConfig.NUM_OUTPUT_CHANNELS),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ImageConfig.MEAN,
            std=ImageConfig.STD
        )
    ])
    return transform


def load_onnx_model(onnx_path):
    """Load ONNX model with GPU support - uses config"""
    # Set up providers from config
    providers = []
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    
    # Use providers from config
    for provider in ONNXConfig.PROVIDERS:
        if provider in available_providers:
            providers.append(provider)
            if 'CUDA' in provider:
                print("✓ Using CUDA (GPU) execution provider")
            elif 'Tensorrt' in provider:
                print("✓ Using TensorRT execution provider")
    
    if not any('CUDA' in p or 'Tensorrt' in p for p in providers):
        print("⚠ GPU providers not available, falling back to CPU")
    
    # Create session options from config
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = ONNXConfig.ENABLE_MEM_PATTERN
    sess_options.enable_cpu_mem_arena = ONNXConfig.ENABLE_CPU_MEM_ARENA
    
    # Load ONNX model
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=providers
    )
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name


def preprocess_image(image_array, transform):
    """
    Preprocess image for inference
    Args:
        image_array: numpy array (RGB format)
        transform: Torchvision transforms
    """
    image = Image.fromarray(image_array)
    
    image_tensor = transform(image)
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    # Convert to numpy array
    image_np = image_batch.numpy().astype(np.float32)
    return image_np


def predict_single_image(session, input_name, output_name, image_array, transform):
    """
    Predict single image
    Args:
        image_array: numpy array (RGB format)
    """
    # Preprocess
    image_np = preprocess_image(image_array, transform)
    
    # Run inference
    outputs = session.run([output_name], {input_name: image_np})
    logits = outputs[0]
    
    # Get probabilities and prediction
    probabilities = softmax(logits[0])
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    # Use class names from config
    class_names = BinaryClassifierConfig.CLASS_NAMES
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'No Finding': float(probabilities[0]),
            'Abnormal': float(probabilities[1])
        }
    }


def softmax(x):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def main():
    parser = argparse.ArgumentParser(description='ONNX Runtime inference for 2-class chest X-ray classification (single image only, GPU enabled)')
    parser.add_argument('--onnx_model', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save results')
    
    args = parser.parse_args()
    
    # Load ONNX model (GPU is always enabled)
    session, input_name, output_name = load_onnx_model(args.onnx_model)
    
    # Get transforms (image size is fixed at 384x384)
    transform = get_transforms()
    
    # Run single image inference
    print(f"\nProcessing image: {args.image}")
    start_time = time.time()
    result = predict_single_image(session, input_name, output_name, args.image, transform)
    inference_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {result['class_name']} (Class {result['predicted_class']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nProbabilities:")
    print(f"  No Finding: {result['probabilities']['No Finding']:.4f}")
    print(f"  Abnormal: {result['probabilities']['Abnormal']:.4f}")
    print(f"\nInference time: {inference_time*1000:.2f} ms")
    
    # Save results if output file specified
    if args.output:
        output_data = {
            'model_path': args.onnx_model,
            'image_path': args.image,
            'image_size': 384,  # Fixed image size
            'use_gpu': True,  # Always enabled
            'inference_time_ms': inference_time * 1000,
            'result': result
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    print('\nInference completed!')


if __name__ == '__main__':
    main()

