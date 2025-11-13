"""
Standalone ONNX Inference - Simple and minimal
Input: image path -> Output: predicted classes
"""

import os
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed!")
    print("Install with: pip install onnxruntime")
    exit(1)

# Import config
from src.config import DiseaseClasses, ImageConfig, ONNXConfig

# Use disease classes and thresholds from config
DISEASE_CLASSES = DiseaseClasses.CLASSES
OPTIMIZED_THRESHOLDS = DiseaseClasses.THRESHOLDS


def get_transforms():
    """Get transforms for inference - uses config"""
    IMAGE_SIZE = ImageConfig.MULTILABEL_SIZE
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


def preprocess_image(image_array):
    """
    Preprocess image for model input
    Args:
        image_array: numpy array (RGB format)
    """
    # Get transforms (same as inference_3class.py)
    transform = get_transforms()
    
    image = Image.fromarray(image_array)
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Convert to numpy and add batch dimension for ONNX
    image_array = image_tensor.numpy()
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def predict_with_session(session, image_array):
    """
    Predict diseases from chest X-ray image using pre-loaded session
    Args:
        session: Pre-loaded ONNX InferenceSession
        image_array: numpy array (RGB format)
    """
    # Preprocess
    image_array = preprocess_image(image_array)
    
    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array.astype(np.float32)})
    
    # Sigmoid
    logits = outputs[0][0]
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Apply per-class optimized thresholds
    predictions = []
    tuberculosis_detected = False
    tuberculosis_prob = 0.0
    
    for i, class_name in enumerate(DISEASE_CLASSES):
        prob = probabilities[i]
        threshold = OPTIMIZED_THRESHOLDS.get(class_name, 0.5)
        if prob >= threshold:
            if class_name == 'Tuberculosis':
                tuberculosis_detected = True
                tuberculosis_prob = float(prob)
            else:
                predictions.append({
                    'class': class_name,
                    'probability': float(prob),
                    'threshold': threshold
                })
    
    # Priority: If Tuberculosis is detected, only return Tuberculosis
    if tuberculosis_detected:
        final_predictions = [{
            'class': 'Tuberculosis',
            'probability': tuberculosis_prob,
            'threshold': OPTIMIZED_THRESHOLDS.get('Tuberculosis', 0.5)
        }]
    else:
        final_predictions = predictions
    
    return {
        'positive_classes': final_predictions,
        'all_probabilities': {DISEASE_CLASSES[i]: float(probabilities[i]) for i in range(len(DISEASE_CLASSES))},
        'tuberculosis_detected': tuberculosis_detected
    }


def predict(onnx_model_path, image_array):
    """
    Predict diseases from chest X-ray image
    Args:
        onnx_model_path: Path to ONNX model
        image_array: numpy array (RGB format)
    """
    # Use providers from config
    providers = ONNXConfig.PROVIDERS
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # Preprocess
    image_array = preprocess_image(image_array)
    
    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array.astype(np.float32)})
    
    # Sigmoid
    logits = outputs[0][0]
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Apply per-class optimized thresholds
    predictions = []
    tuberculosis_detected = False
    tuberculosis_prob = 0.0
    
    for i, class_name in enumerate(DISEASE_CLASSES):
        prob = probabilities[i]
        threshold = OPTIMIZED_THRESHOLDS.get(class_name, 0.5)
        if prob >= threshold:
            if class_name == 'Tuberculosis':
                tuberculosis_detected = True
                tuberculosis_prob = float(prob)
            else:
                predictions.append({
                    'class': class_name,
                    'probability': float(prob),
                    'threshold': threshold
                })
    
    # Priority: If Tuberculosis is detected, only return Tuberculosis
    if tuberculosis_detected:
        final_predictions = [{
            'class': 'Tuberculosis',
            'probability': tuberculosis_prob,
            'threshold': OPTIMIZED_THRESHOLDS.get('Tuberculosis', 0.5)
        }]
    else:
        final_predictions = predictions
    
    return {
        'positive_classes': final_predictions,
        'all_probabilities': {DISEASE_CLASSES[i]: float(probabilities[i]) for i in range(len(DISEASE_CLASSES))},
        'tuberculosis_detected': tuberculosis_detected
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX Inference - Simple output (GPU only, image_size=384)")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--verbose', action='store_true', help='Show all probabilities')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        exit(1)
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        exit(1)
    
    # Run inference (GPU fixed, image_size=384 fixed)
    result = predict(args.model, args.image)
    
    # Print results
    print(f"\nImage: {args.image}")
    print(f"Positive findings: {len(result['positive_classes'])}")
    print("-" * 60)
    
    if len(result['positive_classes']) == 0:
        print("No abnormalities detected")
    else:
        if result['tuberculosis_detected']:
            print("⚠️  Tuberculosis detected (priority output)")
        for pred in result['positive_classes']:
            print(f"  {pred['class']:45s} {pred['probability']:.4f} (threshold: {pred['threshold']})")
    
    # Verbose: show all probabilities
    if args.verbose:
        print("\nAll probabilities:")
        print("-" * 60)
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            thresh = OPTIMIZED_THRESHOLDS.get(cls, 0.5)
            status = "✓" if prob >= thresh else " "
            print(f"  {status} {cls:45s} {prob:.4f}")


if __name__ == "__main__":
    main()


