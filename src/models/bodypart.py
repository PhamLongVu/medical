import argparse
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
from src.config import (
    BodyPartClasses,
    ImageConfig,
    ModelPaths,
    ONNXConfig
)

def preprocess_image(image_np: np.ndarray, image_size: int = ImageConfig.BODYPART_SIZE) -> np.ndarray:
    """Preprocess a single image for ONNX inference"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and convert image
    image = Image.fromarray(image_np).convert('RGB')
    image_tensor = transform(image)
    
    # Convert to numpy and add batch dimension
    image_np = image_tensor.numpy()
    image_np = np.expand_dims(image_np, axis=0)  # [1, 3, H, W]
    
    return image_np


def predict_onnx(ort_session, image_np):
    """Run ONNX inference"""
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    ort_inputs = {input_name: image_np.astype(np.float32)}
    outputs = ort_session.run(None, ort_inputs)
    
    # Get prediction
    logits = outputs[0][0]  # Remove batch dimension
    pred_idx = np.argmax(logits)
    
    # Calculate softmax for confidence
    exp_logits = np.exp(logits - np.max(logits))
    softmax = exp_logits / np.sum(exp_logits)
    confidence = softmax[pred_idx]
    
    return pred_idx, confidence


def main():
    parser = argparse.ArgumentParser(description='Body Part Classification - ONNX Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default=ModelPaths.BODYPART_CLASSIFIER,
                       help=f'Path to ONNX model (default: {ModelPaths.BODYPART_CLASSIFIER})')
    parser.add_argument('--image-size', type=int, default=ImageConfig.BODYPART_SIZE,
                       help=f'Image size (default: {ImageConfig.BODYPART_SIZE})')
    parser.add_argument('--providers', type=str, nargs='+',
                       default=ONNXConfig.PROVIDERS,
                       choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
                       help='ONNX Runtime execution providers')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONNX Inference")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Image:        {args.image}")
    print(f"Image size:   {args.image_size}x{args.image_size}")
    print(f"Providers:    {args.providers}")
    print("=" * 60)
    
    # Create ONNX Runtime session
    print("\nLoading ONNX model...")
    ort_session = ort.InferenceSession(args.model, providers=args.providers)
    print("✓ Model loaded")
    
    # Print model info
    print("\nModel Information:")
    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]
    print(f"  Input name:  {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    
    # Load image
    print(f"\nLoading image...")
    image = cv2.imread(args.image)
    if image is None:
        print(f"✗ Failed to load image: {args.image}")
        return None
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image_rgb.shape}")
    
    # Preprocess image
    print(f"Preprocessing image...")
    image_np = preprocess_image(image_rgb, args.image_size)
    print(f"✓ Image preprocessed: {image_np.shape}")
    
    # Run inference
    print(f"\nRunning inference...")
    pred_idx, confidence = predict_onnx(ort_session, image_np)
    
    # Get class information
    class_name = BodyPartClasses.get_class_name(pred_idx)
    backend = BodyPartClasses.get_backend(pred_idx)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Class Index:  {pred_idx}")
    print(f"Class Name:   {class_name}")
    print(f"Backend:      {backend}")
    print(f"Confidence:   {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 60)
    
    return {
        'class_index': int(pred_idx),
        'class_name': class_name,
        'backend': backend,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    main()

