"""
Optimized ONNX inference for single image with GPU
Fast and simple inference without unnecessary overhead
"""
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

# Import config
from src.config import DetectionClasses, ImageConfig, ONNXConfig

# Use detection classes from config
CLASS_NAMES = DetectionClasses.CLASSES


class FastONNXDetector:
    """Optimized ONNX detector for single image inference"""
    
    def __init__(self, model_path: str, img_size: int = None):
        # Use image size from config if not specified
        self.img_size = img_size if img_size is not None else ImageConfig.DETECTION_SIZE
        
        # Setup ONNX Runtime with GPU - use config
        providers = ONNXConfig.PROVIDERS
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Warmup
        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        self.session.run(self.output_names, {self.input_name: dummy})
        
    def preprocess(self, image: np.ndarray):
        """Fast preprocessing with letterbox"""
        h, w = image.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Normalize and transpose
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        
        return img, scale, (pad_x, pad_y)
    
    def postprocess(self, output, scale, padding, orig_shape, conf_thresh=0.25, iou_thresh=0.45):
        """Fast postprocessing with NMS"""
        # YOLOv11 output: [batch, 22, 8400] where 22 = 4 bbox + 18 classes
        # output shape: [1, 22, 8400]
        # Remove batch dimension and transpose
        predictions = output[0]  # [22, 8400] - remove batch dimension
        predictions = predictions.T  # Transpose to [8400, 22]
        
        # First 4 columns: bbox [x, y, w, h]
        # Remaining 18 columns: class scores
        boxes = predictions[:, :4]  # [8400, 4]
        scores = predictions[:, 4:]  # [8400, 18]
        
        # Get class with max score for each prediction
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        # Filter by confidence
        mask = confidences > conf_thresh
        if not mask.any():
            return []
        
        # Apply mask to filter low-confidence predictions
        boxes = boxes[mask]  # Filter rows
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert box format: center -> corners
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Adjust coordinates
        pad_x, pad_y = padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Clip to image
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        # NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self.nms(boxes_xyxy, confidences, iou_thresh)
        
        # Build results
        detections = []
        for idx in keep:
            cls_id = int(class_ids[idx])
            detections.append({
                'bbox': [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                'confidence': float(confidences[idx]),
                'class_id': cls_id,
                'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            })
        
        return detections
    
    def nms(self, boxes, scores, iou_threshold):
        """Vectorized NMS"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            order = order[1:][iou <= iou_threshold]
        
        return keep
    
    def predict(self, image, conf_thresh=None, iou_thresh=None):
        """Run inference on single image - uses config defaults if not specified"""
        # Use thresholds from config if not specified
        if conf_thresh is None:
            conf_thresh = DetectionClasses.CONFIDENCE_THRESHOLD
        if iou_thresh is None:
            iou_thresh = DetectionClasses.IOU_THRESHOLD
        
        # Preprocess
        img_input, scale, padding = self.preprocess(image)
        
        # Inference
        t0 = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: img_input})
        inference_time = (time.time() - t0) * 1000
        
        # Postprocess
        # outputs is a list, get first element
        detections = self.postprocess(outputs[0], scale, padding, image.shape, conf_thresh, iou_thresh)
        
        return detections, inference_time


def draw_detections(image, detections, class_names=None):
    """Draw bounding boxes on image - uses config for visualization with pretty names"""
    result = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        conf = det['confidence']
        cls_id = det['class_id']
        cls_name = det.get('class_name', f'Class {cls_id}')
        
        # Get display name (pretty format)
        display_name = DetectionClasses.get_display_name(cls_name)
        
        # Color per class
        np.random.seed(cls_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw box - use config thickness
        cv2.rectangle(result, (x1, y1), (x2, y2), color, DetectionClasses.BOX_THICKNESS)
        
        # Create label with display name
        label = f"{display_name}: {conf:.2f}"
        
        # Use config font settings
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                      DetectionClasses.FONT_SCALE, 
                                      DetectionClasses.FONT_THICKNESS)
        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   DetectionClasses.FONT_SCALE, (255, 255, 255), 
                   DetectionClasses.FONT_THICKNESS)
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Fast ONNX inference for single image')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Output image path (default: same as input with _result suffix)')
    parser.add_argument('--conf', type=float, default=None, help=f'Confidence threshold (default: {DetectionClasses.CONFIDENCE_THRESHOLD} from config)')
    parser.add_argument('--iou', type=float, default=None, help=f'IoU threshold for NMS (default: {DetectionClasses.IOU_THRESHOLD} from config)')
    parser.add_argument('--imgsz', type=int, default=None, help=f'Image size (default: {ImageConfig.DETECTION_SIZE} from config)')
    parser.add_argument('--show', action='store_true', help='Show result image')
    parser.add_argument('--no-save', action='store_true', help='Do not save result')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Error: Cannot read image {args.image}")
        return
    
    # Use config defaults if not specified
    conf = args.conf if args.conf is not None else DetectionClasses.CONFIDENCE_THRESHOLD
    iou = args.iou if args.iou is not None else DetectionClasses.IOU_THRESHOLD
    imgsz = args.imgsz if args.imgsz is not None else ImageConfig.DETECTION_SIZE
    
    print("=" * 60)
    print("ONNX Single Image Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")
    print(f"Conf: {conf} | IoU: {iou}")
    print("=" * 60 + "\n")
    
    # Initialize detector (uses config if imgsz not specified)
    print("Loading model...")
    detector = FastONNXDetector(args.model, imgsz)
    
    provider = detector.session.get_providers()[0]
    if 'CUDA' in provider:
        print("✓ Using GPU (CUDA)\n")
    else:
        print("⚠ Using CPU\n")
    
    # Run inference (uses config thresholds)
    print("Running inference...")
    detections, inference_time = detector.predict(image, conf, iou)
    
    print(f"✓ Inference time: {inference_time:.1f}ms")
    print(f"✓ Found {len(detections)} objects\n")
    
    # Print detections
    if detections:
        print("Detections:")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            cls_name = det.get('class_name', f"Class {det['class_id']}")
            print(f"  {i}. {cls_name:<30} | Conf: {det['confidence']:.3f} | Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    # Save result
    if not args.no_save:
        result_img = draw_detections(image, detections)
        
        # Always save to onnx_inference directory
        output_dir = Path("/media/vbdi/ssd2t/Medical/dung/projects/main_det/onnx/outputs/onnx_inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.output:
            # If user specifies output, use it but ensure it's in the output directory
            output_path = output_dir / Path(args.output).name
        else:
            # Use input filename with _result suffix
            input_path = Path(args.image)
            output_path = output_dir / f"{input_path.stem}_result{input_path.suffix}"
        
        cv2.imwrite(str(output_path), result_img)
        print(f"\n✓ Result saved to: {output_path}")
    
    # Show result
    if args.show:
        result_img = draw_detections(image, detections)
        cv2.imshow('Result', result_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("=" * 60)


if __name__ == '__main__':
    main()

