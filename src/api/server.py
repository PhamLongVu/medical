"""
FastAPI Backend for Medical Image Analysis
Flow: Dicom/Image → API KEY → Bodypart Classification → Chestxray Stream (3 stages) → Result
"""

import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import config
from src.config import (
    ModelPaths,
    ImageConfig,
    BodyPartClasses,
    ONNXConfig,
)

# Import processing functions
from src.utils.dicom import dicom_bytes_to_array
from src.models.bodypart import preprocess_image, predict_onnx
from src.pipeline.chestxray import run_full_pipeline

# Import ONNX runtime
import onnxruntime as ort

# =============================================================================
# API Configuration
# =============================================================================
# Import API key management
from src.api.auth import verify_api_key as verify_key_expiration

# Legacy API keys (for backward compatibility)
# New keys should be created using: python -m src.api.auth create <name>
LEGACY_API_KEYS = {
    "test_key_123": {"name": "Test User", "active": True},
    "demo_key_456": {"name": "Demo User", "active": True},
}

# No file saving - all processing done in memory

# =============================================================================
# FastAPI App Setup
# =============================================================================
app = FastAPI(
    title="Medical Image Analysis API",
    description="API for chest X-ray analysis with body part classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Models (loaded once at startup)
# =============================================================================
bodypart_model = None
binary_model_session = None
binary_model_input_name = None
binary_model_output_name = None
binary_model_transform = None
multilabel_model_session = None
detection_model = None

def load_models():
    """Load all models at startup - only once"""
    global bodypart_model
    global binary_model_session, binary_model_input_name, binary_model_output_name, binary_model_transform
    global multilabel_model_session
    global detection_model
    
    print("Loading models...")
    
    # 1. Body part classifier
    bodypart_model = ort.InferenceSession(
        ModelPaths.BODYPART_CLASSIFIER, 
        providers=ONNXConfig.PROVIDERS
    )
    print("✓ Body part classifier loaded")
    
    # 2. Binary classifier (2-class)
    from src.models.binary import load_onnx_model, get_transforms
    binary_model_session, binary_model_input_name, binary_model_output_name = load_onnx_model(
        ModelPaths.BINARY_CLASSIFIER
    )
    binary_model_transform = get_transforms()
    print("✓ Binary classifier loaded")
    
    # 3. Multilabel classifier
    multilabel_model_session = ort.InferenceSession(
        ModelPaths.MULTILABEL_CLASSIFIER,
        providers=ONNXConfig.PROVIDERS
    )
    print("✓ Multilabel classifier loaded")
    
    # 4. Detection model
    from src.models.detection import FastONNXDetector
    detection_model = FastONNXDetector(
        ModelPaths.DETECTION,
        img_size=ImageConfig.DETECTION_SIZE
    )
    print("✓ Detection model loaded")
    
    print("✓ All models loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()
    print("✓ API Server ready!")

# =============================================================================
# API Key Authentication
# =============================================================================
def verify_api_key(x_api_key: str = Header(...)) -> Dict[str, Any]:
    """
    Verify API key from header
    Supports both:
    1. New system: Auto-generated keys with 90-day expiration
    2. Legacy system: Hard-coded keys (backward compatibility)
    """
    # Try new system first (with expiration)
    key_info = verify_key_expiration(x_api_key)
    if key_info:
        return key_info
    
    # Fallback to legacy keys
    if x_api_key in LEGACY_API_KEYS:
        key_info = LEGACY_API_KEYS[x_api_key]
        if key_info.get("active", False):
            return key_info
        else:
            raise HTTPException(
                status_code=403,
                detail="API key is inactive"
            )
    
    # Key not found in either system
    raise HTTPException(
        status_code=401,
        detail="Invalid or expired API key"
    )

# =============================================================================
# Helper Functions
# =============================================================================

def _looks_like_dicom(file_bytes: bytes) -> bool:
    """Quick heuristic to detect DICOM content from bytes."""
    return len(file_bytes) > 132 and file_bytes[128:132] == b'DICM'


def decode_image_from_bytes(file_bytes: bytes, filename: Optional[str]) -> np.ndarray:
    """
    Decode file bytes (image or DICOM) into BGR numpy array without saving to disk.
    """
    extension = Path(filename).suffix.lower() if filename else ''
    is_dicom = extension in ImageConfig.SUPPORTED_DICOM_FORMATS or extension in ['.dcm', '.dicom', '']

    if is_dicom or _looks_like_dicom(file_bytes):
        image = dicom_bytes_to_array(file_bytes)
    else:
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    return image


def classify_bodypart(image_np: np.ndarray) -> Dict[str, Any]:
    """Classify body part of the image"""
    try:
        # Preprocess image
        image_np_processed = preprocess_image(image_np, ImageConfig.BODYPART_SIZE)
        
        # Run inference
        pred_idx, confidence = predict_onnx(bodypart_model, image_np_processed)
        
        # Get class information
        class_name = BodyPartClasses.get_class_name(pred_idx)
        backend = BodyPartClasses.get_backend(pred_idx)
        
        return {
            'class_index': int(pred_idx),
            'class_name': class_name,
            'backend': backend,
            'confidence': float(confidence),
            'is_chestxray': backend == 'chestxray'
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Body part classification failed: {str(e)}"
        )

def convert_to_standard_format(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pipeline results to standard API format
    
    Format:
    {
        "status": "success" | "error",
        "message": "",
        "result": {
            "global": {
                "Abnormal": float,  # always present (probability of abnormal)
                "Tuberculosis": float  # present if Abnormal (TB probability)
            },
            "local": [
                {
                    "name": [str],  # disease names
                    "prob": [float],  # probabilities
                    "start": {"x": int, "y": int},
                    "end": {"x": int, "y": int}
                }
            ]
        },
        "model_version": "0.0.1"
    }
    
    Examples:
        Case 1 - Normal (No Finding):
        {"global": {"Abnormal": 0.0461}, "local": []}
        
        Case 2 - Abnormal (low TB score):
        {"global": {"Abnormal": 0.8977, "Tuberculosis": 0.1234}, "local": [...]}
        
        Case 3 - Abnormal with Tuberculosis (high TB score):
        {"global": {"Abnormal": 0.9997, "Tuberculosis": 0.8523}, "local": [...]}
    """
    try:
        # Check if pipeline failed
        if results['final_result'] is None:
            return {
                "status": "error",
                "message": "Pipeline failed to complete",
                "result": None,
                "model_version": "0.0.1"
            }
        
        final_result = results['final_result']
        status = final_result['status']
        
        # Extract binary classification (global)
        binary_probs = final_result['binary_classification']['probabilities']
        abnormal_prob = binary_probs.get('Abnormal', 0.0)
        
        # Build global result - always include Abnormal score
        global_result = {
            "Abnormal": abnormal_prob
        }
        
        # If Abnormal, always add Tuberculosis score
        if status == 'abnormal' and 'diseases' in final_result:
            diseases = final_result['diseases']
            # Find Tuberculosis score from all_probabilities
            tb_score = 0.0
            if 'all_probabilities' in diseases:
                tb_score = diseases['all_probabilities'].get('Tuberculosis', 0.0)
            else:
                # Fallback: search in positive_classes
                for disease in diseases.get('positive_classes', []):
                    if disease.get('class') == 'Tuberculosis':
                        tb_score = disease.get('probability', 0.0)
                        break
            global_result["Tuberculosis"] = tb_score
        
        # Extract detections (local)
        detections = final_result['lesion_localization']['detections']
        local_results = []
        
        for det in detections:
            bbox = det['bbox']
            local_results.append({
                "name": [det.get('class_name', 'Unknown')],
                "prob": [det['confidence']],
                "start": {
                    "x": int(bbox[0]),
                    "y": int(bbox[1])
                },
                "end": {
                    "x": int(bbox[2]),
                    "y": int(bbox[3])
                }
            })
        
        return {
            "status": "success",
            "message": "",
            "result": {
                "global": global_result,
                "local": local_results
            },
            "model_version": "0.0.1"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to convert results: {str(e)}",
            "result": None,
            "model_version": "0.0.1"
        }


def process_chestxray_pipeline(image: np.ndarray) -> Dict[str, Any]:
    """
    Process chest X-ray through full pipeline
    Args:
        image: numpy array (BGR format from cv2.imread)
    Returns: Results in standard API format
    Optimized: All internal processing uses numpy arrays (no temp files), no visualization in API mode
    Uses pre-loaded global models for faster inference
    """
    try:
        # Run full pipeline with pre-loaded global models (much faster!)
        results = run_full_pipeline(
            image=image,
            binary_session=binary_model_session,
            binary_input_name=binary_model_input_name,
            binary_output_name=binary_model_output_name,
            binary_transform=binary_model_transform,
            multilabel_session=multilabel_model_session,
            detection_model=detection_model
        )
        
        # Check for validation errors
        if 'stage1_validation' in results['pipeline_stages']:
            stage1 = results['pipeline_stages']['stage1_validation']
            if stage1['status'] == 'failed':
                return {
                    "status": "error",
                    "message": stage1.get('error', 'Failed to validate image'),
                    "result": None,
                    "model_version": "0.0.1"
                }
        
        # Convert to standard format
        return convert_to_standard_format(results)
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Pipeline execution failed: {str(e)}",
            "result": None,
            "model_version": "0.0.1"
        }

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Medical Image Analysis API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "bodypart": bodypart_model is not None,
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
    bypass_bodypart: bool = Form(False),
    api_key_info: Dict = Depends(verify_api_key)
):
    """
    Main endpoint: Analyze medical image
    
    Args:
        file: Image or DICOM file
        bypass_bodypart: If True, skip body part classification and force chest X-ray analysis
        api_key_info: API key validation
    
    Returns:
        Standard format:
        {
            "status": "success" | "error",
            "message": "",
            "result": {
                "global": {
                    "Abnormal": float,  # always present
                    "Tuberculosis": float  # present if Abnormal
                },
                "local": [
                    {
                        "name": [str],
                        "prob": [float],
                        "start": {"x": int, "y": int},
                        "end": {"x": int, "y": int}
                    }
                ]
            },
            "model_version": "0.0.1"
        }
        
    Examples:
        - Normal: {"global": {"Abnormal": 0.0461}, "local": []}
        - Abnormal (low TB): {"global": {"Abnormal": 0.8977, "Tuberculosis": 0.1234}, "local": [...]}
        - Abnormal (high TB): {"global": {"Abnormal": 0.9997, "Tuberculosis": 0.8523}, "local": [...]}
    """
    try:
        # Load image either from uploaded file or local path
        if file is not None:
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            filename = file.filename or "uploaded_image"
        elif file_path:
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise HTTPException(status_code=404, detail=f"Input file not found: {file_path}")
            file_bytes = path_obj.read_bytes()
            filename = path_obj.name
        else:
            raise HTTPException(status_code=400, detail="Either file or file_path must be provided")

        image = decode_image_from_bytes(file_bytes, filename)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Failed to load/decode image",
                    "result": None,
                    "model_version": "0.0.1"
                }
            )
        
        # Convert BGR to RGB for bodypart classifier (PIL expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if bypass mode
        if bypass_bodypart:
            # BYPASS MODE: Skip body part classification, force chest X-ray analysis
            result = process_chestxray_pipeline(image)
            
        else:
            # NORMAL MODE: Body part classification first
            bodypart_result = classify_bodypart(image_rgb)
            
            # Check if chest X-ray
            if not bodypart_result['is_chestxray']:
                # Non-chestxray → return error
                result = {
                    "status": "error",
                    "message": f"Image classified as '{bodypart_result['class_name']}' (not chest X-ray)",
                    "result": None,
                    "model_version": "0.0.1"
                }
            else:
                # Chestxray → run full pipeline (use BGR image for pipeline)
                result = process_chestxray_pipeline(image)
        
        return JSONResponse(content=result)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Analysis failed: {str(e)}",
                "result": None,
                "model_version": "0.0.1"
            }
        )

@app.post("/api/v1/bodypart")
async def classify_bodypart_only(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
    api_key_info: Dict = Depends(verify_api_key)
):
    """
    Endpoint: Body part classification only
    """
    try:
        # Load image either from uploaded file or local path
        if file is not None:
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            filename = file.filename or "uploaded_image"
        elif file_path:
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise HTTPException(status_code=404, detail=f"Input file not found: {file_path}")
            file_bytes = path_obj.read_bytes()
            filename = path_obj.name
        else:
            raise HTTPException(status_code=400, detail="Either file or file_path must be provided")

        image = decode_image_from_bytes(file_bytes, filename)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to load/decode image"
            )
        
        # Convert BGR to RGB for bodypart classifier
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Classify body part
        result = classify_bodypart(image_rgb)
        
        return JSONResponse(content={
            'request_id': str(uuid.uuid4()),
            'user': api_key_info['name'],
            'timestamp': datetime.now().isoformat(),
            'input_file': filename,
            'result': result
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

@app.get("/api/v1/results/{request_id}")
async def get_results(
    request_id: str,
    api_key_info: Dict = Depends(verify_api_key)
):
    """
    Get results by request ID (if stored)
    """
    # TODO: Implement result storage and retrieval
    raise HTTPException(
        status_code=501,
        detail="Result retrieval not implemented yet"
    )

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Image Analysis API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

