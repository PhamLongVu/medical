"""
Full Stream Pipeline Configuration
Centralized configuration for all models, paths, and parameters
"""

import os
from pathlib import Path

# =============================================================================
# BASE DIRECTORIES
# =============================================================================
# BASE_DIR now points to project root (parent of src/)
BASE_DIR = Path(__file__).parent.parent.absolute()
PROJECT_ROOT = BASE_DIR

# =============================================================================
# MODEL PATHS
# =============================================================================
class ModelPaths:
    """Paths to ONNX model files"""
    # Body part classifier
    BODYPART_CLASSIFIER = str(BASE_DIR / "weights/bodypart/bodypart_v2.onnx")
    
    # Binary classifier (No Finding / Abnormal)
    BINARY_CLASSIFIER = str(BASE_DIR / "weights/binary/model_2class_simplified.onnx")
    
    # Multi-label classifier (28 diseases)
    MULTILABEL_CLASSIFIER = str(BASE_DIR / "weights/multilabel/convnext_base_384_dynamic_simplified.onnx")
    
    # Object detector (18 lesion types)
    DETECTION = str(BASE_DIR / "weights/detection/detection.onnx")


# =============================================================================
# DIRECTORY PATHS
# =============================================================================
class DirectoryPaths:
    """Working directories for pipeline"""
    # Input/Output directories
    # IMG_CONVERT = str(BASE_DIR / "data/temp/img_convert")  # Converted PNG images
    OUTPUT = str(BASE_DIR / "data/outputs")                # Results and visualizations
    DICOM_TEST = str(BASE_DIR / "data/test/dicom")         # Test DICOM files
    
    # Default input file (can be DICOM or image)
    DEFAULT_INPUT = str(BASE_DIR / "data/test/dicom/DICOM_HOANG VAN MINH_1760674525219.dcm")
    
    # Deprecated (for backward compatibility)
    DEFAULT_DICOM = DEFAULT_INPUT


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
class ImageConfig:
    """Image preprocessing parameters"""
    # Image sizes for different models
    BODYPART_SIZE = 256
    BINARY_CLASSIFIER_SIZE = 384
    MULTILABEL_SIZE = 384
    DETECTION_SIZE = 640
    
    # Normalization parameters (computed from training data)
    MEAN = [0.5029414296150208] * 3  # RGB channels (same value for grayscale)
    STD = [0.2892409563064575] * 3
    
    # Image preprocessing
    GRAYSCALE_TO_RGB = True  # Convert grayscale to 3-channel RGB
    NUM_OUTPUT_CHANNELS = 3
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    SUPPORTED_DICOM_FORMATS = ['.dcm', '.dicom']
    
    @classmethod
    def is_image_file(cls, filepath):
        """Check if file is a supported image format"""
        ext = Path(filepath).suffix.lower()
        return ext in cls.SUPPORTED_IMAGE_FORMATS
    
    @classmethod
    def is_dicom_file(cls, filepath):
        """Check if file is a DICOM format"""
        ext = Path(filepath).suffix.lower()
        return ext in cls.SUPPORTED_DICOM_FORMATS or ext == ''


# =============================================================================
# DISEASE CLASSES (Multi-label Classification)
# =============================================================================
class DiseaseClasses:
    """28 disease classes for multi-label classification"""
    CLASSES = [
        'Aortic enlargement', 
        'Atelectasis', 
        'Calcification', 
        'Cardiomegaly',
        'Clavicle fracture', 
        'Consolidation', 
        'Edema', 
        'Emphysema',
        'Enlarged PA', 
        'Infiltration', 
        'Interstitial lung disease - ILD',
        'Lung cavity', 
        'Lung cyst', 
        'Mediastinal shift', 
        'Nodule/Mass',
        'Opacity', 
        'Other lesion', 
        'Pleural effusion', 
        'Pleural thickening',
        'Pneumothorax', 
        'Pulmonary fibrosis', 
        'Rib fracture', 
        'COPD',
        'Cardiovascular disease', 
        'Lung tumor', 
        'Other', 
        'Pneumonia',
        'Tuberculosis'
    ]
    
    # Optimized F2-score thresholds per class
    THRESHOLDS = {
        'Aortic enlargement': 0.45,
        'Atelectasis': 0.45,
        'Calcification': 0.45,
        'Cardiomegaly': 0.45,
        'Clavicle fracture': 0.60,
        'Consolidation': 0.45,
        'Edema': 0.30,
        'Emphysema': 0.45,
        'Enlarged PA': 0.35,
        'Infiltration': 0.40,
        'Interstitial lung disease - ILD': 0.50,
        'Lung cavity': 0.35,
        'Lung cyst': 0.30,
        'Mediastinal shift': 0.45,
        'Nodule/Mass': 0.45,
        'Opacity': 0.45,
        'Other lesion': 0.45,
        'Pleural effusion': 0.50,
        'Pleural thickening': 0.50,
        'Pneumothorax': 0.40,
        'Pulmonary fibrosis': 0.50,
        'Rib fracture': 0.40,
        'COPD': 0.40,
        'Cardiovascular disease': 0.50,
        'Lung tumor': 0.50,
        'Other': 0.45,
        'Pneumonia': 0.50,
        'Tuberculosis': 0.45
    }
    
    # Priority diseases (highlighted in output)
    PRIORITY_DISEASES = ['Tuberculosis']


# =============================================================================
# DETECTION CLASSES (Object Detection)
# =============================================================================
class DetectionClasses:
    """18 lesion classes for object detection"""
    CLASSES = [
        'Aortic_enlargement',
        'Atelectasis',
        'Calcification',
        'Cardiomegaly',
        'Clavicle_fracture',
        'Consolidation',
        'Emphysema',
        'Enlarged_PA',
        'Infiltration',
        'Interstitial_lung_disease',
        'Nodule_Mass',
        'Opacity',
        'Other_lesion',
        'Pleural_effusion',
        'Pleural_thickening',
        'Pneumothorax',
        'Pulmonary_fibrosis',
        'Rib_fracture'
    ]
    
    # Display names (tên hiển thị đẹp hơn cho visualization)
    DISPLAY_NAMES = {
        'Aortic_enlargement': 'Aortic Enlargement',
        'Atelectasis': 'Atelectasis',
        'Calcification': 'Calcification',
        'Cardiomegaly': 'Cardiomegaly',
        'Clavicle_fracture': 'Clavicle Fracture',
        'Consolidation': 'Consolidation',
        'Emphysema': 'Emphysema',
        'Enlarged_PA': 'Enlarged PA',
        'Infiltration': 'Infiltration',
        'Interstitial_lung_disease': 'Interstitial Lung Disease',
        'Nodule_Mass': 'Nodule/Mass',
        'Opacity': 'Opacity',
        'Other_lesion': 'Other Lesion',
        'Pleural_effusion': 'Pleural Effusion',
        'Pleural_thickening': 'Pleural Thickening',
        'Pneumothorax': 'Pneumothorax',
        'Pulmonary_fibrosis': 'Pulmonary Fibrosis',
        'Rib_fracture': 'Rib Fracture'
    }
    
    @classmethod
    def get_display_name(cls, class_name):
        """Get display name for a class"""
        return cls.DISPLAY_NAMES.get(class_name, class_name.replace('_', ' ').title())
    
    # Detection inference parameters
    CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence for detections
    IOU_THRESHOLD = 0.1         # IoU threshold for NMS
    
    # Visualization settings
    BOX_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1


# =============================================================================
# ONNX RUNTIME CONFIGURATION
# =============================================================================
class ONNXConfig:
    """ONNX Runtime settings"""
    # Execution providers (in priority order)
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Session options
    GRAPH_OPTIMIZATION_LEVEL = 'ORT_ENABLE_ALL'  # Maximum optimization
    ENABLE_MEM_PATTERN = True
    ENABLE_CPU_MEM_ARENA = True
    
    # Prefer GPU
    USE_GPU = True  # Automatically falls back to CPU if GPU not available


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
class PipelineConfig:
    """Pipeline execution parameters"""
    # Parallel execution settings
    ENABLE_PARALLEL = True      # Run Stage 3 & 4 in parallel when abnormal detected
    MAX_WORKERS = 8            # Number of parallel workers
    
    # Output settings
    SAVE_DETECTION_VIS = False  # Save detection visualization (False for speed optimization)
    SAVE_JSON_RESULTS = True    # Save JSON results
    ORGANIZE_BY_FOLDER = True   # Organize outputs in folders by input filename
    
    # Output structure (when ORGANIZE_BY_FOLDER=True):
    # outputs/
    #   └── {input_filename}/
    #       ├── results.json
    #       └── detection_result.png
    
    # Performance settings
    WARMUP_MODELS = False       # Warmup models with dummy input (slower first run, faster subsequent)
    
    # Logging
    VERBOSE = True              # Print detailed progress
    SHOW_PERFORMANCE_METRICS = True  # Show speedup from parallel execution


# =============================================================================
# BODY PART CLASSIFICATION
# =============================================================================
class BodyPartClasses:
    """Body part classification settings"""
    CLASSES = [
        'abdominal',
        'adult',
        'others',
        'pediatric',
        'spine'
    ]
    
    # Backend mapping for each body part
    BACKENDS = [
        'None',
        'chestxray',
        'None',
        'None',
        'spinexr'
    ]
    
    @classmethod
    def get_backend(cls, class_idx):
        """Get backend name for a class index"""
        if 0 <= class_idx < len(cls.BACKENDS):
            return cls.BACKENDS[class_idx]
        return 'None'
    
    @classmethod
    def get_class_name(cls, class_idx):
        """Get class name for a class index"""
        if 0 <= class_idx < len(cls.CLASSES):
            return cls.CLASSES[class_idx]
        return 'unknown'


# =============================================================================
# BINARY CLASSIFICATION
# =============================================================================
class BinaryClassifierConfig:
    """Binary classifier specific settings"""
    CLASS_NAMES = ['No Finding', 'Abnormal']
    
    # Decision threshold (optional, usually use softmax probability)
    DECISION_THRESHOLD = 0.5  # Not actively used, kept for reference


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_model_paths():
    """Get all model paths as a dictionary"""
    return {
        'bodypart': ModelPaths.BODYPART_CLASSIFIER,
        'binary': ModelPaths.BINARY_CLASSIFIER,
        'multilabel': ModelPaths.MULTILABEL_CLASSIFIER,
        'detection': ModelPaths.DETECTION
    }


def get_directory_paths():
    """Get all directory paths as a dictionary"""
    return {
        # 'img_convert': DirectoryPaths.IMG_CONVERT,
        'output': DirectoryPaths.OUTPUT,
        'dicom_test': DirectoryPaths.DICOM_TEST,
        'default_input': DirectoryPaths.DEFAULT_INPUT,
        'default_dicom': DirectoryPaths.DEFAULT_DICOM  # Backward compatibility
    }


# def validate_paths():
#     """Validate that all required paths exist"""
#     issues = []
    
#     # Check model files
#     models = get_model_paths()
#     for name, path in models.items():
#         if not os.path.exists(path):
#             issues.append(f"Model not found: {name} at {path}")
    
#     # Check directories (create if not exist)
#     dirs = [DirectoryPaths.IMG_CONVERT, DirectoryPaths.OUTPUT]
#     for dir_path in dirs:
#         os.makedirs(dir_path, exist_ok=True)
    
#     return issues


def print_config():
    """Print current configuration"""
    print("="*80)
    print("PIPELINE CONFIGURATION")
    print("="*80)
    
    print("\n[Model Paths]")
    for name, path in get_model_paths().items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name:15s}: {path}")
    
    print("\n[Supported Input Formats]")
    print(f"  Image formats: {', '.join(ImageConfig.SUPPORTED_IMAGE_FORMATS)}")
    print(f"  DICOM formats: {', '.join(ImageConfig.SUPPORTED_DICOM_FORMATS)}")
    print(f"  Default input: {DirectoryPaths.DEFAULT_INPUT}")
    
    print("\n[Image Sizes]")
    print(f"  Body Part:         {ImageConfig.BODYPART_SIZE}x{ImageConfig.BODYPART_SIZE}")
    print(f"  Binary Classifier: {ImageConfig.BINARY_CLASSIFIER_SIZE}x{ImageConfig.BINARY_CLASSIFIER_SIZE}")
    print(f"  Multilabel:        {ImageConfig.MULTILABEL_SIZE}x{ImageConfig.MULTILABEL_SIZE}")
    print(f"  Detection:         {ImageConfig.DETECTION_SIZE}x{ImageConfig.DETECTION_SIZE}")
    
    print("\n[Detection Parameters]")
    print(f"  Confidence threshold: {DetectionClasses.CONFIDENCE_THRESHOLD}")
    print(f"  IoU threshold:        {DetectionClasses.IOU_THRESHOLD}")
    
    print("\n[Pipeline Settings]")
    print(f"  Parallel execution:   {PipelineConfig.ENABLE_PARALLEL}")
    print(f"  Max workers:          {PipelineConfig.MAX_WORKERS}")
    print(f"  Save visualization:   {PipelineConfig.SAVE_DETECTION_VIS}")
    
    print("\n[ONNX Runtime]")
    print(f"  Providers: {', '.join(ONNXConfig.PROVIDERS)}")
    print(f"  Optimization: {ONNXConfig.GRAPH_OPTIMIZATION_LEVEL}")
    
    print("="*80 + "\n")


# =============================================================================
# MAIN (for testing)
# =============================================================================
if __name__ == '__main__':
    print_config()
    
    # Validate paths
    issues = validate_paths()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All paths validated successfully!")

