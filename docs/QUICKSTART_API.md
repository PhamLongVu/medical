# 🚀 Quick Start Guide - Medical Image Analysis API
## 🏗️ Architecture Flow

```
┌─────────────┐
│ Dicom/Image │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   API KEY   │ (Authentication - 90 day expiration)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Bodypart            │
│ Classification      │
│ - abdominal         │
│ - adult (chestxray) │
│ - others            │
│ - pediatric         │
│ - spine             │
└──────┬──────────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ Non chestxray│   │ Chestxray    │
│              │   │ Analysis     │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
  ┌────────┐      ┌──────────────┐
  │  None  │      │  4 Statuses: │
  └────────┘      │  - None      │
                  │  - Invalid   │
                  │  - Normal    │
                  │  - Abnormal  │
                  └──────┬───────┘
                         │
                         ▼
                    ┌────────┐
                    │ Result │
                    └────────┘
```

---

## 📋 Features

### ✅ Core Features
- **API Key Authentication** with 90-day auto-expiration
- **Body Part Classification** (5 classes: abdominal, chest, pediatric, spine, others)
- **Automatic Chest X-ray Detection**
- **Bypass Mode** for forced chest X-ray analysis
- **DICOM & Image Format Support** (PNG, JPG, JPEG, DICOM)

### ✅ Chest X-ray Analysis Pipeline
1. **Binary Classification**: No Finding / Abnormal
2. **Multi-label Disease Classification**: 28 diseases
3. **Lesion Detection & Localization**: 18 lesion types
4. **Parallel Execution**: Optimized for speed

### ✅ Result Statuses
| Status | Meaning | Condition |
|--------|---------|-----------|
| **None** | Not chest X-ray | Body part ≠ chest |
| **Invalid** | Processing error | Failed to process |
| **Normal** | No abnormalities | No Finding detected |
| **Abnormal** | Has abnormalities | Diseases/lesions found |

## 🔑 API Key Management (NEW!)

### Generate Your First API Key

```bash
# Create a new API key (expires in 90 days)
./scripts/manage_apikeys.sh create "My Client"

# Output:
# ✓ API key created for 'My Client'
#   Expires: 2026-02-09
#   Key: d9a46d094b696aa538b37e4b6891a6b7759cbd570c5b89c85d1318df7fe2c9ff
#   ⚠️  Save this key! It won't be shown again.
```

**⚠️ Important:** Save the key immediately! It won't be shown again.

📖 **Full Documentation:** [API_KEY_MANAGEMENT.md](API_KEY_MANAGEMENT.md)

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Install FastAPI and related packages
pip install fastapi uvicorn python-multipart

# If you haven't installed the main pipeline dependencies:
pip install numpy opencv-python pillow onnxruntime-gpu torchvision pydicom
```

### 2. Verify Models

Make sure you have all model files in the `weight/` directory:

```
weight/
├── cls_bodypart/
│   └── bodypart_v2.onnx
├── cls_2class/
│   └── model_2class_simplified.onnx
├── cls_multilabel/
│   └── convnext_base_384_dynamic_simplified.onnx
└── detection/
    └── detection.onnx
```

## 🎯 Start the Server

### Option 1: Using the start script (Recommended)

```bash
# Development mode (with auto-reload)
./start_api.sh --dev

# Production mode
./start_api.sh --prod

# Custom port
./start_api.sh --dev --port 8080
```

### Option 2: Direct Python command

```bash
# Development
python api_server.py --reload

# Production
python api_server.py --host 0.0.0.0 --port 8000
```

Server will be available at: **http://localhost:8000**

## 🧪 Test the API

### Option 1: Using Python test client

```bash
python test_api_client.py path/to/your/image.png
```

### Option 2: Using curl script

```bash
./test_api_curl.sh path/to/your/image.png
```

### Option 3: Using curl directly

```bash
# Health check
curl http://localhost:8000/health

# Analyze image
curl -X POST \
  -H "X-API-Key: test_key_123" \
  -F "file=@path/to/image.png" \
  http://localhost:8000/api/v1/analyze
```

### Option 4: Using browser

Open **http://localhost:8000/docs** for interactive API documentation (Swagger UI)

## 📊 Understanding the Response

### Example 1: Chest X-ray with Abnormalities

```json
{
  "request_id": "abc-123",
  "bodypart_classification": {
    "class_name": "adult",
    "backend": "chestxray",
    "is_chestxray": true,
    "confidence": 0.98
  },
  "final_status": "Fail",
  "message": "Abnormalities detected: 3 diseases, 5 lesions",
  "chestxray_analysis": {
    "stage": "Fail",
    "results": { ... }
  }
}
```

### Example 2: Normal Chest X-ray

```json
{
  "request_id": "def-456",
  "bodypart_classification": {
    "class_name": "adult",
    "backend": "chestxray",
    "is_chestxray": true,
    "confidence": 0.99
  },
  "final_status": "Pass",
  "message": "No abnormalities detected",
  "chestxray_analysis": {
    "stage": "Pass",
    "results": { ... }
  }
}
```

### Example 3: Non-Chest X-ray

```json
{
  "request_id": "ghi-789",
  "bodypart_classification": {
    "class_name": "abdominal",
    "backend": "None",
    "is_chestxray": false,
    "confidence": 0.95
  },
  "final_status": "None",
  "message": "Image classified as 'abdominal' (not chest X-ray)",
  "chestxray_analysis": null
}
```

## 🔑 API Keys

Default test keys:
- `test_key_123` - Test User
- `demo_key_456` - Demo User

To add your own keys, edit `api_server.py`:

```python
API_KEYS = {
    "your_custom_key": {"name": "Your Name", "active": True},
}
```

## 📝 Common Use Cases

### 1. Quick Health Check

```bash
curl http://localhost:8000/health
```

### 2. Classify Body Part Only

```bash
curl -X POST \
  -H "X-API-Key: test_key_123" \
  -F "file=@image.png" \
  http://localhost:8000/api/v1/bodypart
```

### 3. Full Chest X-ray Analysis

```bash
curl -X POST \
  -H "X-API-Key: test_key_123" \
  -F "file=@chestxray.png" \
  http://localhost:8000/api/v1/analyze
```

### 4. Batch Processing (Python)

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "test_key_123"

images = ["image1.png", "image2.png", "image3.png"]

for image_path in images:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": f}
        )
        result = response.json()
        print(f"{image_path}: {result['final_status']}")
```

## 🐛 Troubleshooting

### Server won't start

```bash
# Check if port is already in use
lsof -i :8000

# Try a different port
./start_api.sh --dev --port 8080
```

### Import errors

```bash
# Install missing dependencies
pip install fastapi uvicorn python-multipart
```

### CUDA errors

```bash
# Use CPU instead
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

Or edit `config.py`:
```python
ONNXConfig.PROVIDERS = ['CPUExecutionProvider']
```

### Model not found

Check that all models exist:
```bash
ls -lh weight/cls_bodypart/bodypart_v2.onnx
ls -lh weight/cls_2class/model_2class_simplified.onnx
ls -lh weight/cls_multilabel/convnext_base_384_dynamic_simplified.onnx
ls -lh weight/detection/detection.onnx
```

## 📚 More Information

- **Full API Documentation**: See [API_README.md](API_README.md)
- **Interactive Docs**: http://localhost:8000/docs (when server is running)
- **Pipeline Documentation**: See [README.md](README.md)

## 🎉 You're Ready!

Your API is now running and ready to analyze medical images!

Try it out:
```bash
python test_api_client.py dicom_test/DICOM_HOANG\ VAN\ MINH_1760674525219.dcm
```

