"""
Test client for Medical Image Analysis API
"""

import requests
import json
import sys
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "test_key_123"  # Change this to your API key

def test_health_check():
    """Test API health check"""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        print("✓ Health check passed")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"✗ Health check failed: {response.status_code}")
        print(response.text)
    print()

def test_analyze_image(image_path: str):
    """Test full image analysis"""
    print(f"Testing image analysis: {image_path}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"✗ File not found: {image_path}")
        return
    
    # Prepare request
    headers = {
        "X-API-Key": API_KEY
    }
    
    with open(image_path, "rb") as f:
        files = {
            "file": (Path(image_path).name, f, "image/png")
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze",
            headers=headers,
            files=files
        )
    
    # Print results
    if response.status_code == 200:
        result = response.json()
        print("✓ Analysis completed")
        print(f"\nRequest ID: {result['request_id']}")
        print(f"User: {result['user']}")
        print(f"Processing time: {result['processing_time_ms']:.1f}ms")
        
        print("\n--- Body Part Classification ---")
        bp = result['bodypart_classification']
        print(f"Class: {bp['class_name']}")
        print(f"Backend: {bp['backend']}")
        print(f"Confidence: {bp['confidence']:.4f}")
        print(f"Is Chest X-ray: {bp['is_chestxray']}")
        
        print(f"\n--- Final Status: {result['final_status']} ---")
        print(f"Message: {result['message']}")
        
        if result['chestxray_analysis']:
            print("\n--- Chest X-ray Analysis ---")
            cxr = result['chestxray_analysis']
            print(f"Stage: {cxr['stage']}")
            print(f"Message: {cxr['message']}")
            
            if cxr['results']:
                results = cxr['results']
                final = results['final_result']
                
                # Binary classification
                if 'binary_classification' in final:
                    bc = final['binary_classification']
                    print(f"\nBinary Classification: {bc['class_name']} ({bc['confidence']:.4f})")
                
                # Lesions
                lesions = final['lesion_localization']
                print(f"\nLesions detected: {lesions['num_lesions']}")
                if lesions['num_lesions'] > 0:
                    for i, det in enumerate(lesions['detections'][:5], 1):
                        print(f"  {i}. {det['class_name']}: {det['confidence']:.3f}")
                
                # Diseases (if abnormal)
                if final['status'] == 'abnormal' and 'diseases' in final:
                    diseases = final['diseases']
                    print(f"\nDiseases detected: {len(diseases['positive_classes'])}")
                    if diseases['tuberculosis_detected']:
                        print("  ⚠️  TUBERCULOSIS DETECTED")
                    for pred in diseases['positive_classes'][:5]:
                        print(f"  - {pred['class']}: {pred['probability']:.4f}")
        
        print()
        
    else:
        print(f"✗ Analysis failed: {response.status_code}")
        print(response.text)
    print()

def test_bodypart_only(image_path: str):
    """Test body part classification only"""
    print(f"Testing body part classification: {image_path}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"✗ File not found: {image_path}")
        return
    
    # Prepare request
    headers = {
        "X-API-Key": API_KEY
    }
    
    with open(image_path, "rb") as f:
        files = {
            "file": (Path(image_path).name, f, "image/png")
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/bodypart",
            headers=headers,
            files=files
        )
    
    # Print results
    if response.status_code == 200:
        result = response.json()
        print("✓ Classification completed")
        print(f"\nRequest ID: {result['request_id']}")
        print(f"User: {result['user']}")
        
        print("\n--- Result ---")
        res = result['result']
        print(f"Class: {res['class_name']}")
        print(f"Backend: {res['backend']}")
        print(f"Confidence: {res['confidence']:.4f}")
        print(f"Is Chest X-ray: {res['is_chestxray']}")
        print()
        
    else:
        print(f"✗ Classification failed: {response.status_code}")
        print(response.text)
    print()

def main():
    """Main test function"""
    print("="*80)
    print("Medical Image Analysis API - Test Client")
    print("="*80)
    print()
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Analyze image (if provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Full analysis
        test_analyze_image(image_path)
        
        # Body part only
        # test_bodypart_only(image_path)
    else:
        print("Usage: python test_api_client.py <image_path>")
        print("Example: python test_api_client.py dicom_test/sample.dcm")
        print()

if __name__ == "__main__":
    main()

