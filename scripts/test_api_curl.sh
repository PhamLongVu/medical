#!/bin/bash

# Test script for Medical Image Analysis API using curl
# Usage: ./test_api_curl.sh [image_path]

API_URL="http://localhost:8000"
API_KEY="test_key_123"

echo "========================================"
echo "Medical Image Analysis API - Test"
echo "========================================"
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint..."
curl -s "$API_URL/" | jq '.'
echo ""

# Test 2: Health check
echo "2. Testing health check..."
curl -s "$API_URL/health" | jq '.'
echo ""

# Test 3: Analyze image (if provided)
if [ -n "$1" ]; then
    IMAGE_PATH="$1"
    
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Error: Image file not found: $IMAGE_PATH"
        exit 1
    fi
    
    echo "3. Testing image analysis: $IMAGE_PATH"
    echo "   (This may take a while...)"
    
    # Full analysis
    curl -s -X POST \
        -H "X-API-Key: $API_KEY" \
        -F "file=@$IMAGE_PATH" \
        "$API_URL/api/v1/analyze" | jq '.'
    echo ""
    
    # Body part only
    echo "4. Testing body part classification only: $IMAGE_PATH"
    curl -s -X POST \
        -H "X-API-Key: $API_KEY" \
        -F "file=@$IMAGE_PATH" \
        "$API_URL/api/v1/bodypart" | jq '.'
    echo ""
    
else
    echo "3. Skipping image analysis (no image provided)"
    echo ""
    echo "Usage: $0 <image_path>"
    echo "Example: $0 dicom_test/sample.dcm"
fi

echo "========================================"
echo "Tests completed!"
echo "========================================"

