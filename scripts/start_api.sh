#!/bin/bash

# Start Medical Image Analysis API Server
# Usage: ./start_api.sh [--dev|--prod] [--port PORT]

set -e

# Default values
MODE="dev"
PORT=8000
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dev|--prod] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Medical Image Analysis API Server"
echo "========================================"
echo "Mode: $MODE"
echo "Host: $HOST"
echo "Port: $PORT"
echo "========================================"
echo ""

# Go to project root (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1

# Check if required files exist
if [ ! -f "src/api/server.py" ]; then
    echo "Error: src/api/server.py not found"
    echo "Make sure you're running from the project root or restructure completed successfully"
    exit 1
fi

if [ ! -f "src/config.py" ]; then
    echo "Error: src/config.py not found"
    exit 1
fi

# Check if models exist
echo "Checking models..."
if [ ! -f "weights/bodypart/bodypart_v2.onnx" ]; then
    echo "Warning: Body part model not found at weights/bodypart/bodypart_v2.onnx"
fi
if [ ! -f "weights/binary/model_2class_simplified.onnx" ]; then
    echo "Warning: Binary classifier model not found at weights/binary/"
fi
if [ ! -f "weights/multilabel/convnext_base_384_dynamic_simplified.onnx" ]; then
    echo "Warning: Multi-label classifier model not found at weights/multilabel/"
fi
if [ ! -f "weights/detection/detection.onnx" ]; then
    echo "Warning: Detection model not found at weights/detection/"
fi
echo ""

# Start server based on mode
if [ "$MODE" = "dev" ]; then
    echo "Starting development server with auto-reload..."
    echo "API Documentation: http://$HOST:$PORT/docs"
    echo "Press Ctrl+C to stop"
    echo ""
    python run.py --host "$HOST" --port "$PORT" --reload
    
elif [ "$MODE" = "prod" ]; then
    echo "Starting production server..."
    
    # Check if gunicorn is installed
    if ! command -v gunicorn &> /dev/null; then
        echo "Warning: gunicorn not found, using uvicorn instead"
        echo "For production, install gunicorn: pip install gunicorn"
        echo ""
        python run.py --host "$HOST" --port "$PORT"
    else
        echo "Using gunicorn with 4 workers"
        echo "API Documentation: http://$HOST:$PORT/docs"
        echo "Press Ctrl+C to stop"
        echo ""
        gunicorn src.api.server:app \
            --workers 4 \
            --worker-class uvicorn.workers.UvicornWorker \
            --bind "$HOST:$PORT" \
            --timeout 120 \
            --access-logfile - \
            --error-logfile -
    fi
fi

