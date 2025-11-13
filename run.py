#!/usr/bin/env python
"""
Simple runner script for the API server
Usage: python run.py [--host HOST] [--port PORT] [--reload]
"""

import sys
import argparse

# Make sure src is in the path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Medical Image Analysis API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Medical Image Analysis API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()

