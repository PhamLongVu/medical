import argparse
import json
from pathlib import Path

import requests


def call_api_upload(url, api_key, file_path, bypass_bodypart=False):
    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f, "application/octet-stream")}
        data = {"bypass_bodypart": str(bypass_bodypart).lower()}
        headers = {"X-API-Key": api_key}
        response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()


def call_api_bytes(url, api_key, file_path, bypass_bodypart=False):
    file_bytes = Path(file_path).read_bytes()
    files = {"file": (Path(file_path).name, file_bytes, "application/octet-stream")}
    data = {"bypass_bodypart": str(bypass_bodypart).lower()}
    headers = {"X-API-Key": api_key}
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()


def call_api_path(url, api_key, file_path, bypass_bodypart=False):
    data = {
        "bypass_bodypart": str(bypass_bodypart).lower(),
        "file_path": file_path,
    }
    headers = {"X-API-Key": api_key}
    response = requests.post(url, headers=headers, data=data)
    return response.json()


if __name__ == "__main__":
    # Hardcoded parameters (bỏ phần argument đi)
    url = "http://10.149.3.192:8000/api/v1/analyze"
    api_key = "test_key_123"
    file_path = "/media/vbdi/ssd2t/Medical/dung/projects/vindr-xray/dataset/images/vin_png/1.2.840.113564.54.192.168.2.112.6072.2019032422470114.png"  # <-- sửa path này cho đúng file bạn muốn test
    mode = "bytssssses"  # Chọn 1 trong ["upload", "bytes", "path"]
    bypass = False   # True nếu muốn bypass bodypart

    if mode == "upload":
        result = call_api_upload(url, api_key, file_path, bypass_bodypart=bypass)
    elif mode == "bytes":
        result = call_api_bytes(url, api_key, file_path, bypass_bodypart=bypass)
    else:  # path mode
        result = call_api_path(url, api_key, file_path, bypass_bodypart=bypass)

    print(json.dumps(result, indent=2, ensure_ascii=False))
