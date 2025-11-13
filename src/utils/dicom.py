import os 
import argparse
import numpy as np 
import pydicom
import cv2
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom_bytes_to_array(dicom_bytes):
    """
    Convert DICOM bytes to numpy array (in memory, no file saving)
    
    Args:
        dicom_bytes: DICOM file content as bytes
    
    Returns:
        numpy array (BGR format) or None if failed
    """
    try:
        # Read DICOM from bytes
        dcm_file = pydicom.dcmread(BytesIO(dicom_bytes))
        data = apply_voi_lut(dcm_file.pixel_array, dcm_file)

        # Handle MONOCHROME1
        if dcm_file.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data - np.min(data)
        denom = np.max(data)
        if denom == 0:
            data = np.zeros_like(data, dtype=np.uint8)
        else:
            data = (data / denom * 255).astype(np.uint8)
        
        # Convert grayscale to BGR
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        
        return data
    except Exception as e:
        print(f"Error converting DICOM bytes: {e}")
        return None

def dicom2img(dcm_path, out_dir):
    try:
        dcm_file = pydicom.read_file(dcm_path)
        data = apply_voi_lut(dcm_file.pixel_array, dcm_file)

        if dcm_file.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        denom = np.max(data)
        if denom == 0:
            data = np.zeros_like(data, dtype=np.uint8)
        else:
            data = (data / denom * 255).astype(np.uint8)

        sop_instance_uid = dcm_file.get('SOPInstanceUID', None)
        if sop_instance_uid is None:
            sop_instance_uid = os.path.splitext(os.path.basename(str(dcm_path)))[0]

        os.makedirs(out_dir, exist_ok=True)
        image_path = os.path.join(out_dir, f"{sop_instance_uid}.png")
        cv2.imwrite(image_path, data)
        return image_path
    except Exception:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to a single DICOM file to convert')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output PNG')
    args = parser.parse_args()

    output_path = dicom2img(args.input_file, args.output_dir)
    if output_path is None:
        raise SystemExit(1)
    print(output_path)
    raise SystemExit(0)