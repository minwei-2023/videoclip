import torch
import cv2
import ultralytics
import streamlit
import sys

print("-" * 30)
print(f"Python: {sys.version.split()[0]}")
try:
    print(f"OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV: Error - {e}")

try:
    print(f"Ultralytics: {ultralytics.__version__}")
except Exception as e:
    print(f"Ultralytics: Error - {e}")

try:
    print(f"Streamlit: {streamlit.__version__}")
except Exception as e:
    print(f"Streamlit: Error - {e}")

try:
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Running on CPU.")
except Exception as e:
    print(f"Torch: Error - {e}")
print("-" * 30)
