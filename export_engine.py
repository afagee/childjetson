"""
Export model .pt sang TensorRT .engine (chạy trên Jetson, một lần).
Giúp Nano chạy mượt: Python + web + inference.
"""
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import config
from ultralytics import YOLO


def main():
    pt_path = os.path.abspath(config.MODEL_PATH)
    if not os.path.isfile(pt_path):
        print(f"[ERROR] Not found: {pt_path}")
        sys.exit(1)
    imgsz = getattr(config, "YOLO_INPUT_SIZE", 416)
    print(f"[INFO] Exporting {pt_path} to TensorRT (imgsz={imgsz})...")
    model = YOLO(pt_path)
    out = model.export(format="engine", device=0, half=True, imgsz=imgsz)
    print(f"[INFO] Done: {out}")
    base, _ = os.path.splitext(pt_path)
    engine_path = base + ".engine"
    print(f"[INFO] Set in config: MODEL_ENGINE_PATH = '{engine_path}' (or path relative to project)")


if __name__ == "__main__":
    main()
