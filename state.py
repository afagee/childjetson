"""
Global state — car-only.
Nhiều vùng (polygon) theo từng camera; lưu file zones.json.
"""
import json
import threading
from pathlib import Path

import config

# zones_by_camera[camera_key] = [ polygon1, polygon2, ... ], mỗi polygon = [(x,y), ...]
# camera_key = "setup" (ảnh upload) hoặc rtsp_url
zones_by_camera = {}
_zones_lock = threading.Lock()

# Kích thước ảnh setup (để scale zone sang frame camera)
setup_image_width = 0
setup_image_height = 0

# Alert
last_siren_time = 0
current_alert_status = False
alert_timestamp = 0

# Theo dõi di chuyển xe trong zone (để tắt còi khi xe đứng yên hết cooldown)
last_car_centers_in_zone = []  # [(cx, cy), ...] frame trước
last_movement_time = 0.0       # thời điểm cuối cùng thấy xe di chuyển trong zone

# Model & camera
model = None
cap_thread = None
latest_detections = []
frame_counter = 0
last_frame_time = 0

# Frame mới nhất từ detection thread (để /video_feed gửi khi có người xem)
latest_display_frame = None
latest_display_lock = threading.Lock()

# Siren
siren_lock = threading.Lock()
siren_is_on = False
siren_off_at = 0.0  # thời điểm sẽ tắt còi (sau khi kêu thêm SIREN_EXTRA_SEC); 0 = không hẹn

# Path ảnh setup (sau upload)
UPLOAD_DIR_PATH = Path(config.UPLOAD_DIR).resolve()
setup_image_path = None

# Streaming clients: số client /video_feed đang mở
stream_clients = 0
stream_clients_lock = threading.Lock()


def load_zones():
    """Đọc zones.json vào zones_by_camera."""
    global zones_by_camera
    path = getattr(config, "ZONES_PATH", None)
    if not path or not Path(path).is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return
        data = json.loads(raw)
        with _zones_lock:
            zones_by_camera = {k: [[(int(p[0]), int(p[1])) for p in poly] for poly in v] for k, v in data.items()}
    except Exception as e:
        print(f"[STATE] load_zones: {e}")


def save_zones():
    """Ghi zones_by_camera ra zones.json."""
    path = getattr(config, "ZONES_PATH", None)
    if not path:
        return
    try:
        with _zones_lock:
            data = {k: [list(poly) for poly in v] for k, v in zones_by_camera.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[STATE] save_zones: {e}")


# Load zones khi import
load_zones()
