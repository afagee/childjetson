"""
Cấu hình Jetson App — car-only (yolo26s, class car).
Tham số chỉnh qua web được load từ settings.json khi khởi động và khi POST /api/settings.
"""
import os

# --- Model: yolo26s có sẵn class car (COCO id 2) ---
# Đặt các file model (.pt, .engine) trong thư mục con /models của jetson_app
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_APP_DIR, "models", "yolo26s.pt")
MODEL_ENGINE_PATH = os.environ.get(
    "JETSON_APP_ENGINE",
    os.path.join(_APP_DIR, "models", "yolo26s.engine"),
)

# --- Đăng nhập admin (phải đăng nhập để vào app; đã đăng nhập thì config camera không cần mk) ---
ADMIN_PASSWORD = "vifa2204"
SECRET_KEY = os.environ.get("JETSON_APP_SECRET_KEY", "jetson-car-alert-secret-change-in-prod")

# --- Camera: tùy chọn (để trống = chỉ dùng upload ảnh vẽ zone); có thể cấu hình qua web ---
RTSP_URL = os.environ.get("JETSON_APP_RTSP_URL", "")
RTSP_CONFIG_PASSWORD = "vifa2204"  # Chỉ dùng khi chưa đăng nhập; đã đăng nhập thì không yêu cầu
CSI_CAMERA_INDEX = 0

# --- Detection: chỉ xe ---
CLASS_ID_CAR = 2   # COCO car (class 2 trong COCO dataset)
CONF_THRESHOLD = 0.6

# --- Cảnh báo xe: cooldown (chỉnh được qua web / settings.json) ---
CAR_ALERT_COOLDOWN_SEC = 10   # Sau 1 lần phát, không phát lại trong N giây nếu xe không di chuyển

# --- Relay GPIO ---
SIREN_GPIO_CHANNEL = 11
RELAY_CH2_GPIO = 13
SIREN_DURATION = 5
# Còi kêu thêm N giây trước khi tắt (sau khi xe thoát zone hoặc đứng yên hết cooldown)
SIREN_EXTRA_SEC = 5

# --- Video stream (khi bật RTSP) ---
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
PORT = 5000
YOLO_INPUT_SIZE = 416
FRAME_SKIP = 2
JPEG_QUALITY = 75
MAX_FPS = 15
# Giới hạn số client web xem /video_feed cùng lúc (0 = không giới hạn)
MAX_STREAM_CLIENTS = int(os.environ.get("JETSON_APP_MAX_STREAM_CLIENTS", "2"))

# --- Setup image (upload để vẽ zone) ---
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
SETUP_IMAGE_FILENAME = "setup.jpg"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Settings file (điều chỉnh local qua web) ---
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

# --- Zones theo từng camera (lưu file) ---
ZONES_PATH = os.path.join(os.path.dirname(__file__), "zones.json")


def load_settings():
    """Đọc settings.json và ghi đè lên config (rtsp_url, car_alert_cooldown_sec, conf_threshold, siren_duration)."""
    import json
    if not os.path.isfile(SETTINGS_PATH):
        return
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return
        data = json.loads(raw)
        if "rtsp_url" in data:
            globals()["RTSP_URL"] = (data["rtsp_url"] or "").strip()
        if "car_alert_cooldown_sec" in data:
            globals()["CAR_ALERT_COOLDOWN_SEC"] = float(data["car_alert_cooldown_sec"])
        if "conf_threshold" in data:
            globals()["CONF_THRESHOLD"] = float(data["conf_threshold"])
        if "siren_duration" in data:
            globals()["SIREN_DURATION"] = float(data["siren_duration"])
    except Exception as e:
        print(f"[CONFIG] load_settings: {e}")
