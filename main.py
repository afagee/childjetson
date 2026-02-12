"""
Jetson App — Car-only. Entrypoint: load settings, model, (camera nếu có RTSP), relay, Flask.
Detection chạy trong thread riêng để cảnh báo hoạt động kể cả khi không ai mở web.
"""
import os
import sys
import time
import threading

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from flask import Flask

import config
import state
import camera
import routes
import relay
import detection


def init_system():
    config.load_settings()

    # TensorRT model được tạo trong detection thread (lazy-init) để tránh lỗi
    # "cumemalloc failed: invalid device context" (CUDA context phải cùng thread với inference).

    rtsp = getattr(config, "RTSP_URL", "") or ""
    if rtsp and rtsp.strip():
        if state.cap_thread is None or not state.cap_thread.started:
            print("[INFO] Starting camera...")
            state.cap_thread = camera.VideoCaptureThreading(rtsp.strip()).start()
            time.sleep(1.0)
            state.last_frame_time = time.time()
    else:
        print("[INFO] No RTSP_URL: chỉ dùng upload ảnh + vẽ zone, không bật camera")

    relay.init_pins(config.SIREN_GPIO_CHANNEL, config.RELAY_CH2_GPIO)

    # Thread detection luôn chạy: đọc camera, YOLO, bật/tắt còi; không phụ thuộc có ai xem web
    detection_thread = threading.Thread(target=detection.run_detection_loop, daemon=True)
    detection_thread.start()
    print("[INFO] Detection thread started (cảnh báo hoạt động kể cả khi không mở web)")


def main():
    init_system()
    app = Flask(__name__, template_folder=os.path.join(_APP_DIR, "templates"))
    app.secret_key = getattr(config, "SECRET_KEY", "jetson-car-alert-secret")
    routes.register_routes(app)
    print(f"[INFO] Jetson App (car-only) http://0.0.0.0:{config.PORT}")
    app.run(host="0.0.0.0", port=config.PORT, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
