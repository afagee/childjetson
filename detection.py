"""
YOLO detection — car-only.
Còi bật khi xe di chuyển chạm vùng cảnh báo.
Còi tắt khi xe thoát vùng hoặc xe không di chuyển trong thời gian cooldown.
Detection chạy trong thread riêng để cảnh báo hoạt động kể cả khi không ai mở web.
"""
import os
import cv2
import numpy as np
import time
import threading

import config
import state
import siren

# Ngưỡng pixel để coi là xe đã di chuyển (so với frame trước)
MOVE_THRESHOLD_PX = 15

# Đã thử load model (trong detection thread) — tránh spam log khi lỗi
_model_init_tried = False


def _ensure_model():
    """Khởi tạo TensorRT model trong thread hiện tại (detection thread).
    CUDA context phải cùng thread với cuda.mem_alloc, tránh lỗi invalid device context.
    """
    global _model_init_tried
    if state.model is not None:
        return
    if _model_init_tried:
        return
    _model_init_tried = True
    engine_cfg = getattr(config, "MODEL_ENGINE_PATH", "").strip()
    if not engine_cfg:
        print("[DETECTION] MODEL_ENGINE_PATH empty, skip TensorRT load")
        return
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(engine_cfg):
        engine_path = engine_cfg
    else:
        engine_path = os.path.abspath(os.path.join(app_dir, engine_cfg))
    if not os.path.isfile(engine_path):
        print(f"[DETECTION] Engine not found: {engine_path}")
        return
    try:
        # Import trong thread này để pycuda.autoinit chạy trong detection thread
        from trt_yolo import TrtYolo
        state.model = TrtYolo(
            engine_path=engine_path,
            input_size=getattr(config, "YOLO_INPUT_SIZE", 416),
            class_id_car=getattr(config, "CLASS_ID_CAR", 0),
            conf_thres=getattr(config, "CONF_THRESHOLD", 0.6),
        )
        print(f"[DETECTION] TensorRT engine loaded: {engine_path}")
    except Exception as e:
        print(f"[DETECTION] Model load error: {e}")


def _point_in_zone(pt, zone):
    if len(zone) < 3:
        return False
    poly = np.array(zone, np.int32)
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _bbox_intersects_zone(bbox_points, zone):
    for pt in bbox_points:
        if _point_in_zone(pt, zone):
            return True
    return False


def _get_zones_for_frame():
    """Danh sách polygon (đã scale sang frame) cho camera hiện tại. Xe chạm bất kỳ vùng nào thì cảnh báo."""
    import config as cfg
    import json
    from pathlib import Path
    
    # Lấy RTSP URL đang active từ settings.json
    camera_key = ""
    try:
        settings_path = Path(getattr(cfg, "SETTINGS_PATH", ""))
        if settings_path.is_file():
            with open(settings_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                data = json.loads(raw)
                camera_key = (data.get("rtsp_url") or "").strip()
    except Exception:
        pass
    
    # Nếu không có trong settings, dùng từ config
    if not camera_key:
        camera_key = (getattr(cfg, "RTSP_URL", "") or "").strip()
    
    if not camera_key:
        camera_key = "setup"
    
    with state._zones_lock:
        zones = state.zones_by_camera.get(camera_key, [])
        # Không fallback sang "setup" nữa - mỗi camera có zone riêng
    if not zones:
        return []
    sw = state.setup_image_width
    sh = state.setup_image_height
    if sw > 0 and sh > 0 and (sw != config.FRAME_WIDTH or sh != config.FRAME_HEIGHT):
        scale_x = config.FRAME_WIDTH / sw
        scale_y = config.FRAME_HEIGHT / sh
        return [[(int(x * scale_x), int(y * scale_y)) for (x, y) in poly] for poly in zones]
    return [list(poly) for poly in zones]


def _bbox_intersects_any_zone(bbox_points, list_of_zones):
    """True nếu bbox chạm bất kỳ polygon nào trong list."""
    for zone in list_of_zones:
        if len(zone) >= 3 and _bbox_intersects_zone(bbox_points, zone):
            return True
    return False


def process_frame(frame, run_detection=True):
    if frame is None:
        return frame

    display_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    now = time.time()
    zones_for_frame = _get_zones_for_frame()  # list of polygons

    if run_detection and state.model is not None:
        try:
            # Dùng TensorRT YOLO wrapper (TrtYolo) để detect trực tiếp trên frame gốc.
            t_detect_start = time.time()
            detections = state.model.detect(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            t_detect_elapsed = (time.time() - t_detect_start) * 1000  # ms
            if not hasattr(state, '_detect_time_logged'):
                print(f"[DETECTION] Inference time: {t_detect_elapsed:.1f}ms, detections: {len(detections)}")
                state._detect_time_logged = True

            state.latest_detections = []
            car_alert = False
            current_centers_in_zone = []

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bbox_points = [
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2),
                    (cx, cy),
                ]
                is_inside = _bbox_intersects_any_zone(bbox_points, zones_for_frame)
                state.latest_detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": det.get("cls_id", config.CLASS_ID_CAR),
                    "is_inside_car_zone": is_inside,
                })
                if is_inside:
                    car_alert = True
                    current_centers_in_zone.append((cx, cy))

            # Còi bật khi xe di chuyển chạm vùng; tắt khi thoát vùng hoặc đứng yên hết cooldown (sau khi kêu thêm SIREN_EXTRA_SEC)
            cooldown = getattr(config, "CAR_ALERT_COOLDOWN_SEC", 10)
            extra_sec = getattr(config, "SIREN_EXTRA_SEC", 5)
            if not car_alert:
                state.last_car_centers_in_zone = []
                if state.siren_is_on and state.siren_off_at <= 0:
                    state.siren_off_at = now + extra_sec  # kêu thêm N giây rồi tắt
            else:
                state.siren_off_at = 0  # hủy hẹn tắt khi còn xe trong zone
                # Phát hiện di chuyển: xe vừa vào zone hoặc ít nhất một xe trong zone đã dịch chuyển so với frame trước
                last = state.last_car_centers_in_zone
                if not last:
                    movement = True  # vừa chạm vùng
                else:
                    movement = any(
                        min(np.hypot(c[0] - p[0], c[1] - p[1]) for p in last) > MOVE_THRESHOLD_PX
                        for c in current_centers_in_zone
                    )
                state.last_car_centers_in_zone = current_centers_in_zone
                if movement:
                    state.last_movement_time = now
                if (now - state.last_movement_time) < cooldown:
                    state.current_alert_status = True
                    state.alert_timestamp = now
                    siren.turn_on()
                else:
                    if state.siren_is_on and state.siren_off_at <= 0:
                        state.siren_off_at = now + extra_sec  # đứng yên hết cooldown: kêu thêm N giây rồi tắt

            # Tắt còi khi đã hết thời gian kêu thêm
            if state.siren_is_on and state.siren_off_at > 0 and now >= state.siren_off_at:
                siren.turn_off()
                state.siren_off_at = 0
                state.current_alert_status = False

        except Exception as e:
            print(f"[DETECTION ERROR] {e}")

    for det in state.latest_detections:
        x1, y1, x2, y2 = det["bbox"]
        inside = det.get("is_inside_car_zone", False)
        color = (0, 0, 255) if inside else (200, 200, 200)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Vẽ tất cả vùng cảnh báo màu đỏ (tô) + id vùng trên stream
    for idx, poly in enumerate(zones_for_frame):
        if len(poly) < 3:
            continue
        pts = np.array(poly, np.int32)
        overlay = display_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
        cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = "Vung " + str(idx + 1)
        cv2.putText(display_frame, label, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, label, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    if state.current_alert_status:
        cv2.putText(display_frame, "!!! CANH BAO !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)

    return display_frame


def run_detection_loop():
    """Thread luôn chạy: đọc frame camera, chạy detection (còi), lưu frame mới nhất để web xem."""
    _ensure_model()  # Tạo TensorRT model trong thread này để CUDA context đúng
    frame_time = 1.0 / config.MAX_FPS
    while True:
        try:
            if state.cap_thread is None or not state.cap_thread.started:
                time.sleep(0.5)
                continue
            t0 = time.time()
            frame = state.cap_thread.read()
            if frame is None:
                time.sleep(0.1)
                continue
            state.frame_counter += 1
            run_detection = (state.frame_counter % config.FRAME_SKIP == 0)
            processed = process_frame(frame, run_detection=run_detection)
            if processed is None:
                processed = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            with state.latest_display_lock:
                state.latest_display_frame = processed.copy()
            state.last_frame_time = time.time()
            elapsed = time.time() - t0
            if elapsed < frame_time and (frame_time - elapsed) > 0:
                time.sleep(frame_time - elapsed)
        except Exception as e:
            print(f"[DETECTION LOOP ERROR] {e}")
            time.sleep(0.1)


def generate_frames():
    """Stream gửi frame mới nhất từ detection thread (khi có client xem).

    Mỗi client /video_feed tương ứng với một generator riêng.
    Ta đếm số client đang mở để có thể giới hạn số lượng và chỉ encode JPEG khi thực sự có viewer.
    """
    import time as time_module

    frame_time = 1.0 / config.MAX_FPS

    # Đăng ký client mới
    with state.stream_clients_lock:
        state.stream_clients += 1
        current_clients = state.stream_clients
    print(f"[STREAM] New client connected, total={current_clients}")

    try:
        while True:
            try:
                if state.cap_thread is None or not state.cap_thread.started:
                    time_module.sleep(0.5)
                    # Không encode nếu không có camera, vẫn giữ định dạng MJPEG hợp lệ
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + b"\r\n")
                    continue
                with state.latest_display_lock:
                    frame = state.latest_display_frame
                if frame is None:
                    time_module.sleep(0.1)
                    continue
                # Chỉ encode JPEG cho client đang xem
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if not ret:
                    continue
                time_module.sleep(frame_time)
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            except Exception as e:
                print(f"[STREAM ERROR] {e}")
                time_module.sleep(0.1)
    finally:
        # Client đóng kết nối (hoặc generator bị GC)
        with state.stream_clients_lock:
            state.stream_clients = max(0, state.stream_clients - 1)
            remaining = state.stream_clients
        print(f"[STREAM] Client disconnected, total={remaining}")
