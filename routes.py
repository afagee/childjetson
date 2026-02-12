"""
Flask routes — car-only: upload ảnh vẽ zone, settings, car_zone, video_feed (khi có camera).
Cấu hình RTSP qua web yêu cầu mật khẩu. Có RTSP thì có thể chụp frame từ camera để vẽ zone.
"""
import json
import time
from pathlib import Path

import cv2
from flask import Flask, Response, render_template, jsonify, request, session, redirect

import config
import state
import detection
import camera


def register_routes(app: Flask):
    @app.before_request
    def require_admin():
        if request.path in ("/login", "/logout"):
            return None
        if not session.get("admin"):
            if request.path.startswith("/api/"):
                return jsonify(error="Unauthorized", admin=False), 401
            return redirect("/login")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")
        password = (request.form.get("password") or request.form.get("pwd") or "").strip()
        if password == getattr(config, "ADMIN_PASSWORD", ""):
            session["admin"] = True
            return redirect("/")
        return render_template("login.html", error="Mật khẩu không đúng")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/auth")
    def auth():
        return jsonify(admin=session.get("admin", False))

    @app.route("/video_feed")
    def video_feed():
        if state.cap_thread is None or not state.cap_thread.started:
            return ("Camera chưa bật", 204)
        # Giới hạn số client web xem stream cùng lúc để tránh quá tải Jetson
        max_clients = getattr(config, "MAX_STREAM_CLIENTS", 0)
        if max_clients and max_clients > 0:
            # Đọc số client hiện tại từ state (dùng lock để an toàn)
            try:
                with state.stream_clients_lock:
                    current_clients = state.stream_clients
            except Exception:
                current_clients = 0
            if current_clients >= max_clients:
                return (f"Too many viewers (max {max_clients})", 429)
        return Response(
            detection.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/upload_setup_image", methods=["POST"])
    def upload_setup_image():
        if "file" not in request.files and "image" not in request.files:
            return jsonify(success=False, error="No file"), 400
        f = request.files.get("file") or request.files.get("image")
        if not f or f.filename == "":
            return jsonify(success=False, error="No file"), 400
        path = state.UPLOAD_DIR_PATH / config.SETUP_IMAGE_FILENAME
        try:
            f.save(str(path))
        except Exception as e:
            return jsonify(success=False, error=str(e)), 500
        state.setup_image_path = path
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            state.setup_image_height, state.setup_image_width = img.shape[:2]
        else:
            state.setup_image_width = 0
            state.setup_image_height = 0
        return jsonify(
            success=True,
            url="/api/setup_image",
            width=state.setup_image_width,
            height=state.setup_image_height,
        )

    @app.route("/api/capture_setup_from_camera", methods=["POST"])
    def capture_setup_from_camera():
        """Chụp 1 frame từ RTSP, lưu làm ảnh setup để vẽ zone (khi có camera)."""
        if state.cap_thread is None or not state.cap_thread.started:
            return jsonify(success=False, error="Camera chưa bật"), 400
        frame = state.cap_thread.read()
        if frame is None:
            return jsonify(success=False, error="Không đọc được frame"), 400
        path = state.UPLOAD_DIR_PATH / config.SETUP_IMAGE_FILENAME
        display_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        if not cv2.imwrite(str(path), display_frame):
            return jsonify(success=False, error="Không lưu được ảnh"), 500
        state.setup_image_path = path
        state.setup_image_width = config.FRAME_WIDTH
        state.setup_image_height = config.FRAME_HEIGHT
        return jsonify(
            success=True,
            url="/api/setup_image",
            width=state.setup_image_width,
            height=state.setup_image_height,
        )

    @app.route("/api/setup_image")
    def setup_image():
        path = state.UPLOAD_DIR_PATH / config.SETUP_IMAGE_FILENAME
        if not path.is_file():
            return ("", 204)
        try:
            data = path.read_bytes()
            return Response(data, mimetype="image/jpeg")
        except Exception:
            return ("", 204)

    def _apply_rtsp_and_restart_camera(new_rtsp):
        """Dừng camera cũ (nếu có), khởi động lại với new_rtsp nếu không rỗng."""
        if state.cap_thread is not None and state.cap_thread.started:
            try:
                state.cap_thread.stop()
            except Exception:
                pass
            state.cap_thread = None
        if new_rtsp and new_rtsp.strip():
            try:
                state.cap_thread = camera.VideoCaptureThreading(new_rtsp.strip()).start()
                time.sleep(1.0)
                state.last_frame_time = time.time()
            except Exception as e:
                print(f"[ROUTES] Camera start: {e}")
                state.cap_thread = None

    def _rtsp_require_password(data):
        """Cần mật khẩu đúng khi chưa đăng nhập admin."""
        if session.get("admin"):
            return True
        p = data.get("rtsp_password") or data.get("password") or ""
        if p != getattr(config, "RTSP_CONFIG_PASSWORD", ""):
            return False
        return True

    @app.route("/api/settings", methods=["GET", "POST"])
    def settings_api():
        if request.method == "GET":
            path = Path(config.SETTINGS_PATH)
            current = {}
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    current = json.loads(raw) if raw else {}
                except Exception:
                    current = {}
            rtsp_sources = current.get("rtsp_sources") or []
            rtsp_url = current.get("rtsp_url") or config.RTSP_URL or ""
            # Migration: nếu file cũ chỉ có rtsp_url, hiển thị thành 1 item trong list
            if not rtsp_sources and rtsp_url:
                rtsp_sources = [{"name": "Camera", "url": rtsp_url}]
            out = {
                "car_alert_cooldown_sec": config.CAR_ALERT_COOLDOWN_SEC,
                "conf_threshold": config.CONF_THRESHOLD,
                "siren_duration": config.SIREN_DURATION,
                "rtsp_url": rtsp_url,
                "rtsp_sources": rtsp_sources,
            }
            return jsonify(out)
        data = request.get_json()
        if not data:
            return jsonify(success=False, error="No JSON"), 400
        try:
            path = Path(config.SETTINGS_PATH)
            current = {}
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    current = json.loads(raw) if raw else {}
                except Exception:
                    current = {}
            rtsp_updated = False
            new_rtsp = None

            # Cập nhật danh sách RTSP (thêm/sửa/xóa) — cần mật khẩu
            if "rtsp_sources" in data:
                if not _rtsp_require_password(data):
                    return jsonify(success=False, error="Mật khẩu cấu hình RTSP không đúng"), 403
                # Lấy danh sách URL cũ và mới để xác định camera nào bị xóa
                old_urls = set()
                if "rtsp_sources" in current:
                    old_urls = {s.get("url", "").strip() for s in current.get("rtsp_sources", []) if s.get("url", "").strip()}
                
                src = data["rtsp_sources"]
                new_sources = [{"name": str(s.get("name", ""))[:80], "url": (s.get("url") or "").strip()} for s in src if (s.get("url") or "").strip()]
                new_urls = {s["url"] for s in new_sources}
                
                # Xóa zone của các camera không còn trong danh sách
                removed_urls = old_urls - new_urls
                if removed_urls:
                    with state._zones_lock:
                        for removed_url in removed_urls:
                            if removed_url in state.zones_by_camera:
                                del state.zones_by_camera[removed_url]
                    state.save_zones()
                
                current["rtsp_sources"] = new_sources

            # Chọn RTSP đang dùng — cần mật khẩu
            if "rtsp_url" in data:
                if not _rtsp_require_password(data):
                    return jsonify(success=False, error="Mật khẩu cấu hình RTSP không đúng"), 403
                new_rtsp = (data.get("rtsp_url") or "").strip()
                current["rtsp_url"] = new_rtsp
                rtsp_updated = True

            if "car_alert_cooldown_sec" in data:
                current["car_alert_cooldown_sec"] = float(data["car_alert_cooldown_sec"])
            if "conf_threshold" in data:
                current["conf_threshold"] = float(data["conf_threshold"])
            if "siren_duration" in data:
                current["siren_duration"] = float(data["siren_duration"])

            with open(path, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
            config.load_settings()
            if rtsp_updated and new_rtsp is not None:
                _apply_rtsp_and_restart_camera(new_rtsp)
            return jsonify(success=True, settings=current)
        except Exception as e:
            return jsonify(success=False, error=str(e)), 500

    @app.route("/api/alert_status")
    def alert_status():
        return jsonify(
            alert_active=state.current_alert_status,
            timestamp=state.alert_timestamp,
        )

    @app.route("/api/status")
    def status():
        path = state.UPLOAD_DIR_PATH / config.SETUP_IMAGE_FILENAME
        if path.is_file() and (state.setup_image_width <= 0 or state.setup_image_height <= 0):
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is not None:
                    state.setup_image_height, state.setup_image_width = img.shape[:2]
            except Exception:
                pass
        
        # Lấy RTSP URL đang active từ settings.json
        camera_key = ""
        try:
            settings_path = Path(config.SETTINGS_PATH)
            if settings_path.is_file():
                with open(settings_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    data = json.loads(raw)
                    camera_key = (data.get("rtsp_url") or "").strip()
        except Exception:
            pass
        
        if not camera_key:
            camera_key = (config.RTSP_URL or "").strip() or "setup"
        
        zones = state.zones_by_camera.get(camera_key, [])
        zones_count = sum(len(p) for p in zones)
        return jsonify(
            camera_connected=state.cap_thread is not None and state.cap_thread.started,
            car_zone_points=zones_count,
            zones_polygon_count=len(zones),
            setup_image_width=state.setup_image_width,
            setup_image_height=state.setup_image_height,
            model_loaded=state.model is not None,
            rtsp_url_active=camera_key if camera_key != "setup" else "",
        )

    @app.route("/api/car_zone", methods=["GET", "POST", "DELETE"])
    def car_zone_api():
        # Lấy camera_key từ request, nếu không có thì dùng camera đang active
        camera_key = request.args.get("camera_key") or (request.get_json() or {}).get("camera_key")
        
        # Nếu không có camera_key, lấy từ RTSP đang active
        if not camera_key:
            try:
                settings_path = Path(config.SETTINGS_PATH)
                if settings_path.is_file():
                    with open(settings_path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if raw:
                        data = json.loads(raw)
                        camera_key = (data.get("rtsp_url") or "").strip()
            except Exception:
                pass
            
            if not camera_key:
                camera_key = (config.RTSP_URL or "").strip() or "setup"
        
        if request.method == "GET":
            zones = state.zones_by_camera.get(camera_key, [])
            return jsonify(camera_key=camera_key, zones=zones)
        if request.method == "POST":
            data = request.get_json() or {}
            if data.get("new_zone"):
                with state._zones_lock:
                    if camera_key not in state.zones_by_camera:
                        state.zones_by_camera[camera_key] = []
                    state.zones_by_camera[camera_key].append([])
                state.save_zones()
                return jsonify(success=True, camera_key=camera_key, zones=state.zones_by_camera.get(camera_key, []))
            if data.get("point"):
                x = int(data["point"]["x"])
                y = int(data["point"]["y"])
                with state._zones_lock:
                    if camera_key not in state.zones_by_camera:
                        state.zones_by_camera[camera_key] = [[]]
                    if not state.zones_by_camera[camera_key]:
                        state.zones_by_camera[camera_key].append([])
                    state.zones_by_camera[camera_key][-1].append((x, y))
                state.save_zones()
                return jsonify(success=True, camera_key=camera_key, zones=state.zones_by_camera.get(camera_key, []))
            return jsonify(success=False), 400
        # Xóa một vùng (index) hoặc tất cả (không gửi index)
        idx_param = request.args.get("index")
        if idx_param is not None:
            try:
                idx = int(idx_param)
                with state._zones_lock:
                    zones = state.zones_by_camera.get(camera_key, [])
                    if 0 <= idx < len(zones):
                        zones.pop(idx)
                        state.zones_by_camera[camera_key] = zones
                state.save_zones()
                return jsonify(success=True, camera_key=camera_key, zones=state.zones_by_camera.get(camera_key, []))
            except ValueError:
                return jsonify(success=False, error="index invalid"), 400
        with state._zones_lock:
            state.zones_by_camera[camera_key] = []
        state.save_zones()
        return jsonify(success=True, camera_key=camera_key, zones=[])
