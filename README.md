# Jetson App — Car-only Alert

App chạy trên Jetson Nano 4GB: YOLO26s (class car, COCO id 2), **một vùng nguy hiểm cho xe**. Upload 1 ảnh để vẽ zone (không cần RTSP). Khi xe chạm zone → GPIO (còi 12V qua relay). Không lưu ảnh cảnh báo; cooldown cấu hình được qua web.

---

## Hướng dẫn cài đặt trên Jetson Nano

### 1. Yêu cầu

- Jetson Nano 4GB với **JetPack 4.6.x** (Nano không hỗ trợ JetPack 5/6/7)
- Python 3.8 trở lên
- Kết nối internet (để tải dependencies lần đầu)

### 2. Chuẩn bị hệ thống (chung)

```bash
sudo apt-get -y update
sudo apt-get install -y python3-pip libopenblas-dev
```

---

## Hai chế độ cài đặt

- **A. Build-time (trên PC / máy có GPU)**  
  Dùng Ultralytics + PyTorch để **export `.engine`** (TensorRT) từ `.pt`.

- **B. Runtime (trên Jetson Nano)**  
  Chỉ cần **TensorRT + OpenCV + Flask**, load trực tiếp file `.engine` (không cần Ultralytics/PyTorch).

Bạn có thể:
- Train / export trên PC, **copy `.engine` sang Jetson** và chỉ cài phần runtime trên Jetson (khuyến nghị).
- Hoặc cài cả Ultralytics/PyTorch trên Jetson nếu muốn export trực tiếp (nặng, không khuyến khích).

---

## A. Cài build-time để export `.engine` (PC hoặc Jetson có đủ tài nguyên)

### 3A. Cài PyTorch cho Jetson / PC (build-time)

PyTorch phải cài từ NVIDIA wheel (Jetson) hoặc từ PyTorch (PC), **không** nên dùng `pip install torch` bừa bãi trên Jetson.

- Jetson: xem hướng dẫn chính thức  
  [NVIDIA – Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)

### 4A. Cài Ultralytics và các package (build-time)

Trên Jetson, pip không có wheel `torch`/`torchvision` cho ARM64 — nếu cài trực tiếp `ultralytics` sẽ báo lỗi. Cần cài theo thứ tự:

```bash
cd /path/to/ChildAlert

# Cài ultralytics BỎ QUA torch (--no-deps) vì đã cài torch từ NVIDIA ở bước 3
pip3 install ultralytics --no-deps

# Cài các package còn lại
pip3 install -r jetson_app/requirements.txt  # để có Flask, OpenCV, numpy
```

Nếu vẫn lỗi với ultralytics mới (>8.1): thử phiên bản 8.0.x tương thích PyTorch 1.10 (JetPack 4):
```bash
pip3 install "ultralytics>=8.0.0,<8.1" --no-deps
```

### 5A. Export `.engine`

```bash
cd /path/to/ChildAlert
python3 jetson_app/export_engine.py
```

Sau khi chạy xong sẽ sinh file `.engine` cạnh `.pt` (ví dụ `yolo26s.engine`).  
Copy file `.engine` này sang Jetson và cấu hình `MODEL_ENGINE_PATH` (xem phần cấu hình bên dưới).

---

## B. Cài runtime trên Jetson Nano (chỉ chạy `.engine`)

Ở chế độ này **không cần Ultralytics/PyTorch trên Jetson**, chỉ cần:
- TensorRT + CUDA (có sẵn trong JetPack),
- `opencv-python-headless`, `numpy`, `Flask`,
- `Jetson.GPIO` cho relay.

### 3B. Cài Python packages runtime

```bash
cd /path/to/ChildAlert
pip3 install -r jetson_app/requirements.txt
```

### 4B. Cài Jetson.GPIO (cho relay/còi)

```bash
sudo pip3 install Jetson.GPIO
```

### 5B. Chuẩn bị file `.engine`

- Copy file `.engine` đã export (ở bước 5A) vào thư mục `ChildAlert` hoặc một vị trí mà bạn dễ cấu hình.
- Trong `jetson_app/config.py`, cấu hình:

```python
MODEL_ENGINE_PATH = os.environ.get("JETSON_APP_ENGINE", "../yolo26s.engine")
```

Hoặc set qua biến môi trường khi chạy:

```bash
export JETSON_APP_ENGINE=/absolute/path/to/yolo26s.engine
```

### 6. Chạy app

```bash
cd /path/to/ChildAlert
python3 jetson_app/main.py
```

Mở trình duyệt: `http://<ip-jetson>:5000` (thay `<ip-jetson>` bằng IP thực của Jetson trên mạng).

### 8. Tối ưu hiệu năng (tùy chọn)

Trước khi chạy app, chạy lệnh sau để Jetson chạy full power:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Chạy nhanh (đã cài đặt xong)

```bash
cd /path/to/ChildAlert
python3 jetson_app/main.py
```

Upload ảnh → vẽ vùng xe (polygon) → chỉnh cooldown/conf trong form. Khi có RTSP, bật camera và xem live; xe chạm zone thì phát còi (cooldown để không phát lặp nếu xe đứng yên).

---

## Xử lý lỗi "không cài được ultralytics"

Lỗi `Could not find a version that satisfies torchvision>=0.9.0` xảy ra vì PyPI không có torch/torchvision cho ARM64. Giải pháp:

1. **Cài PyTorch + torchvision từ NVIDIA** trước (bước 3)
2. **Cài ultralytics với `--no-deps`:** `pip3 install ultralytics --no-deps`
3. Dùng **ultralytics 8.0.x** nếu PyTorch 1.10: `pip3 install "ultralytics>=8.0,<8.1" --no-deps`
4. **Docker** (khuyến nghị): `sudo docker run -it --ipc=host --runtime=nvidia ultralytics/ultralytics:latest-jetson-jetpack4`

---

## Cấu hình

- **Camera**: `RTSP_URL` (env `JETSON_APP_RTSP_URL`). Để trống = không bật camera, chỉ dùng trang upload ảnh + vẽ zone.
- **Model**: Mặc định `yolo26s.pt` (COCO), chỉ class **car** (`CLASS_ID_CAR = 2`). Có thể dùng TensorRT qua `MODEL_ENGINE_PATH` (env `JETSON_APP_ENGINE`).
- **Thông số chỉnh qua web**: `settings.json` (car_alert_cooldown_sec, conf_threshold, siren_duration). GET/POST `/api/settings`.
- **Relay**: BCM pin cho còi (`SIREN_GPIO_CHANNEL`) và kênh 2 (`RELAY_CH2_GPIO`). Trên PC không có GPIO thì relay no-op.

## Jetson Nano: Python + YOLO + web

Nano 4GB chạy Python, YOLO, Flask và stream video dễ quá tải. Gợi ý:

1. **TensorRT (nên dùng)**  
   Inference .pt chậm; .engine nhanh hơn. Export:  
   `YOLO("path/to/yolo26s.pt").export(format='engine', device=0, half=True, imgsz=416)`  
   Trong config đặt `MODEL_ENGINE_PATH` trỏ tới file .engine.

2. **Giảm tải**  
   - `FRAME_SKIP = 2` hoặc `3`  
   - `YOLO_INPUT_SIZE = 416` (hoặc 320)  
   - `MAX_FPS = 15`, stream 960x540, `JPEG_QUALITY = 75`

3. **Hệ thống**  
   - `sudo nvpmodel -m 0`, `sudo jetson_clocks`
