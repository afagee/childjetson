"""
TensorRT YOLO wrapper — chạy .engine trực tiếp trên Jetson (không cần Ultralytics/PyTorch lúc runtime).

Engine được export từ Ultralytics YOLO (v8/26) với format output 1xNx6:
    (x_center, y_center, width, height, confidence, class_id)
toạ độ tính trên kích thước input của network (thường imgsz x imgsz).
"""

import os
from typing import List, Tuple, Dict

import cv2
import numpy as np

import config

try:
    import tensorrt as trt  # type: ignore
except Exception as e:  # pragma: no cover - chỉ log trên Jetson
    trt = None
    _trt_import_error = e
else:
    _trt_import_error = None

try:
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401  # khởi tạo CUDA context mặc định
except Exception as e:  # pragma: no cover
    cuda = None
    _cuda_import_error = e
else:
    _cuda_import_error = None


class TrtYolo:
    """Wrapper đơn giản cho engine YOLO TensorRT đã export từ Ultralytics."""

    def __init__(
        self,
        engine_path: str,
        input_size: int,
        class_id_car: int,
        conf_thres: float,
    ) -> None:
        if trt is None:
            raise RuntimeError(
                f"TensorRT (tensorrt) chưa được cài / import được: {_trt_import_error}"
            )
        if cuda is None:
            raise RuntimeError(
                f"PyCUDA (pycuda.driver) chưa được cài / import được: {_cuda_import_error}"
            )

        engine_path = engine_path.strip()
        if not os.path.isabs(engine_path):
            # Cho phép cấu hình path tương đối từ thư mục app
            app_dir = os.path.dirname(os.path.abspath(__file__))
            engine_path = os.path.abspath(os.path.join(app_dir, engine_path))
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.engine_path = engine_path
        self.input_size = int(input_size)
        self.class_id_car = int(class_id_car)
        self.conf_thres = float(conf_thres)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        # Kiểm tra CUDA device (pycuda.autoinit đã tạo context rồi)
        try:
            cuda.init()
            device = cuda.Device(0)
            device_name = device.name()
            # Kiểm tra context hiện tại
            try:
                ctx = cuda.Context.get_current()
                print(f"[TRT_YOLO] CUDA context active on device: {device_name}")
            except:
                # Nếu chưa có context, tạo mới
                device.make_context()
                print(f"[TRT_YOLO] CUDA context created on device: {device_name}")
        except Exception as e:
            print(f"[TRT_YOLO WARN] CUDA device check: {e}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        # Log thông tin engine
        print(f"[TRT_YOLO] Engine loaded: {engine_path}")
        print(f"[TRT_YOLO] Input binding: {self.input_index}, Output binding: {self.output_index}")
        print(f"[TRT_YOLO] Input size: {self.input_size}, Class car: {self.class_id_car}, Conf thres: {self.conf_thres}")

        # Giả định 1 input, 1 output
        self.input_index = None
        self.output_index = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_index = i
            else:
                self.output_index = i
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("Unexpected TensorRT bindings (need 1 input, 1 output)")

    # ---- Nội bộ: chạy 1 lần infer trên frame đã resize ----

    def _infer_raw(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Chạy inference trên tensor shape (1,3,H,W) float32 (0–1).
        Trả về numpy array output thô của engine (flatten).
        """
        assert input_tensor.ndim == 4, f"Expected 4D tensor, got {input_tensor.shape}"
        if not input_tensor.flags["C_CONTIGUOUS"]:
            input_tensor = np.ascontiguousarray(input_tensor)

        # Thiết lập shape động nếu engine hỗ trợ
        self.context.set_binding_shape(self.input_index, input_tensor.shape)
        output_shape = self.context.get_binding_shape(self.output_index)

        # Tính kích thước buffer
        input_nbytes = input_tensor.nbytes
        out_size = int(np.prod(output_shape))
        output_tensor = np.empty(out_size, dtype=np.float32)

        d_input = cuda.mem_alloc(input_nbytes)
        d_output = cuda.mem_alloc(output_tensor.nbytes)
        bindings = [None] * self.engine.num_bindings
        bindings[self.input_index] = int(d_input)
        bindings[self.output_index] = int(d_output)

        try:
            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, input_tensor, stream)
            # Chạy inference trên GPU
            success = self.context.execute_v2(bindings)
            if not success:
                print("[TRT_YOLO ERROR] execute_v2 returned False")
            cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
            stream.synchronize()
        except Exception as e:
            print(f"[TRT_YOLO ERROR] Inference failed: {e}")
            raise
        finally:
            d_input.free()
            d_output.free()

        return output_tensor.reshape(tuple(output_shape))

    # ---- Public API ----

    def detect(
        self,
        frame_bgr: np.ndarray,
        out_size: Tuple[int, int] | None = None,
    ) -> List[Dict]:
        """
        Detect xe hơi trên frame BGR gốc.

        - frame_bgr: frame từ OpenCV (H,W,3) BGR.
        - out_size: kích thước output (width, height) để scale bbox. Mặc định = (config.FRAME_WIDTH, config.FRAME_HEIGHT).

        Trả về: list dict {
            'bbox': (x1, y1, x2, y2),
            'cls_id': int,
            'conf': float,
        } với toạ độ trên hệ quy chiếu out_size.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if out_size is None:
            out_w, out_h = config.FRAME_WIDTH, config.FRAME_HEIGHT
        else:
            out_w, out_h = int(out_size[0]), int(out_size[1])

        # Resize về kích thước input cố định (imgsz x imgsz)
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)   # NCHW

        try:
            raw_output = self._infer_raw(img)
        except Exception as e:
            print(f"[TRT_YOLO ERROR] Inference exception: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Debug: in shape output để kiểm tra format (chỉ in lần đầu để tránh spam)
        if not hasattr(self, '_debug_printed'):
            print(f"[TRT_YOLO DEBUG] Output shape: {raw_output.shape}, dtype: {raw_output.dtype}")
            print(f"[TRT_YOLO DEBUG] Output min/max: {raw_output.min():.3f} / {raw_output.max():.3f}")
            self._debug_printed = True

        # Ultralytics TensorRT engine có thể trả về:
        # - Format 1: (1, N, 6) với (x_c, y_c, w, h, conf, cls_id) - đã post-process
        # - Format 2: (1, num_anchors, 4+1+num_classes) - raw output cần decode
        # - Format 3: (1, num_anchors*6) - flatten
        
        if raw_output.ndim == 3:
            # (1, N, 6) hoặc (1, num_anchors, features)
            if raw_output.shape[0] == 1:
                dets = raw_output[0]  # (N, 6) hoặc (num_anchors, features)
            else:
                dets = raw_output.reshape(-1, raw_output.shape[-1])
        elif raw_output.ndim == 2:
            dets = raw_output
        else:
            # Flatten về 2D
            dets = raw_output.reshape(-1, raw_output.shape[-1])

        # Kiểm tra số cột: nếu là 6 thì giả định (x_c, y_c, w, h, conf, cls_id)
        # Nếu nhiều hơn (ví dụ 84 cho YOLOv8 với 80 classes) thì cần decode khác
        num_cols = dets.shape[-1] if dets.ndim > 0 else 0
        if not hasattr(self, '_debug_printed'):
            print(f"[TRT_YOLO DEBUG] Detections shape after reshape: {dets.shape}, num_cols={num_cols}")
            # In sample đầu tiên để debug
            if len(dets) > 0:
                print(f"[TRT_YOLO DEBUG] Sample detection (first row): {dets[0]}")

        results: List[Dict] = []
        
        if num_cols == 6:
            # Format đã post-process: (x_c, y_c, w, h, conf, cls_id)
            # YOLO26 TensorRT engine thường trả về format này
            for det in dets:
                det_list = det.tolist()
                x_c, y_c, w, h, conf, cls_id = det_list[0], det_list[1], det_list[2], det_list[3], det_list[4], det_list[5]
                
                # Kiểm tra confidence
                if conf < self.conf_thres:
                    continue
                
                # Chuyển class_id sang int (có thể là float)
                cls_id_int = int(round(cls_id))
                
                # Lọc chỉ class car nếu được cấu hình
                if self.class_id_car is not None and cls_id_int != self.class_id_car:
                    continue
                
                # Debug: in một vài detection đầu tiên
                if len(results) < 3:
                    print(f"[TRT_YOLO DEBUG] Detection: cls={cls_id_int}, conf={conf:.3f}, xywh=({x_c:.1f},{y_c:.1f},{w:.1f},{h:.1f})")
                # xywh (center) -> xyxy trên hệ imgsz x imgsz
                x1 = x_c - w / 2.0
                y1 = y_c - h / 2.0
                x2 = x_c + w / 2.0
                y2 = y_c + h / 2.0

                # Scale sang out_size
                sx = out_w / float(self.input_size)
                sy = out_h / float(self.input_size)
                x1 = max(0, min(out_w, x1 * sx))
                y1 = max(0, min(out_h, y1 * sy))
                x2 = max(0, min(out_w, x2 * sx))
                y2 = max(0, min(out_h, y2 * sy))

                if x2 <= x1 or y2 <= y1:
                    continue

                results.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "cls_id": cls_id_int,
                        "conf": float(conf),
                    }
                )
        elif num_cols >= 5:
            # Raw YOLO output: (x, y, w, h, obj_conf, ...class_probs...)
            # Giả định format: (x_c, y_c, w, h, obj_conf, class_conf_0, class_conf_1, ...)
            # Hoặc có thể là (x_c, y_c, w, h, conf, cls_id) nhưng cls_id là float
            print(f"[TRT_YOLO DEBUG] Raw format detected, trying decode...")
            for det in dets:
                det_list = det.tolist()
                x_c, y_c, w, h = det_list[0], det_list[1], det_list[2], det_list[3]
                # Tìm confidence và class_id
                if num_cols == 5:
                    # (x, y, w, h, conf) - không có class, giả định class 0
                    conf = det_list[4]
                    cls_id_int = 0
                else:
                    # (x, y, w, h, obj_conf, class_conf_0, class_conf_1, ...)
                    obj_conf = det_list[4]
                    class_confs = np.array(det_list[5:])
                    cls_id_int = int(np.argmax(class_confs))
                    cls_conf = float(class_confs[cls_id_int])
                    conf = obj_conf * cls_conf
                
                if conf < self.conf_thres:
                    continue
                if self.class_id_car is not None and cls_id_int != self.class_id_car:
                    continue
                # xywh (center) -> xyxy trên hệ imgsz x imgsz
                x1 = x_c - w / 2.0
                y1 = y_c - h / 2.0
                x2 = x_c + w / 2.0
                y2 = y_c + h / 2.0

                # Scale sang out_size
                sx = out_w / float(self.input_size)
                sy = out_h / float(self.input_size)
                x1 = max(0, min(out_w, x1 * sx))
                y1 = max(0, min(out_h, y1 * sy))
                x2 = max(0, min(out_w, x2 * sx))
                y2 = max(0, min(out_h, y2 * sy))

                if x2 <= x1 or y2 <= y1:
                    continue

                results.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "cls_id": cls_id_int,
                        "conf": float(conf),
                    }
                )
        else:
            print(f"[TRT_YOLO WARN] Unexpected output format: {dets.shape}, skipping decode")

        if len(results) > 0:
            print(f"[TRT_YOLO] Found {len(results)} detections (conf_thres={self.conf_thres}, class_car={self.class_id_car})")
        elif not hasattr(self, '_no_det_warned'):
            print(f"[TRT_YOLO WARN] No detections found. Check: conf_thres={self.conf_thres}, class_car={self.class_id_car}, output_shape={raw_output.shape}")
            self._no_det_warned = True
        
        return results

            # xywh (center) -> xyxy trên hệ imgsz x imgsz
            x1 = x_c - w / 2.0
            y1 = y_c - h / 2.0
            x2 = x_c + w / 2.0
            y2 = y_c + h / 2.0

            # Scale sang out_size
            sx = out_w / float(self.input_size)
            sy = out_h / float(self.input_size)
            x1 = max(0, min(out_w, x1 * sx))
            y1 = max(0, min(out_h, y1 * sy))
            x2 = max(0, min(out_w, x2 * sx))
            y2 = max(0, min(out_h, y2 * sy))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "cls_id": cls_id_int,
                    "conf": float(conf),
                }
            )

        return results


__all__ = ["TrtYolo"]

