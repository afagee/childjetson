"""
Camera capture module
"""
import cv2
import threading
import time


class VideoCaptureThreading:
    """Thread-safe video capture với buffer tối ưu"""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.latest_frame = None

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.latest_frame = frame.copy()
            # Bỏ sleep để đọc frame nhanh nhất có thể

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
