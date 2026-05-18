import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import json
import os
import threading
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
 
# ──────────────────────────────────────────────
# 설정 (환경변수로 변경 가능)
# ──────────────────────────────────────────────
# USB 웹캠:   "0" (기본값)
# CSI 카메라: "nvarguscamerasrc ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
# IP 카메라:  "rtsp://아이피주소/스트림경로"
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
 
# 인식할 물체 목록 (쉼표로 구분, 영어 소문자)
TARGET_OBJECTS = os.environ.get(
    "TARGET_OBJECTS", "cup,bottle,scissors,phone,box,book"
).split(",")
 
# 발행 주기 (초) — 0.2초 = 5FPS
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "0.2"))
 
# MoonDream2 모델 리비전
MODEL_REVISION = os.environ.get("MODEL_REVISION", "2025-06-21")
 
 
class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
 
        self.pub = self.create_publisher(String, '/detected_objects', 10)
 
        # 모델 로드 (처음 한 번, GPU에 올림)
        self.get_logger().info('MoonDream2 모델 로딩 중... (처음 한 번만, 시간 걸림)')
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=MODEL_REVISION,
            trust_remote_code=True,
            device_map={"": "cuda"}
        )
        self.get_logger().info('MoonDream2 로드 완료!')
 
        # 카메라 초기화
        source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.get_logger().error(f'카메라 열기 실패: {CAMERA_SOURCE}')
        else:
            self.get_logger().info(f'카메라 연결 완료: {CAMERA_SOURCE}')
 
        # 최신 프레임 저장 (백그라운드 캡처 스레드와 공유)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
 
        # 백그라운드 캡처 스레드 시작
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
 
        # 인식 타이머
        self.timer = self.create_timer(POLL_INTERVAL, self.run_detection)
 
        self.get_logger().info(
            f'PerceptionNode 시작 | 인식 대상: {TARGET_OBJECTS} | 주기: {POLL_INTERVAL}s'
        )
 
    def _capture_loop(self):
        """백그라운드에서 카메라 프레임을 계속 캡처"""
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
 
    def run_detection(self):
        """최신 프레임으로 MoonDream2 물체 인식 실행"""
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
 
        # OpenCV BGR → PIL RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
 
        # 이미지 인코딩 (같은 프레임에 여러 쿼리 시 효율적)
        try:
            encoded = self.model.encode_image(pil_image)
        except Exception as e:
            self.get_logger().error(f'이미지 인코딩 실패: {e}')
            return
 
        detected_objects = []
 
        for obj_label in TARGET_OBJECTS:
            try:
                result = self.model.detect(encoded, obj_label)
                objects = result.get("objects", [])
 
                for det in objects:
                    # MoonDream2 반환 좌표: 정규화된 0~1 값
                    x_min = det.get("x_min", 0)
                    y_min = det.get("y_min", 0)
                    x_max = det.get("x_max", 0)
                    y_max = det.get("y_max", 0)
 
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
 
                    detected_objects.append({
                        "label": obj_label,
                        "confidence": 1.0,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "center_2d": {"x": cx, "y": cy},
                        # 3D 위치: depth 카메라 연결 시 실제 값으로 교체
                        "center_3d": {"x": cx, "y": 0.0, "z": cy}
                    })
 
            except Exception as e:
                self.get_logger().warn(f'{obj_label} 인식 오류: {e}')
 
        # /detected_objects 토픽 발행
        msg = String()
        msg.data = json.dumps({"objects": detected_objects}, ensure_ascii=False)
        self.pub.publish(msg)
 
        if detected_objects:
            labels = [o["label"] for o in detected_objects]
            self.get_logger().info(f'인식된 물체: {labels}')
 
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()
 
 
def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
