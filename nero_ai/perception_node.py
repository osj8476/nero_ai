import rclpy
from rclpy.node import Node
from std_msgs.msg import String
 
import cv2
import json
import os
import threading
 
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM, AutoTokenizer  # AutoTokenizer 추가
 
 
# ──────────────────────────────────────────────
# 설정 (환경변수로 변경 가능)
# ──────────────────────────────────────────────
CAMERA_SOURCE  = os.environ.get("CAMERA_SOURCE", "0")
TARGET_OBJECTS = os.environ.get(
    "TARGET_OBJECTS", "cup,bottle,scissors,phone,box,book"
).split(",")
POLL_INTERVAL  = float(os.environ.get("POLL_INTERVAL", "0.2"))
MODEL_REVISION = os.environ.get("MODEL_REVISION", "2025-06-21")  # 최신 revision으로 업데이트
 
 
class PerceptionNode(Node):
 
    def __init__(self):
        super().__init__('perception_node')
 
        self.pub = self.create_publisher(String, '/detected_objects', 10)
 
        self.get_logger().info('MoonDream2 모델 로딩 중...')
 
        # ── 공식 문서 기준: AutoTokenizer도 함께 로드 ──
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=MODEL_REVISION,
            trust_remote_code=True,
            device_map={"": "cuda"}  # Apple Silicon이면 "mps"로 변경
        )
 
        self.get_logger().info('MoonDream2 로드 완료!')
 
        # ── 카메라 초기화 ──
        source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.get_logger().error(f'카메라 열기 실패: {CAMERA_SOURCE}')
        else:
            self.get_logger().info(f'카메라 연결 완료: {CAMERA_SOURCE}')
 
        self.latest_frame = None
        self.frame_lock   = threading.Lock()
 
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()
 
        self.timer = self.create_timer(POLL_INTERVAL, self.run_detection)
        self.get_logger().info(
            f'PerceptionNode 시작 | 대상: {TARGET_OBJECTS} | 주기: {POLL_INTERVAL}s')
 
    # ──────────────────────────────────────────────
    # 카메라 캡처 루프 (별도 스레드)
    # ──────────────────────────────────────────────
    def _capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
 
    # ──────────────────────────────────────────────
    # 객체 탐지 (타이머 콜백)
    # ──────────────────────────────────────────────
    def run_detection(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
 
        # BGR → RGB → PIL Image 변환
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
 
        detected_objects = []
 
        for obj_label in TARGET_OBJECTS:
            try:
                # ── 공식 API: encode_image() 없이 image를 직접 전달 ──
                result  = self.model.detect(pil_image, obj_label)
                objects = result.get("objects", [])
 
                for det in objects:
                    x_min = det.get("x_min", 0)
                    y_min = det.get("y_min", 0)
                    x_max = det.get("x_max", 0)
                    y_max = det.get("y_max", 0)
 
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
 
                    detected_objects.append({
                        "label":      obj_label,
                        "confidence": 1.0,
                        "bbox":       [x_min, y_min, x_max, y_max],
                        "center_2d":  {"x": cx, "y": cy},
                        # 정규화 좌표(0~1) → 미터 단위 근사 변환
                        "center_3d": {
                            "x": round(0.3 + (cx - 0.5) * 0.4, 3),
                            "y": round((0.5 - cy) * 0.3, 3),
                            "z": 0.10
                        }
                    })
 
            except Exception as e:
                self.get_logger().warn(f'{obj_label} 인식 오류: {e}')
 
        msg      = String()
        msg.data = json.dumps({"objects": detected_objects}, ensure_ascii=False)
        self.pub.publish(msg)
 
        if detected_objects:
            labels = [o["label"] for o in detected_objects]
            self.get_logger().info(f'인식된 물체: {labels}')
 
    # ──────────────────────────────────────────────
    # 노드 종료 시 카메라 해제
    # ──────────────────────────────────────────────
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
