#!/usr/bin/env python3
"""
perception_node.py
MoonDream2 기반 객체 인식 노드.
- encode_image 한 번 + 라벨별 detect (테스트 성공 코드 방식)
- FP16 + autocast로 추론 가속
- 320x180 다운스케일 추론, 원본 해상도 좌표 환산
- filter_detections로 오탐 제거
- 좌표 변환은 camera_calibration 모듈로 분리
"""

import os
import json
import threading
import time

import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from PIL import Image as PILImage
from transformers import AutoModelForCausalLM

from nero_ai.camera_calibration import pixel_to_robot_xyz


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
CAMERA_SOURCE  = os.environ.get("CAMERA_SOURCE", "0")
TARGET_OBJECTS = os.environ.get(
    "TARGET_OBJECTS", "cup,bottle,scissors,phone,box,book"
).split(",")
POLL_INTERVAL  = float(os.environ.get("POLL_INTERVAL", "0.3"))
MODEL_REVISION = os.environ.get("MODEL_REVISION", "2025-06-21")

INFER_W = int(os.environ.get("INFER_W", "320"))
INFER_H = int(os.environ.get("INFER_H", "180"))

MIN_BBOX_SIZE = 0.02
DEDUP_THRESH  = 0.08


def filter_detections(objects):
    """크기 필터 + 중복 제거."""
    filtered = []
    for det in objects:
        w = det.get("x_max", 0) - det.get("x_min", 0)
        h = det.get("y_max", 0) - det.get("y_min", 0)
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue
        cx = det["x_min"] + w / 2
        cy = det["y_min"] + h / 2
        too_close = any(
            abs(cx - (f["x_min"] + (f["x_max"]-f["x_min"])/2)) < DEDUP_THRESH and
            abs(cy - (f["y_min"] + (f["y_max"]-f["y_min"])/2)) < DEDUP_THRESH
            for f in filtered
        )
        if not too_close:
            filtered.append(det)
    return filtered


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        # perception은 최신만 중요 → BEST_EFFORT + KEEP_LAST(1)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, '/detected_objects', qos)

        # ── 디바이스 ──
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.get_logger().info(f'GPU: {name} | VRAM: {mem:.1f} GB')
        else:
            self.get_logger().warn('CPU 모드 (느림)')

        # ── 모델 로드 (FP16) ──
        self.get_logger().info('MoonDream2 모델 로딩 중...')
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=MODEL_REVISION,
            trust_remote_code=True,
            device_map={"": self.device},
        )
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.to(torch.float16)
        self.get_logger().info('MoonDream2 로드 완료!')

        # ── 카메라 ──
        source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.get_logger().error(f'카메라 열기 실패: {CAMERA_SOURCE}')
            raise RuntimeError("카메라 초기화 실패")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 워밍업
        for _ in range(30):
            self.cap.read()

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'카메라: {self.frame_w}x{self.frame_h} | 추론: {INFER_W}x{INFER_H}')

        self.latest_frame = None
        self.frame_lock   = threading.Lock()

        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.timer = self.create_timer(POLL_INTERVAL, self.run_detection)
        self.get_logger().info(
            f'PerceptionNode 시작 | 대상: {TARGET_OBJECTS} | 주기: {POLL_INTERVAL}s')

    def _capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def run_detection(self):
        t0 = time.time()

        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        # ── 저해상도 추론용 변환 ──
        small = cv2.resize(frame, (INFER_W, INFER_H))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        detected_objects = []

        try:
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                # 한 번 인코딩 → 라벨별 재사용 (큰 속도 향상)
                try:
                    image_embeds = self.model.encode_image(pil_image)
                    use_embeds = True
                except Exception as e:
                    self.get_logger().warn(f'encode_image 실패, fallback: {e}')
                    use_embeds = False

                for obj_label in TARGET_OBJECTS:
                    obj_label = obj_label.strip()
                    try:
                        target = image_embeds if use_embeds else pil_image
                        result  = self.model.detect(target, obj_label)
                        objects = result.get("objects", [])
                        filtered = filter_detections(objects)

                        for det in filtered:
                            x_min = det.get("x_min", 0)
                            y_min = det.get("y_min", 0)
                            x_max = det.get("x_max", 0)
                            y_max = det.get("y_max", 0)

                            cx_norm = (x_min + x_max) / 2
                            cy_norm = (y_min + y_max) / 2

                            # 원본 픽셀 좌표
                            cx_px = cx_norm * self.frame_w
                            cy_px = cy_norm * self.frame_h

                            # 좌표 변환 모듈로 위임
                            xyz = pixel_to_robot_xyz(
                                cx_px, cy_px,
                                self.frame_w, self.frame_h,
                            )

                            detected_objects.append({
                                "label":     obj_label,
                                "bbox":      [x_min, y_min, x_max, y_max],
                                "center_2d": {"x": cx_norm, "y": cy_norm},
                                "center_3d": xyz,
                            })

                    except Exception as e:
                        self.get_logger().warn(f'{obj_label} 인식 오류: {e}')

        except Exception as e:
            self.get_logger().error(f'추론 실패: {e}')
            return

        msg = String()
        msg.data = json.dumps({"objects": detected_objects}, ensure_ascii=False)
        self.pub.publish(msg)

        elapsed = time.time() - t0
        if detected_objects:
            labels = [o["label"] for o in detected_objects]
            self.get_logger().info(
                f'인식: {labels} ({elapsed:.2f}s)')

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
