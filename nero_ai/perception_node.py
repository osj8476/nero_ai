import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.pub = self.create_publisher(String, '/detected_objects', 10)
        # 1초마다 가짜 인식 결과 발행 (VSS 대신 테스트용)
        self.timer = self.create_timer(1.0, self.publish_fake)
        self.get_logger().info('PerceptionNode 시작 (테스트 모드)')

    def publish_fake(self):
        fake = {
            "objects": [
                {"label": "red cup",
                 "center_3d": {"x": 0.4, "y": 0.0, "z": 0.3},
                 "confidence": 0.95},
                {"label": "bottle",
                 "center_3d": {"x": 0.5, "y": 0.1, "z": 0.3},
                 "confidence": 0.88}
            ]
        }
        msg = String()
        msg.data = json.dumps(fake)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(PerceptionNode())
    rclpy.shutdown()
