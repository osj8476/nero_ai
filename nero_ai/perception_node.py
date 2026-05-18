import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import os

CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "/dev/video0")
VSS_URL = os.environ.get("VSS_URL", "http://localhost:1984/api/v1/query")
VSS_PROMPT = "List all objects on the table with their 3D positions. Return JSON only."
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "0.5"))


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.pub = self.create_publisher(String, '/detected_objects', 10)
        self.vss_connected = False
        self.timer = self.create_timer(POLL_INTERVAL, self.poll_vss)
        self.get_logger().info(f'PerceptionNode 시작 — VSS: {VSS_URL} | 카메라: {CAMERA_SOURCE}')
        self.get_logger().info('VSS 연결 대기 중...')

    def poll_vss(self):
        try:
            response = requests.post(
                VSS_URL,
                json={"prompt": VSS_PROMPT, "stream": False},
                timeout=2.0
            )
            response.raise_for_status()
            data = response.json()
            if "objects" not in data:
                data = {"objects": [], "raw": str(data)}
            msg = String()
            msg.data = json.dumps(data, ensure_ascii=False)
            self.pub.publish(msg)
            if not self.vss_connected:
                self.vss_connected = True
                self.get_logger().info('VSS 연결 성공! 물체 인식 시작.')
            objects = data.get("objects", [])
            if objects:
                labels = [o.get("label", "unknown") for o in objects]
                self.get_logger().info(f'인식된 물체: {labels}')
        except requests.exceptions.ConnectionError:
            if self.vss_connected:
                self.get_logger().warn('VSS 연결 끊김. 재연결 시도 중...')
                self.vss_connected = False
            else:
                self.get_logger().warn('VSS 아직 준비 안 됨. docker compose up vss 실행했는지 확인.')
        except requests.exceptions.Timeout:
            self.get_logger().warn('VSS 응답 시간 초과. 다음 주기에 재시도.')
        except requests.exceptions.HTTPError as e:
            self.get_logger().error(f'VSS HTTP 오류: {e}')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'VSS JSON 파싱 실패: {e}')
            msg = String()
            msg.data = json.dumps({"objects": [], "raw": response.text})
            self.pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'예상치 못한 오류: {e}')


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
