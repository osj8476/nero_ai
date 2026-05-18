import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import time

# 그리퍼 열림/닫힘 값 (Jetson에서 실제 값 확인 후 수정)
# TODO: ros2 interface show agx_arm_msgs/msg/HandCmd 로 필드 확인 필요
GRIPPER_OPEN  = 1.0
GRIPPER_CLOSE = 0.0

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, 10)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command, 10)

        # 이동 퍼블리셔
        self.pub_move = self.create_publisher(PoseStamped, '/control/move_p', 10)

        # 그리퍼 퍼블리셔 (String으로 임시 — HandCmd 구조 확인 후 교체)
        # TODO: from agx_arm_msgs.msg import HandCmd
        self.pub_hand = self.create_publisher(String, '/control/hand_raw', 10)

        self.target_label = None
        self.target_position = None
        self.state = 'idle'  # idle → moving → gripping → done

        self.get_logger().info('PlanningNode 준비 완료')

    def on_command(self, msg: String):
        cmd = json.loads(msg.data)
        self.target_label = cmd.get('target_label')
        self.state = 'idle'
        self.get_logger().info(f'목표 설정: {self.target_label}')

    def on_objects(self, msg: String):
        if not self.target_label or self.state != 'idle':
            return

        objects = json.loads(msg.data).get('objects', [])
        for obj in objects:
            if obj['label'] == self.target_label:
                self.get_logger().info(f'물체 발견! {obj}')
                pos = obj['center_3d']
                self.target_position = pos
                self.state = 'moving'
                self.execute_pick(pos)
                break

    def execute_pick(self, pos: dict):
        """물체 집기 시퀀스: 접근 → 그리퍼 열기 → 내려가기 → 그리퍼 닫기 → 들어올리기"""

        # 1단계: 물체 위 10cm로 접근
        self.get_logger().info('1단계: 물체 위로 접근')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.10)

        # 2단계: 그리퍼 열기
        self.get_logger().info('2단계: 그리퍼 열기')
        self.send_gripper(GRIPPER_OPEN)

        # 3단계: 물체 위치로 내려가기
        self.get_logger().info('3단계: 물체로 내려가기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.02)

        # 4단계: 그리퍼 닫기 (집기)
        self.get_logger().info('4단계: 그리퍼 닫기 (집기)')
        self.send_gripper(GRIPPER_CLOSE)

        # 5단계: 들어올리기
        self.get_logger().info('5단계: 들어올리기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.15)

        self.state = 'done'
        self.target_label = None
        self.get_logger().info('픽업 시퀀스 완료!')

    def send_move(self, x: float, y: float, z: float):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 0.707
        pose.pose.orientation.y = 0.707
        self.pub_move.publish(pose)
        self.get_logger().info(f'move_p 발행: x={x:.2f} y={y:.2f} z={z:.2f}')

    def send_gripper(self, value: float):
        # TODO: HandCmd 필드 확인 후 실제 메시지로 교체
        # Jetson에서: ros2 interface show agx_arm_msgs/msg/HandCmd
        msg = String()
        msg.data = json.dumps({"gripper": value})
        self.pub_hand.publish(msg)
        self.get_logger().info(f'그리퍼 명령: {value}')

def main():
    rclpy.init()
    rclpy.spin(PlanningNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
