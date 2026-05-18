import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json

GRIPPER_OPEN  = 0.04  # 4cm 열림
GRIPPER_CLOSE = 0.0   # 완전 닫힘


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, 10)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command, 10)

        self.pub_move = self.create_publisher(PoseStamped, '/control/move_p', 10)
        self.pub_gripper = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.target_label = None
        self.state = 'idle'

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
                self.state = 'moving'
                self.execute_pick(obj['center_3d'])
                break

    def execute_pick(self, pos: dict):
        x, y, z = pos['x'], pos['y'], pos['z']

        # 1단계: 물체 위 10cm 접근
        self.get_logger().info('1단계: 물체 위로 접근')
        self.send_move(x, y, z + 0.10)

        # 2단계: 그리퍼 열기
        self.get_logger().info('2단계: 그리퍼 열기')
        self.send_gripper(GRIPPER_OPEN)

        # 3단계: 물체로 내려가기
        self.get_logger().info('3단계: 물체로 내려가기')
        self.send_move(x, y, z + 0.02)

        # 4단계: 그리퍼 닫기
        self.get_logger().info('4단계: 그리퍼 닫기')
        self.send_gripper(GRIPPER_CLOSE)

        # 5단계: 들어올리기
        self.get_logger().info('5단계: 들어올리기')
        self.send_move(x, y, z + 0.15)

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
        traj = JointTrajectory()
        traj.joint_names = ['gripper_joint1', 'gripper_joint2']
        pt = JointTrajectoryPoint()
        pt.positions = [value, value]
        pt.time_from_start = Duration(sec=1)
        traj.points = [pt]
        self.pub_gripper.publish(traj)
        self.get_logger().info(f'그리퍼: {"열기" if value > 0 else "닫기"} ({value}m)')


def main():
    rclpy.init()
    rclpy.spin(PlanningNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
