import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import json

# 그리퍼 설정 (agx_gripper 기준)
GRIPPER_OPEN  = 0.08   # 8cm 열림 (최대 0.1m)
GRIPPER_CLOSE = 0.01   # 거의 닫힘
GRIPPER_FORCE = 1.5    # 집는 힘 (0.5~3.0N)

# 픽업 각 단계 대기시간 (초)
STEP_DELAYS = {
    'approach':    3.0,
    'open_hand':   1.5,
    'descend':     2.5,
    'close_hand':  1.5,
    'lift':        2.5,
}

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, 10)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command, 10)

        # 이동 제어 (point-to-point)
        self.pub_move = self.create_publisher(
            PoseStamped, '/control/move_p', 10)

        # 그리퍼 제어 (/control/joint_states — 공식 방식)
        self.pub_gripper = self.create_publisher(
            JointState, '/control/joint_states', 10)

        self.target_label = None
        self.target_pos   = None
        self.state        = 'idle'
        self.step_timer   = None

        self.get_logger().info('PlanningNode 준비 완료')

    # ── 명령 수신 ──────────────────────────────
    def on_command(self, msg: String):
        cmd = json.loads(msg.data)
        self.target_label = cmd.get('target_label')
        self.state = 'idle'
        self.get_logger().info(f'목표 설정: {self.target_label}')

    # ── 물체 인식 수신 ─────────────────────────
    def on_objects(self, msg: String):
        if not self.target_label or self.state != 'idle':
            return
        objects = json.loads(msg.data).get('objects', [])
        for obj in objects:
            if obj['label'] == self.target_label:
                self.get_logger().info(f'물체 발견! {obj}')
                self.target_pos   = obj['center_3d']
                self.target_label = None
                self._step_approach()
                break

    # ── 픽업 시퀀스 (타이머 상태머신) ────────────
    def _step_approach(self):
        pos = self.target_pos
        self.state = 'approach'
        self.get_logger().info('1단계: 물체 위로 접근')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.10)
        self._schedule(STEP_DELAYS['approach'], self._step_open)

    def _step_open(self):
        self.state = 'open_hand'
        self.get_logger().info('2단계: 그리퍼 열기')
        self.send_gripper(GRIPPER_OPEN)
        self._schedule(STEP_DELAYS['open_hand'], self._step_descend)

    def _step_descend(self):
        pos = self.target_pos
        self.state = 'descend'
        self.get_logger().info('3단계: 물체로 내려가기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.01)
        self._schedule(STEP_DELAYS['descend'], self._step_close)

    def _step_close(self):
        self.state = 'close_hand'
        self.get_logger().info('4단계: 그리퍼 닫기 (집기)')
        self.send_gripper(GRIPPER_CLOSE, force=GRIPPER_FORCE)
        self._schedule(STEP_DELAYS['close_hand'], self._step_lift)

    def _step_lift(self):
        pos = self.target_pos
        self.state = 'lift'
        self.get_logger().info('5단계: 들어올리기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.20)
        self._schedule(STEP_DELAYS['lift'], self._step_done)

    def _step_done(self):
        self.state      = 'idle'
        self.target_pos = None
        self.get_logger().info('✅ 픽업 시퀀스 완료!')

    # ── 타이머 헬퍼 ───────────────────────────
    def _schedule(self, delay: float, callback):
        if self.step_timer:
            self.step_timer.cancel()
        self.step_timer = self.create_timer(delay, lambda: self._fire(callback))

    def _fire(self, callback):
        self.step_timer.cancel()
        self.step_timer = None
        callback()

    # ── 발행 함수 ─────────────────────────────
    def send_move(self, x: float, y: float, z: float):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 0.707
        pose.pose.orientation.y = 0.707
        self.pub_move.publish(pose)
        self.get_logger().info(f'move_p: x={x:.3f} y={y:.3f} z={z:.3f}')

    def send_gripper(self, width: float, force: float = 1.0):
        """
        width: 0.0 ~ 0.1 (m)
        force: 0.5 ~ 3.0 (N)
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['gripper']
        msg.position = [float(width)]
        msg.effort   = [float(force)]
        self.pub_gripper.publish(msg)
        self.get_logger().info(
            f'그리퍼: {"열기" if width > 0.05 else "닫기"} '
            f'(width={width:.3f}m, force={force:.1f}N)')


def main():
    rclpy.init()
    rclpy.spin(PlanningNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
