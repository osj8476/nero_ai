#!/usr/bin/env python3
"""
planning_node.py
- pick / place 시퀀스 분리
- 시퀀스 시작 직전 최신 물체 위치 갱신
- 완료/실패를 /pick_result 로 발행
- STEP_DELAYS 넉넉히 (피드백 토픽 없으므로 보수적)
"""

import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5

# 보수적 대기시간 (실제 로봇 피드백 없으므로 넉넉히)
STEP_DELAYS = {
    'approach':   5.0,
    'open_hand':  2.0,
    'descend':    3.5,
    'close_hand': 2.0,
    'lift':       3.5,
    'transit':    5.0,
    'release':    2.0,
}


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_perception = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, qos_perception)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command, qos_cmd)

        self.pub_move = self.create_publisher(
            PoseStamped, '/control/move_p', qos_cmd)
        self.pub_gripper = self.create_publisher(
            JointState, '/control/joint_states', qos_cmd)
        self.pub_result = self.create_publisher(
            String, '/pick_result', qos_cmd)

        # 최신 인식 결과 캐시 (시퀀스 시작 직전 사용)
        self.latest_objects: list = []

        # 현재 시퀀스 상태
        self.target_label = None
        self.target_pos   = None
        self.action       = None
        self.place_pos    = None
        self.state        = 'idle'
        self.step_timer   = None

        self.get_logger().info('PlanningNode 준비 완료')

    # ── 토픽 콜백 ────────────────────────────
    def on_objects(self, msg: String):
        """항상 최신 인식 결과 캐시. 시퀀스 트리거는 on_command에서만."""
        try:
            self.latest_objects = json.loads(msg.data).get('objects', [])
        except json.JSONDecodeError:
            pass

    def on_command(self, msg: String):
        if self.state != 'idle':
            self.get_logger().warn(
                f'이미 작업 진행 중 (state={self.state}). 명령 무시.')
            self._publish_result("failed", "busy")
            return

        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self._publish_result("failed", "invalid_command_json")
            return

        action = cmd.get('action', 'pick')

        # ────────────────────────────────────────────────────────────────
        # [BUG FIX] self.action 설정 전에 실패하면 _publish_result에서
        # self.action이 None인 채로 발행되던 문제 수정.
        # action을 로컬 변수로 먼저 검증한 뒤, 성공 확정 시점에만
        # self.action에 할당한다.
        # ────────────────────────────────────────────────────────────────
        if action == 'pick':
            label = cmd.get('target_label')
            pos = self._find_object(label)
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음. 픽업 취소.")
                # self.action 은 여전히 None → 직접 action 값 넘겨서 발행
                self._publish_result("failed", f"object_not_found:{label}",
                                     action_override=action)
                return
            # 검증 완료 후 상태 기록
            self.action       = action
            self.target_label = label
            self.target_pos   = pos
            self.get_logger().info(f'PICK 시작: {label} @ {pos}')
            self._step_approach()

        elif action == 'place':
            place_pos = cmd.get('place_pos')
            if place_pos is None:
                self._publish_result("failed", "no_place_pos",
                                     action_override=action)
                return
            # 검증 완료 후 상태 기록
            self.action    = action
            self.place_pos = place_pos
            self.get_logger().info(f'PLACE 시작 @ {place_pos}')
            self._step_transit_to_place()

        else:
            self._publish_result("failed", f"unknown_action:{action}",
                                 action_override=action)

    def _find_object(self, label: str):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None

    # ── PICK 시퀀스 ──────────────────────────
    def _step_approach(self):
        pos = self.target_pos
        self.state = 'approach'
        self.get_logger().info('1/5: 물체 위로 접근')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.10)
        self._schedule(STEP_DELAYS['approach'], self._step_open)

    def _step_open(self):
        self.state = 'open_hand'
        self.get_logger().info('2/5: 그리퍼 열기')
        self.send_gripper(GRIPPER_OPEN)
        self._schedule(STEP_DELAYS['open_hand'], self._step_descend)

    def _step_descend(self):
        pos = self.target_pos
        self.state = 'descend'
        self.get_logger().info('3/5: 물체로 내려가기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.01)
        self._schedule(STEP_DELAYS['descend'], self._step_close)

    def _step_close(self):
        self.state = 'close_hand'
        self.get_logger().info('4/5: 그리퍼 닫기 (집기)')
        self.send_gripper(GRIPPER_CLOSE, force=GRIPPER_FORCE)
        self._schedule(STEP_DELAYS['close_hand'], self._step_lift)

    def _step_lift(self):
        pos = self.target_pos
        self.state = 'lift'
        self.get_logger().info('5/5: 들어올리기')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.20)
        self._schedule(STEP_DELAYS['lift'], self._pick_done)

    def _pick_done(self):
        self.get_logger().info('✅ PICK 완료')
        self._publish_result("success", "pick_complete")
        self._reset()

    # ── PLACE 시퀀스 ─────────────────────────
    def _step_transit_to_place(self):
        pos = self.place_pos
        self.state = 'transit'
        self.get_logger().info('1/2: place 위치로 이동')
        self.send_move(pos['x'], pos['y'], pos['z'] + 0.10)
        self._schedule(STEP_DELAYS['transit'], self._step_release)

    def _step_release(self):
        self.state = 'release'
        self.get_logger().info('2/2: 그리퍼 열기 (놓기)')
        self.send_gripper(GRIPPER_OPEN)
        self._schedule(STEP_DELAYS['release'], self._place_done)

    def _place_done(self):
        self.get_logger().info('✅ PLACE 완료')
        self._publish_result("success", "place_complete")
        self._reset()

    # ── 공통 ────────────────────────────────
    def _reset(self):
        self.state        = 'idle'
        self.target_label = None
        self.target_pos   = None
        self.action       = None
        self.place_pos    = None
        if self.step_timer:
            self.step_timer.cancel()
            self.step_timer = None

    def _publish_result(self, status: str, reason: str = "",
                        action_override: str = None):
        # ────────────────────────────────────────────────────────────────
        # [BUG FIX] action_override: on_command에서 self.action 할당 전
        # 실패(early return) 시 action 값을 직접 전달받아 사용.
        # self.action이 None인 채로 발행되는 문제 방지.
        # ────────────────────────────────────────────────────────────────
        action_to_report = action_override if action_override is not None else self.action
        msg = String()
        msg.data = json.dumps({
            "status": status,
            "reason": reason,
            "action": action_to_report,
        }, ensure_ascii=False)
        self.pub_result.publish(msg)

    def _schedule(self, delay: float, callback):
        if self.step_timer:
            self.step_timer.cancel()
        self.step_timer = self.create_timer(delay, lambda: self._fire(callback))

    def _fire(self, callback):
        self.step_timer.cancel()
        self.step_timer = None
        callback()

    # ── 발행 함수 ───────────────────────────
    def send_move(self, x: float, y: float, z: float):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        # ⚠️ TODO: end-effector orientation 하드코딩.
        # 실제 로봇 설치 방향에 따라 아래 값을 조정해야 함.
        # (수직 하향 그리퍼 기준 예시값 — 환경에 맞게 변경 필요)
        pose.pose.orientation.w = 0.707
        pose.pose.orientation.y = 0.707

        self.pub_move.publish(pose)
        self.get_logger().info(f'move_p: x={x:.3f} y={y:.3f} z={z:.3f}')

    def send_gripper(self, width: float, force: float = 1.0):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['gripper']
        msg.position = [float(width)]
        msg.effort   = [float(force)]
        self.pub_gripper.publish(msg)
        self.get_logger().info(
            f'그리퍼: {"열기" if width > 0.05 else "닫기"} '
            f'(width={width:.3f}, force={force:.1f})')


def main():
    rclpy.init()
    try:
        rclpy.spin(PlanningNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
