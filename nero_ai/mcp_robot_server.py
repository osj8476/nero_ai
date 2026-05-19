#!/usr/bin/env python3
"""
mcp_robot_server.py
AgileX Nero + Gripper 픽앤플레이스용 MCP 서버.
FastMCP 서버 안에서 ROS2 브리지 노드를 데몬 스레드로 spin.
pick_object / list_detected_objects Tool 호출 시 기존 /arm_command 토픽 발행.
기존 perception_node / planning_node 는 무변경.
"""

import os
import json
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from mcp.server.fastmcp import FastMCP


class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos)
        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0
        self.get_logger().info('RosBridgeNode 준비 완료')

    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get("objects", [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')

    def get_objects(self):
        with self._objects_lock:
            return list(self._latest_objects), self._last_obj_stamp

    def publish_arm_command(self, target_label: str, action: str = "pick"):
        payload = {"target_label": target_label, "action": action}
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'/arm_command 발행: {payload}')
        return payload


_ros_node: Optional[RosBridgeNode] = None
_ros_ready = threading.Event()


def _ros_spin_thread():
    global _ros_node
    rclpy.init()
    _ros_node = RosBridgeNode()
    _ros_ready.set()
    try:
        rclpy.spin(_ros_node)
    finally:
        _ros_node.destroy_node()
        rclpy.shutdown()


def _ensure_ros():
    if not _ros_ready.is_set():
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if _ros_node is None:
            raise RuntimeError("ROS2 bridge 노드 초기화 실패")


mcp = FastMCP("agilex-nero-pnp")


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전이 인식 중인 물체 목록을 조회한다.
    로봇에게 무언가를 집으라고 시키기 전에 반드시 먼저 호출해서
    실제로 어떤 물체가 있는지, 라벨이 무엇인지 확인하라.

    Returns:
        JSON 문자열. {"objects":[{"label":"cup","center_3d":{...}}], "age_sec":0.18}
        objects 가 빈 배열이면 인식된 물체가 없다는 뜻이다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [{"label": o.get("label", "?"),
             "center_3d": o.get("center_3d", {})} for o in objects]
    return json.dumps({"objects": slim, "age_sec": age}, ensure_ascii=False)


@mcp.tool()
def pick_object(target_label: str) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다 (pick 동작).
    내부적으로 planning_node 의 상태머신을 트리거한다.

    Args:
        target_label: 집을 물체 라벨. list_detected_objects() 가 반환한
                       objects 안에 존재하는 label 이어야 한다. 영어 소문자.

    Returns:
        성공: {"status":"dispatched","target_label":"...","action":"pick"}
        실패: {"status":"rejected","reason":"..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()
    objects, _ = _ros_node.get_objects()
    available = {o.get("label", "").lower() for o in objects}
    if available and target_label not in available:
        return json.dumps({
            "status": "rejected",
            "reason": f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                      f"인식된 물체: {sorted(available)}",
        }, ensure_ascii=False)
    payload = _ros_node.publish_arm_command(target_label, action="pick")
    return json.dumps({"status": "dispatched", **payload}, ensure_ascii=False)


@mcp.tool()
def get_system_status() -> str:
    """로봇/비전 브리지의 현재 연결 상태를 점검한다 (헬스체크용).

    Returns:
        {"ros_bridge":"up","vision_objects":3,"vision_age_sec":0.2}
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({
        "ros_bridge": "up",
        "vision_objects": len(objects),
        "vision_age_sec": age,
    }, ensure_ascii=False)


def main():
    _ensure_ros()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
