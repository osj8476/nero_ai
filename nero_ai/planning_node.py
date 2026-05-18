import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, 10)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command, 10)
        self.pub = self.create_publisher(
            PoseStamped, '/control/move_p', 10)
        self.target_label = None
        self.get_logger().info('PlanningNode 준비 완료')

    def on_command(self, msg: String):
        cmd = json.loads(msg.data)
        self.target_label = cmd.get('target_label')
        self.get_logger().info(f'목표 설정: {self.target_label}')

    def on_objects(self, msg: String):
        if not self.target_label:
            return
        objects = json.loads(msg.data).get('objects', [])
        for obj in objects:
            if obj['label'] == self.target_label:
                self.get_logger().info(f'물체 발견! {obj}')
                self.send_move(obj['center_3d'])
                self.target_label = None
                break

    def send_move(self, pos: dict):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pos['x']
        pose.pose.position.y = pos['y']
        pose.pose.position.z = pos['z'] + 0.05
        pose.pose.orientation.w = 0.707
        pose.pose.orientation.y = 0.707
        self.pub.publish(pose)
        self.get_logger().info(f'move_p 발행: x={pos["x"]:.2f} y={pos["y"]:.2f} z={pos["z"]:.2f}')

def main():
    rclpy.init()
    rclpy.spin(PlanningNode())
    rclpy.shutdown()
