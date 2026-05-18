import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from google import genai
import json, os

class CommandParserNode(Node):
    def __init__(self):
        super().__init__('command_parser')

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY 환경변수가 없습니다!')
        self.client = genai.Client(api_key=api_key)

        self.pub = self.create_publisher(String, '/arm_command', 10)
        self.sub = self.create_subscription(
            String, '/user_command', self.on_command, 10)
        self.get_logger().info('CommandParser 준비 완료 (Gemini 모드)')

    def on_command(self, msg: String):
        user_text = msg.data
        self.get_logger().info(f'명령 수신: {user_text}')

        prompt = f"""
당신은 로봇 팔 제어 시스템입니다.
아래 JSON만 출력하세요. 다른 텍스트, 마크다운 절대 금지.
{{"target_label": "물체이름(영어 소문자)", "action": "pick"}}

명령: {user_text}
"""
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw = resp.text.strip().strip('```json').strip('```').strip()
            cmd = json.loads(raw)

            out = String()
            out.data = json.dumps(cmd)
            self.pub.publish(out)
            self.get_logger().info(f'파싱 결과: {cmd}')

        except Exception as e:
            self.get_logger().error(f'Gemini 오류: {e}')

def main():
    rclpy.init()
    rclpy.spin(CommandParserNode())
    rclpy.shutdown()
