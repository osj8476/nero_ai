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

        self.pub     = self.create_publisher(String, '/arm_command', 10)
        self.sub_cmd = self.create_subscription(
            String, '/user_command', self.on_command, 10)
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects, 10)

        self.latest_objects = []
        self.get_logger().info('CommandParser 준비 완료 (Gemini + MoonDream2 모드)')

    def on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.latest_objects = data.get("objects", [])
        except json.JSONDecodeError:
            pass

    def on_command(self, msg: String):
        user_text = msg.data
        self.get_logger().info(f'명령 수신: {user_text}')

        if self.latest_objects:
            scene_info = ", ".join([
                f"{o.get('label','?')} at {o.get('center_3d', {})}"
                for o in self.latest_objects
            ])
        else:
            scene_info = "장면 정보 없음"

        prompt = f"""당신은 로봇 팔 제어 시스템입니다.
아래 JSON만 출력하세요. 다른 텍스트, 마크다운 절대 금지.
{{"target_label": "물체이름(영어 소문자)", "action": "pick"}}

현재 장면의 물체: {scene_info}
명령: {user_text}

장면에 있는 물체 중에서만 선택하세요."""

        # 수정: resp 스코프 버그 수정
        resp = None
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw = resp.text.strip().strip('```json').strip('```').strip()
            cmd = json.loads(raw)

            out      = String()
            out.data = json.dumps(cmd)
            self.pub.publish(out)
            self.get_logger().info(f'파싱 결과: {cmd}')

        except json.JSONDecodeError as e:
            raw_text = resp.text if resp else "응답 없음"
            self.get_logger().error(f'JSON 파싱 실패: {e} / 원문: {raw_text}')
        except Exception as e:
            self.get_logger().error(f'Gemini 오류: {e}')


def main():
    rclpy.init()
    rclpy.spin(CommandParserNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
