#!/usr/bin/env python3
"""
command_parser.py (MCP 클라이언트 버전 — 전면 개편)
억지 프롬프트/JSON 파싱 제거. Gemini SDK 에 MCP 세션을 tools=[session] 로
전달 → Automatic Function Calling 으로 자연어 → Tool 자동 호출.
/user_command 수신 → Gemini+MCP → (서버가) /arm_command 발행.
"""

import os
import asyncio
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=[os.environ.get(
        "MCP_SERVER_PATH",
        os.path.join(os.path.dirname(__file__), "mcp_robot_server.py"),
    )],
    env=os.environ.copy(),
)

SYSTEM_INSTRUCTION = (
    "너는 AgileX Nero 로봇 팔 픽앤플레이스 시스템의 두뇌다. "
    "사용자의 한국어/영어 자연어 명령을 받아 적절한 도구를 호출해 로봇을 제어한다. "
    "물체를 집으라는 요청을 받으면, 먼저 list_detected_objects 로 현재 장면에 "
    "어떤 물체가 있는지 확인한 뒤, 그 안의 라벨로 pick_object 를 호출하라. "
    "장면에 없는 물체를 요청받으면 집으려 시도하지 말고 사용자에게 알려라."
)


class CommandParserNode(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__('command_parser')
        self._loop = loop
        self.sub_cmd = self.create_subscription(
            String, '/user_command', self.on_command, 10)
        self.pub_reply = self.create_publisher(String, '/assistant_reply', 10)
        self.get_logger().info(
            'CommandParser 준비 완료 (Gemini + MCP Tool Call 모드)')

    def on_command(self, msg: String):
        user_text = msg.data
        self.get_logger().info(f'명령 수신: {user_text}')
        asyncio.run_coroutine_threadsafe(
            self._handle(user_text), self._loop)

    async def _handle(self, user_text: str):
        try:
            reply = await run_llm_turn(user_text)
            self.get_logger().info(f'처리 완료: {reply}')
            out = String()
            out.data = reply
            self.pub_reply.publish(out)
        except Exception as e:
            self.get_logger().error(f'LLM/MCP 처리 오류: {e}')


_session: ClientSession | None = None
_session_ready = asyncio.Event()
_gemini: genai.Client | None = None


async def _session_owner():
    global _session
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _session = session
            _session_ready.set()
            await asyncio.Event().wait()


async def run_llm_turn(user_text: str) -> str:
    await _session_ready.wait()
    response = await _gemini.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_text,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[_session],
        ),
    )
    return response.text or "(완료)"


def _ros_spin(node: Node):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


async def main_async():
    global _gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다!")
    _gemini = genai.Client(api_key=api_key)

    owner_task = asyncio.create_task(_session_owner())
    await _session_ready.wait()

    rclpy.init()
    loop = asyncio.get_running_loop()
    node = CommandParserNode(loop)
    spin_thread = threading.Thread(
        target=_ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    node.get_logger().info('MCP 세션 연결 완료. /user_command 대기 중...')

    try:
        await asyncio.Event().wait()
    finally:
        owner_task.cancel()
        node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
