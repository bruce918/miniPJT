"""버스 도착 정보 에이전트: 버스 MCP 서버의 도구로 ReAct 에이전트를 만든다.

먼저 다른 터미널에서 `python -m src.tools.bus_mcp.server`로 MCP 서버를 띄워 두세요.
단독 실행(정류소 ID를 물어 조회하는 콘솔): `python -m src.agents.bus_arrival.agent`
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from src.agents.bus_arrival.prompts import BUS_PROMPT, SYSTEM_PROMPT
from src.shared.llm import get_text, make_llm

load_dotenv()

# 독립 실행 중인 MCP 서버 주소 (src/tools/bus_mcp/server.py의 기본값과 같음)
BUS_MCP_URL = os.getenv("BUS_MCP_URL", "http://127.0.0.1:8765/mcp")


async def build_agent(prompt: str = SYSTEM_PROMPT, extra_tools=(), **agent_kwargs):
    """MCP 서버에 접속해 도구를 받고, 그 도구로 ReAct 에이전트를 만들어 (agent, tools)를 돌려줍니다.

    에이전트는 공공데이터포털이나 API 키를 전혀 모르고, 서버가 노출한 도구만 호출합니다.
    채팅 워크플로(src/workflows/chat)는 build_bus_agent()를 쓴다.
    extra_tools: MCP 도구 외에 붙일 로컬 도구 (예: 표를 그리는 render_table)
    """
    client = MultiServerMCPClient({
        "gg-bus": {"url": BUS_MCP_URL, "transport": "streamable_http"},
    })
    try:
        tools = await client.get_tools()
    except Exception as e:
        raise RuntimeError(
            f"버스 MCP 서버({BUS_MCP_URL})에 연결하지 못했습니다 ({type(e).__name__}). "
            "다른 터미널에서 'python -m src.tools.bus_mcp.server'로 서버를 먼저 실행했는지 확인하세요."
        ) from e

    tools = [*tools, *extra_tools]
    return create_react_agent(make_llm(), tools, prompt=prompt, **agent_kwargs), tools


async def build_bus_agent():
    """채팅 워크플로용 버스 도착 에이전트 (BUS_PROMPT + MCP 도구 + render_table)."""
    from src.tools.table.tool import render_table
    agent, _ = await build_agent(BUS_PROMPT, extra_tools=[render_table], name="bus_agent")
    return agent


async def main():
    try:
        agent, tools = await build_agent()
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"MCP 도구 {len(tools)}개: {[t.name for t in tools]}")

    # 사용자에게 정류소 ID를 묻고, 조회 결과를 알려주는 것을 반복합니다 (빈 입력이나 q로 종료)
    while True:
        station_id = (await asyncio.to_thread(input, "\n정류소 ID를 입력하세요 (종료: q): ")).strip()
        if station_id.lower() in ("", "q", "quit", "exit"):
            break
        if not station_id.isdigit():
            print("정류소 ID는 숫자(9자리)입니다. 다시 입력해 주세요.")
            continue

        question = f"정류소 ID {station_id}에 지금 도착 예정인 버스를 알려주세요."
        async for event in agent.astream({"messages": [("user", question)]}, stream_mode="updates"):
            for node, update in event.items():
                for m in update.get("messages", []):
                    if getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            print(f"  도구 호출: {tc['name']}({tc['args']})")
                    elif m.type == "tool":
                        print(f"  도구 결과: [{m.name}] {get_text(m)[:80]!r}")
                    elif m.content:
                        print(f"\n답변: {get_text(m)}")


if __name__ == "__main__":
    asyncio.run(main())
