# bus_agent.py - 경기도 실시간 버스 정보를 MCP 서버(bus_server.py)에서 조회하는 LangGraph Agent
"""먼저 다른 터미널에서 python bus_server.py 로 MCP 서버를 띄워 두세요."""
import asyncio
import os
import sys

from dotenv import load_dotenv
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 독립 실행 중인 MCP 서버 주소 (bus_server.py의 기본값과 같음)
BUS_MCP_URL = os.getenv("BUS_MCP_URL", "http://127.0.0.1:8765/mcp")

# 도구 2개를 호출해 요약하는 단순한 작업이라 가볍고 한도가 따로 잡히는 Haiku를 기본으로 씁니다.
# 다른 모델을 쓰려면 .env에 BUS_MODEL_ID를 넣으세요 (예: us.anthropic.claude-sonnet-4-5-20250929-v1:0)
BUS_MODEL_ID = os.getenv("BUS_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

SYSTEM_PROMPT = """경기도 버스 안내 봇입니다. 도구 결과에만 근거해 짧게 답하세요.
- 정류소 이름이 나오면 존재 여부를 짐작하지 말고 반드시 search_bus_station으로 ID부터 찾기
- 정류소의 버스 전체: get_bus_arrival_list(정류소 ID 9자리)
- 한 노선: get_bus_arrival_item(목록 결과의 routeId, staOrder 사용)
검색 결과가 2건 이상이면 도착 정보를 조회하지 말고, 후보를 번호 목록(이름, 정류소번호, 시군)으로 보여 주며 하나를 고르게 다시 물어보세요. 사용자가 번호나 정류소번호로 고르면 그 정류소만 조회하세요. 1건일 때만 바로 조회하세요.
predictTime은 도착까지 분이고 빈 값은 정보 없음이니, 정보 있는 노선 위주로 답하세요."""


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


async def build_agent(prompt: str = SYSTEM_PROMPT, extra_tools=(), **agent_kwargs):
    """MCP 서버에 접속해 도구를 받고, 그 도구로 ReAct 에이전트를 만들어 (agent, tools)를 돌려줍니다.

    에이전트는 공공데이터포털이나 API 키를 전혀 모르고, 서버가 노출한 도구만 호출합니다.
    웹 챗봇(chat_graph.py) 같은 입출력 쪽은 이 함수만 불러다 쓰면 됩니다.
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
            "다른 터미널에서 'python bus_server.py'로 서버를 먼저 실행했는지 확인하세요."
        ) from e

    tools = [*tools, *extra_tools]
    return create_react_agent(make_llm(), tools, prompt=prompt, **agent_kwargs), tools


def make_llm(max_tokens: int = 600, model_id: str | None = None) -> ChatBedrockConverse:
    """Bedrock Claude 모델. 에이전트·분류기(chat_graph.classify)·평가 판사(evaluate.py)가 같이 씁니다."""
    return ChatBedrockConverse(
        model=model_id or BUS_MODEL_ID,
        region_name="us-east-1",
        max_tokens=max_tokens,   # 답변이 길어지지 않게 출력 상한을 둡니다
        # 한도 초과(ThrottlingException)에는 간격을 늘려 재시도합니다
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
    )


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
