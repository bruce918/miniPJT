"""도착 표 새로고침용: 모델을 거치지 않고 버스 MCP 서버의 get_bus_arrival_list를 바로 불러 표를 다시 만든다.

화면의 도착 표 ↻ 버튼 -> GET /api/arrivals?station_id=... -> 여기.
모델(LLM)을 부르지 않으므로 빠르고, 대화 기록에 같은 질문이 쌓이지 않는다.
표는 채팅 응답과 같은 함수(table_tool.arrivals_table)로 만들어 모양이 같다.
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from bus_agent import BUS_MCP_URL
from table_tool import _text, arrivals_table_from_text

_tool = None
_lock = asyncio.Lock()


async def _arrival_tool():
    """MCP 서버에서 get_bus_arrival_list 도구를 한 번 받아 재사용한다."""
    global _tool
    async with _lock:
        if _tool is None:
            client = MultiServerMCPClient({"gg-bus": {"url": BUS_MCP_URL, "transport": "streamable_http"}})
            tools = await client.get_tools()
            _tool = next(t for t in tools if t.name == "get_bus_arrival_list")
    return _tool


async def refresh_arrivals(station_id: str) -> dict:
    """정류소 ID의 도착 표. 실패하면 RuntimeError(사용자에게 보여 줄 문장)."""
    try:
        tool = await _arrival_tool()
        result = await tool.ainvoke({"station_id": station_id})
    except Exception as e:
        raise RuntimeError(f"도착 정보를 다시 불러오지 못했어요 ({type(e).__name__}).") from e
    raw = _text(result[0] if isinstance(result, tuple) else result)
    table = arrivals_table_from_text(station_id, raw)
    if table is None:
        raise RuntimeError(raw.split(".")[0][:80] or "도착 정보를 다시 불러오지 못했어요.")
    return table
