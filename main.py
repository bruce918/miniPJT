"""경기도 버스 도착 정보 챗봇 서버.

정류소 이름을 말하면 버스 MCP 서버(bus_server.py)의 도구로 정류소를 찾고 도착 정보를 알려 준다.

실행:
    pip install -r requirements.txt
    1) 터미널 A: python bus_server.py          (MCP 서버, http://127.0.0.1:8765/mcp)
    2) 터미널 B: uvicorn main:app --reload     (챗봇)
브라우저에서 http://localhost:8000 접속.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from bus_agent import SYSTEM_PROMPT, build_agent, get_text

BASE_DIR = Path(__file__).parent

WEB_PROMPT = SYSTEM_PROMPT + """

[웹 챗봇 답변 형식]
- 일반 텍스트로 짧게 답하세요. 마크다운(**, #, 표)은 쓰지 마세요. 줄바꿈과 '-' 목록은 괜찮습니다.
- 노선마다 한 줄로: 노선번호(행선지) N분 후, M정류장 전. 다음 차가 있으면 이어서 적으세요.
- 후보 목록은 한 줄에 하나씩 '번호. 이름 (시군, 정류소번호 N, ID 9자리)' 형식으로 쓰세요.
  다음 질문에서는 이전 도구 결과가 보이지 않으므로, 사용자가 고르면 목록에 적힌 ID로 바로 get_bus_arrival_list를 호출하세요.
- 버스와 관계없는 질문에는 경기도 버스 도착 정보만 안내할 수 있다고 답하세요.
- 정류소 이름을 모르면 '정류소 이름을 알려 주세요 (예: 미금역)'라고 물어보세요."""

KEEP_LAST_MESSAGES = 6  # 모델에 보내는 이전 대화 상한 (토큰 절약)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    reply: str


_agent = None
_agent_lock = asyncio.Lock()


async def get_agent():
    """MCP 서버에 처음 접속할 때 에이전트를 만든다. 서버가 늦게 떠도 다음 요청에서 다시 시도한다."""
    global _agent
    async with _agent_lock:
        if _agent is None:
            _agent, _ = await build_agent(WEB_PROMPT)
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_agent()
        print("버스 MCP 서버 연결 완료")
    except RuntimeError as e:
        print(f"[경고] {e} (첫 질문 때 다시 시도합니다)")
    yield


app = FastAPI(title="miniPJT 버스 챗봇", lifespan=lifespan)


async def generate_reply(message: str, history: list[Message]) -> str:
    try:
        agent = await get_agent()
    except RuntimeError as e:
        return str(e)

    # 브라우저가 보내 준 대화 기록으로 맥락(후보 정류소 번호 선택 등)을 이어간다
    recent = [m for m in history if m.role in ("user", "assistant") and m.content][-KEEP_LAST_MESSAGES:]
    while recent and recent[0].role != "user":
        recent.pop(0)
    messages = [(m.role, m.content) for m in recent] + [("user", message)]

    try:
        result = await agent.ainvoke({"messages": messages}, {"recursion_limit": 12})
    except Exception as e:
        print(f"에이전트 오류: {type(e).__name__}: {e}")
        return f"조회 중 오류가 났습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요."
    return get_text(result["messages"][-1]) or "답변을 만들지 못했습니다."


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="정류소 이름을 알려 주세요 (예: 미금역).")
    return ChatResponse(reply=await generate_reply(message[:500], req.history))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "bus_agent": _agent is not None}
