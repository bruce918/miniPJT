"""경기도 버스 도착 정보 챗봇 서버.

질문은 LangGraph 그래프(chat_graph.py)로 보낸다. classify 노드가 질문을 분류해
버스 도착 에이전트(MCP 도구 사용) 또는 범위 밖 안내 노드로 보낸다.

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

from chat_graph import build_chat_graph, run_chat

BASE_DIR = Path(__file__).parent


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None  # classify 결과 (디버깅·테스트용)
    tables: list[dict] = []    # render_table 도구가 만든 표 (UI가 그린다. 각 표의 text는 평문 버전)


_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """MCP 서버에 처음 접속할 때 그래프를 만든다. 서버가 늦게 떠도 다음 요청에서 다시 시도한다."""
    global _graph
    async with _graph_lock:
        if _graph is None:
            _graph = await build_chat_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_graph()
        print("버스 MCP 서버 연결 완료, 챗 그래프 준비 완료")
    except RuntimeError as e:
        print(f"[경고] {e} (첫 질문 때 다시 시도합니다)")
    yield


app = FastAPI(title="miniPJT 버스 챗봇", lifespan=lifespan)


async def generate_reply(message: str, history: list[Message]) -> ChatResponse:
    try:
        graph = await get_graph()
    except RuntimeError as e:
        return ChatResponse(reply=str(e))

    try:
        turn = await run_chat(graph, message, [(m.role, m.content) for m in history])
    except Exception as e:
        print(f"그래프 오류: {type(e).__name__}: {e}")
        return ChatResponse(reply=f"조회 중 오류가 났습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요.")
    return ChatResponse(reply=turn.reply, intent=turn.intent, tables=turn.tables)


@app.get("/")
def index() -> FileResponse:
    # UI를 고친 뒤 브라우저가 예전 index.html을 쓰지 않도록 매번 다시 확인하게 한다
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="정류소 이름을 알려 주세요 (예: 미금역).")
    return await generate_reply(message[:500], req.history)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chat_graph": _graph is not None}
