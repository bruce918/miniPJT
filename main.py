"""경기도 버스 도착 정보 챗봇 서버.

질문은 LangGraph 그래프(chat_graph.py)로 보낸다. classify 노드가 질문을 분류해
버스 도착 에이전트(MCP 도구 사용), 정책 안내 에이전트(RAG 문서 검색) 또는 범위 밖 안내 노드로 보낸다.

실행:
    pip install -r requirements.txt
    1) 터미널 A: python bus_server.py          (MCP 서버, http://127.0.0.1:8765/mcp)
    2) 터미널 B: uvicorn main:app --reload     (챗봇)
브라우저에서 http://localhost:8000 접속.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from arrivals_api import refresh_arrivals
from chat_graph import build_chat_graph, run_chat
from observability import send_feedback, tracing_enabled

BASE_DIR = Path(__file__).parent


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    session_id: str | None = Field(default=None, max_length=64)  # 대화 ID (LangSmith Threads로 묶임)
    client: Literal["web", "test"] = "web"                       # 트레이스 구분용 (run_tests.py는 test)


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None  # classify 결과 (디버깅·테스트용)
    tables: list[dict] = []    # render_table 도구가 만든 표 (UI가 그린다. 각 표의 text는 평문 버전)
    sources: list[dict] = []   # 정책 답변의 근거 문서 출처 (retrieve_docs 결과)
    trace_id: str | None = None  # LangSmith 루트 실행 ID (피드백·디버깅용)


class FeedbackRequest(BaseModel):
    trace_id: UUID
    score: Literal[0, 1]  # 1=좋아요, 0=별로
    comment: str | None = Field(default=None, max_length=500)


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


def error_reply(e: Exception) -> str:
    """그래프 오류를 사용자에게 보여 줄 문장으로 바꾼다."""
    text = str(e)
    if type(e).__name__ == "ThrottlingException" or "ThrottlingException" in text:
        if "per day" in text:
            return ("오늘 쓸 수 있는 AI 모델 사용량을 모두 써서 지금은 답할 수 없어요. 한도가 풀리면 다시 시도해 주세요. "
                    "이미 띄운 도착 표의 ↻ 새로고침은 AI 없이 동작해요.")
        return "지금 요청이 몰려 답하지 못했어요. 1분쯤 뒤에 다시 시도해 주세요."
    return f"조회 중 오류가 났습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요."


async def generate_reply(message: str, history: list[Message], source: str, session_id: str | None) -> ChatResponse:
    try:
        graph = await get_graph()
    except RuntimeError as e:
        return ChatResponse(reply=str(e))

    try:
        turn = await run_chat(graph, message, [(m.role, m.content) for m in history],
                              source=source, session_id=session_id)
    except Exception as e:
        print(f"그래프 오류: {type(e).__name__}: {e}")
        return ChatResponse(reply=error_reply(e))
    return ChatResponse(reply=turn.reply, intent=turn.intent, tables=turn.tables, sources=turn.sources,
                        trace_id=turn.trace_id)


@app.get("/")
def index() -> FileResponse:
    # UI를 고친 뒤 브라우저가 예전 index.html을 쓰지 않도록 매번 다시 확인하게 한다
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="정류소 이름을 알려 주세요 (예: 미금역).")
    return await generate_reply(message[:500], req.history, req.client, req.session_id)


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """답변에 대한 사용자 피드백을 LangSmith의 해당 트레이스(chat_turn)에 붙인다."""
    if not tracing_enabled():
        raise HTTPException(503, "트레이싱이 꺼져 있어 피드백을 기록할 수 없습니다.")
    try:
        await asyncio.to_thread(send_feedback, str(req.trace_id), req.score, req.comment)
    except Exception as e:
        print(f"피드백 기록 실패: {type(e).__name__}: {e}")
        raise HTTPException(502, "피드백을 기록하지 못했습니다.")
    return {"ok": True}


@app.get("/api/arrivals")
async def arrivals(station_id: str = Query(pattern=r"^\d{9}$")) -> dict:
    """도착 표 새로고침 (↻ 버튼). 모델을 거치지 않고 MCP 도착 조회만 다시 해 표를 돌려준다."""
    try:
        return {"table": await refresh_arrivals(station_id)}
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chat_graph": _graph is not None, "tracing": tracing_enabled()}
