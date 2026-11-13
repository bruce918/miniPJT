"""경기도 버스 도착 정보 챗봇 서버.

질문은 LangGraph 워크플로(src/workflows/chat)로 보낸다. 계획 노드가 질문을 단계로 나눠
버스 도착 에이전트(MCP 도구 사용), 정책 안내 에이전트(RAG 문서 검색) 또는 범위 밖 안내 노드로 보낸다.

실행:
    pip install -r requirements.txt
    1) 터미널 A: python -m src.tools.bus_mcp.server        (MCP 서버, http://127.0.0.1:8765/mcp)
    2) 터미널 B: uvicorn src.channels.web.app:app --reload     (챗봇)
브라우저에서 http://localhost:8000 접속.
"""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware

from src.auth import router as auth
from src.channels.web.schemas import ChatRequest, ChatResponse, FeedbackRequest, Message
from src.guardrails import filters as guardrails
from src.memory import store as memory
from src.shared.observability import send_feedback, tracing_enabled
from src.tools.bus_mcp.client import refresh_arrivals
from src.workflows.chat.graph import build_chat_graph, run_chat

STATIC_DIR = Path(__file__).parent / "static"


def ui_version() -> str:
    """화면 파일(index.html)의 버전. 열려 있는 페이지가 예전 화면이면 새로고침을 안내하는 데 쓴다."""
    return hashlib.sha256((STATIC_DIR / "index.html").read_bytes()).hexdigest()[:12]


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
    auth.init_db()
    if not auth.enabled():
        print("[안내] GOOGLE_CLIENT_ID/SECRET·SESSION_SECRET이 없어 로그인 없이 실행합니다 (즐겨찾기는 브라우저에만 저장).")
    try:
        await get_graph()
        print("버스 MCP 서버 연결 완료, 챗 그래프 준비 완료")
    except RuntimeError as e:
        print(f"[경고] {e} (첫 질문 때 다시 시도합니다)")
    yield


app = FastAPI(title="miniPJT 버스 챗봇", lifespan=lifespan)
# 구글 로그인 과정의 임시 값(state·nonce·PKCE)만 담는 서명 쿠키. 로그인 상태 자체는 auth의 세션 쿠키가 맡는다.
app.add_middleware(SessionMiddleware, secret_key=auth.SESSION_SECRET or "login-disabled",
                   session_cookie="ggbus_oauth", max_age=600, same_site="lax")
app.include_router(auth.router)


def error_reply(e: Exception) -> str:
    """그래프 오류를 사용자에게 보여 줄 문장으로 바꾼다."""
    text = str(e)
    if type(e).__name__ == "ThrottlingException" or "ThrottlingException" in text:
        if "per day" in text:
            return ("오늘 쓸 수 있는 AI 모델 사용량을 모두 써서 지금은 답할 수 없어요. 한도가 풀리면 다시 시도해 주세요. "
                    "이미 띄운 도착 표의 ↻ 새로고침은 AI 없이 동작해요.")
        return "지금 요청이 몰려 답하지 못했어요. 1분쯤 뒤에 다시 시도해 주세요."
    return f"조회 중 오류가 났습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요."


def memory_key(request: Request, client: str, session_id: str | None) -> str | None:
    """장기 메모리 키: 로그인한 웹 사용자와 텔레그램 채팅만 기억한다 (비로그인 웹·테스트는 기억하지 않음)."""
    if client == "test":
        return None
    user = auth.current_user(request)
    if user:
        return f"user:{user['id']}"
    if client == "telegram" and session_id:
        return f"tg:{session_id}"
    return None


async def generate_reply(message: str, history: list[Message], source: str, session_id: str | None,
                         user_key: str | None = None) -> ChatResponse:
    # 입력 가드레일: 인젝션은 모델을 부르지 않고 막고, 개인정보는 가린 문장으로만 모델·트레이스에 보낸다
    checked = guardrails.check_input(message)
    if checked.blocked:
        print(f"[가드레일] 프롬프트 인젝션 차단 ({checked.blocked}, source={source})")  # 질문 내용은 남기지 않는다
        return ChatResponse(reply=guardrails.BLOCKED_REPLY, intent="blocked",
                            guard={"blocked": True, "pii": checked.pii, "masked": checked.text})
    past, dropped = guardrails.clean_history([(m.role, m.content) for m in history])
    if dropped:
        print(f"[가드레일] 대화 기록에서 인젝션 메시지 {dropped}건 제외 (source={source})")

    try:
        graph = await get_graph()
    except RuntimeError as e:
        return ChatResponse(reply=str(e))

    try:
        turn = await run_chat(graph, checked.text, past, source=source, session_id=session_id,
                              guard_tags=[f"guard:pii:{k}" for k in checked.pii]
                              + (["guard:history_dropped"] if dropped else []),
                              user_key=user_key)
    except Exception as e:
        print(f"그래프 오류: {type(e).__name__}: {e}")
        return ChatResponse(reply=error_reply(e))
    guard = {"blocked": False, "pii": checked.pii, "masked": checked.text} if checked.pii else None
    return ChatResponse(reply=turn.reply, intent=turn.intent, tables=turn.tables, sources=turn.sources,
                        trace_id=turn.trace_id, guard=guard, parts=turn.parts, plan=turn.plan)


@app.get("/")
def index() -> FileResponse:
    # UI를 고친 뒤 브라우저가 예전 index.html을 쓰지 않도록 매번 다시 확인하게 한다
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="정류소 이름을 알려 주세요 (예: 미금역).", ui_version=ui_version())
    res = await generate_reply(message[:500], req.history, req.client, req.session_id,
                               memory_key(request, req.client, req.session_id))
    res.ui_version = ui_version()
    return res


def _memory_owner(request: Request, session_id: str | None) -> str:
    """기억을 보거나 지울 대상: 로그인 사용자, 아니면 텔레그램 채팅(session_id가 tg-로 시작)."""
    user = auth.current_user(request)
    if user:
        return f"user:{user['id']}"
    if session_id and session_id.startswith("tg-"):
        return f"tg:{session_id}"
    raise HTTPException(401, "로그인이 필요합니다.")


@app.get("/api/memory")
async def get_memory(request: Request, session_id: str | None = Query(default=None, max_length=64)) -> dict:
    """버스봇이 기억하는 내용 (장기 메모리)."""
    mem = await memory.load(await memory.get_store(), _memory_owner(request, session_id))
    return {"summary": memory.summarize(mem), "profile": mem.get("profile") or {},
            "stations": [{"id": k, "name": v.get("name"), "count": v.get("count")}
                         for k, v in sorted((mem.get("stations") or {}).items(), key=lambda kv: -kv[1].get("count", 0))]}


@app.delete("/api/memory", dependencies=[Depends(auth.same_origin)])
async def delete_memory(request: Request, session_id: str | None = Query(default=None, max_length=64)) -> dict:
    """버스봇의 기억을 모두 지운다."""
    removed = await memory.forget(await memory.get_store(), _memory_owner(request, session_id))
    return {"ok": True, "removed": removed}


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
    return {"status": "ok", "chat_graph": _graph is not None, "tracing": tracing_enabled(),
            "ui_version": ui_version()}
