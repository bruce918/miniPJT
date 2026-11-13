"""장기 메모리와 보류 중인 계획 (LangGraph Store, SQLite `data/memory.db`).

LangGraph의 장기 메모리 패턴을 따른다: 그래프를 store와 함께 컴파일하면 노드가 `store` 인자로 같은 저장소를 받는다.
- 읽기(recall): 계획 노드가 질문을 나누기 전에 사용자 기억을 읽어, 빠진 조건(예: 요금 구분)을 채운다.
- 쓰기(remember): 답을 돌려준 뒤 배경 작업으로 기록한다 (응답을 늦추지 않음).

네임스페이스
- ("memory", <사용자 키>)       "profile"  : 사용자가 스스로 밝힌 지속 정보 (요금 구분, 결제 수단, 이동 지원, 사는 곳)
                                 "stations" : 도착 정보를 조회한 정류소별 횟수 (모델 없이 규칙으로 기록)
- ("pending_plan", <대화 키>)  "plan"     : 정류소를 골라야 해서 멈춘 계획의 남은 단계 (30분 지나면 버림)

사용자 키는 로그인한 웹 사용자(`user:<id>`)와 텔레그램 채팅(`tg:<해시>`)에만 있다. 로그인하지 않은 웹 사용자와
테스트 요청은 장기 메모리를 쓰지 않는다 (보류 계획은 대화 ID로 쓴다).
"""

import asyncio
import os
import time
from pathlib import Path

import aiosqlite
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.base import BaseStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from src.memory.prompts import EXTRACT_PROMPT, SELF_INFO
from src.memory.schemas import MemoryUpdate
from src.shared.paths import DATA_DIR

DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or DATA_DIR / "memory.db")
PENDING_TTL = 30 * 60   # 보류 계획 유효 시간 (초)
TOP_STATIONS = 3        # 기억 요약에 넣는 자주 찾는 정류소 수

_store: AsyncSqliteStore | None = None
_lock = asyncio.Lock()


async def get_store() -> AsyncSqliteStore:
    global _store
    async with _lock:
        if _store is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(DB_PATH, isolation_level=None)
            _store = AsyncSqliteStore(conn)
            await _store.setup()
    return _store


# ---------- 읽기 ----------

PROFILE_LABELS = {"fare_group": "요금 구분", "payment": "결제 수단", "mobility": "이동 지원", "home_area": "사는 곳·자주 가는 곳"}


async def load(store: BaseStore, user_key: str | None) -> dict:
    """{"profile": {...}, "stations": {id: {...}}} (없으면 빈 dict)."""
    if not user_key:
        return {}
    items = await store.asearch(("memory", user_key))
    return {i.key: i.value for i in items}


def summarize(mem: dict) -> str:
    """계획 노드에 넣을 기억 요약 문장. 기억이 없으면 빈 문자열."""
    lines = [f"- {PROFILE_LABELS[k]}: {v}" for k, v in (mem.get("profile") or {}).items()
             if k in PROFILE_LABELS and v]
    stations = sorted((mem.get("stations") or {}).items(), key=lambda kv: -kv[1].get("count", 0))[:TOP_STATIONS]
    if stations:
        lines.append("- 자주 찾는 정류소: " + ", ".join(
            f"{s['name']}(ID {sid}, {s['count']}회)" for sid, s in stations))
    return "\n".join(lines)


# ---------- 보류 계획 ----------

async def save_pending(store: BaseStore, session_key: str | None, steps: list[dict], question: str,
                       paused_query: str) -> None:
    """정류소 선택 때문에 멈춘 계획을 저장한다. question=원래 복합 질문, paused_query=멈춘 단계의 질문."""
    if session_key and steps:
        await store.aput(("pending_plan", session_key), "plan",
                         {"steps": steps, "question": question, "paused_query": paused_query, "ts": time.time()})


async def pop_pending(store: BaseStore, session_key: str | None) -> dict | None:
    """보류 계획({steps, question, paused_query})을 꺼내고 지운다 (없거나 오래됐으면 None)."""
    if not session_key:
        return None
    item = await store.aget(("pending_plan", session_key), "plan")
    if item is None:
        return None
    await store.adelete(("pending_plan", session_key), "plan")
    if time.time() - item.value.get("ts", 0) > PENDING_TTL or not item.value.get("steps"):
        return None
    return item.value


# ---------- 쓰기 ----------

_extract_chain = None


def _extractor():
    global _extract_chain
    if _extract_chain is None:
        from src.shared.llm import make_llm
        prompt = ChatPromptTemplate.from_messages([SystemMessage(EXTRACT_PROMPT), ("human", "{message}")])
        _extract_chain = (prompt | make_llm(max_tokens=150).with_structured_output(MemoryUpdate)
                          ).with_config(run_name="memory_extract")
    return _extract_chain


async def remember(store: BaseStore, user_key: str | None, message: str, parts: list[dict]) -> None:
    """턴이 끝난 뒤 장기 메모리를 갱신한다 (배경 작업)."""
    if not user_key:
        return
    ns = ("memory", user_key)
    # 1) 조회한 정류소: 도착 표의 정류소 ID로 횟수를 센다 (규칙, 모델 없음)
    looked = [t for p in parts for t in p.get("tables", []) if t.get("kind") == "arrivals" and t.get("station_id")]
    if looked:
        item = await store.aget(ns, "stations")
        stations = dict(item.value) if item else {}
        for t in looked:
            sid = t["station_id"]
            s = stations.get(sid, {"name": t.get("title", "").replace(" 도착 정보", ""), "count": 0})
            stations[sid] = {**s, "count": s.get("count", 0) + 1, "last": time.time()}
        await store.aput(ns, "stations", stations)
    # 2) 사용자가 밝힌 자기 정보: 해당 표현이 있을 때만 모델로 뽑는다
    if SELF_INFO.search(message):
        try:
            update: MemoryUpdate = await _extractor().ainvoke({"message": message})
        except Exception as e:
            print(f"[메모리] 추출 실패: {type(e).__name__}")
            return
        found = {k: v for k, v in update.model_dump().items() if v}
        if found:
            item = await store.aget(ns, "profile")
            await store.aput(ns, "profile", {**(item.value if item else {}), **found})
            print(f"[메모리] 기억함: {sorted(found)}")


async def forget(store: BaseStore, user_key: str) -> int:
    items = await store.asearch(("memory", user_key))
    for i in items:
        await store.adelete(("memory", user_key), i.key)
    return len(items)
