"""버스 도착 답변의 근거 점검 (규칙 기반, LLM 없음).

평가(evaluate.py)에서 모델이 도구 없이 도착 시간을 지어내거나, 검색 없이 지식으로 답하는 경우가 나와
답변을 내보내기 전에 규칙으로 한 번 점검하고 어기면 한 번만 다시 시키도록 했다.
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.bus_arrival.prompts import CORRECTIONS, UNVERIFIED_REPLY
from src.shared.llm import get_text

ARRIVAL_TOOLS = {"get_bus_arrival_list", "get_bus_arrival_item"}
DATA_TOOLS = {"search_bus_station", *ARRIVAL_TOOLS}
MAX_RETRIES = 1
MINUTES_RE = re.compile(r"\d+\s*분")
ASK_STATION_RE = re.compile(r"정류(소|장).{0,6}(이름|번호).{0,8}알려")
# "경기도 버스 도착 정보만", "경기도 내 정류소만", "경기도 ... 범위를 벗어나" 같은 범위 밖 안내
GYEONGGI_ONLY_RE = re.compile(r"경기도.{0,20}(만\s|만$|만[.,]|밖|범위|벗어)")


def _current_turn(messages):
    """마지막 사용자(Human) 메시지 뒤의 메시지들 = 이번 시도에서 에이전트가 만든 메시지."""
    idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[idx + 1:]


def _station_hits(turn) -> int:
    """이번 시도에서 search_bus_station이 찾은 정류소 수 (가장 최근 검색 기준)."""
    for m in reversed(turn):
        if isinstance(m, ToolMessage) and m.name == "search_bus_station":
            return get_text(m).count('"id":')
    return 0


def find_violation(messages) -> str | None:
    """이번 시도의 답변이 근거 규칙을 어겼으면 이유를, 아니면 None을 돌려준다."""
    turn = _current_turn(messages)
    used = {m.name for m in turn if isinstance(m, ToolMessage)}
    text = next((get_text(m).strip() for m in reversed(turn) if isinstance(m, AIMessage) and get_text(m).strip()), "")
    if MINUTES_RE.search(text) and not (used & ARRIVAL_TOOLS):
        return "arrival_without_tool"
    if _station_hits(turn) and GYEONGGI_ONLY_RE.search(text):
        return "gyeonggi_notice_with_results"
    if not (used & DATA_TOOLS) and (not ASK_STATION_RE.search(text) or _names_place(_question(messages))):
        # 정류소 이름을 되묻는 답은 질문에 이름이 전혀 없을 때만 허용한다
        # ("쿄쿄쿄없는정류장 버스 언제 와?"에 검색 없이 "정류소 이름을 알려 주세요"라고 답하던 문제)
        return "answer_without_search"
    return None


# 질문에서 이 말들을 지우고도 글자가 남으면 정류소 이름(또는 번호)을 말한 것으로 본다
GENERIC_WORDS_RE = re.compile(r"버스|언제|와요|와|오나요|와\?|도착|정보|알려\s*주세요|알려\s*줘|알려|몇\s*분|정류장|정류소|"
                              r"남았어|남았|지금|좀|요|[?!.,~\s]")


def _question(messages) -> str:
    """이 단계의 사용자 질문 (자동 점검 메시지는 건너뛴다)."""
    return next((get_text(m) for m in reversed(messages)
                 if isinstance(m, HumanMessage) and not get_text(m).startswith("[자동 점검]")), "")


def _names_place(question: str) -> bool:
    return len(GENERIC_WORDS_RE.sub("", question)) >= 2


async def verify(state: dict) -> dict:
    """단계 서브그래프의 verify 노드: 위반이면 한 번 다시 시키고, 그래도 어기면 안전하게 바꾼다."""
    violation = find_violation(state["messages"])
    retries = state.get("retries", 0)
    if violation is None:
        return {}
    if retries < MAX_RETRIES:
        print(f"[verify] {violation} -> 에이전트에게 다시 시킵니다")
        return {"messages": [HumanMessage(CORRECTIONS[violation])], "retries": retries + 1}
    print(f"[verify] 다시 시켜도 {violation}")
    if violation == "arrival_without_tool":
        # 근거 없는 도착 시간은 내보내지 않는다
        return {"messages": [AIMessage(UNVERIFIED_REPLY)]}
    hits = _station_hits(_current_turn(state["messages"]))
    if violation == "gyeonggi_notice_with_results" and hits >= 2:
        # 검색 결과가 있으니 안내 문구 대신 후보 안내로 바꾼다 (표는 run_chat의 안전망이 그린다)
        return {"messages": [AIMessage(f"{hits}곳이 있어요. 표에서 번호를 골라 주세요.")]}
    return {}  # 검색만 건너뛴 답(예: 경기도 밖 지명 안내)은 위험하지 않아 그대로 둔다
