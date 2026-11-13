"""정책 답변 점검 (규칙 기반): 문서를 검색했는지, 요금 합계 같은 계산을 calculate 도구로 했는지."""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.policy_qa.prompts import POLICY_CORRECTIONS, POLICY_UNVERIFIED_REPLY
from src.shared.llm import get_text

MAX_RETRIES = 1
# "830원 × 2", "1,650 x 3", "1650*3" 같은 곱셈 표현 (계산을 직접 했다는 신호)
CALC_RE = re.compile(r"\d[\d,]*\s*원?\s*[×xX*]\s*\d")


async def verify_policy(state: dict) -> dict:
    """정책 답변 점검: 문서를 검색했는지, 요금 합계 같은 계산을 calculate 도구로 했는지.

    단계 서브그래프의 대화 기록에는 도구 메시지가 없으므로, 메시지 전체의 도구 호출이 곧 이 단계의 호출이다
    (다시 시킨 뒤에도 앞서 한 검색을 인정한다).
    """
    msgs = state["messages"]
    used = {m.name for m in msgs if isinstance(m, ToolMessage)}
    text = next((get_text(m) for m in reversed(msgs) if isinstance(m, AIMessage) and get_text(m).strip()), "")
    problem = None
    if "retrieve_docs" not in used:
        problem = "no_search"
    elif CALC_RE.search(text) and "calculate" not in used:
        problem = "no_calc"
    if problem is None:
        return {}
    retries = state.get("retries", 0)
    if retries < MAX_RETRIES:
        print(f"[verify_policy] {problem} -> 다시 시킵니다")
        return {"messages": [HumanMessage(POLICY_CORRECTIONS[problem])], "retries": retries + 1}
    if problem == "no_search":
        return {"messages": [AIMessage(POLICY_UNVERIFIED_REPLY)]}
    print("[verify_policy] 다시 시켜도 계산 도구를 쓰지 않음")
    return {}
