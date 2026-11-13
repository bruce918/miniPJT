"""계획 단계: 질문을 단계로 나누는 LCEL 체인과 보류 계획을 잇는 규칙."""

import re

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.shared.llm import make_llm
from src.workflows.chat.prompts import CLASSIFY_PROMPT
from src.workflows.chat.schemas import Classification, Plan

CLASSIFY_CONTEXT = 4  # 계획할 때 함께 보는 직전 메시지 수

# 정류소 후보를 고르는 짧은 답: "1번", "2", "7043", "정류소 ID 206000222 버스 언제 와?"(웹 즐겨찾기·텔레그램 버튼)
SELECTION_RE = re.compile(r"^\s*(\d{1,2}\s*번?|\d{4,5}|정류소 ID \d{9}.*)\s*$")
PENDING_NOTE_RE = re.compile(r"\s*고르시면 이어서 .*?도 알려 드릴게요\.")


def build_classify_chain():
    """classify LCEL 체인: 프롬프트 템플릿 | 모델(Pydantic 구조화 출력).

    입력 {"messages": 직전 대화 + 마지막 사용자 메시지} -> 출력 Classification.
    시스템 프롬프트는 SystemMessage 객체로 넣어 중괄호 등이 템플릿 변수로 해석되지 않게 한다.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(CLASSIFY_PROMPT),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | make_llm(max_tokens=100).with_structured_output(Classification)
    return chain.with_config(run_name="classify")


def build_plan_chain():
    """plan LCEL 체인: (계획 프롬프트 + 사용자 기억) | 모델(Pydantic 구조화 출력 Plan).

    시스템 문장은 변수로 통째로 넣는다 (값 안의 중괄호는 템플릿으로 해석되지 않는다).
    """
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), MessagesPlaceholder("messages")])
    chain = prompt | make_llm(max_tokens=400).with_structured_output(Plan)
    return chain.with_config(run_name="plan")
