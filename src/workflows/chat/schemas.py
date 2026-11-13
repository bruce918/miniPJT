"""채팅 워크플로의 입출력 스키마: 의도(Intent), 계획(Plan), 그래프 상태, 한 턴의 결과(ChatTurn)."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 기능이 늘면 여기에 intent를 추가하고, 에이전트를 단계 서브그래프(graph.py)에 등록한다.
Intent = Literal["bus_arrival", "policy_qa", "out_of_scope"]
MAX_STEPS = 3


class Classification(BaseModel):
    intent: Intent = Field(description="질문 유형")


class PlanStep(BaseModel):
    intent: Intent = Field(description="이 단계의 종류")
    query: str = Field(description="이 단계만 처리할 질문 문장")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS, description="실행 순서대로 나눈 단계")


class ChatState(MessagesState):
    """한 단계(질문 하나)를 처리하는 서브그래프의 상태."""
    intent: Intent
    retries: int  # verify/verify_policy가 에이전트로 되돌려 보낸 횟수


class PlanState(TypedDict, total=False):
    """바깥 그래프(계획 -> 실행 반복)의 상태."""
    question: str                    # 사용자 메시지 (가드레일을 거친 것)
    history: list[tuple[str, str]]   # 최근 대화 (role, content)
    memory: str                      # 장기 메모리 요약 (계획에만 쓴다)
    resume: list[dict]               # 지난 턴에 정류소 선택 때문에 멈춘 계획의 남은 단계
    rewrite: dict                    # 단계가 볼 대화 기록에서 바꿔 쓸 문장 {원래 복합 질문: 멈춘 단계의 질문}
    steps: list[dict]                # 아직 실행하지 않은 단계 [{intent, query}]
    plan: list[dict]                 # 이번 턴에 세운 전체 계획 (기록용)
    parts: list[dict]                # 실행한 단계별 결과 (화면에 차례로 보여 준다)


@dataclass
class ChatTurn:
    """한 턴의 결과. 웹 API(src/channels/web)와 평가(src/evaluation)가 같이 쓴다.

    reply·tables·sources·messages는 모든 단계를 합친 값이고, parts는 단계별 결과(화면이 차례로 그린다)다.
    """
    reply: str
    intent: str | None
    tables: list[dict]
    messages: list = field(default_factory=list)  # 이번 턴에 새로 생긴 메시지 (도구 호출·결과 포함)
    sources: list[dict] = field(default_factory=list)  # retrieve_docs가 찾은 문서 출처 (정책 답변일 때)
    trace_id: str | None = None  # LangSmith 루트 실행(chat_turn) ID. 트레이싱이 꺼져 있으면 None
    parts: list[dict] = field(default_factory=list)  # 단계별 결과 [{intent, query, reply, tables, sources, pending?}]
    plan: list[dict] = field(default_factory=list)   # 이번 턴에 세운 계획 [{intent, query}]
