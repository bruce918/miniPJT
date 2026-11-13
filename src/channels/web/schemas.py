"""웹 API 입출력 스키마 (/api/chat, /api/feedback)."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default=[], max_length=40)  # 클라이언트가 보내는 값이라 길이를 제한한다
    session_id: str | None = Field(default=None, max_length=64)  # 대화 ID (LangSmith Threads로 묶임)
    client: Literal["web", "test", "telegram"] = "web"           # 트레이스 구분용 (src/evaluation/run_tests.py는 test, 텔레그램 봇은 telegram)


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None  # classify 결과 (디버깅·테스트용)
    tables: list[dict] = []    # render_table 도구가 만든 표 (UI가 그린다. 각 표의 text는 평문 버전)
    sources: list[dict] = []   # 정책 답변의 근거 문서 출처 (retrieve_docs 결과)
    trace_id: str | None = None  # LangSmith 루트 실행 ID (피드백·디버깅용)
    # 가드레일 결과: pii=가린 개인정보 종류, masked=가린 질문(화면이 대화 기록에 이 값을 쓴다), blocked=차단 여부
    guard: dict | None = None
    ui_version: str | None = None  # 지금 서버의 화면 버전 (페이지가 예전 것이면 새로고침 안내)
    # Plan-Execute: 단계별 결과(화면이 차례로 그린다)와 이번 턴의 계획. 단계가 하나면 parts도 하나다.
    parts: list[dict] = []
    plan: list[dict] = []


class FeedbackRequest(BaseModel):
    trace_id: UUID
    score: Literal[0, 1]  # 1=좋아요, 0=별로
    comment: str | None = Field(default=None, max_length=500)
