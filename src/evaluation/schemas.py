"""평가 스크립트의 구조화 출력 스키마 (test_queries.csv LLM 판사)."""

from pydantic import BaseModel, Field


class ItemVerdict(BaseModel):
    index: int = Field(description="항목 번호 (1부터)")
    passed: bool = Field(description="기대 항목이면 답변이 지켰는지, 금지 항목이면 답변에 나오지 않았는지")
    reason: str = Field(description="판정 근거 한 문장")


class CaseJudgement(BaseModel):
    traits: list[ItemVerdict] = Field(description="expected_traits 항목별 판정 (순서대로 모두)")
    forbidden: list[ItemVerdict] = Field(description="forbidden 항목별 판정 (순서대로 모두, 없으면 빈 목록)")
