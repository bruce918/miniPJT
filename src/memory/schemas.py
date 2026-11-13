"""장기 메모리 스키마: 대화에서 뽑는 사용자 정보."""

from typing import Literal

from pydantic import BaseModel, Field


class MemoryUpdate(BaseModel):
    """사용자가 자기 자신에 대해 밝힌, 다음 대화에도 유효한 정보. 해당 없으면 null."""
    fare_group: Literal["성인", "청소년", "어린이", "어르신", "장애인"] | None = Field(
        default=None, description="본인의 버스 요금 구분 (예: '고등학생이에요' -> 청소년)")
    payment: str | None = Field(default=None, description="주로 쓰는 결제 수단 (예: 교통카드, 경기패스)")
    mobility: str | None = Field(default=None, description="이동 지원이 필요한 사정 (예: 휠체어, 유모차)")
    home_area: str | None = Field(default=None, description="사는 동네나 자주 타는 곳 (예: 분당 정자동)")
