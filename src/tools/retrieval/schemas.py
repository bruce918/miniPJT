"""정책 문서 검색의 스키마: 검색 설정, 청크·검색 결과, LCEL 체인의 구조화 출력."""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class RetrievalConfig:
    """검색 단계 설정. 기본값이 운영 설정이고, 평가에서 단계를 하나씩 꺼 비교한다."""
    use_bm25: bool = True
    use_dense: bool = True
    use_expansion: bool = True
    use_rerank: bool = True
    candidates: int = 10      # 검색기·질문별로 가져오는 후보 수
    rerank_pool: int = 12     # 리랭킹에 넘기는 후보 수
    top_k: int = 4            # 최종 반환 청크 수
    min_rerank_score: int = 3  # 리랭킹 점수(0~10)가 이보다 낮으면 버린다


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    category: str
    heading: str
    text: str
    sources: list[dict] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        return f"{self.title} > {self.heading}\n{self.text}"

    def citation(self) -> str:
        src = self.sources[0] if self.sources else {}
        name = src.get("name", "출처 미상")
        return f"{name} ({src['url']})" if src.get("url") else name


@dataclass
class Hit:
    chunk: Chunk
    rrf: float = 0.0
    rerank: int | None = None


class ExpandedQuery(BaseModel):
    queries: list[str] = Field(description="원 질문과 뜻이 같지만 표현이 다른 검색 질문 2개")
    keywords: list[str] = Field(description="문서에 나올 법한 동의어·관련 용어 3~6개 (예: 갈아타기 -> 환승, 환승할인)")


class RerankScore(BaseModel):
    id: int = Field(description="후보 번호")
    score: int = Field(ge=0, le=10, description="질문에 답하는 데 필요한 정보가 있는 정도 (0=무관, 10=정확히 답함)")


class RerankResult(BaseModel):
    scores: list[RerankScore]
