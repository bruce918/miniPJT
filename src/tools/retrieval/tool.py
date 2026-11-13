"""retrieve_docs 도구: 정책 에이전트가 부르는 문서 검색 (검색기는 한 번만 만들어 재사용)."""

import asyncio

from langchain_core.tools import tool

from src.tools.retrieval.retriever import Retriever
from src.tools.retrieval.schemas import Hit

_retriever: Retriever | None = None
_retriever_lock = asyncio.Lock()


async def get_retriever() -> Retriever:
    global _retriever
    async with _retriever_lock:
        if _retriever is None:
            _retriever = await Retriever.create()
    return _retriever


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "관련 문서를 찾지 못했습니다. 문서에 없는 내용은 답하지 말고, 120 경기도 콜센터 등 공식 창구를 안내하세요."
    return "\n\n".join(
        f"[{n}] {h.chunk.title} > {h.chunk.heading}\n출처: {h.chunk.citation()}\n{h.chunk.text}"
        for n, h in enumerate(hits, 1)
    )


@tool(response_format="content_and_artifact")
async def retrieve_docs(query: str) -> tuple[str, list[dict]]:
    """경기도 버스 정책·리서치 문서(요금, 환승, 저상·심야버스, 분실물, 민원, 광역버스, 공공버스, 교통비 환급, 버스정보, 연구 보고서)를 검색한다.

    Args:
        query: 사용자 질문 그대로 또는 검색할 내용 (예: "환승 할인 시간 제한")
    """
    retriever = await get_retriever()
    result = await retriever.search(query)
    hits = result["hits"]
    artifact = [{"id": h.chunk.id, "title": h.chunk.title, "heading": h.chunk.heading,
                 "citation": h.chunk.citation(),
                 # 화면의 바로가기 링크용 (문서 머리말의 첫 번째 출처)
                 "source_name": (h.chunk.sources[0] if h.chunk.sources else {}).get("name", ""),
                 "url": (h.chunk.sources[0] if h.chunk.sources else {}).get("url", ""),
                 "rerank": h.rerank, "rrf": round(h.rrf, 4)} for h in hits]
    return format_hits(hits), artifact
