"""경기도 버스 정책·리서치 문서 RAG: 하이브리드 검색 + 쿼리 확장 + LLM 리랭킹.

    질문 ─▶ 쿼리 확장 (LCEL: 다른 표현 2개 + 동의어 키워드)
         ─▶ 질문마다 BM25(kiwipiepy 형태소) 와 임베딩(Titan v2) 검색
         ─▶ RRF(Reciprocal Rank Fusion)로 모든 순위 목록을 합침
         ─▶ 상위 후보를 LLM이 관련도 0~10점으로 다시 매김 (LCEL 리랭킹)
         ─▶ 상위 k개 청크 + 출처

문서: docs/policy/*.md (YAML 머리말 + '## 소제목' 단위로 청크).
임베딩은 .rag_index/embeddings.json 에 청크 내용 해시로 캐시한다 (문서가 바뀐 청크만 다시 계산).
Bedrock 리랭크 모델은 이 계정에서 권한이 없어 LLM 리랭킹을 쓴다.

에이전트에는 retrieve_docs 도구(tool.py)로 붙는다 (src/agents/policy_qa).
RetrievalConfig로 각 단계를 켜고 끌 수 있어 단계별 기여를 비교할 수 있다 (src/evaluation/eval_retrieval.py).
"""

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_aws import BedrockEmbeddings
from langsmith import traceable
from rank_bm25 import BM25Okapi

from src.shared.llm import make_llm
from src.shared.paths import POLICY_DOCS_DIR, RAG_INDEX_DIR
from src.tools.retrieval.prompts import EXPAND_PROMPT, RERANK_PROMPT
from src.tools.retrieval.schemas import Chunk, ExpandedQuery, Hit, RerankResult, RetrievalConfig

load_dotenv()

DOCS_DIR = POLICY_DOCS_DIR
INDEX_DIR = RAG_INDEX_DIR
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
MAX_CHUNK_CHARS = 900   # 소제목 본문이 이보다 길면 문단 단위로 나눈다
RRF_K = 60              # RRF 상수 (원 논문 기본값)


# ---------- 문서 적재·청크 ----------

def _split_frontmatter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, raw


def _split_long(text: str) -> list[str]:
    """긴 본문을 문단 단위로 MAX_CHUNK_CHARS 안쪽으로 묶는다."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) > MAX_CHUNK_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        title = meta.get("title") or path.stem
        sections = re.split(r"^## +", body, flags=re.M)
        intro, sections = sections[0], sections[1:]
        intro = re.sub(r"^# .*$", "", intro, flags=re.M).strip()
        items = ([("개요", intro)] if intro else []) + [
            (s.split("\n", 1)[0].strip(), s.split("\n", 1)[1].strip() if "\n" in s else "") for s in sections
        ]
        for heading, text in items:
            if not text:
                continue
            for n, part in enumerate(_split_long(text)):
                chunks.append(Chunk(
                    id=f"{path.stem}#{len(chunks)}",
                    doc=path.name, title=title, category=meta.get("category", ""),
                    heading=heading + (f" ({n + 1})" if n else ""), text=part,
                    sources=meta.get("sources") or [],
                ))
    return chunks


# ---------- BM25 (형태소 기반) ----------

KEEP_POS = ("NN", "VV", "VA", "XR", "SL", "SN", "MAG")  # 명사·동사/형용사 어간·어근·외국어·숫자
_kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form.lower() for t in _kiwi.tokenize(text) if t.tag.startswith(KEEP_POS)]


# ---------- 임베딩 (캐시) ----------

def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingStore:
    def __init__(self, model_id: str = EMBED_MODEL_ID):
        self.embeddings = BedrockEmbeddings(model_id=model_id, region_name="us-east-1")
        self.path = INDEX_DIR / "embeddings.json"
        self.cache: dict[str, list[float]] = {}
        if self.path.exists():
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(f"{EMBED_MODEL_ID}\n{text}".encode()).hexdigest()[:24]

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        missing = [c for c in chunks if self.key(c.search_text) not in self.cache]
        if missing:
            vecs = await self.embeddings.aembed_documents([c.search_text for c in missing])
            for c, v in zip(missing, vecs):
                self.cache[self.key(c.search_text)] = v
            INDEX_DIR.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[self.key(c.search_text)] for c in chunks]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)


# ---------- LCEL 체인: 쿼리 확장 · 리랭킹 ----------

def build_expand_chain(llm=None):
    chain = EXPAND_PROMPT | (llm or make_llm(max_tokens=400)).with_structured_output(ExpandedQuery)
    return chain.with_config(run_name="query_expansion")


def build_rerank_chain(llm=None):
    chain = RERANK_PROMPT | (llm or make_llm(max_tokens=800)).with_structured_output(RerankResult)
    return chain.with_config(run_name="rerank")


# ---------- 검색기 ----------

class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]], store: EmbeddingStore):
        self.chunks = chunks
        self.vectors = vectors
        self.store = store
        self.bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks]) if chunks else None
        self.expand_chain = build_expand_chain()
        self.rerank_chain = build_rerank_chain()

    @classmethod
    async def create(cls, docs_dir: Path = DOCS_DIR) -> "Retriever":
        chunks = load_chunks(docs_dir)
        store = EmbeddingStore()
        vectors = await store.embed_chunks(chunks) if chunks else []
        return cls(chunks, vectors, store)

    # 각 검색 단계는 LangSmith에 하위 실행으로 남는다 (rag_search 아래 bm25 · dense · rrf_fusion).
    # 결과는 트레이스에서 읽기 쉽도록 청크 ID·소제목·점수를 담은 dict 목록으로 돌려준다.

    def _ranked(self, idxs, scores) -> list[dict]:
        return [{"idx": i, "id": self.chunks[i].id, "heading": self.chunks[i].heading, "score": round(float(scores[i]), 4)}
                for i in idxs]

    @traceable(name="bm25", run_type="retriever", process_inputs=lambda d: {"query": d["query"], "n": d["n"]})
    def _bm25_rank(self, query: str, n: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return self._ranked([i for i in ranked[:n] if scores[i] > 0], scores)

    @traceable(name="dense", run_type="retriever", process_inputs=lambda d: {"query": d["query"], "n": d["n"]})
    async def _dense_rank(self, query: str, n: int) -> list[dict]:
        q = await self.store.embed_query(query)
        sims = [_cosine(q, v) for v in self.vectors]
        return self._ranked(sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:n], sims)

    @traceable(name="rrf_fusion", run_type="chain",
               process_inputs=lambda d: {"rankings": [[x["id"] for x in r] for r in d["rankings"]]})
    def _rrf(self, rankings: list[list[dict]]) -> list[dict]:
        """RRF: 각 목록에서의 순위 r에 대해 1/(k + r)을 더한다."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for r, item in enumerate(ranking, 1):
                fused[item["idx"]] = fused.get(item["idx"], 0.0) + 1.0 / (RRF_K + r)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        return self._ranked(order, fused)

    @traceable(name="rag_search", run_type="retriever",
               process_inputs=lambda d: {"question": d["question"], "config": asdict(d.get("cfg") or RetrievalConfig())},
               process_outputs=lambda out: {
                   "documents": [{"page_content": h.chunk.text, "type": "Document",
                                  "metadata": {"id": h.chunk.id, "title": h.chunk.title, "heading": h.chunk.heading,
                                               "rrf": round(h.rrf, 4), "rerank": h.rerank}} for h in out["hits"]],
                   "queries": out["queries"], "keywords": out["keywords"]})
    async def search(self, question: str, cfg: RetrievalConfig | None = None) -> dict:
        """질문으로 청크를 찾는다. 반환: {"hits": [Hit], "queries": [...], "keywords": [...]}"""
        cfg = cfg or RetrievalConfig()
        if not self.chunks:
            return {"hits": [], "queries": [question], "keywords": []}

        queries, keywords = [question], []
        if cfg.use_expansion:
            exp: ExpandedQuery = await self.expand_chain.ainvoke({"question": question})
            queries += [q for q in exp.queries if q.strip()][:2]
            keywords = [k for k in exp.keywords if k.strip()][:6]

        # 순위 목록들을 모은다: 질문별 BM25(키워드를 붙여서) + 질문별 임베딩
        rankings: list[list[dict]] = []
        if cfg.use_bm25:
            kw = " ".join(keywords)
            rankings += [self._bm25_rank(f"{q} {kw}", cfg.candidates) for q in queries]
        if cfg.use_dense:
            rankings += await asyncio.gather(*(self._dense_rank(q, cfg.candidates) for q in queries))

        hits = [Hit(self.chunks[x["idx"]], rrf=x["score"]) for x in self._rrf(rankings)]

        if cfg.use_rerank and hits:
            pool = hits[:cfg.rerank_pool]
            listing = "\n\n".join(f"[{n}] {h.chunk.title} > {h.chunk.heading}\n{h.chunk.text[:600]}"
                                  for n, h in enumerate(pool))
            rr: RerankResult = await self.rerank_chain.ainvoke({"question": question, "candidates": listing})
            score_of = {s.id: s.score for s in rr.scores}
            for n, h in enumerate(pool):
                h.rerank = score_of.get(n, 0)
            hits = sorted(pool, key=lambda h: (h.rerank, h.rrf), reverse=True)
            hits = [h for h in hits if h.rerank >= cfg.min_rerank_score]

        return {"hits": hits[:cfg.top_k], "queries": queries, "keywords": keywords}


# ---------- 에이전트 도구 ----------

