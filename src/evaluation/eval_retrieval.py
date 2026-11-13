"""정책 문서 검색(src/tools/retrieval) 품질을 구성별로 비교한다 (ablation).

라벨 세트(retrieval_set.json)의 질문마다 정답 문서(+선택: 소제목 키워드)를 두고,
검색 단계를 하나씩 더해 가며 Hit@1, Hit@k, MRR을 잰다.

    bm25          BM25(형태소)만
    dense         임베딩(Titan v2)만
    hybrid        BM25 + 임베딩, RRF 결합
    +expansion    hybrid + 쿼리 확장 (LCEL)
    +rerank       hybrid + 쿼리 확장 + LLM 리랭킹 (LCEL)  <- 운영 설정

실행: python -m src.evaluation.eval_retrieval            (결과: 콘솔 표 + eval_results/retrieval_<시각>.json)
"""

import asyncio
import json
import sys
from datetime import datetime
import os
from pathlib import Path

# 평가는 사람이 기다리지 않으므로 분당 한도 초과에 더 오래 재시도한다 (bus_agent.make_llm이 읽음)
os.environ.setdefault("BEDROCK_MAX_ATTEMPTS", "8")

from src.shared.observability import eval_tracing
from src.shared.paths import EVAL_RESULTS_DIR
from src.tools.retrieval.retriever import Retriever
from src.tools.retrieval.schemas import RetrievalConfig

BASE_DIR = Path(__file__).parent
TOP_K = 4

CONFIGS = {
    "bm25":       RetrievalConfig(use_dense=False, use_expansion=False, use_rerank=False, top_k=TOP_K),
    "dense":      RetrievalConfig(use_bm25=False, use_expansion=False, use_rerank=False, top_k=TOP_K),
    "hybrid":     RetrievalConfig(use_expansion=False, use_rerank=False, top_k=TOP_K),
    "+expansion": RetrievalConfig(use_rerank=False, top_k=TOP_K),
    "+rerank":    RetrievalConfig(top_k=TOP_K),
}


def is_relevant(hit, item: dict) -> bool:
    if hit.chunk.doc not in item["docs"]:
        return False
    kw = item.get("heading_contains")
    return not kw or any(k in hit.chunk.heading or k in hit.chunk.text for k in kw)


def score(hits, item: dict) -> dict:
    rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h, item)), None)
    return {"hit1": float(rank == 1), "hitk": float(rank is not None), "rr": 1.0 / rank if rank else 0.0, "rank": rank}


async def main() -> int:
    data = json.loads((BASE_DIR / "retrieval_set.json").read_text(encoding="utf-8"))
    items = data["queries"]
    retriever = await Retriever.create()
    print(f"청크 {len(retriever.chunks)}개, 질문 {len(items)}개, top_k={TOP_K}\n")

    results = {}
    for name, cfg in CONFIGS.items():
        rows = []
        for item in items:
            res = await retriever.search(item["question"], cfg)
            s = score(res["hits"], item)
            rows.append({"id": item["id"], "question": item["question"], **s,
                         "top": [f"{h.chunk.doc} > {h.chunk.heading}" for h in res["hits"]]})
        n = len(rows)
        results[name] = {
            "hit@1": sum(r["hit1"] for r in rows) / n,
            f"hit@{TOP_K}": sum(r["hitk"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
            "misses": [r for r in rows if not r["hitk"]],
            "rows": rows,
        }
        r = results[name]
        print(f"{name:11} hit@1={r['hit@1']:.2f}  hit@{TOP_K}={r[f'hit@{TOP_K}']:.2f}  MRR={r['mrr']:.2f}"
              f"  놓친 질문 {len(r['misses'])}개")

    best = results["+rerank"]
    if best["misses"]:
        print("\n운영 설정(+rerank)에서 놓친 질문:")
        for m in best["misses"]:
            print(f"  {m['id']} {m['question']} -> {m['top']}")

    out = EVAL_RESULTS_DIR
    out.mkdir(exist_ok=True)
    path = out / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {path}")
    return 0


if __name__ == "__main__":
    with eval_tracing("retrieval") as project:
        print(f"[트레이스] LangSmith 프로젝트: {project}")
        sys.exit(asyncio.run(main()))
