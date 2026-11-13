"""LangSmith 트레이스(Observability) 설정을 한곳에 모은다.

.env의 LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT만 있으면 LangChain/LangGraph 실행은
자동으로 기록된다. 여기서는 그 기록을 보기 좋게 구조화한다.

    chat_turn (루트, 질문 1건)                       <- run_chat에 @traceable
      metadata: session_id(대화 ID → LangSmith Threads), source(web/test/eval), intent, app
      tags: source:web, intent:policy_qa ...
      └ chat_graph (LangGraph)
          ├ classify (LCEL)
          ├ bus_agent / policy_agent (ReAct) ─ 도구 호출 (MCP 도구, render_table, retrieve_docs)
          │                                   └ rag_search (retriever) ─ query_expansion · bm25 · dense · rrf_fusion · rerank
          └ verify / verify_policy

- 평가 스크립트는 eval_tracing()으로 별도 프로젝트(<프로젝트>-eval)에 기록해 실제 사용 기록과 섞이지 않게 한다.
- 응답에 trace_id(루트 실행 ID)를 돌려주고, 사용자 피드백(👍/👎)을 그 실행에 붙인다.
- 질문 내용을 외부로 보내고 싶지 않으면 .env에 LANGSMITH_HIDE_INPUTS=true / LANGSMITH_HIDE_OUTPUTS=true.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from langsmith import Client, tracing_context
from langsmith.run_helpers import get_current_run_tree

load_dotenv()

APP_NAME = "gg-bus-chatbot"


def tracing_enabled() -> bool:
    return (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower() == "true" \
        and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def project_name() -> str:
    return os.getenv("LANGSMITH_PROJECT") or "default"


def annotate_current_run(metadata: dict | None = None, tags: list[str] | None = None) -> str | None:
    """지금 실행 중인 트레이스 실행에 메타데이터·태그를 붙이고 그 실행 ID를 돌려준다 (트레이싱이 꺼져 있으면 None)."""
    run = get_current_run_tree()
    if run is None:
        return None
    if metadata:
        run.add_metadata(metadata)
    if tags:
        run.add_tags(tags)
    return str(run.id)


@contextmanager
def eval_tracing(kind: str):
    """평가 실행을 별도 프로젝트(<프로젝트>-eval, EVAL_LANGSMITH_PROJECT로 변경 가능)에 기록한다."""
    project = os.getenv("EVAL_LANGSMITH_PROJECT") or f"{project_name()}-eval"
    with tracing_context(project_name=project, tags=["eval", f"eval:{kind}"], metadata={"app": APP_NAME}):
        yield project


_client: Client | None = None


def send_feedback(trace_id: str, score: int, comment: str | None = None) -> None:
    """사용자 피드백(1=좋아요, 0=별로)을 루트 실행에 붙인다."""
    global _client
    if _client is None:
        _client = Client()
    _client.create_feedback(trace_id, key="user_score", score=score, comment=comment)
