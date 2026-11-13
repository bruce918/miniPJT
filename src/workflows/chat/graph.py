"""채팅 워크플로 (LangGraph Plan-Execute).

    바깥 그래프:  START -> recall (장기 메모리·보류 계획 읽기)
                       -> plan   (질문을 최대 3단계로 나눔, planner.py)
                       -> execute (단계 하나 실행) --남은 단계--> execute / END
    단계 서브그래프 (intent별):
        bus_arrival (src/agents/bus_arrival) -> verify        -> END 또는 1번만 다시
        policy_qa   (src/agents/policy_qa)   -> verify_policy -> END 또는 1번만 다시
        out_of_scope (고정 안내, LLM 호출 없음) -> END

새 기능은 schemas.Intent에 값을 추가하고 에이전트를 단계 서브그래프에 등록한다.
"""

import asyncio
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langsmith import traceable

from src.agents.bus_arrival.agent import build_bus_agent
from src.agents.bus_arrival.prompts import BUS_PROMPT
from src.agents.bus_arrival.verify import verify
from src.agents.policy_qa.agent import build_policy_agent
from src.agents.policy_qa.prompts import POLICY_PROMPT
from src.agents.policy_qa.verify import verify_policy
from src.guardrails.filters import check_output
from src.memory import store as memory
from src.shared.llm import get_text
from src.shared.observability import APP_NAME, annotate_current_run
from src.tools.calculator.tool import fix_totals
from src.tools.table.tool import fallback_tables
from src.workflows.chat.planner import (CLASSIFY_CONTEXT, PENDING_NOTE_RE, SELECTION_RE, build_plan_chain)
from src.workflows.chat.prompts import CLASSIFY_PROMPT, OUT_OF_SCOPE_REPLY, PLAN_PROMPT
from src.workflows.chat.schemas import MAX_STEPS, ChatState, ChatTurn, Plan, PlanState


async def build_chat_graph():
    """Plan-Execute 그래프를 컴파일한다.

    바깥 그래프: recall(장기 메모리·보류 계획 읽기) -> plan(질문을 단계로 나눔) -> execute(단계 하나 실행) 반복.
    단계 하나는 서브그래프(step_graph)가 처리한다: intent에 맞는 에이전트 -> verify(근거 점검, 필요하면 한 번 다시).
    정류소 후보를 골라야 하는 단계에서 멈추면 남은 단계를 저장해 두고, 사용자가 고른 다음 턴에 이어서 실행한다.
    """
    bus_agent = await build_bus_agent()
    policy_agent = await build_policy_agent()
    plan_chain = build_plan_chain()
    store = await memory.get_store()

    async def out_of_scope(state: ChatState) -> dict:
        return {"messages": [AIMessage(OUT_OF_SCOPE_REPLY)]}

    def back_or_end(node: str):
        def route(state: ChatState) -> str:
            return node if isinstance(state["messages"][-1], HumanMessage) else END
        route.__name__ = f"retry_or_end_{node}"  # 트레이스에 보이는 이름
        return route

    def route_by_intent(state: ChatState) -> str:
        return state["intent"]

    ROUTES = {"bus_arrival": "bus_arrival", "policy_qa": "policy_qa", "out_of_scope": "out_of_scope"}

    step = StateGraph(ChatState)
    step.add_node("bus_arrival", bus_agent)  # 컴파일된 ReAct 에이전트를 서브그래프 노드로 사용 (messages 키 공유)
    step.add_node("verify", verify)
    step.add_node("policy_qa", policy_agent)  # retrieve_docs(RAG)를 쓰는 정책 안내 에이전트
    step.add_node("verify_policy", verify_policy)
    step.add_node("out_of_scope", out_of_scope)
    step.add_conditional_edges(START, route_by_intent, ROUTES)
    step.add_edge("bus_arrival", "verify")
    step.add_conditional_edges("verify", back_or_end("bus_arrival"), {"bus_arrival": "bus_arrival", END: END})
    step.add_edge("policy_qa", "verify_policy")
    step.add_conditional_edges("verify_policy", back_or_end("policy_qa"), {"policy_qa": "policy_qa", END: END})
    step.add_edge("out_of_scope", END)
    step_graph = step.compile(name="step")

    # ---- 바깥 그래프 노드 ----

    async def recall(state: PlanState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """장기 메모리(사용자 기억)와 보류 중인 계획을 읽는다."""
        cfg = config.get("configurable", {})
        mem = await memory.load(store, cfg.get("user_key"))
        pending = await memory.pop_pending(store, cfg.get("session_key"))
        resume = pending["steps"] if pending else []
        # 이어서 실행할 때 단계 에이전트가 기록 속 복합 질문을 보고 다른 단계의 질문(요금 등)까지 언급하지 않도록,
        # 그 사용자 메시지를 멈춘 단계의 질문으로 바꿔 보여 준다 (예: "오리역 도착…요금 알려줘" -> "오리역 버스 도착 정보")
        rewrite = {pending["question"]: pending["paused_query"]} if pending and pending.get("paused_query") else {}
        return {"memory": memory.summarize(mem), "resume": resume, "rewrite": rewrite, "parts": []}

    async def plan(state: PlanState) -> dict:
        resume = state.get("resume") or []
        if resume and SELECTION_RE.match(state["question"]):
            # 보류 계획이 있고 후보를 고른 답("1번", "7043", "정류소 ID 2060...")이면 모델 없이 바로 잇는다
            steps = [{"intent": "bus_arrival", "query": state["question"]}] + resume
            return {"steps": steps, "plan": steps}
        system = PLAN_PROMPT + (f"\n\n[사용자 기억 내용]\n{state['memory']}" if state.get("memory") else "")
        recent = convert_to_messages(state["history"][-CLASSIFY_CONTEXT:]) + [HumanMessage(state["question"])]
        result: Plan = await plan_chain.ainvoke({"system": system, "messages": recent})
        steps = [s.model_dump() for s in result.steps][:MAX_STEPS]
        if resume and steps[0]["intent"] == "bus_arrival":
            # 사용자가 정류소를 골랐으면(또는 다른 정류소를 말했으면) 그 단계 바로 뒤에 멈춰 둔 단계를 잇는다
            known = {s["query"] for s in steps}
            steps = steps[:1] + [r for r in resume if r["query"] not in known] + steps[1:]
        elif resume:
            print(f"[plan] 화제가 바뀌어 보류 계획 {len(resume)}단계를 버립니다")
        return {"steps": steps, "plan": steps}

    async def execute(state: PlanState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """계획의 다음 단계 하나를 서브그래프로 실행하고 결과(part)를 쌓는다."""
        current, rest = state["steps"][0], state["steps"][1:]
        rewrite = state.get("rewrite") or {}
        past = [(r, rewrite.get(c, c) if r == "user" else c) for r, c in state["history"]]
        inputs = past + [("user", current["query"])]
        out = await step_graph.ainvoke({"messages": inputs, "intent": current["intent"]},
                                       {"recursion_limit": 20, "run_name": f"step:{current['intent']}"})
        part = make_part(current, out["messages"][len(inputs):])
        session_key = config.get("configurable", {}).get("session_key")
        if rest and needs_choice(part) and session_key:
            # 정류소를 골라야 다음으로 갈 수 있다: 남은 단계를 저장하고 이번 턴은 여기서 멈춘다
            await memory.save_pending(store, session_key, rest, state["question"], current["query"])
            part["reply"] += " 고르시면 이어서 " + ", ".join(f"'{r['query']}'" for r in rest) + "도 알려 드릴게요."
            part["pending"] = [r["query"] for r in rest]
            rest = []
        return {"parts": state["parts"] + [part], "steps": rest}

    def more_steps(state: PlanState) -> str:
        return "execute" if state.get("steps") else END

    graph = StateGraph(PlanState)
    graph.add_node("recall", recall)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", more_steps, {"execute": "execute", END: END})
    return graph.compile(name="chat_graph", store=store)


KEEP_LAST_MESSAGES = 6  # 그래프에 보내는 이전 대화 상한 (토큰 절약)

# 화면은 평문으로 보여 주므로 모델이 프롬프트를 어기고 쓴 마크다운 기호를 지운다.
# 줄바꿈과 '-' 목록은 평문으로도 자연스러워 남긴다.
_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_CODE = re.compile(r"`([^`]*)`")


def strip_markdown(text: str) -> str:
    text = _MD_BOLD.sub(r"\2", text)
    text = text.replace("**", "")  # 짝이 맞지 않고 남은 별표
    text = _MD_HEADING.sub("", text)
    return _MD_CODE.sub(r"\1", text)


def needs_choice(part: dict) -> bool:
    """정류소 후보 표만 보여 주고 끝난 단계 (사용자가 골라야 다음으로 갈 수 있다)."""
    kinds = {t.get("kind") for t in part["tables"]}
    return "stations" in kinds and "arrivals" not in kinds


def make_part(step: dict, turn: list) -> dict:
    """단계 하나의 메시지에서 답변·표·출처를 꺼낸다."""
    # 모델이 render_table 호출과 함께 답을 먼저 쓰고 마지막 메시지를 비우는 경우가 있어,
    # 마지막 메시지가 비면 이번 단계의 가장 최근 AI 텍스트를 쓴다.
    ai_texts = [get_text(m).strip() for m in turn if isinstance(m, AIMessage)]
    reply = strip_markdown(next((t for t in reversed(ai_texts) if t), "")) or "답변을 만들지 못했습니다."
    tables = [m.artifact for m in turn if isinstance(m, ToolMessage) and m.name == "render_table" and m.artifact]
    if not tables:
        tables = fallback_tables(turn)
        if tables:
            print(f"[안전망] 모델이 render_table을 호출하지 않아 서버가 표를 만들었습니다: {tables[0]['kind']}")
    # 후보 표를 보여 줄 때는 말풍선에 목록을 다시 늘어놓지 않도록 짧은 안내로 고정한다
    # (모델이 프롬프트를 어기고 후보 전체를 나열하는 경우가 있어서)
    stations = next((t for t in tables if t.get("kind") == "stations"), None)
    if stations and not any(t.get("kind") == "arrivals" for t in tables):
        reply = f"{len(stations['rows'])}곳이 있어요. 표에서 골라 주세요."
    # 정책 답변의 출처: 모델이 쓰게 하지 않고 retrieve_docs 결과에서 꺼낸다 (지어낸 출처 방지)
    sources, seen = [], set()
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "retrieve_docs" and m.artifact:
            for d in m.artifact:
                if d["citation"] not in seen:
                    seen.add(d["citation"])
                    url = d.get("url", "")
                    sources.append({"title": d["title"], "heading": d["heading"], "citation": d["citation"],
                                    "name": d.get("source_name", ""),
                                    # 화면에서 링크로 여는 주소는 http(s)만 넘긴다
                                    "url": url if url.startswith(("https://", "http://")) else ""})
    # 요금 합계 식("단가×인원 + … = 합계원")은 다시 계산해 틀린 합계를 바로잡는다 (모델 산수 실수 방지)
    reply, n_fixed = fix_totals(reply)
    if n_fixed:
        print(f"[계산 교정] 틀린 합계 {n_fixed}건을 고쳤습니다")
        annotate_current_run(tags=["fix:arithmetic"])
    # 출력 가드레일: 비밀값·시스템 프롬프트 유출은 답 전체를 바꾸고, 개인정보는 가린다
    out = check_output(reply, (BUS_PROMPT, POLICY_PROMPT, CLASSIFY_PROMPT, PLAN_PROMPT))
    if out.leak or out.pii:
        print(f"[가드레일] 답변 필터링: leak={out.leak}, pii={out.pii}")
        annotate_current_run(tags=[f"guard:leak:{out.leak}"] if out.leak else [f"guard:out_pii:{k}" for k in out.pii])
    return {"intent": step["intent"], "query": step["query"], "reply": out.text, "tables": tables,
            "sources": sources, "messages": turn}


def _trace_inputs(inputs: dict) -> dict:
    """루트 트레이스에 남길 입력 (그래프 객체·긴 대화 기록은 빼고 요약)."""
    history = inputs.get("history") or []
    return {"message": inputs.get("message"), "history_turns": len(history),
            "last_assistant": next((c for r, c in reversed(history) if r == "assistant"), None)}


def _trace_outputs(turn: "ChatTurn | None") -> dict:
    if not isinstance(turn, ChatTurn):  # 실행이 예외로 끝나면 출력이 없다 (오류는 LangSmith가 따로 기록)
        return {}
    return {"reply": turn.reply, "intent": turn.intent, "plan": turn.plan,
            "tables": [t.get("kind") for t in turn.tables],
            "sources": [s["citation"] for s in turn.sources]}


_background: set = set()  # 배경 작업(장기 메모리 쓰기)이 끝나기 전에 사라지지 않게 잡아 둔다


@traceable(name="chat_turn", run_type="chain", metadata={"app": APP_NAME},
           process_inputs=_trace_inputs, process_outputs=_trace_outputs)
async def run_chat(graph, message: str, history: list[tuple[str, str]],
                   source: str = "web", session_id: str | None = None,
                   guard_tags: list[str] | None = None, user_key: str | None = None) -> ChatTurn:
    """사용자 메시지 하나를 그래프에 넣고 답변·표를 꺼낸다. history는 (role, content) 목록.

    LangSmith에는 질문 1건이 chat_turn 루트 실행 하나로 남는다. session_id는 대화 단위 묶음(Threads)이자
    보류 계획의 키, user_key는 장기 메모리의 키(로그인 사용자·텔레그램 채팅만), source는 web/test/eval 구분에 쓴다.
    """
    trace_id = annotate_current_run(
        metadata={"session_id": session_id or "", "source": source},
        tags=[f"source:{source}", *(guard_tags or [])],
    )
    # 브라우저가 보내 준 대화 기록으로 맥락(후보 정류소 번호 선택 등)을 이어간다
    # 보류 계획 예고("고르시면 이어서 …도 알려 드릴게요")는 서버가 이어서 처리하므로 에이전트가 보지 않게 지운다
    # (남겨 두면 도착 단계 에이전트가 요금 질문까지 답하려 하거나 못 한다고 덧붙인다)
    recent = [(r, PENDING_NOTE_RE.sub("", c)) for r, c in history
              if r in ("user", "assistant") and c][-KEEP_LAST_MESSAGES:]
    while recent and recent[0][0] != "user":
        recent.pop(0)

    result = await graph.ainvoke(
        {"question": message, "history": recent},
        {"recursion_limit": 20, "run_name": "chat_graph",
         "configurable": {"session_key": session_id, "user_key": user_key}},
    )
    parts = result["parts"]
    intents = [p["intent"] for p in parts]
    intent = intents[0] if len(set(intents)) == 1 else "multi"
    annotate_current_run(metadata={"intent": intent, "plan": result.get("plan")},
                         tags=[f"intent:{intent}", f"plan_steps:{len(result.get('plan') or [])}"]
                         + (["plan:paused"] if any(p.get("pending") for p in parts) else []))
    # 장기 메모리 쓰기는 응답을 늦추지 않게 배경에서 한다
    if user_key:
        task = asyncio.create_task(memory.remember(await memory.get_store(), user_key, message, parts))
        _background.add(task)
        task.add_done_callback(_background.discard)
    return ChatTurn(
        reply="\n\n".join(p["reply"] for p in parts),
        intent=intent,
        tables=[t for p in parts for t in p["tables"]],
        messages=[m for p in parts for m in p["messages"]],
        sources=[s for p in parts for s in p["sources"]],
        trace_id=trace_id,
        parts=[{k: v for k, v in p.items() if k != "messages"} for p in parts],
        plan=result.get("plan") or [],
    )
