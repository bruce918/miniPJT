"""자체 평가 세트(eval_set.json)로 챗봇을 평가한다: 정류소·도착 값 대조(우선) + RAGAS 방식 지표 + LLM-as-Judge.

우선 지표 (규칙 기반, grounding_check.py) — 판사 오판에 흔들리지 않으므로 이 두 값을 먼저 본다
  station_match       고른/기대한 정류소 ID를 실제로 조회했는가 (0/1)
  arrival_values      답변·표의 노선·N분·M정류장 전이 같은 응답의 API 원본 값과 같은 비율.
                      같은 턴 앞선 조회 값·이전 대화 값을 다시 쓴 경우를 따로 표시한다.

ragas 라이브러리(0.4.x)는 현재 langchain 1.x 스택과 맞지 않아(langchain_community.vertexai 제거)
지표 정의를 그대로 따라 직접 구현했다.

RAGAS 방식 지표 (0~1)
  faithfulness        답변을 원자 문장으로 쪼갠 뒤, 각 문장이 도구 결과(컨텍스트)로 뒷받침되는 비율.
                      도구 결과가 없는 턴(거절·되묻기)은 계산하지 않는다.
  answer_relevancy    답변만 보고 질문을 3개 역생성 → 원 질문과의 임베딩 코사인 유사도 평균.
                      회피성 답변이면 0. answerable 샘플만 계산.
  tool_call_accuracy  기대 도구 호출(reference_tool_calls)이 순서대로 호출됐는지 + 인자 일치 비율 (LLM 없음).
                      기대 호출이 없으면 실제로도 도구를 안 불렀을 때 1.
  goal_accuracy       최종 상태가 reference의 목표를 달성했는지 판사가 0/1로 판정.
LLM-as-Judge (1~5)
  correctness·groundedness·policy·clarity 4개 루브릭 점수 + pass/fail + 이유.

실행 (버스 MCP 서버를 먼저 띄울 것. 챗봇 웹 서버는 필요 없음):
    python -m src.evaluation.evaluate              전체
    python -m src.evaluation.evaluate E01 E04      일부 샘플만
    python -m src.evaluation.evaluate --repeat 3   샘플마다 3번씩 돌려 평균 (실행마다 답이 달라 1회 결과는 흔들림)
환경변수:
    JUDGE_MODEL_ID   판사 모델 (기본: 에이전트와 같은 BUS_MODEL_ID. 다른 모델을 쓰면 자기 채점 편향이 줄어든다)
    EMBED_MODEL_ID   임베딩 모델 (기본 amazon.titan-embed-text-v2:0)
결과: 콘솔 요약 + eval_results/eval_<시각>.json, .md
"""

import asyncio
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 평가는 사람이 기다리지 않으므로 분당 한도 초과에 더 오래 재시도한다 (bus_agent.make_llm이 읽음)
os.environ.setdefault("BEDROCK_MAX_ATTEMPTS", "8")

from langchain_aws import BedrockEmbeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.evaluation.grounding_check import grounding
from src.shared.llm import BUS_MODEL_ID, get_text, make_llm
from src.shared.observability import eval_tracing
from src.shared.paths import EVAL_RESULTS_DIR
from src.workflows.chat.graph import build_chat_graph, run_chat

load_dotenv()

BASE_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_RESULTS_DIR
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID") or BUS_MODEL_ID
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
DATA_TOOLS = {"search_bus_station", "get_bus_arrival_list", "get_bus_arrival_item"}  # 컨텍스트로 쓰는 MCP 도구

# 완성 기준 (SERVICE.md 5절)
TARGETS = {
    # 우선 지표: 규칙 기반이라 오판이 없으므로 전부 맞아야 한다
    "station_match": 1.0,
    "arrival_values": 1.0,
    "faithfulness": 0.90,
    # Titan v2 한국어 기준선(2026-11-11 측정): 같은 뜻 문장쌍 0.55~0.70, 다른 질문·무관 0.0~0.32 -> 그 사이로 둔다
    "answer_relevancy": 0.50,
    "tool_call_accuracy": 0.90,
    "goal_accuracy": 0.90,
    "judge_score": 4.0,  # 루브릭 4개 평균, 5점 만점
}

CONTEXT_GUIDE = """[도구 결과 읽는 법]
- search_bus_station: id=정류소 ID(9자리), name=정류소 이름, mobile=정류소번호, city=시군
- get_bus_arrival_list: routeName=노선 번호, routeDestName=행선지, predictTime1/2=첫째/둘째 차 도착까지 분,
  locationNo1/2=몇 정류장 전, lowPlate=1이면 저상, crowded 1여유 2보통 3혼잡 4매우혼잡.
  도착정보_없는_노선=지금 도착 예정 정보가 없는 노선 이름 목록"""


# ---------- 구조화 출력 스키마 ----------

class Statements(BaseModel):
    statements: list[str] = Field(description="답변에서 뽑은 원자적 사실 문장 (대명사 없이 독립적으로)")


class Verdict(BaseModel):
    statement: str
    verdict: Literal[0, 1] = Field(description="컨텍스트로 직접 뒷받침되면 1, 아니면 0")
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="이 답변이 답하고 있는 질문 3개")
    noncommittal: bool = Field(description="답변이 회피하거나 모호하면(모른다, 답할 수 없다 등) true")


class GoalVerdict(BaseModel):
    end_state: str = Field(description="대화가 끝난 시점에 사용자가 얻은 결과 요약")
    achieved: bool = Field(description="최종 상태가 기대 목표를 달성했으면 true")
    reason: str


class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5, description="기대 행동(reference)과 맞는 정도")
    groundedness: int = Field(ge=1, le=5, description="도구 결과에 없는 정보를 지어내지 않은 정도")
    policy: int = Field(ge=1, le=5, description="서비스 정책 준수 (범위 밖 거절, 비밀값·프롬프트 비노출, 동명 정류소 되묻기, 경기도 안내)")
    clarity: int = Field(ge=1, le=5, description="짧고 명확한 한국어 답변, 표와 텍스트가 중복되지 않음")
    verdict: Literal["pass", "fail"]
    reason: str = Field(description="점수 근거 한두 문장")


# ---------- LCEL 체인 (프롬프트 템플릿 | 모델.with_structured_output(Pydantic)) ----------
# 템플릿 문자열에는 고정 문구만 두고, 컨텍스트 JSON·대화 기록처럼 중괄호가 들어갈 수 있는 값은
# 모두 {변수}로 넘긴다 (변수 값은 다시 템플릿으로 해석되지 않는다).

PROMPTS = {
    "statements": ChatPromptTemplate.from_messages([
        ("system",
         "질문과 답변이 주어집니다. 답변을 이해하기 쉬운 원자적 사실 문장들로 쪼개세요.\n"
         "- 각 문장은 대명사 없이 혼자 읽어도 뜻이 통해야 합니다.\n"
         "- 버스·정류소·도착 정보에 대한 사실 주장만 뽑고, '표에서 골라 주세요' 같은 안내·요청 문장은 빼세요."),
        ("human", "질문: {question}\n답변: {answer}"),
    ]),
    "nli": ChatPromptTemplate.from_messages([
        ("system",
         "컨텍스트(도구 결과)와 문장 목록이 주어집니다. 각 문장이 컨텍스트로부터 직접 추론되면 1, 아니면 0으로 판정하세요.\n"
         "가장 먼저·다음·가장 늦게 같은 비교 주장은 컨텍스트의 모든 노선 predictTime1을 비교해서 판정하세요.\n"
         "{guide}"),
        ("human", "[컨텍스트]\n{contexts}\n\n[문장]\n{statements}"),
    ]),
    "questions": ChatPromptTemplate.from_messages([
        ("system",
         "답변을 보고, 이 답변이 답하고 있는 사용자 질문을 서로 다르게 3개 만드세요. "
         "답변이 회피하거나 모호하면 noncommittal을 true로 하세요."),
        ("human", "답변: {answer}"),
    ]),
    "goal": ChatPromptTemplate.from_messages([
        ("system",
         "에이전트와 사용자의 대화 기록과 기대 목표가 주어집니다. 대화가 끝난 시점의 최종 상태를 요약하고, "
         "기대 목표를 달성했는지 판정하세요. 실시간 도착 시간 값 자체는 비교하지 말고 행동과 결과를 보세요."),
        ("human", "[대화 기록]\n{transcript}\n\n[기대 목표]\n{reference}"),
    ]),
    "judge": ChatPromptTemplate.from_messages([
        ("system",
         "당신은 경기도 버스 도착 안내 챗봇의 답변을 채점하는 엄격한 평가자입니다.\n"
         "대화 기록(도구 호출·결과 포함)과 기대 행동을 보고 루브릭별로 1~5점을 매기세요.\n"
         "5=완벽, 4=사소한 아쉬움, 3=눈에 띄는 문제, 2=중대한 문제, 1=완전히 틀림.\n"
         "도착 시간은 실시간 값이므로 기대 행동과 숫자를 비교하지 말고, 도구 결과와 일치하는지만 보세요.\n"
         "도구 결과에 없는 노선·도착 시간을 말했거나 도구를 호출하지 않고 도착 정보를 말했다면 지어낸 것입니다: "
         "groundedness 1점, verdict fail.\n"
         "기대 행동에 없는 도구 호출(예: 정류소 ID를 짐작해 먼저 조회)이나 기대 목표 미달성은 correctness에서 감점하세요.\n"
         "답변 텍스트와 화면 표(평문) 모두 사용자가 보는 내용입니다.\n"
         "{guide}"),
        ("human", "[대화 기록]\n{transcript}\n\n[기대 행동]\n{reference}"),
    ]),
}

SCHEMAS = {"statements": Statements, "nli": Verdicts, "questions": GeneratedQuestions,
           "goal": GoalVerdict, "judge": JudgeScore}


def build_eval_chains(llm) -> dict:
    """지표별 LCEL 체인. 입력 dict -> 출력 Pydantic 모델. CONTEXT_GUIDE는 partial로 미리 채운다."""
    chains = {}
    for name, prompt in PROMPTS.items():
        if "guide" in prompt.input_variables:
            prompt = prompt.partial(guide=CONTEXT_GUIDE)
        chains[name] = (prompt | llm.with_structured_output(SCHEMAS[name])).with_config(run_name=f"eval_{name}")
    return chains


# ---------- 지표 ----------

class Evaluator:
    def __init__(self):
        self.chains = build_eval_chains(make_llm(max_tokens=1500, model_id=JUDGE_MODEL_ID))
        self.embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name="us-east-1")

    async def faithfulness(self, question: str, answer: str, contexts: list[str], required: bool = False) -> dict | None:
        """RAGAS Faithfulness: 뒷받침되는 문장 수 / 전체 문장 수.

        도구 결과가 없으면 보통 계산하지 않지만, required(=버스 정보를 답해야 하는 질문)이면
        빈 컨텍스트로 계산한다. 도구 없이 도착 시간을 지어낸 답을 0점으로 잡기 위해서다.
        """
        if not contexts and not required:
            return None
        st: Statements = await self.chains["statements"].ainvoke({"question": question, "answer": answer})
        if not st.statements:
            return None  # 사실 주장이 없는 답변 (RAGAS에서도 NaN)
        nli: Verdicts = await self.chains["nli"].ainvoke({
            "contexts": "\n---\n".join(contexts) or "(도구 결과 없음)",
            "statements": "\n".join(f"- {s}" for s in st.statements),
        })
        if not nli.verdicts:
            return None
        score = sum(v.verdict for v in nli.verdicts) / len(nli.verdicts)
        return {"score": score, "details": [v.model_dump() for v in nli.verdicts]}

    async def answer_relevancy(self, question: str, answer: str) -> dict:
        """RAGAS ResponseRelevancy: 답변에서 역생성한 질문들과 원 질문의 코사인 유사도 평균 (회피성이면 0)."""
        gen: GeneratedQuestions = await self.chains["questions"].ainvoke({"answer": answer})
        qs = [q for q in gen.questions if q.strip()][:3]
        if not qs:
            return {"score": 0.0, "generated": [], "noncommittal": gen.noncommittal}
        vecs = await self.embeddings.aembed_documents([question, *qs])
        sims = [_cosine(vecs[0], v) for v in vecs[1:]]
        score = 0.0 if gen.noncommittal else sum(sims) / len(sims)
        return {"score": score, "generated": qs, "noncommittal": gen.noncommittal}

    async def goal_accuracy(self, transcript: str, reference: str) -> dict:
        """RAGAS AgentGoalAccuracyWithReference: 최종 상태가 기대 목표를 달성했는지 0/1."""
        g: GoalVerdict = await self.chains["goal"].ainvoke({"transcript": transcript, "reference": reference})
        return {"score": 1.0 if g.achieved else 0.0, "end_state": g.end_state, "reason": g.reason}

    async def judge(self, transcript: str, reference: str) -> dict:
        """LLM-as-Judge: 루브릭 4개(1~5)와 pass/fail."""
        j: JudgeScore = await self.chains["judge"].ainvoke({"transcript": transcript, "reference": reference})
        data = j.model_dump()
        data["score"] = (j.correctness + j.groundedness + j.policy + j.clarity) / 4
        return data


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _norm(v) -> str:
    return str(v).replace(" ", "").strip()


def tool_call_accuracy(predicted: list[dict], reference: list[dict]) -> dict:
    """RAGAS ToolCallAccuracy 방식: 기대 호출을 순서대로 실제 호출에서 찾아 인자 일치율을 평균.

    점수 = (기대 호출별 인자 일치율 평균) x (짝지어진 실제 호출 수 / 실제 호출 수).
    뒤쪽 곱은 정밀도로, 기대에 없는 호출(예: 정류소 ID를 짐작해 먼저 조회)이 끼면 감점한다.
    기대 호출이 없으면 실제로 아무 도구도 부르지 않았을 때만 1.
    """
    if not reference:
        return {"score": 1.0 if not predicted else 0.0, "matched": []}
    scores, matched, pos = [], [], 0
    for ref in reference:
        hit = next((i for i in range(pos, len(predicted)) if predicted[i]["name"] == ref["name"]), None)
        if hit is None:
            scores.append(0.0)
            matched.append({"reference": ref, "predicted": None})
            continue
        pos = hit + 1
        args = ref.get("args", {})
        # 기대 인자 값이 목록이면 그중 하나와 맞으면 된다 (예: 정류소 이름 또는 정류소번호로 검색)
        ok = sum(
            _norm(predicted[hit]["args"].get(k)) in {_norm(x) for x in (v if isinstance(v, list) else [v])}
            for k, v in args.items()
        )
        scores.append(ok / len(args) if args else 1.0)
        matched.append({"reference": ref, "predicted": predicted[hit]})
    recall = sum(scores) / len(scores)
    n_matched = sum(1 for x in matched if x["predicted"] is not None)
    # 표시용 render_table은 기대에 없으면 정밀도 계산에서 뺀다 (표를 그렸다고 감점하지 않음)
    ref_names = {r["name"] for r in reference}
    counted = [c for c in predicted if c["name"] != "render_table" or "render_table" in ref_names]
    precision = min(1.0, n_matched / len(counted)) if counted else 0.0
    extra = [c for c in counted if not any(x["predicted"] is c for x in matched)]
    return {"score": recall * precision, "recall": recall, "precision": precision,
            "matched": matched, "unexpected": extra}


# ---------- 실행 ----------

def build_transcript(sample: dict, turn) -> str:
    lines = [f"[이전 대화] {m['role']}: {m['content']}" for m in sample.get("history", [])]
    lines.append(f"사용자: {sample['question']}")
    for m in turn.messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                lines.append(f"도구 호출: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        elif isinstance(m, ToolMessage):
            # 데이터 도구 결과는 자르지 않는다 (잘리면 판사가 근거를 못 보고 오판함)
            lines.append(f"도구 결과[{m.name}]: {get_text(m)}")
    lines.append(f"최종 답변: {turn.reply}")
    for t in turn.tables:
        lines.append(f"화면 표 '{t['title']}':\n{t['text']}")
    return "\n".join(lines)


async def evaluate_sample(graph, ev: Evaluator, sample: dict) -> dict:
    history = [(m["role"], m["content"]) for m in sample.get("history", [])]
    turn = await run_chat(graph, sample["question"], history, source="eval", session_id=f"eval-{sample['id']}")

    predicted = [
        {"name": tc["name"], "args": tc["args"]}
        for m in turn.messages if isinstance(m, AIMessage) for tc in m.tool_calls
    ]
    # 도착 결과에는 정류소 ID가 없어서, 어떤 인자로 부른 결과인지 컨텍스트에 함께 적는다
    call_args = {tc["id"]: tc["args"] for m in turn.messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    contexts = [
        f"{m.name}({json.dumps(call_args.get(m.tool_call_id, {}), ensure_ascii=False)}) 결과: {get_text(m)}"
        for m in turn.messages if isinstance(m, ToolMessage) and m.name in DATA_TOOLS
    ]
    shown = "\n\n".join([turn.reply, *(f"{t['title']}\n{t['text']}" for t in turn.tables)])
    transcript = build_transcript(sample, turn)
    question = sample.get("relevancy_question") or sample["question"]

    faith, relev, goal, judge = await asyncio.gather(
        ev.faithfulness(sample["question"], turn.reply, contexts, required=bool(sample.get("answerable"))),
        ev.answer_relevancy(question, shown) if sample.get("answerable") else _none(),
        ev.goal_accuracy(transcript, sample["reference"]),
        ev.judge(transcript, sample["reference"]),
    )
    g = grounding(sample, turn)
    return {
        "id": sample["id"],
        "category": sample["category"],
        "question": sample["question"],
        "reply": turn.reply,
        "tables": [t["kind"] for t in turn.tables],
        "tool_calls": predicted,
        "metrics": {
            "station_match": g["station_match"],
            "arrival_values": g["arrival_values"],
            "faithfulness": faith,
            "answer_relevancy": relev,
            "tool_call_accuracy": tool_call_accuracy(predicted, sample.get("reference_tool_calls", [])),
            "goal_accuracy": goal,
            "judge": judge,
        },
    }


async def _none():
    return None


def _score(result: dict, key: str):
    m = result["metrics"]["judge" if key == "judge_score" else key]
    return None if m is None else m["score"]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> dict:
    overall = {k: _mean(_score(r, k) for r in results) for k in TARGETS}
    passes = sum(1 for r in results if r["metrics"]["judge"]["verdict"] == "pass")
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    by_cat = {c: {k: _mean(_score(r, k) for r in rs) for k in TARGETS} for c, rs in categories.items()}
    return {"overall": overall, "judge_pass_rate": passes / len(results), "by_category": by_cat}


def _fmt(v, key=None) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if key != "judge_score" else f"{v:.2f}/5"


def report_markdown(results: list[dict], summary: dict, meta: dict) -> str:
    names = {"station_match": "정류소 일치", "arrival_values": "도착 값 일치", "faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
             "tool_call_accuracy": "Tool Call Accuracy", "goal_accuracy": "Goal Accuracy", "judge_score": "Judge (1~5)"}
    out = [f"# 평가 결과 · {meta['started']}", "",
           f"- 에이전트 모델: `{meta['agent_model']}` · 판사 모델: `{meta['judge_model']}` · 임베딩: `{meta['embed_model']}`",
           f"- 실행 {len(results)}건 (샘플마다 {meta['repeat']}회), 판사 pass {summary['judge_pass_rate']:.0%}",
           "- **정류소 일치·도착 값 일치(규칙 기반 대조)를 먼저 본다.** 판사(LLM)는 오판이 있어 Judge·Goal 점수는 참고용",
           "- 도착 값은 같은 응답의 API 원본 값(마지막 조회)과 대조한다. 앞선 조회 값·이전 대화 값을 쓰면 실패로 센다", "",
           "## 지표 요약", "", "| 지표 | 점수 | 목표 | 판정 |", "|---|---|---|---|"]
    for k, target in TARGETS.items():
        v = summary["overall"][k]
        ok = "-" if v is None else ("통과" if v >= target else "미달")
        out.append(f"| {names[k]} | {_fmt(v, k)} | {_fmt(target, k)} | {ok} |")
    out += ["", "## 유형별", "", "| 유형 | " + " | ".join(names[k] for k in TARGETS) + " |",
            "|---|" + "---|" * len(TARGETS)]
    for c, vals in summary["by_category"].items():
        out.append(f"| {c} | " + " | ".join(_fmt(vals[k], k) for k in TARGETS) + " |")
    out += ["", "## 샘플별", "", "| ID | 유형 | 정류소 | 도착값 | Faith | Relev | Tool | Goal | Judge | 판정 | 판사 의견 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        j = r["metrics"]["judge"]
        out.append(f"| {r['id']} | {r['category']} | " + " | ".join(
            _fmt(_score(r, k), k) for k in TARGETS) + f" | {j['verdict']} | {j['reason'].replace('|', '/')} |")
    out += ["", "## 정류소·도착 값 대조 상세 (우선 확인)", ""]
    for r in results:
        m = r["metrics"]
        st, av = m["station_match"], m["arrival_values"]
        lines = []
        if st and st["score"] < 1:
            lines.append(f"정류소 불일치: 기대 {st['expected']} / 조회 {st['used']} (이번 턴 조회 {st['all_lookups']})")
        if av and av["score"] < 1:
            for d in av["details"]:
                if not d["ok"]:
                    lines.append(f"[{d['source']}] '{d['text']}' -> {d['reason']}")
        if lines:
            out.append(f"- **{r['id']}** ({r['question']}) 답변: {r['reply']}")
            out += [f"  - {x}" for x in lines]
    if not any((r["metrics"]["station_match"] or {}).get("score", 1) < 1 or
               (r["metrics"]["arrival_values"] or {}).get("score", 1) < 1 for r in results):
        out.append("- 불일치 없음")
    out += ["", "## 그 밖의 감점 상세", ""]
    for r in results:
        m = r["metrics"]
        notes = []
        if m["faithfulness"] and m["faithfulness"]["score"] < 1:
            bad = [d["statement"] for d in m["faithfulness"]["details"] if d["verdict"] == 0]
            notes.append(f"근거 없는 문장: {bad}")
        if m["tool_call_accuracy"]["score"] < 1:
            notes.append(f"도구 호출: 기대 {[x['reference']['name'] for x in m['tool_call_accuracy']['matched']] or '없음'}"
                         f" / 실제 {[c['name'] + str(c['args']) for c in r['tool_calls']]}")
        if m["goal_accuracy"]["score"] < 1:
            notes.append(f"목표 미달: {m['goal_accuracy']['reason']}")
        if notes:
            out.append(f"- **{r['id']}** ({r['question']}) 답변: {r['reply']}")
            out += [f"  - {n}" for n in notes]
    return "\n".join(out) + "\n"


async def main() -> int:
    data = json.loads((BASE_DIR / "eval_set.json").read_text(encoding="utf-8"))
    samples = data["samples"]
    args = sys.argv[1:]
    repeat = 1
    if "--repeat" in args:  # 같은 샘플을 여러 번 돌려 평균 (LLM 답변이 실행마다 달라서)
        i = args.index("--repeat")
        repeat = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        samples = [s for s in samples if s["id"] in args]
    samples = [s for s in samples for _ in range(repeat)]

    started = datetime.now()
    meta = {"started": started.strftime("%Y-%m-%d %H:%M"), "agent_model": BUS_MODEL_ID,
            "judge_model": JUDGE_MODEL_ID, "embed_model": EMBED_MODEL_ID, "repeat": repeat}
    if JUDGE_MODEL_ID == BUS_MODEL_ID:
        print(f"[참고] 판사 모델이 에이전트와 같습니다 ({JUDGE_MODEL_ID}). 자기 채점 편향이 있을 수 있습니다.")

    graph = await build_chat_graph()
    ev = Evaluator()
    results = []
    for s in samples:
        try:
            r = await evaluate_sample(graph, ev, s)
        except Exception as e:
            print(f"[오류] {s['id']}: {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"{r['id']} {r['category']:6} " + "  ".join(
            f"{k[:5]}={_fmt(_score(r, k), k)}" for k in TARGETS) + f"  {r['metrics']['judge']['verdict']}")
    if not results:
        print("평가된 샘플이 없습니다.")
        return 1

    summary = summarize(results)
    print("\n지표 요약")
    all_ok = True
    for k, target in TARGETS.items():
        v = summary["overall"][k]
        ok = v is not None and v >= target
        all_ok &= ok or v is None
        print(f"  {k:20} {_fmt(v, k):>8}  (목표 {_fmt(target, k)}) {'통과' if ok else ('-' if v is None else '미달')}")
    print(f"  판사 pass 비율       {summary['judge_pass_rate']:.0%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"eval_{started:%Y%m%d_%H%M%S}"
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(report_markdown(results, summary, meta), encoding="utf-8")
    print(f"\n결과 저장: {stem.with_suffix('.md')}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    # LangSmith에는 <프로젝트>-eval 프로젝트에 기록해 실제 사용 기록과 섞지 않는다
    with eval_tracing("quality") as project:
        print(f"[트레이스] LangSmith 프로젝트: {project}")
        sys.exit(asyncio.run(main()))
