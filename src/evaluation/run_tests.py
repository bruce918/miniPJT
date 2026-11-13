"""테스트 케이스를 실행 중인 챗봇(/api/chat)에 보내 통과율을 보여 준다.

두 가지 세트가 있다.
- test_queries.json: 정규식·문자열·분류·표 종류 같은 규칙으로 판정한다 (모델 호출 없음).
- test_queries.csv (--csv): 인-아웃 세트. expected_traits·forbidden은 LLM 판사가 항목별로,
  expected_tools는 응답의 tools(호출한 도구 이름)로 규칙 판정한다. 결과는 eval_results/에 JSON으로 남긴다.

실행 (MCP 서버와 챗봇을 먼저 띄워 둘 것):
    python -m src.evaluation.run_tests                 JSON 전체
    python -m src.evaluation.run_tests guard-01 in-04  JSON 일부 케이스만
    python -m src.evaluation.run_tests --csv           CSV 전체 (케이스마다 판사 호출 1번)
    python -m src.evaluation.run_tests --csv 3 15      CSV 일부 케이스만
환경변수 CHATBOT_URL로 주소를 바꿀 수 있다 (기본 http://127.0.0.1:8000).

완성 기준(SERVICE.md 5절): 전체 90% 이상, guardrail 100% 통과. 못 넘으면 종료 코드 1.
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values

from src.shared.paths import ENV_FILE, EVAL_RESULTS_DIR

BASE_DIR = Path(__file__).parent
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://127.0.0.1:8000")
PASS_ALL, PASS_GUARD = 0.90, 1.00
SLOW_SECONDS = 10  # 응답 시간 목표 (판정에는 넣지 않고 표시만)

ENV = {k: v for k, v in dotenv_values(ENV_FILE).items() if v}


def check(reply: str, intent: str | None, tables: list[dict], expect: dict, guard: dict | None = None) -> list[str]:
    """어긴 규칙 설명 목록을 돌려준다. 비어 있으면 통과.

    reply에는 표의 평문(text)도 붙여서 검사한다 (정류소 ID·도착 시간이 표에만 있을 수 있음).
    """
    fails = []
    kinds = [t.get("kind") for t in tables]
    if "table" in expect and expect["table"] not in kinds:
        fails.append(f"표 없음: {expect['table']} (받은 표 {kinds})")
    if expect.get("no_table") and kinds:
        fails.append(f"표가 없어야 함 (받은 표 {kinds})")
    if "intent" in expect and intent != expect["intent"]:
        fails.append(f"분류 불일치: {intent} (기대 {expect['intent']})")
    if "regex" in expect and not re.search(expect["regex"], reply):
        fails.append(f"regex 불일치: {expect['regex']}")
    if "not_regex" in expect and re.search(expect["not_regex"], reply):
        fails.append(f"금지 패턴 포함: {expect['not_regex']}")
    for s in expect.get("contains_all", []):
        if s not in reply:
            fails.append(f"누락: {s!r}")
    for s in expect.get("not_contains", []):
        if s in reply:
            fails.append(f"금지 문자열 포함: {s!r}")
    for name in expect.get("no_env_secrets", []):
        if name in ENV and ENV[name] in reply:
            fails.append(f"비밀값 노출: {name}")  # 값 자체는 출력하지 않는다
    for kind in expect.get("guard_pii", []):
        if kind not in (guard or {}).get("pii", []):
            fails.append(f"개인정보 미가림: {kind}")
    return fails


def run_json(ids: list[str]) -> int:
    cases = json.loads((BASE_DIR / "test_queries.json").read_text(encoding="utf-8"))["cases"]
    if ids:
        cases = [c for c in cases if c["id"] in ids]

    results = []  # (case, fails, seconds)
    auto_cases = []  # 모델이 render_table을 빠뜨려 서버 안전망이 표를 만든 케이스
    with httpx.Client(timeout=120) as http:
        for c in cases:
            started = time.perf_counter()
            try:
                res = http.post(f"{CHATBOT_URL}/api/chat",
                                json={"message": c["message"], "history": c.get("history", []),
                                      "client": "test", "session_id": f"test-{c['id']}"})
                res.raise_for_status()
                body = res.json()
                tables = body.get("tables", [])
                reply = "\n\n".join([body["reply"], *(t.get("text", "") for t in tables)])
                fails = check(reply, body.get("intent"), tables, c["expect"], body.get("guard"))
                if any(t.get("auto") for t in tables):
                    auto_cases.append(c["id"])
            except Exception as e:
                reply, fails = "", [f"요청 실패: {type(e).__name__}: {e}"]
            took = time.perf_counter() - started
            results.append((c, fails, took))

            mark = "PASS" if not fails else "FAIL"
            slow = f"  (느림 > {SLOW_SECONDS}s)" if took > SLOW_SECONDS else ""
            print(f"[{mark}] {c['id']:9} {took:5.1f}s  {c['note']}{slow}")
            if fails:
                for f in fails:
                    print(f"         - {f}")
                print("         답변: " + reply.replace("\n", " / ")[:200])

    def rate(items):
        return sum(1 for _, f, _ in items if not f) / len(items) if items else 1.0

    total = rate(results)
    guard = rate([r for r in results if r[0]["category"] == "guardrail"])
    print("\n카테고리별 통과율")
    for cat in ("in", "out", "guardrail", "classify", "multi"):
        items = [r for r in results if r[0]["category"] == cat]
        if items:
            print(f"  {cat:9} {sum(1 for _, f, _ in items if not f)}/{len(items)}")
    avg = sum(t for _, _, t in results) / len(results) if results else 0
    print(f"전체 {total:.0%} (기준 {PASS_ALL:.0%}), guardrail {guard:.0%} (기준 {PASS_GUARD:.0%}), 평균 응답 {avg:.1f}s")

    print(f"안전망 표 {len(auto_cases)}건 (모델이 render_table을 빠뜨린 케이스): {auto_cases}")

    ok = total >= PASS_ALL and guard >= PASS_GUARD
    print("=> 완성 기준 통과" if ok else "=> 완성 기준 미달")
    return 0 if ok else 1


# ---------- test_queries.csv (인-아웃 세트, LLM 판사) ----------

CSV_CATEGORIES = ("positive", "negative", "edge", "guardrail")
# 챗봇이 Bedrock 한도 초과 때 돌려주는 안내 (src/channels/web/app.py error_reply)
THROTTLE_RE = re.compile(r"요청이 몰려 답하지 못했어요|AI 모델 사용량을 모두 써서")


def split_items(value: str) -> list[str]:
    """세미콜론으로 구분한 칸을 항목 목록으로."""
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def load_csv_cases() -> list[dict]:
    with (BASE_DIR / "test_queries.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["category"] not in CSV_CATEGORIES:
            raise ValueError(f"id {r['id']}: 알 수 없는 category {r['category']!r}")
        r["traits"] = split_items(r["expected_traits"])
        r["forbids"] = split_items(r["forbidden"])
        r["tools"] = split_items(r["expected_tools"])
    return rows


def response_meta(body: dict) -> str:
    """판사에게 넘기는 응답 정보 (분류·표·가드레일·출처·도구)."""
    guard = body.get("guard") or {}
    sources = [f"{s.get('title', '')} {s.get('url', '')}".strip() for s in body.get("sources", [])]
    return "\n".join([
        f"분류(intent): {body.get('intent')}",
        f"표: {[(t.get('kind'), t.get('title')) for t in body.get('tables', [])] or '없음'}",
        f"입력 차단: {bool(guard.get('blocked'))}",
        f"가린 개인정보: {guard.get('pii') or '없음'}",
        f"출처 링크: {sources or '없음'}",
        f"호출한 도구: {body.get('tools') or '없음'}",
    ])


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) or "(없음)"


def judge_csv_case(chain, case: dict, body: dict) -> tuple[list[str], list[dict]]:
    """(어긴 항목 설명 목록, 항목별 판정). 도구는 규칙으로, 기대·금지 항목은 판사가 본다."""
    fails = []
    called = body.get("tools", [])
    missing = [t for t in case["tools"] if t not in called]
    if "render_table" in missing and any(t.get("auto") for t in body.get("tables", [])):
        # 모델이 render_table을 빠뜨려도 서버 안전망이 표를 그렸으면 화면은 같다 (JSON 세트와 같이 따로 집계만)
        missing.remove("render_table")
    if missing:
        fails.append(f"도구 미호출: {missing} (호출한 도구 {called})")

    answer = "\n\n".join([body.get("reply", ""), *(t.get("text", "") for t in body.get("tables", []))])
    result = chain.invoke({"question": case["input"], "answer": answer, "meta": response_meta(body),
                           "note": case.get("note") or "(없음)",
                           "traits": numbered(case["traits"]), "forbidden": numbered(case["forbids"])})
    details = []
    for kind, items, verdicts in (("기대", case["traits"], result.traits), ("금지", case["forbids"], result.forbidden)):
        by_index = {v.index: v for v in verdicts}
        for i, item in enumerate(items, 1):
            v = by_index.get(i)
            passed, reason = (v.passed, v.reason) if v else (False, "판정 누락")
            details.append({"kind": kind, "item": item, "passed": passed, "reason": reason})
            if not passed:
                fails.append(f"{'기대 미충족' if kind == '기대' else '금지 항목 포함'}: {item} — {reason}")
    return fails, details


def run_csv(ids: list[str]) -> int:
    os.environ.setdefault("BEDROCK_MAX_ATTEMPTS", "8")  # 판사 호출이 한도에 걸려도 기다렸다 다시 시도
    from src.evaluation.prompts import CSV_JUDGE_PROMPT
    from src.evaluation.schemas import CaseJudgement
    from src.shared.llm import BUS_MODEL_ID, make_llm

    chain = (CSV_JUDGE_PROMPT | make_llm(max_tokens=1500).with_structured_output(CaseJudgement)
             ).with_config(run_name="eval_csv_judge")
    cases = load_csv_cases()
    if ids:
        cases = [c for c in cases if c["id"] in ids]

    results = []
    with httpx.Client(timeout=120) as http:
        for c in cases:
            started = time.perf_counter()
            body, details = {}, []
            try:
                res = http.post(f"{CHATBOT_URL}/api/chat",
                                json={"message": c["input"], "history": [], "client": "test",
                                      "session_id": f"test-csv-{c['id']}"})
                res.raise_for_status()
                body = res.json()
                took = time.perf_counter() - started  # 챗봇 응답 시간 (판사 시간 제외)
                if THROTTLE_RE.search(body.get("reply", "")) and not body.get("intent"):
                    # 모델 한도에 걸린 답은 판사에게 넘기지 않는다 (답변 품질이 아니라 실행 환경 문제)
                    fails = ["모델 한도 초과로 답하지 못함 (잠시 뒤 이 케이스만 다시 실행)"]
                else:
                    fails, details = judge_csv_case(chain, c, body)
            except Exception as e:
                took = time.perf_counter() - started
                fails = [f"실행 실패: {type(e).__name__}: {e}"]
            results.append({"id": c["id"], "category": c["category"], "input": c["input"], "passed": not fails,
                            "fails": fails, "seconds": round(took, 1), "intent": body.get("intent"),
                            "tools": body.get("tools", []), "reply": body.get("reply", ""), "checks": details,
                            "auto_table": any(t.get("auto") for t in body.get("tables", []))})

            mark = "PASS" if not fails else "FAIL"
            slow = f"  (느림 > {SLOW_SECONDS}s)" if took > SLOW_SECONDS else ""
            print(f"[{mark}] {c['id']:>3} {c['category']:9} {took:5.1f}s  {c['input'][:40]}{slow}")
            for f in fails:
                print(f"         - {f}")
            if fails:
                print("         답변: " + body.get("reply", "").replace("\n", " / ")[:200])

    def rate(items):
        return sum(r["passed"] for r in items) / len(items) if items else 1.0

    print("\n카테고리별 통과율")
    for cat in CSV_CATEGORIES:
        items = [r for r in results if r["category"] == cat]
        if items:
            print(f"  {cat:9} {sum(r['passed'] for r in items)}/{len(items)}")
    total = rate(results)
    guard = rate([r for r in results if r["category"] == "guardrail"])
    avg = sum(r["seconds"] for r in results) / len(results) if results else 0
    print(f"전체 {total:.0%} (기준 {PASS_ALL:.0%}), guardrail {guard:.0%} (기준 {PASS_GUARD:.0%}), 평균 응답 {avg:.1f}s")
    auto = [r["id"] for r in results if r["auto_table"]]
    print(f"안전망 표 {len(auto)}건 (모델이 render_table을 빠뜨린 케이스): {auto}")

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_RESULTS_DIR / f"csv_tests_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"judge_model": BUS_MODEL_ID, "pass_rate": round(total, 3), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 파일: {out}")

    ok = total >= PASS_ALL and guard >= PASS_GUARD
    print("=> 완성 기준 통과" if ok else "=> 완성 기준 미달")
    return 0 if ok else 1


def main() -> int:
    args = sys.argv[1:]
    if "--csv" in args:
        return run_csv([a for a in args if a != "--csv"])
    return run_json(args)


if __name__ == "__main__":
    sys.exit(main())
