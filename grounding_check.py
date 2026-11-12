"""정류소·도착 값 대조 (규칙 기반, LLM 없음) — 평가에서 판사 점수보다 우선해서 보는 지표.

판사(LLM)는 행선지를 정류소로 착각하거나 둘째 차 시간을 첫째 차로 읽는 등 오판이 있어,
"고른 정류소를 실제로 조회했는가"와 "답변·표의 도착 값이 같은 응답의 API 원본 값과 같은가"를 규칙으로 직접 확인한다.

station_match
    기대 정류소 ID(reference_tool_calls의 get_bus_arrival_list station_id)와
    이번 턴에서 마지막으로 성공한 도착 조회의 정류소 ID가 같으면 1.
arrival_values
    답변 문장과 도착 표의 (노선, N분, M정류장 전)을 하나씩 뽑아,
    같은 턴의 마지막 도착 조회 결과(원본 스냅숏)와 대조한 일치 비율.
    맞지 않는 값은 원인을 표시한다:
      stale_snapshot  같은 턴의 앞선 조회 값(그 사이 도착 정보가 바뀜)을 썼다
      stale_history   이전 대화에 있던 숫자를 다시 썼다 (새로 조회한 값이 아님)
      unknown_route   조회 결과에 없는 노선을 말했다
      mismatch        조회 결과와 다른 값을 말했다
"""

import json
import re

from langchain_core.messages import AIMessage, ToolMessage

from bus_agent import get_text

ROUTE_RE = re.compile(r"([A-Za-z가-힣]*\d+(?:-\d+)?)\s*번")
MIN_RE = re.compile(r"(\d+)\s*분")
STOP_RE = re.compile(r"(\d+)\s*정류장")
LINE_ROUTE_RE = re.compile(r"^\s*([A-Za-z가-힣]*\d+(?:-\d+)?)\s*\(")
CLAUSE_SPLIT_RE = re.compile(r"[.!?,/]|그리고|그 다음|그다음|이어서")


def _route_key(name) -> str:
    """'1241(예약)' -> '1241', 'N8844' -> 'n8844'. 답변 속 '1241번'과 맞추기 위해 괄호·공백을 뗀다."""
    return re.sub(r"\(.*?\)|\s", "", str(name)).lower()


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------- 원본 스냅숏 ----------

def arrival_snapshots(messages) -> list[dict]:
    """이번 턴의 도착 조회 결과를 호출 순서대로. [{"station_id", "live": {노선: 값}, "idle": set}]"""
    args = {tc["id"]: tc["args"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    snaps = []
    for m in messages:
        if not (isinstance(m, ToolMessage) and m.name == "get_bus_arrival_list"):
            continue
        raw = get_text(m)
        start = raw.find("{")
        if start < 0:
            continue  # 오류 문장 (예: 없는 정류소 ID)
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError:
            continue
        live = {}
        for r in data.get("도착정보_있는_노선", []):
            live[_route_key(r.get("routeName"))] = {
                "minutes": {x for x in (_int(r.get("predictTime1")), _int(r.get("predictTime2"))) if x is not None},
                "stops": {x for x in (_int(r.get("locationNo1")), _int(r.get("locationNo2"))) if x is not None},
                "first": (_int(r.get("predictTime1")), _int(r.get("locationNo1"))),
                "second": (_int(r.get("predictTime2")), _int(r.get("locationNo2"))),
            }
        idle = {_route_key(x) for x in data.get("도착정보_없는_노선", [])}
        snaps.append({"station_id": str(args.get(m.tool_call_id, {}).get("station_id", "")), "live": live, "idle": idle})
    return snaps


# ---------- 답변·표에서 주장 뽑기 ----------

def _routes_in(clause: str) -> list[str]:
    routes = ROUTE_RE.findall(clause)
    if not routes:  # 표 형식 줄: '250(궁내동) 1분 · 1정류장 전'
        m = LINE_ROUTE_RE.match(clause)
        routes = [m.group(1)] if m else []
    return [_route_key(r) for r in routes]


def extract_claims(text: str) -> list[dict]:
    """문장을 절 단위로 나눠 (노선들, 분, 정류장 수)를 뽑는다.

    노선 없이 시간만 있는 절('다음 차는 17분 후')은 같은 줄 앞 절의 노선을 잇는다. 줄이 바뀌면 잇지 않는다.
    """
    claims = []
    for line in text.splitlines():
        claims += _line_claims(line)
    return claims


def _line_claims(line: str) -> list[dict]:
    claims, last_routes = [], []
    for clause in CLAUSE_SPLIT_RE.split(line):
        routes = _routes_in(clause)
        minutes = [int(x) for x in MIN_RE.findall(clause)]
        stops = [int(x) for x in STOP_RE.findall(clause)]
        if routes:
            last_routes = routes
        elif minutes or stops:
            routes = last_routes
        if routes and (minutes or stops):
            claims.append({"source": "답변", "text": clause.strip(), "routes": routes, "minutes": minutes, "stops": stops})
    return claims


def table_claims(tables: list[dict]) -> list[dict]:
    """도착 표의 행 하나하나를 주장으로 본다 ('N분 · M정류장 전')."""
    out = []
    for t in tables:
        if t.get("kind") != "arrivals":
            continue
        for row in t.get("rows", []):
            for col in ("first", "second"):
                cell = row.get(col) or ""
                m, s = MIN_RE.search(cell), STOP_RE.search(cell)
                if m:
                    out.append({"source": f"표({col})", "text": f"{row.get('route')} {cell}",
                                "routes": [_route_key(row.get("route"))], "minutes": [int(m.group(1))],
                                "stops": [int(s.group(1))] if s else [], "slot": col})
    return out


# ---------- 대조 ----------

def _matches(claim: dict, snap: dict) -> bool:
    if not all(r in snap["live"] for r in claim["routes"]):
        return False
    vals = [snap["live"][r] for r in claim["routes"]]
    if claim.get("slot"):  # 표는 첫째/둘째 차 칸까지 정확히 맞아야 한다
        exp = vals[0][claim["slot"]]
        return claim["minutes"] == [exp[0]] and (not claim["stops"] or claim["stops"] == [exp[1]])
    minutes_ok = all(any(m in v["minutes"] for v in vals) for m in claim["minutes"])
    stops_ok = all(any(s in v["stops"] for v in vals) for s in claim["stops"])
    return minutes_ok and stops_ok


def _in_history(claim: dict, history_text: str) -> bool:
    """같은 노선과 같은 분이 이전 대화의 한 절에 함께 있었는지."""
    for c in extract_claims(history_text):
        if set(claim["routes"]) & set(c["routes"]) and set(claim["minutes"]) & set(c["minutes"]):
            return True
    return False


def check_arrival_values(reply: str, tables: list[dict], snaps: list[dict], history_text: str) -> dict | None:
    claims = extract_claims(reply) + table_claims(tables)
    if not claims:
        return None
    if not snaps:
        # 도착 조회 없이 도착 시간을 말함 = 전부 근거 없음
        details = [{**c, "ok": False, "reason": "stale_history" if _in_history(c, history_text) else "no_lookup"}
                   for c in claims]
        return {"score": 0.0, "checked": len(claims), "details": details}
    last, earlier = snaps[-1], snaps[:-1]
    details = []
    for c in claims:
        if _matches(c, last):
            details.append({**c, "ok": True})
            continue
        if any(_matches(c, s) for s in earlier):
            reason = "stale_snapshot"
        elif _in_history(c, history_text):
            reason = "stale_history"
        elif not all(r in last["live"] or r in last["idle"] for r in c["routes"]):
            reason = "unknown_route"
        else:
            reason = "mismatch"
        details.append({**c, "ok": False, "reason": reason})
    ok = sum(d["ok"] for d in details)
    return {"score": ok / len(details), "checked": len(details), "details": details}


def expected_station(sample: dict) -> list[str] | None:
    for call in reversed(sample.get("reference_tool_calls", [])):
        if call["name"] == "get_bus_arrival_list" and "station_id" in call.get("args", {}):
            v = call["args"]["station_id"]
            return [str(x) for x in (v if isinstance(v, list) else [v])]
    return None


def check_station(sample: dict, snaps: list[dict]) -> dict | None:
    expected = expected_station(sample)
    if expected is None:
        return None
    used = snaps[-1]["station_id"] if snaps else None
    return {"score": 1.0 if used in expected else 0.0, "expected": expected,
            "used": used, "all_lookups": [s["station_id"] for s in snaps]}


def grounding(sample: dict, turn) -> dict:
    """evaluate.py에서 부르는 진입점. turn은 chat_graph.ChatTurn."""
    snaps = arrival_snapshots(turn.messages)
    history_text = "\n".join(m["content"] for m in sample.get("history", []) if m["role"] == "assistant")
    return {
        "station_match": check_station(sample, snaps),
        "arrival_values": check_arrival_values(turn.reply, turn.tables, snaps, history_text),
    }
