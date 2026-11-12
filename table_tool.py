"""정류소 후보 목록·도착 정보를 화면용 표로 만드는 로컬 도구 (render_table).

표 데이터는 LLM이 옮겨 적지 않는다. 같은 턴에 이미 호출된 MCP 도구 결과
(search_bus_station / get_bus_arrival_list)를 그래프 상태에서 직접 읽어 표로 바꾸므로
값이 틀어질 일이 없다.

- LLM에게는 짧은 확인 문장만 돌려준다 (content).
- 표 자체는 ToolMessage.artifact로 실려 main.py가 응답의 tables로 꺼내 UI에 그린다.
- 표마다 text(평문)도 함께 만든다. 브라우저가 이 평문을 대화 기록에 넣어
  다음 턴에 "4번"처럼 고를 때 정류소 ID를 알 수 있게 한다.
"""

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

SOURCE_TOOL = {"stations": "search_bus_station", "arrivals": "get_bus_arrival_list"}
CROWDED = {1: "여유", 2: "보통", 3: "혼잡", 4: "매우 혼잡"}


def _text(content) -> str:
    """MCP 도구 결과 content는 문자열이거나 텍스트 블록 리스트다."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _latest_result(messages, tool_name: str):
    """이번 턴에서 가장 최근에 성공한 tool_name 결과(JSON)를 찾는다. 없으면 None."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == tool_name:
            raw = _text(m.content)
            start = min((i for i in (raw.find("["), raw.find("{")) if i >= 0), default=-1)
            if start < 0:
                return None  # "찾지 못했습니다" 같은 오류 문장
            try:
                return json.loads(raw[start:])
            except json.JSONDecodeError:
                return None
    return None


def _search_total(messages) -> int | None:
    """search_bus_station 결과 앞의 '(총 N건 중 10건)'에서 전체 건수 N을 꺼낸다."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == SOURCE_TOOL["stations"]:
            found = re.match(r"\(총 (\d+)건", _text(m.content).strip())
            return int(found.group(1)) if found else None
    return None


BASE_DIR = Path(__file__).parent
STATIONS_PATH = BASE_DIR / "gg_stations.json"
NEXT_STOP_CACHE = BASE_DIR / ".cache" / "next_stops.json"  # MCP 서버(next_stop.py)가 쓰는 다음 정류장 캐시
_station_by_id: dict | None = None


def _station(station_id) -> dict | None:
    """정류소 ID -> {name, mobile, city} (gg_stations.json, 처음 쓸 때 한 번만 읽음)."""
    global _station_by_id
    if _station_by_id is None:
        try:
            with open(STATIONS_PATH, encoding="utf-8") as f:
                _station_by_id = {str(s["id"]): s for s in json.load(f)}
        except (OSError, json.JSONDecodeError):
            _station_by_id = {}
    return _station_by_id.get(str(station_id))


def _next_stop(station_id) -> str:
    try:
        with open(NEXT_STOP_CACHE, encoding="utf-8") as f:
            return json.load(f).get(str(station_id), {}).get("next", "")
    except (OSError, json.JSONDecodeError):
        return ""


def _arrival_station_id(messages) -> str | None:
    """이번 턴 마지막 도착 조회(get_bus_arrival_list)에 넘긴 정류소 ID."""
    calls = {tc["id"]: tc["args"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == SOURCE_TOOL["arrivals"]:
            sid = calls.get(m.tool_call_id, {}).get("station_id")
            return str(sid) if sid else None
    return None


def station_title(station_id, fallback: str) -> tuple[str, str]:
    """정류소 ID -> ("○○ 도착 정보", 다음 정류장). 모르는 ID면 (fallback, "")."""
    info = _station(station_id) if station_id else None
    if not info:
        return fallback, ""
    return f"{info['name']} 도착 정보", _next_stop(station_id)


def arrival_title(messages, fallback: str) -> tuple[str, str, str | None]:
    """도착 표 제목은 모델이 아니라 실제로 조회한 정류소 ID의 이름으로 정한다 (ID·번호 대신 이름).

    이름이 같은 정류소를 구분하도록 다음 정류장을 알면 함께 돌려준다 (표 아래 안내에 쓴다).
    반환: (제목, 다음 정류장, 정류소 ID) — 정류소 ID는 화면의 새로고침 버튼이 쓴다.
    """
    sid = _arrival_station_id(messages)
    title, next_stop = station_title(sid, fallback)
    return title, next_stop, sid


def arrivals_table_from_text(station_id: str, raw: str) -> dict | None:
    """get_bus_arrival_list 결과 문자열로 도착 표를 만든다 (새로고침 API용). 오류 문장이면 None."""
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(raw[start:])
    except json.JSONDecodeError:
        return None
    title, next_stop = station_title(station_id, "도착 정보")
    return {**arrivals_table(title, data, next_stop), "station_id": str(station_id)}


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def stations_table(title: str, stations: list[dict], total: int | None = None) -> dict:
    rows = [
        {"no": i, "name": s["name"], "next": s.get("next", ""), "city": s.get("city", ""),
         "mobile": s.get("mobile", ""), "id": s["id"]}
        for i, s in enumerate(stations, 1)
    ]
    # 다음 정류장(방향 구분)은 MCP 서버가 알아냈을 때만 칸을 만든다
    has_next = any(r["next"] for r in rows)
    text = "\n".join(
        f"{r['no']}. {r['name']} ({r['city']}, "
        + (f"다음 정류장 {r['next']}, " if r["next"] else "")
        + f"정류소번호 {r['mobile']}, ID {r['id']})"
        for r in rows
    )
    columns = [
        {"key": "no", "label": "번호", "align": "center"},
        {"key": "name", "label": "정류소"},
        *([{"key": "next", "label": "다음 정류장"}] if has_next else []),
        {"key": "city", "label": "시군"},
        {"key": "mobile", "label": "정류소번호", "align": "center"},
        {"key": "id", "label": "정류소 ID", "align": "center"},
    ]
    return {
        "kind": "stations",
        "title": title,
        "columns": columns,
        "rows": rows,
        "note": (f"총 {total}곳 중 {len(rows)}곳만 보여 드려요. 원하는 곳이 없으면 이름을 더 자세히 입력해 주세요. " if total and total > len(rows) else "")
                + "행을 누르면 그 정류소를 고릅니다.",
        "text": text,
    }


def _eta(minutes, stops) -> str:
    if minutes is None:
        return ""
    s = f"{minutes}분"
    return s + (f" · {stops}정류장 전" if stops is not None else "")


def arrivals_table(title: str, data: dict, next_stop: str = "") -> dict:
    live = sorted(data.get("도착정보_있는_노선", []), key=lambda r: _num(r.get("predictTime1")) or 999)
    rows = []
    for r in live:
        first, second = _num(r.get("predictTime1")), _num(r.get("predictTime2"))
        tags = []
        if _num(r.get("lowPlate1")) == 1:
            tags.append("저상")
        if _num(r.get("crowded1")) in CROWDED:
            tags.append(CROWDED[_num(r["crowded1"])])
        rows.append({
            "route": str(r.get("routeName", "")),
            "dest": r.get("routeDestName", ""),
            "first": _eta(first, _num(r.get("locationNo1"))),
            "second": _eta(second, _num(r.get("locationNo2"))),
            "info": " · ".join(tags),
            "minutes": first,  # UI가 곧 도착(2분 이하) 강조에 쓴다
        })
    idle = data.get("도착정보_없는_노선", [])
    note = " · ".join(x for x in (
        "지금 도착 예정인 버스가 없어요" if not rows else "",
        f"다음 정류장 → {next_stop} 방향" if next_stop else "",
        f"도착 정보 없는 노선: {', '.join(map(str, idle))}" if idle else "",
    ) if x)
    lines = [f"{r['route']}({r['dest']}) {r['first']}" + (f", 다음 {r['second']}" if r["second"] else "") for r in rows]
    return {
        "kind": "arrivals",
        "title": title,
        "columns": [
            {"key": "route", "label": "노선"},
            {"key": "dest", "label": "방면"},
            {"key": "first", "label": "도착 예정"},
            {"key": "second", "label": "다음 차"},
            {"key": "info", "label": "차량"},
        ],
        "rows": rows,
        "note": note,
        "text": "\n".join(lines + ([note] if note else [])),
    }


def fallback_tables(messages) -> list[dict]:
    """모델이 render_table을 빠뜨렸을 때 서버가 대신 만드는 표 (안전망).

    이번 턴 메시지에 도착 정보(노선 1개 이상)가 있으면 도착 표, 아니면 후보 2곳 이상일 때 후보 표.
    auto=True로 표시해 모델이 도구를 빠뜨린 빈도를 알 수 있게 한다.
    """
    arrivals = _latest_result(messages, SOURCE_TOOL["arrivals"])
    if isinstance(arrivals, dict) and arrivals.get("도착정보_있는_노선"):
        title, next_stop, sid = arrival_title(messages, "도착 정보")
        return [{**arrivals_table(title, arrivals, next_stop), "station_id": sid, "auto": True}]
    stations = _latest_result(messages, SOURCE_TOOL["stations"])
    if isinstance(stations, list) and len(stations) >= 2:
        return [{**stations_table("정류소 후보", stations), "auto": True}]
    return []


@tool(response_format="content_and_artifact")
def render_table(
    kind: Literal["stations", "arrivals"],
    title: str,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict | None]:
    """방금 조회한 결과를 사용자 화면에 표로 그린다. 데이터는 직전 도구 결과에서 자동으로 가져오므로 넘길 필요 없다.

    Args:
        kind: "stations" = 직전 search_bus_station 결과(정류소 후보 목록), "arrivals" = 직전 get_bus_arrival_list 결과(도착 정보)
        title: 표 제목 (예: "미금역 정류소 후보", "미금역.2001아울렛 도착 정보")
    """
    data = _latest_result(state["messages"], SOURCE_TOOL[kind])
    if not data:
        return f"표를 그릴 데이터가 없습니다. 먼저 {SOURCE_TOOL[kind]}를 호출해 결과를 받은 뒤 다시 호출하세요.", None
    if kind == "stations":
        table = stations_table(title, data, _search_total(state["messages"]))
        return (f"정류소 {len(table['rows'])}곳을 표로 화면에 보여 줬습니다. 답변에서 목록을 다시 나열하지 마세요. "
                f"답변의 개수는 표에 보이는 {len(table['rows'])}곳으로 말하세요."), table
    # 제목은 모델이 준 값(ID나 다른 이름일 수 있음) 대신 실제로 조회한 정류소의 이름으로 정한다
    title, next_stop, sid = arrival_title(state["messages"], title)
    table = {**arrivals_table(title, data, next_stop), "station_id": sid}
    # 요약할 때 모델이 순서를 직접 비교하지 않도록, 이미 정렬된 앞쪽 노선을 그대로 알려 준다
    first = [f"{r['route']}번 {r['first']}" for r in table["rows"][:3]]
    order = ", ".join(first) if first else "없음"
    return (f"노선 {len(table['rows'])}개 도착 정보를 표로 화면에 보여 줬습니다. 답변에서 표 내용을 반복하지 마세요.\n"
            f"도착이 빠른 순서(이 순서 그대로 요약에 쓰세요): {order}"), table
