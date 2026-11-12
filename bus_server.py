# bus_server.py - 경기도 버스도착정보(공공데이터포털) MCP 서버 (fastmcp)
"""노출하는 도구 3개:
  - search_bus_station(name, city)  정류소 이름 -> stationId (공공데이터 정류소 CSV에서 만든 gg_stations.json, API 호출 없음)
공공데이터포털 '경기도_버스도착정보 조회' 서비스 도구 2개:
  - get_bus_arrival_list(station_id)                       정류소의 버스 도착 목록 (버스도착정보목록조회)
  - get_bus_arrival_item(station_id, route_id, sta_order)  특정 노선 1건의 도착 정보 (버스도착정보항목조회)

필요한 환경변수(.env): DATA_GO_KR_KEY (공공데이터포털 일반 인증키, Decoding 키)
실행:
  python bus_server.py          독립 서버로 실행 (streamable-http, 기본 http://127.0.0.1:8765/mcp)
  python bus_server.py stdio    stdio 모드 (클라이언트가 자식 프로세스로 띄울 때)
환경변수 BUS_MCP_HOST / BUS_MCP_PORT 로 주소를 바꿀 수 있습니다.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

import next_stop

load_dotenv()

mcp = FastMCP("gg-bus")

STATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gg_stations.json")
_stations_cache: list | None = None


def _stations() -> list:
    """gg_stations.json(make_stations.py가 만든 경기도 정류소 목록)을 처음 쓸 때 한 번만 읽습니다."""
    global _stations_cache
    if _stations_cache is None:
        with open(STATIONS_PATH, encoding="utf-8") as f:
            _stations_cache = json.load(f)
    return _stations_cache


BASE_URL = "https://apis.data.go.kr/6410000/busarrivalservice/v2"
MAX_STATIONS = 10
MAX_ROWS = 20  # 도구 결과가 길어지면 LLM이 헷갈리므로 상위 몇 건만 돌려줍니다


# 모델이 답변에 쓰는 필드만 남깁니다 (원본은 필드가 30여 개이고 대부분 빈 값)
KEEP_FIELDS = ("routeName", "routeId", "staOrder", "routeDestName",
               "predictTime1", "predictTime2", "locationNo1", "locationNo2",
               "remainSeatCnt1", "remainSeatCnt2", "plateNo1", "plateNo2",
               "lowPlate1", "lowPlate2", "crowded1", "crowded2")


def _slim(row: dict) -> dict:
    return {k: row[k] for k in KEEP_FIELDS if row.get(k) not in ("", None)}


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))   # 들여쓰기 없이 한 줄로


def _minutes(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 9999


_station_ids: set | None = None


def _known_station(station_id: str) -> bool | None:
    """정류소 ID가 정류소 목록에 있는지. 목록 파일이 없으면 None(확인 불가)."""
    global _station_ids
    try:
        if _station_ids is None:
            _station_ids = {s["id"] for s in _stations()}
    except OSError:
        return None
    return station_id in _station_ids


def _call(operation: str, **params) -> str:
    """공공데이터포털을 호출해 사람이 읽을 수 있는 문장으로 돌려줍니다. 예외를 던지지 않습니다."""
    key = os.getenv("DATA_GO_KR_KEY")
    if not key:
        return "DATA_GO_KR_KEY가 설정되지 않았습니다. .env에 공공데이터포털 인증키를 넣어 주세요."
    try:
        res = requests.get(
            f"{BASE_URL}/{operation}",
            params={"serviceKey": key, "format": "json", **params},
            timeout=10,
        )
        if res.status_code != 200:
            # 인증키 오류 등은 본문에 이유가 있으므로 그대로 전달합니다 (키는 URL에만 있고 본문에는 없음)
            reason = res.json().get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
            return (f"API 인증/요청 오류 (HTTP {res.status_code}): {reason.get('returnAuthMsg') or reason.get('errMsg') or res.text[:100]}. "
                    "일시적인 장애가 아니므로 재시도하지 말고 사용자에게 인증키/활용신청 상태 확인을 안내하세요.")
        data = res.json()["response"]
    except Exception as e:
        return f"버스 정보 API 호출에 실패했습니다 ({type(e).__name__}: {e}). 잠시 후 다시 시도하세요."

    header = data.get("msgHeader", {})
    body = data.get("msgBody") or {}
    if not body:
        return f"조회 결과가 없습니다 ({header.get('resultMessage', '결과 없음')}). 정류소 ID 등 입력값을 확인하세요."
    rows = next((v for v in body.values() if isinstance(v, list)), None)
    if rows is None:   # 항목조회는 목록이 아니라 객체 1건으로 옵니다
        return _dump(_slim(next(iter(body.values()))))
    # 도착 정보가 있는 노선만 자세히, 없는 노선은 이름만 모아서 토큰을 아낍니다.
    # 모델이 "가장 먼저 오는 버스"를 직접 비교하다 틀리지 않도록 도착이 빠른 순으로 정렬해 둡니다.
    live = sorted(
        (_slim(r) for r in rows if r.get("predictTime1") not in ("", None)),
        key=lambda r: (_minutes(r.get("predictTime1")), _minutes(r.get("locationNo1"))),
    )
    idle = [str(r.get("routeName")) for r in rows if r.get("predictTime1") in ("", None)]
    result = {"정렬": "도착정보_있는_노선은 도착이 빠른 순서입니다 (첫 항목이 가장 먼저 옴)",
              "도착정보_있는_노선": live[:MAX_ROWS]}
    if idle:
        result["도착정보_없는_노선"] = idle
    return _dump(result)


@mcp.tool()
def search_bus_station(name: str, city: str = "") -> str:
    """정류소 이름이나 정류소번호로 정류소 ID(stationId)를 찾는다. 같은 이름이 여러 개면 방향별 정류소다.

    Args:
        name: 정류소 이름 또는 일부 (예: '미금역'), 또는 정류소 표지판의 정류소번호 숫자 (예: '7079')
        city: 시군으로 좁힐 때 (예: '성남시'). 비우면 전체
    """
    name = name.strip()
    if not name:
        return "정류소 이름이 비어 있습니다. 이름을 입력하세요."
    try:
        pool = [s for s in _stations() if not city or city in s["city"]]
    except OSError:
        return "정류소 데이터(gg_stations.json)가 없습니다. make_stations.py로 먼저 만들어 주세요."
    if name.isdigit():
        # 숫자만 오면 정류소번호(mobile)로 찾는다. 정류소번호는 시군 간에 겹칠 수 있어 여러 건일 수 있다.
        hits = [s for s in pool if s["mobile"] == name] or [s for s in pool if s["id"] == name]
        if not hits:
            return f"정류소번호 '{name}'인 정류소를 찾지 못했습니다. 경기도 내 정류소만 조회할 수 있습니다."
        return _dump(_with_next_stop(hits[:MAX_STATIONS]))
    # 이름이 정확히 같은 것 -> 이름으로 시작하는 것 -> 이름이 포함된 것 순으로 보여 줍니다
    exact = [s for s in pool if s["name"] == name]
    starts = [s for s in pool if s["name"].startswith(name) and s["name"] != name]
    inner = [s for s in pool if name in s["name"] and not s["name"].startswith(name)]
    hits = exact + starts + inner
    if not hits:
        return (f"'{name}' 정류소를 찾지 못했습니다. 경기도 내 정류소만 조회할 수 있으니 "
                "경기도 정류소 이름을 입력하도록 안내하세요 (이름을 줄여 입력해도 됩니다).")
    note = f"(총 {len(hits)}건 중 {MAX_STATIONS}건)\n" if len(hits) > MAX_STATIONS else ""
    return note + _dump(_with_next_stop(hits[:MAX_STATIONS]))


def _with_next_stop(hits: list[dict]) -> list[dict]:
    """후보가 2곳 이상이면 방향 구분용 '다음 정류장'(next)을 붙인다. 모르면 붙이지 않는다.

    정류소 목록 캐시의 dict를 바꾸지 않도록 복사해서 돌려준다.
    """
    if len(hits) < 2:
        return hits
    nexts = next_stop.lookup_many([s["id"] for s in hits])
    return [{**s, "next": nexts[s["id"]]} if nexts.get(s["id"]) else dict(s) for s in hits]


@mcp.tool()
def get_bus_arrival_list(station_id: str) -> str:
    """정류소의 버스 도착 목록 (도착이 빠른 순). predictTime1/2는 도착까지 분, locationNo는 몇 정류장 전, remainSeatCnt는 빈 좌석. 한 노선만 필요하면 get_bus_arrival_item.

    Args:
        station_id: 정류소 ID (숫자 9자리). search_bus_station 결과나 이전 후보 목록에 나온 ID만 쓴다. 짐작하지 말 것
    """
    station_id = station_id.strip()
    if _known_station(station_id) is False:
        # 모델이 정류소번호(4~5자리) 등으로 ID를 짐작해 부르면 API를 부르지 않고 바로 알려 준다
        return (f"'{station_id}'는 정류소 목록에 없는 정류소 ID입니다. ID를 짐작하지 말고 "
                "search_bus_station으로 정류소 이름이나 정류소번호를 검색해 ID를 확인하세요.")
    return _call("getBusArrivalListv2", stationId=station_id)


@mcp.tool()
def get_bus_arrival_item(station_id: str, route_id: str, sta_order: str) -> str:
    """한 노선의 도착 정보. route_id, sta_order는 get_bus_arrival_list 결과의 routeId, staOrder.

    Args:
        station_id: 정류소 ID
        route_id: 노선 ID
        sta_order: 노선 내 정류소 순번
    """
    return _call("getBusArrivalItemv2", stationId=station_id, routeId=route_id, staOrder=sta_order)


if __name__ == "__main__":
    if sys.argv[1:] == ["stdio"]:
        mcp.run()
    else:
        # 로컬에서만 접속되도록 기본은 127.0.0.1에 바인딩합니다 (인증이 없으므로 외부 노출 금지)
        mcp.run(
            transport="streamable-http",
            host=os.getenv("BUS_MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("BUS_MCP_PORT", "8765")),
        )
