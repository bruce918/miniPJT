"""정류소의 '다음 정류장' 이름을 구한다 (같은 이름 정류소의 방향을 구분하려고).

정류소 목록(gg_stations.json)에는 정류소 사이의 순서가 없어서 공공데이터포털 API 두 개를 쓴다.
  1) 경기도_정류소 조회  getBusStationViaRouteListv2(stationId)  -> 그 정류소를 지나는 노선 목록
  2) 경기도_버스노선 조회 getBusRouteStationListv2(routeId)      -> 노선의 정류소 순서
앞쪽 노선 최대 ROUTES_PER_STATION개에서 바로 다음 정류소를 찾고, 갈리면 많이 나온 쪽을 쓴다.

- 노선별 정류소 순서는 프로세스 메모리에, 정류소별 결과는 .cache/next_stops.json에 7일간 캐시한다 (일일 호출 한도 절약).
- API를 쓸 수 없으면(활용신청 전, 장애) 빈 값을 돌려주고, 인증 오류는 10분 동안 다시 부르지 않는다.
  호출하는 쪽은 빈 값이면 '다음 정류장' 칸을 안 보여 주면 된다.
"""

import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()

STATION_URL = "https://apis.data.go.kr/6410000/busstationservice/v2/getBusStationViaRouteListv2"
ROUTE_URL = "https://apis.data.go.kr/6410000/busrouteservice/v2/getBusRouteStationListv2"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "next_stops.json")
CACHE_TTL = 7 * 24 * 3600
ROUTES_PER_STATION = 2
TIMEOUT = 5
AUTH_BACKOFF = 600  # 인증 오류(활용신청 전 등) 뒤 다시 시도하기까지 초

_lock = threading.Lock()
_route_stations: dict[str, list[dict]] = {}
_cache: dict[str, dict] | None = None
_disabled_until = 0.0
_last_error = ""


def status() -> dict:
    """진단용: API를 쓸 수 있는 상태인지."""
    return {"available": time.time() >= _disabled_until, "last_error": _last_error}


def _items(body) -> list[dict]:
    """msgBody 안의 목록(한 건이면 dict로 오기도 한다)을 리스트로."""
    for v in (body or {}).values():
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
    return []


def _get(url: str, **params) -> list[dict] | None:
    """API 호출. 실패하면 None (예외를 던지지 않는다)."""
    global _disabled_until, _last_error
    if time.time() < _disabled_until:
        return None
    key = os.getenv("DATA_GO_KR_KEY")
    if not key:
        _last_error = "DATA_GO_KR_KEY 없음"
        return None
    try:
        res = requests.get(url, params={"serviceKey": key, "format": "json", **params}, timeout=TIMEOUT)
        if res.status_code in (401, 403):
            # 활용신청 전이거나 승인 반영 전: 한동안 부르지 않는다
            _disabled_until = time.time() + AUTH_BACKOFF
            _last_error = f"HTTP {res.status_code} (활용신청·승인 반영 확인 필요)"
            return None
        res.raise_for_status()
        return _items(res.json().get("response", {}).get("msgBody"))
    except Exception as e:  # 네트워크 오류, JSON 형식 오류 등
        _last_error = f"{type(e).__name__}: {e}"[:200]
        return None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def _route_station_list(route_id: str) -> list[dict] | None:
    if route_id in _route_stations:
        return _route_stations[route_id]
    rows = _get(ROUTE_URL, routeId=route_id)
    if rows is None:
        return None
    rows = sorted(rows, key=lambda r: int(r.get("stationSeq") or 0))
    _route_stations[route_id] = rows
    return rows


def _next_on_route(route_id: str, station_id: str, sta_order=None) -> str | None:
    rows = _route_station_list(route_id)
    if not rows:
        return None
    # 경유 노선 목록의 staOrder(이 노선에서 이 정류소의 순번)가 있으면 그 위치를 쓴다.
    # 순환 노선처럼 같은 정류소를 두 번 지나도 정확한 방향의 다음 정류장을 찾기 위해서다.
    if sta_order not in (None, ""):
        for i, r in enumerate(rows):
            if str(r.get("stationSeq")) == str(sta_order) and str(r.get("stationId")) == station_id:
                return str(rows[i + 1].get("stationName", "")) if i + 1 < len(rows) else "종점"
    for i, r in enumerate(rows):
        if str(r.get("stationId")) == station_id:
            return str(rows[i + 1].get("stationName", "")) if i + 1 < len(rows) else "종점"
    return None


def lookup(station_id: str) -> str:
    """정류소 ID의 다음 정류장 이름. 모르면 빈 문자열."""
    station_id = str(station_id)
    with _lock:
        hit = _load_cache().get(station_id)
    if hit and time.time() - hit.get("at", 0) < CACHE_TTL:
        return hit.get("next", "")

    routes = _get(STATION_URL, stationId=station_id)
    if not routes:
        return hit.get("next", "") if hit else ""  # 조회 실패 시 기한이 지난 캐시라도 쓴다
    votes = Counter()
    for r in routes[:ROUTES_PER_STATION]:
        nxt = _next_on_route(str(r.get("routeId")), station_id, r.get("staOrder"))
        if nxt:
            votes[nxt] += 1
    if not votes:
        return ""
    name = votes.most_common(1)[0][0]
    with _lock:
        _load_cache()[station_id] = {"next": name, "at": time.time()}
        _save_cache()
    return name


def lookup_many(station_ids: list[str]) -> dict[str, str]:
    """여러 정류소를 동시에 조회 (후보 표용)."""
    if time.time() < _disabled_until:
        return {sid: _load_cache().get(sid, {}).get("next", "") for sid in station_ids}
    with ThreadPoolExecutor(max_workers=5) as pool:
        return dict(zip(station_ids, pool.map(lookup, station_ids)))
