"""구글 로그인(OpenID Connect) + 서버 세션 쿠키 + 사용자별 즐겨찾기.

흐름:
    /auth/google/login     -> 구글 로그인 화면 (state·nonce·PKCE는 authlib가 만들고 임시 쿠키에 보관)
    /auth/google/callback  -> code를 토큰으로 바꾸고 id_token 검증 -> 사용자 찾기/만들기 -> 세션 쿠키 발급
    /auth/logout           -> 세션 삭제
    /api/me                -> 로그인 상태
    /api/favorites[...]    -> 로그인한 사용자의 즐겨찾기 (비로그인은 화면이 localStorage에 저장)

- 사용자는 구글 고유 ID(sub)로만 구분한다. 이메일은 요청하지 않는다 (범위: openid profile).
- 세션 쿠키(ggbus_sid)는 HttpOnly·SameSite=Lax. DB에는 세션 ID의 SHA-256 해시만 저장한다.
- 쓰기 요청은 Origin 헤더가 이 서버와 다르면 거절한다 (CSRF 방어 보강).

필요한 .env: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SESSION_SECRET
선택: GOOGLE_REDIRECT_URI (기본 http://localhost:8000/auth/google/callback), APP_DB_PATH
"""

import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from authlib.integrations.starlette_client import OAuth, OAuthError
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = Path(os.getenv("APP_DB_PATH") or BASE_DIR / "data" / "app.db")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

SESSION_COOKIE = "ggbus_sid"
SESSION_TTL = 30 * 24 * 3600  # 30일
MAX_FAVORITES = 100
STATION_ID = PathParam(pattern=r"^\d{9}$")


def enabled() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET") and SESSION_SECRET)


oauth = OAuth()
if enabled():
    oauth.register(
        "google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile", "code_challenge_method": "S256"},
    )


# ---------- DB ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    UNIQUE (provider, provider_user_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS favorites (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    station_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mobile TEXT NOT NULL DEFAULT '',
    city TEXT NOT NULL DEFAULT '',
    next TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, station_id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as conn, conn:
        conn.executescript(SCHEMA)


def _hash(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


def _upsert_user(provider: str, provider_user_id: str, name: str) -> int:
    now = int(time.time())
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO users (provider, provider_user_id, name, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (provider, provider_user_id) DO UPDATE SET name = excluded.name",
            (provider, provider_user_id, name, now),
        )
        row = conn.execute("SELECT id FROM users WHERE provider = ? AND provider_user_id = ?",
                           (provider, provider_user_id)).fetchone()
        return row["id"]


def _new_session(user_id: int) -> str:
    sid = secrets.token_urlsafe(32)
    now = int(time.time())
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        conn.execute("INSERT INTO sessions (id_hash, user_id, expires_at) VALUES (?, ?, ?)",
                     (_hash(sid), user_id, now + SESSION_TTL))
    return sid


def current_user(request: Request) -> dict | None:
    """세션 쿠키로 로그인한 사용자를 찾는다. 없거나 만료면 None."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT u.id, u.name FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.id_hash = ? AND s.expires_at > ?",
            (_hash(sid), int(time.time())),
        ).fetchone()
    return dict(row) if row else None


def require_user(request: Request) -> dict:
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "로그인이 필요합니다.")
    return user


def same_origin(request: Request) -> None:
    """쓰기 요청의 Origin이 이 서버와 다르면 거절한다 (SameSite 쿠키에 더한 CSRF 방어)."""
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "허용되지 않은 출처입니다.")


# ---------- 로그인 ----------

router = APIRouter()


@router.get("/auth/google/login")
async def google_login(request: Request):
    if not enabled():
        raise HTTPException(503, "구글 로그인이 설정되지 않았습니다 (.env 확인).")
    # 등록한 리다이렉트 URI와 다른 주소(예: 127.0.0.1)로 들어왔으면 그 주소로 옮겨서 시작한다.
    # 그래야 state를 담은 임시 쿠키가 콜백 때 함께 돌아온다.
    target = urlsplit(REDIRECT_URI)
    if request.url.netloc != target.netloc or request.url.scheme != target.scheme:
        return RedirectResponse(f"{target.scheme}://{target.netloc}/auth/google/login")
    return await oauth.google.authorize_redirect(request, REDIRECT_URI)


@router.get("/auth/google/callback")
async def google_callback(request: Request):
    if not enabled():
        raise HTTPException(503, "구글 로그인이 설정되지 않았습니다 (.env 확인).")
    try:
        token = await oauth.google.authorize_access_token(request)  # state·nonce·id_token 서명 검증 포함
    except OAuthError as e:
        print(f"구글 로그인 실패: {e.error}")
        return RedirectResponse("/?login=failed")
    info = token.get("userinfo") or {}
    sub = info.get("sub")
    if not sub:
        return RedirectResponse("/?login=failed")
    name = (info.get("name") or info.get("given_name") or "사용자")[:40]
    sid = _new_session(_upsert_user("google", sub, name))
    request.session.clear()  # 로그인 과정용 임시 값(state 등) 정리
    resp = RedirectResponse("/?login=ok")
    resp.set_cookie(SESSION_COOKIE, sid, max_age=SESSION_TTL, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https", path="/")
    return resp


@router.post("/auth/logout", dependencies=[Depends(same_origin)])
async def logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        with closing(_connect()) as conn, conn:
            conn.execute("DELETE FROM sessions WHERE id_hash = ?", (_hash(sid),))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/api/me")
async def me(request: Request) -> dict:
    user = current_user(request)
    return {"login_enabled": enabled(), "user": {"name": user["name"]} if user else None}


# ---------- 즐겨찾기 ----------

class FavoriteIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    mobile: str = Field(default="", max_length=20)
    city: str = Field(default="", max_length=20)
    next: str = Field(default="", max_length=80)


class FavoriteItem(FavoriteIn):
    id: str = Field(pattern=r"^\d{9}$")


class MergeRequest(BaseModel):
    items: list[FavoriteItem] = Field(default=[], max_length=MAX_FAVORITES)


def _list_favorites(user_id: int) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT station_id AS id, name, mobile, city, next FROM favorites "
            "WHERE user_id = ? ORDER BY created_at, rowid", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def _add_favorites(user_id: int, items: list[FavoriteItem]) -> None:
    """이미 있는 정류소는 그대로 두고 새 정류소만 더한다 (최대 MAX_FAVORITES)."""
    now = int(time.time())
    with closing(_connect()) as conn, conn:
        count = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id = ?", (user_id,)).fetchone()[0]
        for it in items:
            if count >= MAX_FAVORITES:
                break
            cur = conn.execute(
                "INSERT OR IGNORE INTO favorites (user_id, station_id, name, mobile, city, next, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, it.id, it.name, it.mobile, it.city, it.next, now),
            )
            count += cur.rowcount


@router.get("/api/favorites")
async def list_favorites(user: dict = Depends(require_user)) -> dict:
    return {"items": _list_favorites(user["id"])}


@router.put("/api/favorites/{station_id}", dependencies=[Depends(same_origin)])
async def put_favorite(body: FavoriteIn, station_id: str = STATION_ID,
                       user: dict = Depends(require_user)) -> dict:
    items = _list_favorites(user["id"])
    if len(items) >= MAX_FAVORITES and all(f["id"] != station_id for f in items):
        raise HTTPException(409, f"즐겨찾기는 {MAX_FAVORITES}곳까지 저장할 수 있어요.")
    _add_favorites(user["id"], [FavoriteItem(id=station_id, **body.model_dump())])
    return {"items": _list_favorites(user["id"])}


@router.delete("/api/favorites/{station_id}", dependencies=[Depends(same_origin)])
async def delete_favorite(station_id: str = STATION_ID, user: dict = Depends(require_user)) -> dict:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM favorites WHERE user_id = ? AND station_id = ?", (user["id"], station_id))
    return {"items": _list_favorites(user["id"])}


@router.post("/api/favorites/merge", dependencies=[Depends(same_origin)])
async def merge_favorites(body: MergeRequest, user: dict = Depends(require_user)) -> dict:
    """로그인 직후 이 브라우저(localStorage)의 즐겨찾기를 계정으로 합친다."""
    _add_favorites(user["id"], body.items)
    return {"items": _list_favorites(user["id"])}
