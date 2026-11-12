"""경기 버스봇 텔레그램 봇 (롱 폴링).

웹 챗봇 서버(main.py)의 /api/chat·/api/arrivals를 그대로 불러 쓴다. 질문 분류·MCP 도착 조회·정책 검색은
모두 챗봇 서버가 하고, 이 봇은 텔레그램 메시지 <-> 챗봇 API 변환만 맡는다.

실행 (MCP 서버와 챗봇 서버가 먼저 떠 있어야 한다):
    python -m telegram_bot

.env:
    TELEGRAM_BOT_TOKEN       (필수) BotFather에서 받은 토큰
    CHATBOT_URL              (선택) 챗봇 서버 주소, 기본 http://127.0.0.1:8000
    TELEGRAM_ALLOWED_USERS   (선택) 쓸 수 있는 텔레그램 사용자 ID 목록(쉼표 구분). 비우면 누구나 쓸 수 있다.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv

from . import render, table_image

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://127.0.0.1:8000").rstrip("/")
ALLOWED_USERS = {int(x) for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").replace(" ", "").split(",") if x.isdigit()}
MAX_HISTORY = 12  # 대화 기록은 최근 6턴(사용자·봇 메시지 12개)만 챗봇 서버에 보낸다

log = logging.getLogger("telegram_bot")

# 채팅방별 상태 (봇을 다시 켜면 비워진다)
histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)  # 같은 방의 질문은 순서대로 처리


class TelegramError(RuntimeError):
    pass


class Bot:
    def __init__(self, token: str):
        self.api = f"https://api.telegram.org/bot{token}"
        self.tg = httpx.AsyncClient(timeout=httpx.Timeout(70.0))
        self.chatbot = httpx.AsyncClient(base_url=CHATBOT_URL, timeout=httpx.Timeout(150.0))

    async def call(self, method: str, **params):
        r = await self.tg.post(f"{self.api}/{method}", json={k: v for k, v in params.items() if v is not None})
        data = r.json()
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('error_code')} {data.get('description')}")
        return data["result"]

    async def send(self, chat_id: int, text: str, markup: dict | None = None):
        return await self.call("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML",
                               reply_markup=markup, disable_web_page_preview=True)

    async def upload(self, method: str, fields: dict, png: bytes, name: str = "table"):
        """사진을 올리는 요청 (multipart). fields의 dict 값은 JSON 문자열로 보낸다."""
        data = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v))
                for k, v in fields.items() if v is not None}
        r = await self.tg.post(f"{self.api}/{method}", data=data, files={name: (f"{name}.png", png, "image/png")})
        result = r.json()
        if not result.get("ok"):
            raise TelegramError(f"{method}: {result.get('error_code')} {result.get('description')}")
        return result["result"]

    async def send_table(self, chat_id: int, table: dict, caption: str, markup: dict | None, fallback):
        """표를 그림으로 보낸다. 한글 글꼴이 없거나 그리기·전송에 실패하면 글자 메시지(fallback)로 보낸다."""
        if table_image.available():
            try:
                png = await asyncio.to_thread(table_image.render, table, render.now_label()
                                              if table.get("kind") == "arrivals" else "")
                return await self.upload("sendPhoto", {"chat_id": chat_id, "caption": caption,
                                                       "parse_mode": "HTML", "reply_markup": markup}, png, "photo")
            except Exception as e:
                log.warning("표 그림 전송 실패, 글자로 보냄: %s", e)
        return await self.send(chat_id, *fallback)

    # ---------- 챗봇 서버 ----------

    async def ask(self, chat_id: int, query: str):
        """질문 하나를 챗봇 서버에 보내고 답(표 포함)을 텔레그램으로 보낸다."""
        await self.call("sendChatAction", chat_id=chat_id, action="typing")
        payload = {
            "message": query,
            "history": list(histories[chat_id]),
            # LangSmith에서 대화 단위로 묶되, 텔레그램 채팅 ID를 그대로 남기지 않도록 해시한다
            "session_id": "tg-" + hashlib.sha256(str(chat_id).encode()).hexdigest()[:16],
            "client": "telegram",
        }
        try:
            r = await self.chatbot.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            log.warning("챗봇 서버 오류: %s", e)
            await self.send(chat_id, "버스봇 서버에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.")
            return

        reply, tables, sources = data.get("reply", ""), data.get("tables", []), data.get("sources", [])
        guard = data.get("guard") or {}
        if guard.get("pii"):
            # 서버 가드레일이 개인정보를 가렸으면 알려 주고, 기록에도 가린 문장만 남긴다
            await self.send(chat_id, render.escape(f"🔒 개인정보({', '.join(guard['pii'])})는 가려서 처리했어요. "
                                                   "채팅에 개인정보를 보내지 마세요."))
            query = guard.get("masked") or query
        stations = next((t for t in tables if t.get("kind") == "stations"), None)
        arrivals = next((t for t in tables if t.get("kind") == "arrivals"), None)
        if arrivals:
            # 웹과 같이 도착 표가 있으면 모델의 요약 문장 없이 표만 보여 준다
            await self.send_table(chat_id, arrivals, render.arrivals_caption(arrivals),
                                  render.arrivals_markup(arrivals), render.arrivals_message(arrivals))
        elif stations:
            await self.send_table(chat_id, render.stations_for_image(stations), render.stations_caption(reply),
                                  render.stations_buttons(stations), render.stations_message(stations, reply))
        else:
            await self.send(chat_id, *render.reply_message(reply, sources, data.get("intent") == "policy_qa"))
        if guard.get("blocked"):
            return  # 차단된 질문은 기록에 남기지 않는다
        histories[chat_id].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": render.history_content(reply, tables)},
        ])

    async def refresh(self, chat_id: int, message_id: int, station_id: str, is_photo: bool) -> str:
        """도착 메시지의 새로고침 버튼: 모델 없이 도착 정보만 다시 받아 그 메시지(그림 또는 글자)를 고친다."""
        try:
            r = await self.chatbot.get("/api/arrivals", params={"station_id": station_id})
            if r.status_code != 200:
                return (r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json")
                        else None) or "갱신하지 못했어요."
            table = r.json()["table"]
        except httpx.HTTPError:
            return "버스봇 서버에 연결하지 못했어요."
        try:
            if is_photo and table_image.available():
                png = await asyncio.to_thread(table_image.render, table, render.now_label())
                media = {"type": "photo", "media": "attach://table",
                         "caption": render.arrivals_caption(table), "parse_mode": "HTML"}
                await self.upload("editMessageMedia", {"chat_id": chat_id, "message_id": message_id, "media": media,
                                                       "reply_markup": render.arrivals_markup(table)}, png)
            else:
                text, markup = render.arrivals_message(table)
                await self.call("editMessageText", chat_id=chat_id, message_id=message_id, text=text,
                                parse_mode="HTML", reply_markup=markup, disable_web_page_preview=True)
        except TelegramError as e:
            if "not modified" not in str(e):  # 같은 분 안에 다시 눌러 내용이 같으면 그대로 둔다
                raise
        return "도착 정보를 갱신했어요."

    # ---------- 업데이트 처리 ----------

    async def handle(self, update: dict):
        if "callback_query" in update:
            await self.on_callback(update["callback_query"])
        elif "message" in update and update["message"].get("text"):
            await self.on_message(update["message"])

    def allowed(self, user: dict | None) -> bool:
        return not ALLOWED_USERS or (user or {}).get("id") in ALLOWED_USERS

    async def on_message(self, msg: dict):
        chat_id, text = msg["chat"]["id"], msg["text"].strip()
        if not self.allowed(msg.get("from")):
            await self.send(chat_id, f"이 봇은 허용된 사용자만 쓸 수 있어요. (내 텔레그램 ID: {msg.get('from', {}).get('id')})")
            return
        command = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
        if command in ("/start", "/help", "/menu"):
            await self.send(chat_id, *render.welcome())
            return
        if command == "/reset":
            histories.pop(chat_id, None)
            await self.send(chat_id, "대화 기록을 지웠어요. 정류소 이름을 보내 주세요.")
            return
        async with locks[chat_id]:
            await self.ask(chat_id, text[:500])

    async def on_callback(self, cq: dict):
        data, msg = cq.get("data", ""), cq.get("message") or {}
        chat_id = msg.get("chat", {}).get("id")
        if chat_id is None or not self.allowed(cq.get("from")):
            await self.call("answerCallbackQuery", callback_query_id=cq["id"], text="허용된 사용자만 쓸 수 있어요.")
            return
        kind, _, value = data.partition(":")
        if kind == "ref" and value.isdigit():
            note = await self.refresh(chat_id, msg["message_id"], value, bool(msg.get("photo")))
            await self.call("answerCallbackQuery", callback_query_id=cq["id"], text=note)
            return
        await self.call("answerCallbackQuery", callback_query_id=cq["id"])  # 버튼의 로딩 표시를 끈다
        if kind == "menu":
            await self.send(chat_id, "궁금한 것을 골라 보세요. 정류소 이름을 보내면 도착 정보를 알려 드려요.",
                            render.menu_keyboard())
        elif kind == "q" and value.isdigit() and int(value) < len(render.QUICK_QUESTIONS):
            question = render.QUICK_QUESTIONS[int(value)]
            await self.send(chat_id, f"💬 {render.escape(question)}")
            async with locks[chat_id]:
                await self.ask(chat_id, question)
        elif kind == "pick" and value.isdigit():
            # 후보 버튼: 웹의 즐겨찾기처럼 정류소 ID로 정확히 조회한다
            async with locks[chat_id]:
                await self.ask(chat_id, f"정류소 ID {value} 버스 언제 와?")

    async def run(self):
        me = await self.call("getMe")
        await self.call("setMyCommands", commands=[
            {"command": "start", "description": "처음 메뉴 보기"},
            {"command": "reset", "description": "대화 기록 지우기"},
        ])
        log.info("@%s 시작 (챗봇 서버 %s, 허용 사용자 %s)", me["username"], CHATBOT_URL,
                 len(ALLOWED_USERS) or "제한 없음")
        offset = None
        while True:
            try:
                updates = await self.call("getUpdates", offset=offset, timeout=50,
                                          allowed_updates=["message", "callback_query"])
            except TelegramError as e:
                if " 409 " in f" {e} ":
                    sys.exit("같은 토큰으로 다른 곳에서 봇이 이미 실행 중입니다 (409). 그쪽을 먼저 끄세요.")
                log.warning("getUpdates 실패: %s", e)
                await asyncio.sleep(5)
                continue
            except httpx.HTTPError as e:
                log.warning("텔레그램 연결 오류: %s", e)
                await asyncio.sleep(5)
                continue
            for u in updates:
                offset = u["update_id"] + 1
                asyncio.create_task(self._safe(u))  # 느린 질문이 다른 방을 막지 않게 따로 처리

    async def _safe(self, update: dict):
        try:
            await self.handle(update)
        except Exception:
            log.exception("업데이트 처리 실패")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 요청 URL에 토큰이 들어가므로 로그에 남기지 않는다
    if not TOKEN:
        sys.exit(".env에 TELEGRAM_BOT_TOKEN이 없습니다.")
    try:
        asyncio.run(Bot(TOKEN).run())
    except KeyboardInterrupt:
        pass
