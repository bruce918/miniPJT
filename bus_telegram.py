# bus_telegram.py - 텔레그램 봇(Bruce918_bot)에 질문하면 경기도 버스 도착 정보를 답해 주는 Agent
"""텔레그램 Bot API를 롱폴링(getUpdates)으로 받아 bus_agent의 MCP 에이전트에게 넘깁니다.

필요한 환경변수(.env):
  TELEGRAM_BOT_TOKEN   BotFather가 발급한 Bruce918_bot 토큰
  TELEGRAM_CHAT_ID     (선택) 이 채팅의 메시지에만 응답. 없으면 누구에게나 응답하므로 지정을 권장
  (DATA_GO_KR_KEY는 MCP 서버 쪽에서만 필요합니다)
실행: 1) 터미널 A: python bus_server.py   2) 터미널 B: python bus_telegram.py   (Ctrl+C로 종료)
"""
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv
from langchain_core.messages.utils import trim_messages
from langgraph.checkpoint.memory import MemorySaver

from bus_agent import SYSTEM_PROMPT, build_agent, get_text

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_CHAT = os.getenv("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"

TELEGRAM_PROMPT = SYSTEM_PROMPT + """

[텔레그램 답변 형식]
- 일반 텍스트로 짧게 답하세요. 마크다운(**, #)은 쓰지 마세요.
- 정류소 ID를 모르면 '정류소 ID(숫자 9자리)를 보내주세요'라고 물어보세요."""

KEEP_LAST_MESSAGES = 6   # 모델에 보내는 대화 기록 상한 (도구 결과가 쌓여 토큰이 늘어나는 것을 막음)


def trim_history(state):
    """대화가 길어져도 최근 메시지만 모델에 보냅니다. 저장된 기록은 그대로 둡니다."""
    trimmed = trim_messages(
        state["messages"], strategy="last", token_counter=len, max_tokens=KEEP_LAST_MESSAGES,
        start_on="human", include_system=False,
    )
    return {"llm_input_messages": trimmed}


WELCOME = "경기도 버스 도착 정보 봇입니다.\n정류소 ID(숫자 9자리)를 보내 주세요. 예: 228000704"


async def send(http: httpx.AsyncClient, chat_id: int, text: str):
    # 텔레그램 메시지 한 건은 4096자 제한이라 넘치면 잘라서 보냅니다
    await http.post(f"{API}/sendMessage", json={"chat_id": chat_id, "text": text[:4000]})


async def main():
    if not TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다. .env에 BotFather 토큰을 넣어 주세요.")

    # 채팅방마다 thread_id를 나눠 대화 맥락(정류소 ID 등)을 이어갑니다
    try:
        agent, tools = await build_agent(
            TELEGRAM_PROMPT, pre_model_hook=trim_history, checkpointer=MemorySaver()
        )
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"MCP 도구 {len(tools)}개 준비 완료. 텔레그램 메시지를 기다립니다...")

    offset = None
    async with httpx.AsyncClient(timeout=40) as http:
        while True:
            try:
                res = await http.get(f"{API}/getUpdates", params={"timeout": 30, "offset": offset})
                updates = res.json().get("result", [])
            except httpx.HTTPError as e:
                print(f"getUpdates 실패, 3초 뒤 재시도: {type(e).__name__}")
                await asyncio.sleep(3)
                continue

            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = (msg.get("chat") or {}).get("id")
                if not text or chat_id is None:
                    continue
                if ALLOWED_CHAT and str(chat_id) != ALLOWED_CHAT:
                    print(f"허용되지 않은 채팅 무시: {chat_id}")
                    continue
                if text.startswith("/start"):
                    await send(http, chat_id, WELCOME)
                    continue

                print(f"[{chat_id}] 질문: {text}")
                try:
                    result = await agent.ainvoke(
                        {"messages": [("user", text)]},
                        {"configurable": {"thread_id": str(chat_id)}},
                    )
                    answer = get_text(result["messages"][-1]) or "답변을 만들지 못했습니다."
                except Exception as e:
                    answer = f"조회 중 오류가 났습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요."
                    print(f"  오류: {e}")
                await send(http, chat_id, answer)


if __name__ == "__main__":
    asyncio.run(main())
