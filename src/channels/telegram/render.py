"""챗봇 응답(/api/chat의 reply·tables·sources)을 텔레그램 메시지(HTML)와 버튼으로 바꾼다.

웹 화면은 표를 그리지만 텔레그램은 표가 없으므로, 휴대폰에서 읽기 쉬운 줄 목록으로 바꾼다.
- 정류소 후보 표 -> 후보 목록 + 후보마다 고르기 버튼 (callback "pick:<정류소ID>")
- 도착 표 -> 노선별 한 줄 + 새로고침 버튼 (callback "ref:<정류소ID>")
- 정책 답변 출처 -> 링크 목록

모든 값은 html.escape로 이스케이프한다 (parse_mode=HTML).
"""

import re
from datetime import datetime
from html import escape

MAX_TEXT = 4000  # 텔레그램 메시지 한도(4096자)보다 조금 작게

QUICK_ARRIVAL = ["미금역 버스 언제 와?", "7079 도착"]
QUICK_POLICY = [
    ("요금", "청소년 버스비 얼마야?"),
    ("환승", "환승 몇 분 안에 해야 할인돼?"),
    ("교통카드", "학생 할인 받으려면 카드 등록해야 돼?"),
    ("분실물", "버스에 지갑 두고 내렸어"),
    ("민원", "기사가 정류장 그냥 지나쳤어 어디 신고해?"),
    ("경기패스", "경기패스 환급 얼마나 받아?"),
    ("어르신·장애인", "어르신 지하철 무료 카드 어디서 만들어?"),
    ("버스정보", "버스 도착 예정 시간이 왜 자꾸 틀려?"),
    ("광역버스", "광역버스 서서 타도 돼?"),
    ("공공버스", "경기도 공공버스가 뭐야?"),
]
# 버튼 callback_data는 64바이트까지라 질문 문장 대신 번호를 싣는다 ("q:<번호>")
QUICK_QUESTIONS = QUICK_ARRIVAL + [q for _, q in QUICK_POLICY]

WEB_ONLY_NOTE = "행을 누르면 그 정류소를 고릅니다."


def clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    """[[(버튼 글자, callback_data), ...], ...] -> 텔레그램 inline_keyboard."""
    return {"inline_keyboard": [[{"text": t[:60], "callback_data": d} for t, d in row] for row in rows]}


def welcome() -> tuple[str, dict]:
    text = ("안녕하세요! 경기도 버스 안내봇이에요 🚌\n"
            "정류소 이름(예: 미금역)이나 정류소번호(예: 7079)를 보내면 지금 오는 버스를 알려 드려요.\n"
            "요금·환승·분실물 같은 이용 방법도 물어보세요. 아래에서 골라도 돼요.")
    return text, menu_keyboard()


def menu_keyboard() -> dict:
    rows = [[(q, f"q:{i}") for i, q in enumerate(QUICK_ARRIVAL)]]
    policy = [(label, f"q:{len(QUICK_ARRIVAL) + i}") for i, (label, _) in enumerate(QUICK_POLICY)]
    rows += [policy[i:i + 3] for i in range(0, len(policy), 3)]
    return keyboard(rows)


def stations_message(table: dict, reply: str) -> tuple[str, dict]:
    reply = reply.replace("표에서 골라 주세요.", "").strip()  # 웹용 문구 (텔레그램은 버튼으로 고른다)
    lines = [escape(reply), "", f"<b>{escape(table.get('title') or '정류소 후보')}</b>"]
    for r in table.get("rows", []):
        extra = " · ".join(x for x in (
            f"→ {r['next']}" if r.get("next") else "",
            r.get("city", ""),
            f"정류소번호 {r['mobile']}" if r.get("mobile") else "",
        ) if x)
        lines.append(f"{r['no']}. <b>{escape(r['name'])}</b>" + (f"\n    {escape(extra)}" if extra else ""))
    lines += ["", escape(stations_for_image(table)["note"])]
    return clip("\n".join(lines)), stations_buttons(table)


def stations_buttons(table: dict) -> dict:
    rows = []
    for r in table.get("rows", []):
        label = f"{r['no']}. {r['name']}" + (f" → {r['next']}" if r.get("next") else "")
        rows.append([(label, f"pick:{r['id']}")])
    return keyboard(rows)


def stations_for_image(table: dict) -> dict:
    """그림에 넣을 후보 표: 웹용 안내("행을 누르면…")를 버튼 안내로 바꾼다."""
    note = (table.get("note") or "").replace(WEB_ONLY_NOTE, "").strip()
    return {**table, "note": (note + " " if note else "") + "아래 버튼을 눌러 정류소를 고르세요."}


def stations_caption(reply: str) -> str:
    return escape(reply.replace("표에서 골라 주세요.", "아래 버튼에서 골라 주세요."))


def arrivals_caption(table: dict) -> str:
    return f"<b>{escape(table.get('title') or '도착 정보')}</b> · {now_label()}"


def arrivals_markup(table: dict) -> dict | None:
    sid = table.get("station_id")
    return keyboard([[("↻ 새로고침", f"ref:{sid}")]]) if sid else None


def now_label() -> str:
    now = datetime.now()
    return f"{'오전' if now.hour < 12 else '오후'} {now.hour % 12 or 12}:{now.minute:02d} 기준"


def arrivals_message(table: dict) -> tuple[str, dict | None]:
    lines = [f"<b>{escape(table.get('title') or '도착 정보')}</b>  <i>{now_label()}</i>", ""]
    for r in table.get("rows", []):
        soon = r.get("minutes") is not None and r["minutes"] <= 2
        head = f"{'🔴' if soon else '🚌'} <b>{escape(r['route'])}</b>"
        if r.get("dest"):
            head += f" ({escape(r['dest'])} 방면)"
        body = f"    <b>{escape(r['first'])}</b>" if r.get("first") else "    도착 정보 없음"
        if r.get("second"):
            body += f"  /  다음 {escape(r['second'])}"
        if r.get("info"):
            body += f"  ·  {escape(r['info'])}"
        lines += [head, body]
    if table.get("note"):
        lines += ["", f"<i>{escape(table['note'])}</i>"]
    sid = table.get("station_id")
    kb = keyboard([[("↻ 새로고침", f"ref:{sid}")]]) if sid else None
    return clip("\n".join(lines)), kb


def reply_message(reply: str, sources: list[dict], show_menu_button: bool) -> tuple[str, dict | None]:
    lines = [escape(reply)]
    links = [s for s in sources if str(s.get("url", "")).startswith(("http://", "https://"))]
    if sources:
        lines += ["", "<b>출처</b>"]
        for s in sources:
            # citation 끝의 "(https://...)"는 링크로 걸리므로 글자에서는 뺀다
            name = escape(re.sub(r"\s*\(https?://[^)]*\)\s*$", "", s.get("citation") or s.get("title") or "출처"))
            url = s.get("url", "")
            lines.append(f'· <a href="{escape(url, quote=True)}">{name}</a>' if s in links else f"· {name}")
    kb = keyboard([[("다른 질문 보기", "menu")]]) if show_menu_button else None
    return clip("\n".join(lines)), kb


def history_content(reply: str, tables: list[dict]) -> str:
    """웹과 같은 방식: 답변 + 표의 평문(정류소 ID 포함)을 대화 기록에 남겨 다음 턴에 번호로 고를 수 있게 한다."""
    return "\n\n".join([reply, *(f"{t.get('title', '')}\n{t.get('text', '')}" for t in tables)])
