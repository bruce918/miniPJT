"""챗봇 응답의 표(정류소 후보·도착 정보)를 PNG 이미지로 그린다 (Pillow).

텔레그램 메시지에는 표 서식이 없으므로, 웹 화면과 비슷한 표를 그림으로 만들어 사진으로 보낸다.
- 열 정의(columns)와 행(rows)은 챗봇 서버가 웹에 주는 것과 같은 값을 그대로 쓴다.
- 긴 칸(정류소 이름·방면·다음 정류장)은 폭을 제한하고 줄바꿈한다.
- 2분 이내 도착은 빨간색으로 강조한다 (웹과 같은 기준).
- 한글 글꼴이 없으면 available()이 False가 되고, 봇은 글자 목록으로 대신 보낸다.

글꼴은 TELEGRAM_TABLE_FONT(.env)로 지정할 수 있다. 없으면 Windows 맑은 고딕, 리눅스 나눔고딕·Noto CJK,
macOS Apple SD 고딕 순으로 찾는다.
"""

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2  # 2배로 그려 폰 화면에서도 글자가 선명하게 보이게 한다

REGULAR_FONTS = [
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
BOLD_FONTS = [
    r"C:\Windows\Fonts\malgunbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

# 색 (웹 화면 톤에 맞춤)
BG = "#FFFFFF"
TITLE_BG = "#FEE500"
TITLE_FG = "#3A2E00"
HEAD_BG = "#F3F4F6"
LINE = "#E5E7EB"
ZEBRA = "#FAFAFB"
TEXT = "#111827"
MUTED = "#6B7280"
SOON = "#E5484D"

HIDDEN_COLUMNS = {"id"}            # 정류소 ID(9자리)는 사람에게 필요 없어 그림에서는 뺀다 (버튼이 대신 씀)
WRAP_COLUMNS = {"name": 210, "next": 190, "dest": 150, "info": 120}  # 줄바꿈할 칸의 최대 폭(1배 기준 px)
PAD = 16          # 바깥 여백
CELL_X = 12       # 칸 좌우 여백
CELL_Y = 9        # 칸 위아래 여백
MIN_WIDTH = 420   # 표 최소 폭


def _find(candidates: list[str]) -> str | None:
    custom = os.getenv("TELEGRAM_TABLE_FONT")
    for path in ([custom] if custom else []) + candidates:
        if path and Path(path).is_file():
            return path
    return None


def available() -> bool:
    return _find(REGULAR_FONTS) is not None


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = (_find(BOLD_FONTS) if bold else None) or _find(REGULAR_FONTS)
    return ImageFont.truetype(path, size * S)


def _wrap(text: str, font, max_w: float) -> list[str]:
    """글자 단위로 max_w 안에 들어가게 줄을 나눈다 (가능하면 공백·점 뒤에서 끊는다)."""
    lines, line = [], ""
    for ch in text:
        if font.getlength(line + ch) <= max_w or not line:
            line += ch
            continue
        cut = max(line.rfind(" "), line.rfind("."), line.rfind("·"))
        if cut > 0 and cut < len(line) - 1:
            lines.append(line[: cut + 1].rstrip())
            line = line[cut + 1:].lstrip() + ch
        else:
            lines.append(line)
            line = ch
    if line or not lines:
        lines.append(line)
    return lines


def render(table: dict, stamp: str = "") -> bytes:
    """표 하나를 PNG 바이트로 그린다. stamp는 제목 오른쪽에 붙는 "오후 3:05 기준" 같은 문구."""
    body_font, head_font = _font(15), _font(13, bold=True)
    bold_font, title_font, small_font = _font(15, bold=True), _font(17, bold=True), _font(12)
    line_h = int(15 * S * 1.45)

    columns = [c for c in table.get("columns", []) if c["key"] not in HIDDEN_COLUMNS]
    rows = table.get("rows", [])
    soon_rows = {i for i, r in enumerate(rows) if r.get("minutes") is not None and r["minutes"] <= 2}

    # 칸 폭: 내용의 실제 폭(머리글 포함), 줄바꿈 칸은 최대 폭으로 제한
    widths = []
    for c in columns:
        natural = max([head_font.getlength(c["label"])]
                      + [bold_font.getlength(str(r.get(c["key"], ""))) for r in rows])
        limit = WRAP_COLUMNS.get(c["key"])
        widths.append(min(natural, limit * S) if limit else natural)
    widths = [int(w) + 2 * CELL_X * S for w in widths]
    table_w = sum(widths)
    if table_w < MIN_WIDTH * S:  # 칸이 적으면 남는 폭을 고르게 나눈다
        extra = (MIN_WIDTH * S - table_w) // max(len(widths), 1)
        widths = [w + extra for w in widths]
        table_w = sum(widths)

    # 행마다 칸 내용을 줄바꿈해 높이를 정한다
    cells = []
    for r in rows:
        row_cells = []
        for c, w in zip(columns, widths):
            text = str(r.get(c["key"], "") or "")
            font = bold_font if c["key"] in ("route", "first") else body_font
            row_cells.append((_wrap(text, font, w - 2 * CELL_X * S), font))
        cells.append(row_cells)
    row_hs = [max(len(lines) for lines, _ in rc) * line_h + 2 * CELL_Y * S for rc in cells]
    head_h = int(13 * S * 1.45) + 2 * CELL_Y * S

    title = table.get("title", "")
    title_h = int(17 * S * 1.5) + 2 * 12 * S
    note_lines = _wrap(table.get("note", "") or "", small_font, table_w) if table.get("note") else []
    note_h = (len(note_lines) * int(12 * S * 1.5) + 12 * S) if note_lines else 0
    empty_h = line_h + 2 * CELL_Y * S if not rows else 0

    width = table_w + 2 * PAD * S
    height = title_h + PAD * S + head_h + sum(row_hs) + empty_h + note_h + PAD * S
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # 제목 띠
    d.rectangle([0, 0, width, title_h], fill=TITLE_BG)
    ty = (title_h - int(17 * S * 1.3)) // 2
    d.text((PAD * S, ty), title, font=title_font, fill=TITLE_FG)
    if stamp:
        sw = small_font.getlength(stamp)
        d.text((width - PAD * S - sw, ty + 4 * S), stamp, font=small_font, fill=TITLE_FG)

    # 머리글
    x0, y = PAD * S, title_h + PAD * S
    d.rectangle([x0, y, x0 + table_w, y + head_h], fill=HEAD_BG)
    x = x0
    for c, w in zip(columns, widths):
        tw = head_font.getlength(c["label"])
        tx = x + (w - tw) / 2 if c.get("align") == "center" else x + CELL_X * S
        d.text((tx, y + CELL_Y * S), c["label"], font=head_font, fill=MUTED)
        x += w
    y += head_h

    # 행
    for i, (rc, rh) in enumerate(zip(cells, row_hs)):
        if i % 2:
            d.rectangle([x0, y, x0 + table_w, y + rh], fill=ZEBRA)
        d.line([x0, y, x0 + table_w, y], fill=LINE, width=S)
        x = x0
        for (lines, font), c, w in zip(rc, columns, widths):
            color = SOON if (i in soon_rows and c["key"] == "first") else TEXT
            if c["key"] in ("city", "mobile", "info", "second"):
                color = MUTED if color == TEXT else color
            for j, line in enumerate(lines):
                lw = font.getlength(line)
                tx = x + (w - lw) / 2 if c.get("align") == "center" else x + CELL_X * S
                d.text((tx, y + CELL_Y * S + j * line_h), line, font=font, fill=color)
            x += w
        y += rh
    if not rows:
        d.text((x0 + CELL_X * S, y + CELL_Y * S), "표시할 내용이 없어요.", font=body_font, fill=MUTED)
        y += empty_h
    d.line([x0, y, x0 + table_w, y], fill=LINE, width=S)

    # 표 아래 안내
    if note_lines:
        y += 12 * S
        for line in note_lines:
            d.text((x0, y), line, font=small_font, fill=MUTED)
            y += int(12 * S * 1.5)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
