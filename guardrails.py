"""입출력 가드레일: 개인정보(PII) 가리기 · 프롬프트 인젝션 차단 · 비밀값/시스템 프롬프트 유출 차단.

모델(LLM)을 부르기 전에 규칙으로 걸러 비용 없이 빠르게 막고, 모델이 내놓은 답도 사용자에게 가기 전에 한 번 더 본다.
classify 노드의 out_of_scope 분류(LLM 판단)는 규칙을 빠져나간 요청을 막는 두 번째 방어선으로 그대로 둔다.

입력 (main.py에서 그래프 호출 전)
- check_input(message): PII를 [전화번호] 같은 표시로 바꾸고, 인젝션 패턴이면 blocked로 돌려준다.
  가린 문장으로만 모델·LangSmith에 보내므로 개인정보가 외부 서비스로 나가지 않는다.
- clean_history(history): 브라우저·텔레그램이 보내 준 대화 기록도 같은 규칙으로 가리고,
  인젝션 문장이 든 메시지는 뺀다 (기록은 클라이언트가 보내므로 조작될 수 있다).

출력 (chat_graph.run_chat에서 답을 돌려주기 전)
- check_output(reply, prompts): .env 비밀값이나 시스템 프롬프트 원문이 섞였으면 답 전체를 안내 문구로 바꾸고,
  주민등록번호·카드번호 등은 항상, 전화번호·이메일은 정책 문서에 없는 것만 가린다
  (문서에 나온 콜센터 번호 같은 공개 연락처는 그대로 보여 준다).
"""

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).parent
POLICY_DOCS = BASE_DIR / "docs" / "policy"

BLOCKED_REPLY = ("이 요청은 처리할 수 없어요. 경기도 버스 도착 정보와 버스 이용 방법(요금·환승 등)만 안내할 수 있어요. "
                 "정류소 이름을 알려 주세요 (예: 미금역).")
LEAK_REPLY = "답변에 공개할 수 없는 내용이 섞여 있어 보여 드릴 수 없어요. 질문을 바꿔서 다시 물어봐 주세요."
MAX_HISTORY_CHARS = 2000  # 기록 메시지 하나의 최대 길이


# ---------- 개인정보(PII) ----------

def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total, alt = total + n, not alt
    return total % 10 == 0


# (종류, 정규식, 검증 함수). 버스 데이터(정류소 ID 9자리, 정류소번호 4~5자리, 노선 "720-2")와 겹치지 않게
# 자릿수·앞자리·구분자를 좁게 잡았다.
PII_RULES = [
    ("주민등록번호",
     re.compile(r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s?-?\s?[1-8]\d{6}(?!\d)"), None),
    ("카드번호", re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)"),
     lambda m: _luhn(re.sub(r"\D", "", m))),
    ("운전면허번호", re.compile(r"(?<!\d)\d{2}-\d{2}-\d{6}-\d{2}(?!\d)"), None),
    ("여권번호", re.compile(r"(?<![A-Za-z0-9])[MSRODG](?:\d{8}|\d{3}[A-Z]\d{4})(?![A-Za-z0-9])"), None),
    ("계좌번호", re.compile(r"(?<=계좌)(?:번호)?\s*[:은는]?\s*\d[\d-]{8,18}\d"), None),
    ("휴대전화번호", re.compile(r"(?<!\d)01[016789][-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"), None),
    ("전화번호", re.compile(r"(?<!\d)0(?:2|[3-6][1-5]|70)[-. )]\s?\d{3,4}[-. ]\d{4}(?!\d)"), None),
    ("이메일", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), None),
]
# 답변에서는 공개 문서에 나온 연락처면 가리지 않는 종류
PUBLIC_OK = {"전화번호", "이메일", "휴대전화번호"}


@lru_cache(maxsize=1)
def _public_contacts() -> frozenset[str]:
    """정책 문서(docs/policy)에 나온 전화번호·이메일 (공식 연락처라 답변에 나와도 된다)."""
    found = set()
    for path in POLICY_DOCS.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for kind, rx, _ in PII_RULES:
            if kind in PUBLIC_OK:
                found.update(re.sub(r"\D", "", m.group()) if kind != "이메일" else m.group().lower()
                             for m in rx.finditer(text))
    return frozenset(found)


def mask_pii(text: str, allow_public: bool = False) -> tuple[str, list[str]]:
    """PII를 [종류]로 바꾼다. 반환: (가린 문장, 찾은 종류 목록)."""
    kinds = []
    public = _public_contacts() if allow_public else frozenset()

    for kind, rx, valid in PII_RULES:
        def repl(m, kind=kind, valid=valid):
            value = m.group()
            if valid and not valid(value):
                return value
            key = value.lower() if kind == "이메일" else re.sub(r"\D", "", value)
            if kind in PUBLIC_OK and key in public:
                return value
            if kind not in kinds:
                kinds.append(kind)
            return f"[{kind}]"
        text = rx.sub(repl, text)
    return text, kinds


# ---------- 프롬프트 인젝션 ----------

# 한국어는 띄어쓰기가 제각각이라 공백을 지운 문장에 맞춘다.
KO_INJECTION = [
    r"(이전|위의?|앞의?|기존|모든|지금까지)의?(모든)?(지시|지침|명령|규칙|프롬프트|설정|대화)(사항)?[을를은는과와]?(모두|전부|다|싹)?(무시|잊어|잊고|따르지마|취소|무효|리셋|초기화)",
    r"(시스템|개발자|숨겨진|내부)(프롬프트|메시지|지시|지침|명령)",
    r"(너의?|네|당신의?|봇의?)(프롬프트|지시문|지침|규칙|설정|초기지시)(을|를|이|가)?(보여|알려|출력|공개|말해|복사|그대로)",
    r"(프롬프트|지시문|지침)(원문|전문|내용)(을|를)?(보여|알려|출력|공개|말해)",
    r"지금부터(너는|넌|당신은|네가)",
    r"(탈옥|제약(을)?(해제|풀어|없이)|필터(를)?(해제|우회|끄|꺼)|검열(없이|해제)|안전장치(를)?(해제|끄|꺼|무시))",
    r"(관리자|개발자|디버그|탈옥|무제한)모드",
    r"역할(을|극)?(바꿔|변경|해줘|놀이)",
]
EN_INJECTION = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|any|your|system)\b.{0,20}\b(instructions?|prompts?|rules|messages?|guidelines|directions)",
    r"\b(system|developer|hidden|initial)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat|output|leak|dump)\b.{0,30}\b(prompt|instructions|system message)\b",
    r"\b(jailbreak|developer mode|dan mode|do anything now)\b",
    r"\byou are now\b|\bact as (an?|the)\b|\bpretend (to be|you are)\b",
    r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?(system|inst)\]|^\s*(system|assistant)\s*:|#{2,}\s*(system|instruction)",
]
_KO = [re.compile(p) for p in KO_INJECTION]
_EN = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in EN_INJECTION]


def injection_reason(text: str) -> str | None:
    """인젝션 패턴과 맞으면 맞은 규칙 이름(로그용)을, 아니면 None."""
    compact = re.sub(r"\s+", "", text)
    for i, rx in enumerate(_KO):
        if rx.search(compact):
            return f"ko_{i}"
    for i, rx in enumerate(_EN):
        if rx.search(text):
            return f"en_{i}"
    return None


# ---------- 입력 ----------

@dataclass
class InputCheck:
    text: str                                   # 모델에 보낼 문장 (PII를 가린 것)
    pii: list[str] = field(default_factory=list)
    blocked: str | None = None                  # 차단 사유 (인젝션 규칙 이름)


def check_input(message: str) -> InputCheck:
    masked, kinds = mask_pii(message)
    return InputCheck(text=masked, pii=kinds, blocked=injection_reason(message))


def clean_history(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """대화 기록의 PII를 가리고 인젝션 문장이 든 메시지는 뺀다. 반환: (정리한 기록, 뺀 개수)."""
    cleaned, dropped = [], 0
    for role, content in history:
        content = content[:MAX_HISTORY_CHARS]
        if injection_reason(content):
            dropped += 1
            continue
        cleaned.append((role, mask_pii(content)[0]))
    return cleaned, dropped


# ---------- 출력 ----------

SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD)", re.IGNORECASE)
LEAK_WINDOW = 25      # 프롬프트 원문과 비교하는 조각 길이 (공백 제외 글자 수)
LEAK_MIN_CHARS = 60   # 프롬프트와 같은 조각이 덮는 글자가 이만큼 넘으면 유출로 본다
#                       (주제 목록 한 줄을 되풀이하는 정도의 정상 답변은 걸리지 않게)


def _secret_values() -> list[str]:
    return [v for k, v in os.environ.items() if SECRET_NAME.search(k) and v and len(v) >= 8]


@lru_cache(maxsize=8)
def _prompt_windows(prompts: tuple[str, ...]) -> frozenset[str]:
    """프롬프트 원문의 LEAK_WINDOW 글자 조각 집합. 모델에게 그대로 말하라고 시킨 따옴표 안 문구는 뺀다."""
    windows = set()
    for p in prompts:
        # 따옴표 안 문구 자리는 구분 문자로 남겨, 그 앞뒤 문장이 이어져 가짜 조각이 생기지 않게 한다
        p = re.sub(r"\s+", "", re.sub(r"\"[^\"]*\"|'[^']*'", "\x00", p))
        for part in p.split("\x00"):
            windows.update(part[i:i + LEAK_WINDOW] for i in range(len(part) - LEAK_WINDOW + 1))
    return frozenset(windows)


@dataclass
class OutputCheck:
    text: str
    pii: list[str] = field(default_factory=list)
    leak: str | None = None  # "secret" | "prompt"


def check_output(reply: str, prompts: tuple[str, ...] = ()) -> OutputCheck:
    if any(v in reply for v in _secret_values()):
        return OutputCheck(text=LEAK_REPLY, leak="secret")
    if prompts:
        compact = re.sub(r"\s+", "", reply)
        windows = _prompt_windows(prompts)
        covered = set()
        for i in range(len(compact) - LEAK_WINDOW + 1):
            if compact[i:i + LEAK_WINDOW] in windows:
                covered.update(range(i, i + LEAK_WINDOW))
        if len(covered) >= LEAK_MIN_CHARS:
            return OutputCheck(text=LEAK_REPLY, leak="prompt")
    masked, kinds = mask_pii(reply, allow_public=True)
    return OutputCheck(text=masked, pii=kinds)
