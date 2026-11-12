"""계산 도구 (calculate). 정책 에이전트가 요금 합계 같은 계산을 직접 하지 않고 이 도구로 하게 한다.

모델이 "830원 × 2 + 1,650원 × 3"을 직접 계산하면 자주 틀린다 (실제로 9,960원·6,310원으로 답한 적이 있다, 정답 6,610원).
eval을 쓰지 않고 사칙연산만 허용하는 AST 계산기다.
"""

import ast
import operator
import re

from langchain_core.tools import tool

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("사칙연산(+ - * /)과 숫자, 괄호만 쓸 수 있습니다.")


def compute(expression: str) -> float:
    expr = expression.replace(",", "").replace("원", "").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)  # "830 x 2"
    if len(expr) > 200:
        raise ValueError("식이 너무 깁니다.")
    return _eval(ast.parse(expr, mode="eval"))


# 답변 속 "단가 × 인원 + … = 합계원" 식 (예: "어린이 900원×2 + 성인 1,700원×3 = 6,300원",
# "어린이(830원) 2명 + 성인(1,650원) 3명 = 6,610원")
_TERM = r"(?:[가-힣]+\s*)?\(?\s*(\d[\d,]*)\s*원?\s*\)?\s*(?:(?:[×xX*]\s*)?(\d+)\s*명|[×xX*]\s*(\d+)\s*명?)?"
_TERM_RE = re.compile(_TERM)
_SUM_RE = re.compile(rf"((?:{_TERM})(?:\s*\+\s*(?:{_TERM}))*)\s*=\s*(\d[\d,]*)(\s*원)")


def fix_totals(text: str) -> tuple[str, int]:
    """답변 속 요금 합계 식을 다시 계산해, 틀린 합계를 맞는 값으로 바꾼다. 반환: (고친 답변, 고친 개수).

    모델이 계산 도구를 쓰라는 지시를 일부만 따르는 경우(카드 합계는 도구로, 현금 합계는 직접)가 있어
    답변을 내보내기 전에 규칙으로 한 번 더 확인한다. 곱셈이나 덧셈이 없는 식(검증할 것이 없음)은 건드리지 않는다.
    """
    fixed = 0

    def repl(m: re.Match) -> str:
        nonlocal fixed
        expr, stated, unit = m.group(1), m.group(m.lastindex - 1), m.group(m.lastindex)
        terms = [t for t in _TERM_RE.finditer(expr) if t.group(1)]
        has_op = len(terms) > 1 or any(t.group(2) or t.group(3) for t in terms)
        if not has_op:
            return m.group(0)
        total = sum(int(t.group(1).replace(",", "")) * int(t.group(2) or t.group(3) or 1) for t in terms)
        if total == int(stated.replace(",", "")):
            return m.group(0)
        fixed += 1
        return f"{expr.rstrip()} = {total:,}{unit}"

    return _SUM_RE.sub(repl, text), fixed


@tool
def calculate(expression: str) -> str:
    """사칙연산 계산기. 요금 합계처럼 숫자를 더하거나 곱해야 하면 직접 계산하지 말고 반드시 이 도구를 쓰세요.

    Args:
        expression: 계산식 (예: "830*2 + 1650*3"). 숫자, + - * /, 괄호만 쓴다. 쉼표·"원"·"×"는 알아서 처리한다.
    """
    try:
        value = compute(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"계산할 수 없는 식입니다: {e}"
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return f"{expression} = {text}"
