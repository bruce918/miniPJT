"""버스 이용 정책 에이전트: 정책 문서 검색(retrieve_docs)과 계산(calculate) 도구를 쓰는 ReAct 에이전트."""

from langgraph.prebuilt import create_react_agent

from src.agents.policy_qa.prompts import POLICY_PROMPT
from src.shared.llm import make_llm
from src.tools.calculator.tool import calculate
from src.tools.retrieval.tool import get_retriever, retrieve_docs


async def build_policy_agent():
    await get_retriever()  # 문서 색인(임베딩 캐시)을 미리 만들어 첫 질문이 느리지 않게 한다
    return create_react_agent(make_llm(), [retrieve_docs, calculate], prompt=POLICY_PROMPT, name="policy_agent")
