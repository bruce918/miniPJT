"""정책 문서 검색의 LCEL 프롬프트: 쿼리 확장·리랭킹."""

from langchain_core.prompts import ChatPromptTemplate

EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "경기도 버스 이용 정책 문서를 검색하려 합니다. 사용자 질문을 검색이 잘 되도록 확장하세요.\n"
     "- queries: 같은 뜻의 다른 표현 2개 (구어체 질문은 문서체로도 바꿔 보기)\n"
     "- keywords: 문서에 쓰일 법한 공식 용어·동의어 (예: 갈아타기→환승, 휠체어→저상버스·교통약자, "
     "물건 두고 내림→분실물, 밤늦게→심야버스)"),
    ("human", "질문: {question}"),
])

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "질문과 검색된 문서 조각 후보들이 주어집니다. 각 후보가 질문에 답하는 데 필요한 정보를 얼마나 담고 있는지 "
     "0~10점으로 매기세요. 주제만 비슷하고 답이 없으면 3점 이하입니다. 모든 후보 번호에 점수를 주세요."),
    ("human", "질문: {question}\n\n[후보]\n{candidates}"),
])
