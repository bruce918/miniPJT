"""여러 에이전트·체인이 같이 쓰는 LLM 클라이언트 설정 (AWS Bedrock Claude)."""

import os

from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

# 도구 2개를 호출해 요약하는 단순한 작업이라 가볍고 한도가 따로 잡히는 Haiku를 기본으로 씁니다.
# 다른 모델을 쓰려면 .env에 BUS_MODEL_ID를 넣으세요 (예: us.anthropic.claude-sonnet-4-5-20250929-v1:0)
BUS_MODEL_ID = os.getenv("BUS_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


def make_llm(max_tokens: int = 600, model_id: str | None = None) -> ChatBedrockConverse:
    """Bedrock Claude 모델. 에이전트·계획 체인·RAG 체인·평가 판사가 같이 씁니다."""
    return ChatBedrockConverse(
        model=model_id or BUS_MODEL_ID,
        region_name="us-east-1",
        max_tokens=max_tokens,   # 답변이 길어지지 않게 출력 상한을 둡니다
        # 한도 초과(ThrottlingException)에는 간격을 늘려 재시도합니다.
        # 채팅은 사용자가 기다리므로 기본 3번(일일 한도 초과면 재시도해도 소용없음), 평가 스크립트는 8번으로 올려 씁니다.
        config=Config(retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "3")), "mode": "adaptive"}),
    )
