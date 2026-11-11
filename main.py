"""챗봇 뼈대 서버.

실행:
    pip install -r requirements.txt
    uvicorn main:app --reload
브라우저에서 http://localhost:8000 접속.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent

app = FastAPI(title="miniPJT Chatbot")


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    reply: str


def generate_reply(message: str, history: list[Message]) -> str:
    """여기에 실제 에이전트(LLM 호출, 도구 사용, 가드레일)를 붙이면 된다.

    지금은 동작 확인용으로 받은 메시지를 그대로 돌려준다.
    """
    return f"(에코) {message}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return ChatResponse(reply=generate_reply(req.message, req.history))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
