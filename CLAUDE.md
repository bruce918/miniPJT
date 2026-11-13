# CLAUDE.md

경기도 버스 안내 AI 에이전트(웹 챗봇 + 텔레그램 봇). 이 문서는 `src/` 폴더 구조와 파일 배치 규칙을 정한다.
사용자와의 대화·진행 안내·요약은 모두 한국어로 한다.

## 실행

명령은 모두 **프로젝트 루트**에서 실행한다 (`src`를 패키지로 import하므로 `python -m`을 쓴다).

```bash
python -m src.tools.bus_mcp.server               # 1) 버스 MCP 서버 (127.0.0.1:8765/mcp)
uvicorn src.channels.web.app:app --port 8000     # 2) 웹 챗봇 (http://localhost:8000)
python -m src.channels.telegram                  # 3) (선택) 텔레그램 봇
```

```bash
python -m src.evaluation.test_guardrails         # 규칙 단위 시험 (모델·서버 불필요)
python -m src.evaluation.run_tests               # 회귀 테스트 (1·2번 서버 필요)
python -m src.evaluation.run_tests --csv         # 인-아웃 세트 test_queries.csv (LLM 판사, 1·2번 서버 필요)
python -m src.evaluation.evaluate                # 품질 평가 (RAGAS 방식 + LLM 판사, 비용 큼)
python -m src.evaluation.eval_retrieval          # 검색 평가 (ablation)
```

## 폴더 구조

```
src/
├── shared/            공통: LLM 클라이언트(llm.py), 경로(paths.py), 트레이싱(observability.py)
├── agents/            질문 하나를 처리하는 ReAct 에이전트 (도메인별 폴더)
│   ├── bus_arrival/   버스 도착 정보: agent.py · prompts.py · verify.py
│   └── policy_qa/     버스 이용 정책: agent.py · prompts.py · verify.py
├── tools/             에이전트가 부르는 도구 (도메인별 폴더)
│   ├── bus_mcp/       버스 MCP 서버(server.py), 다음 정류장(next_stop.py), 서버 클라이언트(client.py), 정류소 데이터
│   ├── retrieval/     정책 문서 RAG: retriever.py · tool.py(retrieve_docs) · prompts.py · schemas.py
│   ├── table/         render_table 표 도구
│   └── calculator/    calculate 도구 + 답변 속 합계 교정
├── workflows/
│   └── chat/          채팅 Plan-Execute 그래프: graph.py · planner.py · prompts.py · schemas.py
├── memory/            장기 메모리(LangGraph Store)·보류 계획: store.py · prompts.py · schemas.py
├── guardrails/        입출력 필터 (PII 가리기, 인젝션 차단, 유출 차단): filters.py
├── auth/              Google 로그인·세션·즐겨찾기 API: router.py · schemas.py
├── channels/          사용자 접점
│   ├── web/           FastAPI 앱(app.py) · API 스키마(schemas.py) · 화면(static/index.html)
│   └── telegram/      텔레그램 봇 (bot.py · render.py · table_image.py, 별도 requirements.txt)
└── evaluation/        회귀 테스트·평가 스크립트, 판사 프롬프트·스키마, 데이터셋(*.json·*.csv)
```

## 각 폴더의 역할

| 폴더 | 역할 | 의존해도 되는 곳 |
|---|---|---|
| `shared/` | 둘 이상의 도메인이 쓰는 것만: `make_llm`·`get_text`·모델 ID(llm.py), 프로젝트 경로(paths.py), LangSmith 설정(observability.py) | 외부 라이브러리만 (다른 `src` 폴더를 import하지 않는다) |
| `agents/<도메인>/` | 에이전트 생성(`agent.py`), 시스템 프롬프트·고정 문구·자동 점검 메시지(`prompts.py`), 답변 근거 점검 노드(`verify.py`) | `shared`, `tools` |
| `tools/<도메인>/` | 에이전트가 호출하는 도구와 그 구현·데이터. 도구 하나가 쓰는 프롬프트·스키마도 같은 폴더에 둔다 | `shared` (`table`은 `bus_mcp`의 정류소 데이터를 읽는다) |
| `workflows/chat/` | 에이전트를 엮는 LangGraph 그래프. 계획 체인(`planner.py`), 분류·계획 프롬프트(`prompts.py`), 상태·계획·결과 스키마(`schemas.py`) | `agents`, `tools`, `memory`, `guardrails`, `shared` |
| `memory/` | 장기 메모리 읽기·쓰기와 보류 계획 저장 (SQLite `data/memory.db`) | `shared` |
| `guardrails/` | 모델 입력 전·답변 출력 전 규칙 필터 | `shared` |
| `auth/` | 로그인·세션·즐겨찾기 API 라우터 (SQLite `data/app.db`) | `shared` |
| `channels/` | 외부 진입점. 웹은 FastAPI 앱이 워크플로를 부르고, 텔레그램은 웹 API를 HTTP로 부른다 | 모든 도메인 (텔레그램은 HTTP만) |
| `evaluation/` | 실행 중인 서버나 워크플로를 시험·평가한다. 테스트 데이터셋(json·csv)도 여기 둔다 | 모든 도메인 |

의존 방향은 `channels → workflows → agents → tools → shared`이다. 반대 방향 import는 만들지 않는다.

## 파일 배치 규칙

1. **모든 소스 코드는 `src/` 아래에 둔다.** 루트에는 문서(`README.md`, `SERVICE.md`, `CLAUDE.md`), 설정(`requirements.txt`, `.env`), 정책 원문(`docs/policy/`)만 둔다.
2. **기술 역할이 아니라 도메인으로 나눈다.** `utils/`, `models/`, `prompts/` 같은 전역 폴더를 만들지 않는다. 새 기능은 그 기능의 도메인 폴더를 만든다.
3. **도메인 폴더 안의 파일 이름은 아래를 따른다.** 해당 내용이 없으면 파일을 만들지 않는다.
   - `agent.py` — 에이전트 생성 (에이전트 도메인)
   - `tool.py` — LangChain `@tool` 정의 (도구 도메인)
   - `graph.py` — LangGraph 그래프 조립 (워크플로 도메인)
   - `prompts.py` — 시스템 프롬프트, LCEL 프롬프트 템플릿, 사용자에게 보이는 고정 문구, 자동 점검 메시지
   - `schemas.py` — Pydantic 모델(구조화 출력·API 입출력), dataclass, TypedDict 상태, Literal 타입
   - `verify.py` — 답변 근거를 점검하는 규칙 노드
   - 그 밖의 실행 로직은 역할이 드러나는 이름으로 (`retriever.py`, `planner.py`, `store.py`, `filters.py`, `router.py`, `server.py`, `client.py`)
4. **프롬프트 문자열과 Pydantic/dataclass 정의는 로직 파일에 두지 않는다.** 각 도메인의 `prompts.py`와 `schemas.py`로 옮긴다.
5. **`shared/`에는 둘 이상의 도메인이 실제로 쓰는 것만 둔다.** 한 도메인만 쓰면 그 도메인 폴더에 둔다.
6. **실행 중에 생기는 파일은 루트 아래 정해진 곳에만 쓴다** (`shared/paths.py`): `data/`(DB), `.cache/`, `.rag_index/`, `eval_results/`. 모두 `.gitignore` 대상이다. 도메인 고정 데이터(예: `tools/bus_mcp/gg_stations.json`)는 그 도메인 폴더에 둔다.
7. **import는 `src.`로 시작하는 절대 경로를 쓴다** (`from src.shared.llm import make_llm`). 단 `channels/telegram`은 독립 실행 패키지라 내부 상대 import(`from . import render`)를 쓴다.
8. **새 의도(intent)를 붙이는 순서**
   1. `workflows/chat/schemas.py`의 `Intent`에 값을 추가한다
   2. `agents/<새 도메인>/`에 agent·prompts·verify를 만든다
   3. `workflows/chat/graph.py`의 단계 서브그래프에 노드와 경로를 등록한다
   4. `workflows/chat/prompts.py`의 분류 규칙에 설명을 넣는다

## 작업 규칙

- 커밋·푸시는 사용자가 요청할 때만 한다. `src/channels/telegram/` 변경은 본체와 **따로 커밋**한다.
- 코드를 바꾸면 `python -m src.evaluation.test_guardrails`와 `python -m src.evaluation.run_tests`(서버 재시작 후)로 확인한다.
- 화면(`static/index.html`)을 바꾸면 서버의 `ui_version`이 바뀌어, 열려 있던 페이지에 새로고침 안내가 뜬다.
- `history.md`(작업 기록)는 개인 파일이라 `.gitignore`에 있다.
