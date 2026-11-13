"""프로젝트 경로. 실행 중에 만들어지는 파일(캐시·DB·평가 결과)은 모두 프로젝트 루트 아래에 둔다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트 (src/의 부모)
ENV_FILE = ROOT / ".env"
DATA_DIR = ROOT / "data"                      # 사용자·세션·즐겨찾기 DB, 장기 메모리 DB (.gitignore)
CACHE_DIR = ROOT / ".cache"                   # 다음 정류장 캐시 (.gitignore)
RAG_INDEX_DIR = ROOT / ".rag_index"           # 문서 임베딩 캐시 (.gitignore)
POLICY_DOCS_DIR = ROOT / "docs" / "policy"    # 정책 문서 (RAG 원문)
EVAL_RESULTS_DIR = ROOT / "eval_results"      # 평가 결과 (.gitignore)
NEXT_STOP_CACHE = CACHE_DIR / "next_stops.json"
