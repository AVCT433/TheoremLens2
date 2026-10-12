# 코드 컨벤션

루트 `CLAUDE.md`에서 분리한 상세 규칙. 코드를 작성·수정할 때 참조한다.

## 공통 규칙

- 페이지 번호는 어디서나 **1-based PDF 페이지 인덱스**이며, 책에 인쇄된 쪽수가 아니다.
- 모든 Gemini 호출은 `theoremlens/llm.py`를 거친다(재시도, API 키 확인 `LLMNotConfigured`, 임베딩 배치 처리). `genai.Client`를 다른 곳에서 만들지 않는다.
- `index/db.py`는 `RLock`으로 보호되는 단일 공유 커넥션을 쓴다. 쓰기에는 `tx()`를 쓴다.
- 사용자에게 보이는 텍스트와 LLM 프롬프트는 한국어다.

## 설정

`.env`(`.env.example` 참고)에서 읽는다: `GEMINI_API_KEY`, 모델 이름(`THEOREMLENS_ANSWER_MODEL`, `THEOREMLENS_FAST_MODEL`, `THEOREMLENS_EMBED_MODEL`), `THEOREMLENS_TEXT_EXTRACTOR`(`auto|marker|pymupdf|gemini`), `THEOREMLENS_MAX_UPLOAD_MB`(업로드 상한, 기본 300). 런타임 데이터(SQLite DB, Chroma 인덱스, 책별 파일)는 gitignore된 `data/`에 둔다.

선택 설치: `uv pip install -e ".[marker]"`는 text-PDF의 LaTeX 품질 추출용 Marker를 설치한다(torch 포함).

## 코드 스타일 (ruff: `ruff check`, `ruff format`로 강제)

- 줄 길이 120, 들여쓰기 4칸, 큰따옴표.
- 모든 모듈은 한국어 docstring, 그다음 `from __future__ import annotations`로 시작한다. import 순서는 stdlib → 서드파티 → 상대 import(`from .. import llm`).
- 의존 방향 `app → rag → index → ingest → (llm, config)`를 지키고, 위쪽으로 import하지 않는다.
- 공개 함수는 타입 힌트를 모두 단다(`list[int]`, `X | None` 같은 builtin 제네릭). API 경계(요청/응답 본문, LLM JSON)는 pydantic 모델, 내부 값 객체는 `@dataclass`.
- private 헬퍼는 `_` 접두사. 모듈 로거는 `log = logging.getLogger(__name__)`이며 `print`는 쓰지 않는다.
- 도메인 에러는 `RuntimeError`를 상속한다(예: `LLMNotConfigured`). `except Exception: pass`는 금지.
- 설정은 `get_settings()`로만 읽고, `config.py` 밖에서 `os.getenv`를 쓰지 않는다.
- 순수 로직(`structure.py`, 검증)은 네트워크·DB 접근 없이 유지해 단위 테스트가 가능하게 한다.
- 주석은 *무엇*이 아니라 *왜*를 한국어로 쓴다. 식별자는 영어.

## 테스트

- 단위 테스트는 네트워크를 쓰지 않으며 Gemini는 가짜로 대체한다. 계층, 격리 fixture, 모듈별 점검은 `tests/CLAUDE.md`, 품질 평가는 `tests/eval/CLAUDE.md`를 본다.
