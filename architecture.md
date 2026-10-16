# TheoremLens 아키텍처

수학 전공책 PDF를 올리면 **그 책의 내용만 근거로**, 인용과 함께 답하는 로컬 웹앱의 설계 문서다. 목표는 환각 최소화다. 이 문서는 구현(`theoremlens/`, `web/`)을 기준으로 쓰였으며, 구조를 바꾸는 변경을 할 때는 코드와 함께 갱신한다. 규칙 요약은 루트 `CLAUDE.md`, 폴더별 규칙은 각 폴더의 `CLAUDE.md`에 있다.

## 1. 개요와 데이터 흐름

```
PDF 업로드
  └─ [ingest/extract]    MinerU(middle_json) → 쪽별 Markdown (pages 테이블)
  └─ [ingest/structure]  수학 단위 분할 → ChunkDraft (label, refs, 장/절, 쪽 범위)
  └─ [index]             SQLite chunks(+FTS5)  /  Chroma 벡터(로컬 임베딩)
질문
  └─ [rag/retrieve]      BM25 + 벡터 → RRF, 라벨 직접 조회, 1-hop 인용 확장, 근거 부족 판정
  └─ [rag/answer]        Gemini(JSON 스키마, temperature 0) → 주장(claims)과 [S#] 인용
  └─ [rag/verify]        quote 규칙 검사 → LLM 지지 검사 → 주장 제거/경고
  └─ [app → web]         답변, 인용 원문(DB), 원본 쪽 이미지
```

- **의존 방향**: `app → rag → index → ingest → (llm, embedder, config)`. 아래 계층은 위 계층을 import하지 않는다. 예외적으로 `rag/retrieve`가 `ingest.structure.TYPE_ALIASES`를 import하는 것은 이 방향에 맞다.
- **저장소 역할**: SQLite가 원본이고(books, pages, chunks, messages), Chroma는 임베딩만 가진다. 청크의 `id`가 두 저장소의 조인 키다.
- **쪽 번호**는 어디서나 **1부터 시작하는 PDF 쪽 인덱스**이며 책에 인쇄된 쪽 번호가 아니다.

## 2. 설정과 데이터 레이아웃 (`config.py`)

`Settings`는 `frozen` 데이터클래스이고 `get_settings()`(lru_cache)로만 읽는다. 환경변수는 `config.py` 밖에서 읽지 않는다.

| 필드 | 환경변수 | 기본값 |
|---|---|---|
| `api_key` | `GEMINI_API_KEY` | `""` |
| `answer_model` | `THEOREMLENS_ANSWER_MODEL` | `gemini-3.8-flash` |
| `fast_model` | `THEOREMLENS_FAST_MODEL` | `gemini-3.8-flash` |
| `embed_model` | `THEOREMLENS_EMBED_MODEL` | `BAAI/bge-m3` (로컬 sentence-transformers 모델) |
| `embed_device` | `THEOREMLENS_EMBED_DEVICE` | `auto` (`auto|cpu|cuda`) |
| `embed_batch` | `THEOREMLENS_EMBED_BATCH` | `8` |
| `embed_max_seq_length` | `THEOREMLENS_EMBED_MAX_SEQ` | `1024` |
| `embed_query_prefix`, `embed_doc_prefix` | `THEOREMLENS_EMBED_QUERY_PREFIX`, `THEOREMLENS_EMBED_DOC_PREFIX` | `""` (e5·nomic 같은 접두어 모델용) |
| `mineru_tier` | `THEOREMLENS_MINERU_TIER` | `advanced` (`flash|basic|standard|advanced`) |
| `mineru_ocr_mode` | `THEOREMLENS_MINERU_OCR_MODE` | `auto` (`auto|txt|ocr`) |
| `data_dir` | `THEOREMLENS_DATA_DIR` | `<repo>/data` |
| `max_upload_mb` | `THEOREMLENS_MAX_UPLOAD_MB` | `300` |
| `top_k` | (고정) | `8` |
| `min_similarity` | (고정) | `0.55` (코사인 유사도 하한) |

```
data/                      (gitignored)
  theoremlens.sqlite
  chroma/
  books/<book_id>/book.pdf   # 서버가 정한 경로. 클라이언트 파일명은 쓰지 않는다
```

## 3. LLM 계층 (`llm.py`, `embedder.py`)

Gemini 호출은 전부 `llm.py`를 거치며, `genai.Client`는 여기서만 만든다. 임베딩은 Gemini가 아니라 로컬 모델이며 `embedder.py`가 맡는다.

- `generate_json(model, system, contents, schema) -> str`: temperature 0, `response_mime_type="application/json"`, 스키마 제약 출력.
- `generate_text(model, system, contents) -> str`.
- **재시도**: 총 4회, 실패 시 1·2·4초 대기. `LLMNotConfigured`(키 없음)는 재시도하지 않고 즉시 올린다.
- **`embedder.embed(texts, task) -> list[list[float]]`** (`task`는 `"document"` / `"query"`): sentence-transformers 모델을 프로세스당 한 번 로드(락)하고, L2 정규화한 벡터를 입력 순서대로 돌려준다. 접두어 설정을 task별로 붙이고, 빈 입력이면 모델을 로드하지 않는다. 디바이스 `auto`는 CUDA가 있으면 cuda, 없으면 cpu. API 키·네트워크가 필요 없다(모델 최초 다운로드 제외).
- 테스트에서는 이 모듈의 속성만 patch한다(`tests/CLAUDE.md`).

## 4. 추출 (`ingest/extract.py`)

- **변환**: 텍스트 레이어 쪽도 스캔 쪽도 구분하지 않고 모든 쪽을 MinerU(`mineru-kit parse --format middle_json`)가 변환한다(`extract_mineru`). `mineru` 패키지는 프로젝트 의존성이고, 실행 파일은 현재 인터프리터 환경의 `mineru-kit`를 쓴다(`_run_mineru`가 subprocess로 호출하는 유일한 지점). tier와 OCR 모드는 `mineru_tier`, `mineru_ocr_mode`로 정한다. Gemini는 쓰지 않는다.
- **쪽 번호**: middle_json의 `pages[].page_idx`(0-based)에 `+1`해 `PageText.page_no`(1-based)로 둔다. 요청한 쪽이 출력에 없거나 JSON을 읽을 수 없거나 종료 코드가 0이 아니면 `RuntimeError`를 던지고, PyMuPDF 등으로 대체하지 않는다. 파이프라인이 책을 `error`로 만들고, 재시도하면 이미 저장된 쪽은 건너뛴다.
- **블록 → Markdown** (`blocks_to_markdown`, 순수 함수): `doc_title`/`paragraph_title`은 `level`만큼 `#`, `text`/`page_footnote`는 글자 그대로 이어 붙이되 `equation_inline`은 `$…$`, `equation`은 `$$
…
$$`, `list`는 항목마다 한 줄, `table`은 `table_body` HTML을 두고 `<eq>…</eq>`만 `$…$`로 바꾼다. `image`/`chart`는 캡션만 `[그림: 캡션]`으로 남기고 본문(도형 속 글자·차트 표)은 버린다. `header`와 `page_number`는 버린다. 알 수 없는 타입은 경고 로그를 남기고 텍스트만 유지한다. 블록은 빈 줄로 잇는다.
- **PyMuPDF**는 쪽 수(`page_count`)와 미리보기 PNG(`render_page_png`)에만 쓴다.
- **출력**: `PageText(page_no, markdown, method)` (`method`: `mineru`). `extract(pdf, on_page, skip)`는 `skip`에 든 쪽을 뺀 나머지를 한 번에 변환하고(파이프라인이 100쪽 배치로 부른다), 남은 쪽이 없으면 MinerU를 부르지 않는다. 이전 버전이 저장한 `marker|pymupdf|gemini` 쪽은 DB에 그대로 남는다.
- 추출 결과는 **원문 그대로**다. 요약하거나 고치지 않는다. 이후 인용 검증이 이 텍스트와 대조되기 때문이다.

## 5. 구조 분석 (`ingest/structure.py`)

순수 로직이다(네트워크·DB 없음). `build_chunks(pages, max_chars=2000) -> list[ChunkDraft]`.

- **단위 유형**: definition, theorem, lemma, proposition, corollary, claim, example, exercise, remark, proof, text. 한국어 키워드(정의, 정리, 보조정리, 명제, 따름정리, 예제, 연습문제, 문제, 증명)는 영어 유형으로 정규화한다. 키워드 표 `TYPE_ALIASES`는 `retrieve`와 공유하는 단일 출처다.
- **단위 시작 판정** (`_parse_unit`): `#`과 굵게 표시를 걷어낸 뒤, 긴 키워드부터 매칭한다(`보조정리`가 `정리`보다 먼저). 키워드 뒤에 한글이 바로 오면 제외하고, 영어 키워드는 대문자로 시작해야 한다. 통과 조건은 굵게 닫힘, 뒤 내용 없음, 또는 뒤가 `.`/`:`인 경우다. 번호 없는 머리말(`## 연습문제`)은 종결 표시가 있어야만 단위로 본다. 제목과 단위를 구분하기 위해서다.
- **정리 + 증명**: theorem-like(theorem, lemma, proposition, corollary, claim) 뒤의 `Proof`/`증명`/`Proof of …`는 같은 청크에 이어 붙는다. `∎ □ ■ \blacksquare \square \qed Q.E.D.`가 나오면 증명이 끝나고, 이후 일반 문단은 새 `text` 청크가 된다.
- **디스플레이 수식 보존**: 한 줄의 `$$` 개수가 홀수이면 수식 안/밖 상태를 토글한다. 수식 안에서 시작한 줄은 구조 판정을 모두 건너뛰고 현재 청크에 붙인다. 이 상태는 쪽 경계를 넘어 유지된다. 따라서 `$$…$$`는 쪼개지지 않는다.
- **label / display_label**: `label`은 정규화된 영어 소문자 `"{unit_type} {번호}"`(예: `theorem 3.2`, 한국어 원문 `보조정리 3.2`도 `lemma 3.2`)다. 번호가 없거나 proof이면 `None`이다. `display_label`은 원문 키워드와 번호, 선택적 `(이름)`이다(`정리 3.2`, `Theorem 2.3 (Heine-Borel)`).
- **refs**: `extract_refs(text, own_label)`이 본문 속 `키워드 번호`를 정규화 라벨로 모은다. 중복과 자기 자신은 뺀다. 산문 속 언급(`Theorem 9.9`)도 포함한다.
- **chapter / section** (`_parse_heading`): `#`~`######` 헤딩, 또는 80자 이하의 평문 `Chapter N`/`제N장`. `Chapter`/`제N장`이거나 `#` 레벨 1이거나 `N`으로 시작하면 chapter, `N.M`으로 시작하면 section이다. chapter가 바뀌면 section을 초기화한다. 제목만 있는 청크는 뒤따르는 단위가 흡수한다.
- **긴 본문 분할**: `text` 청크만 `max_chars` 이상일 때 빈 줄 경계에서 나눈다. 단위 청크(정리+증명)는 길이와 무관하게 쪼개지 않는다.
- **쪽 범위**: 각 줄에 쪽 번호를 붙여 두고, flush 때 첫/끝 비공백 줄에서 `page_start`/`page_end`를 정한다.
- **불변식**(모든 fixture에 테스트로 고정): 청크당 `$$` 짝수, 정리와 증명은 한 청크, 청크를 이으면 원문 내용 손실 없음, `ord`는 0부터 연속, `page_start <= page_end`, 빈 청크 없음.
- `ChunkDraft.as_row()`가 `db.replace_chunks`가 받는 dict(`ord, unit_type, label, display_label, chapter, section, page_start, page_end, text, refs`)를 만든다.

## 6. 인제스트 파이프라인 (`ingest/pipeline.py`)

`ingest_book(book_id, pdf_path)`는 예외를 밖으로 던지지 않는다. 실패하면 `status="error"`와 메시지를 기록한다.

```
queued → extracting (진행률 0~0.5) → indexing (0.5~1.0) → ready
                              └──────── 실패 시 error ────────┘
```

1. `num_pages` 기록, `extracting`.
2. 이미 `pages`에 있는 쪽을 제외하고 **100쪽 배치**(`EXTRACT_BATCH`)로 추출해 쪽마다 `db.save_page`(INSERT OR REPLACE). 중간에 실패해도 추출한 쪽이 남고, 재실행하면 건너뛴다. 진행률은 `0.5 × 완료쪽/전체`.
3. `$$`가 홀수인 쪽은 경고 로그(추출 오류 신호; 이후 내용이 한 청크로 뭉칠 수 있다).
4. `structure.build_chunks`. 청크가 0개면 `IngestError("PDF에서 추출된 텍스트가 없습니다.")`.
5. `indexing` → `db.replace_chunks`(전부 교체) → `vectors.index_chunks`(해당 책 벡터를 지운 뒤 다시 추가; 진행률 `0.5 + 0.5f`).
6. `ready`, `progress=1.0`.

재실행 시 캐시되는 것은 추출 결과(pages)뿐이고, 청크와 벡터는 매번 다시 만든다.

**재시작 복구**: 앱 기동(lifespan) 때 백그라운드 스레드가 `recover_interrupted()`로 `queued`/`extracting`/`indexing`에 멈춘 책을 찾는다. `data/books/{id}/book.pdf`가 있으면 `ingest_book`을 다시 돌려 이어 하고(추출된 쪽은 건너뜀), 없으면 `status="error"`와 사유 메시지로 바꾼다. `ready`/`error` 책은 건드리지 않는다.

**삭제와의 경쟁**: `pipeline.purge_book(id)`(벡터 → SQLite → 파일)가 유일한 삭제 경로이고 `DELETE /books/{id}`가 이를 부른다. 파이프라인은 쪽 저장, 청크 교체, 벡터 배치마다, 그리고 `ready` 기록 직전에 책이 아직 있는지 확인하고(`BookDeleted`), 모듈 락으로 이 쓰기들과 삭제가 서로 끼어들지 않게 한다. 삭제를 알아챈 파이프라인은 자신이 남긴 벡터·행·파일을 치우고 조용히 끝난다(error로 기록하지 않음).

## 7. 저장소 (`index/`)

### SQLite (`db.py`)
단일 공유 연결(`check_same_thread=False`, WAL, `foreign_keys=ON`)을 `RLock`으로 보호한다. 쓰기는 `tx()`(예외 시 rollback)를 쓴다.

| 테이블 | 핵심 컬럼 |
|---|---|
| `books` | `id`(uuid hex), `title`, `filename`, `num_pages`, `status`, `progress`, `message`, `created_at` |
| `pages` | `(book_id, page_no)` PK, `markdown`, `method`. `ON DELETE CASCADE` |
| `chunks` | `id` AUTOINCREMENT, `book_id`, `ord`, `unit_type`, `label`, `display_label`, `chapter`, `section`, `page_start`, `page_end`, `text`, `refs`(JSON 문자열). 인덱스 `(book_id, label)`, `(book_id, ord)` |
| `chunks_fts` | FTS5(`text`, `display_label`), `content='chunks'`, `tokenize='unicode61'` |
| `messages` | `id`, `conversation_id`, `role`, `content`, `meta`(JSON), `created_at` |

- FTS 동기화는 `chunks_ai`(INSERT), `chunks_ad`(DELETE) **트리거**가 맡는다. UPDATE 트리거는 없으므로 청크는 갱신하지 않고 `replace_chunks`로 통째로 교체한다.
- `fts_search(query, book_ids, limit)`: `\w+` 토큰 중 2자 이상이거나 숫자인 것만 쓰고, 각각 `"tok"*`(따옴표로 FTS 문법 오류 방지, 접두 일치로 한국어 조사 변형 흡수)로 감싸 `OR`로 묶는다. **bm25는 낮을수록 좋다.**
- `chunks_by_label(labels, book_ids)`는 `label` 완전 일치로 조회한다(정규화된 영어 라벨 전제).
- `delete_book`은 chunks, pages, books를 지운다(FTS는 트리거가 정리). 벡터는 호출자가 `vectors.delete_book`을 함께 불러야 한다.

### Chroma (`vectors.py`)
컬렉션 `chunks__<임베딩 모델 slug>`(cosine, `PersistentClient`)에 청크 id(문자열)로 임베딩과 `{book_id}` 메타데이터만 저장한다. 모델마다 컬렉션이 달라 차원·의미가 다른 벡터가 섞이지 않는다.

- `embed_text_for(chunk)`: `"장 / 절 / display_label"` 헤더 + 줄바꿈 + 본문. 이 형식을 바꾸면 전체 재색인이 필요하다.
- `index_chunks`는 64개씩 `document`로 임베딩한다. `search`는 `query`로 임베딩하고 `1 - distance`(높을수록 유사)를 돌려준다. 책이 하나면 `book_id` 같음, 여럿이면 `$in` 필터.
- `has_book(book_id)`: 현재 모델의 컬렉션에 책의 벡터가 있는가. `drop_legacy()`: Gemini 시절의 `chunks` 컬렉션을 지운다.
- **임베딩 모델 교체**: `THEOREMLENS_EMBED_MODEL`을 바꾸면 새 컬렉션이 비어 있다. 서버 시작 때 복구 스레드가 `recover_interrupted` 재개 뒤 `pipeline.reindex_missing()`을 불러, `ready`인데 벡터가 없는 책을 SQLite 청크로 다시 임베딩한다(재추출·Gemini 호출 없음). 재임베딩 동안 책은 `indexing`이고, 실패하면 `error`와 사유가 남는다. 벡터가 갖춰지면 옛 `chunks` 컬렉션을 지운다. 새 모델은 유사도 분포가 다르므로 `min_similarity`를 평가 러너로 다시 튜닝한다.

## 8. 검색 (`rag/retrieve.py`)

`retrieve(question, book_ids, top_k=None) -> Retrieval(sources, not_found, best_similarity)`.

1. `book_ids`가 비면 not_found.
2. 벡터 검색과 BM25를 각각 `top_k × 3`개 후보로 실행. `best`는 벡터 최고 유사도.
3. **라벨 직접 조회**: 질문 속 `Theorem 3.2`, `정리 3.2`를 `extract_labels`로 정규화 라벨로 바꿔 `chunks_by_label`로 가져온다(`via="label"`, 맨 앞).
4. **근거 부족 판정**: 라벨 히트가 없고 `best < min_similarity(0.55)`이면 `not_found`를 돌려준다. 이 경로에서 **LLM을 호출하지 않는다**. 라벨 히트가 있으면 유사도가 낮아도 통과한다.
5. **RRF 융합**(`rrf`, k=60): 순위표마다 `1/(k+rank)`를 합산한다. 한 순위표 안의 중복 id는 한 번만 센다. 동점은 먼저 나온 것이 앞선다. 라벨 히트를 뺀 남은 자리(`top_k - 라벨 히트 수`)만큼 `via="search"`로 추가한다.
6. **1-hop 인용 확장**: 위에서 고른 청크의 `refs`를 **인용한 청크와 같은 책**에서 찾아 `via="cited"`로 추가한다. 최대 4개이며 확장된 청크의 refs는 따라가지 않는다(2-hop 방지). `top_k`에는 포함되지 않아 총 소스는 최대 `top_k + 4`다.
7. 추가된 순서가 `S1, S2, …` 번호가 된다.

`Source`는 `sid, chunk_id, book_id, book_title, page_start, page_end, label, display_label, unit_type, chapter, section, text, via`를 가진다.

## 9. 답변 (`rag/answer.py`, `rag/prompts.py`)

`answer(question, book_ids, history=None, fast=False, on_unsupported="drop") -> AnswerResponse`.

- `retrieve`가 not_found이면 `NOT_FOUND_MESSAGE`를 즉시 돌려준다(LLM 호출 없음).
- 모델: `fast=True`면 `fast_model`, 아니면 `answer_model`. temperature 0, `ANSWER_SCHEMA = {answer_markdown, claims:[{text, source_ids, quote}], not_found}`.
- **시스템 프롬프트 규칙**: 주어진 발췌만 근거로 삼는다. 모든 주장은 `claims`에 넣고 `source_ids`는 주어진 `S#`만 쓴다. `quote`는 발췌 본문에서 그대로 복사한다. 본문에는 `[S#]`를 표시한다. 책 밖 설명은 `## 보충(책 외부)` 섹션에만 쓰고 claims에 넣지 않는다. 답할 수 없으면 `not_found=true`. 이전 대화는 근거가 아니다.
- **컨텍스트 조립** (`build_contents`): 선택적 이전 대화 블록 + `## 발췌`(`[S1] 책: … | 쪽: 12-13 | 라벨: … | 위치: 장 > 절` 다음 본문) + `## 질문`.
- **history**: user/assistant 역할만, 비어 있지 않은 항목 중 **최근 6개 메시지**. "맥락 참고용, 근거 아님"으로 표시한다.
- **후속 질문 검색**: 단독 질문으로 먼저 `retrieve`하고, not_found이면 **직전 사용자 질문**을 앞에 붙인 질의로 한 번 더 `retrieve`한다(결정적, LLM 질의 재작성 없음; assistant 턴은 쓰지 않는다). 그래도 not_found면 LLM 호출 없이 반환한다. 프롬프트의 질문은 원문 그대로다.
- 응답 JSON을 해석하지 못하면 `AnswerError`. 모델이 `not_found`를 주면 그대로 반영한다.
- **응답 모델(pydantic)**: `AnswerResponse(answer_markdown, claims[ClaimOut], citations[CitationOut], not_found, outside_book, verification)`.
  - `ClaimOut(text, display_text, source_ids, quote, status, reason)`
  - `CitationOut(sid, book_id, book_title, page_start, page_end, label, display_label, chapter, section, text)`
  - `VerificationOut(llm_checked, dropped, flagged)`
- **인용 원문은 DB에서만**: `citations[].text`는 검색된 청크(`Source.text`)이고 모델 출력이 아니다. 검증을 통과한 claim이 인용한 출처만 `S#` 순으로 담는다.
- **`outside_book`**: 경고 처리되지 않은(verified/unchecked) claim이 하나도 없으면 True이고, 이때 `answer_markdown` 앞에 `## 보충(책 외부)`를 붙인다. 근거 없는 답이 책 내용처럼 보이지 않게 하기 위함이다.

## 10. 검증 (`rag/verify.py`)

결정적 규칙을 먼저, LLM은 두 번째 의견으로 쓴다.

1. **정규화** (`normalize`): 소문자화, `\left \right \displaystyle \big` 및 `\, \; \: \! \\` 제거, `\leq→\le`, `\geq→\ge`, `\ldots/\cdots→\dots`, 공백·`$`·중괄호 제거.
2. **퍼지 매칭** (`match_score`): 정규화한 quote가 출처에 그대로 있으면 100, 아니면 rapidfuzz `partial_ratio`. 임계값 `MATCH_THRESHOLD = 90`(실제 책으로 튜닝하지 않았다).
3. **규칙 검사** (`check_quotes`, 네트워크 없음): 사유는 `no_source`, `no_quote`, `unknown_source`, `quote_not_found`, `ok`. 인용한 출처 중 하나라도 일치하면 통과.
4. **LLM 지지 검사** (`llm_support_check`): 규칙 통과분만 `fast_model`에 보내 `{results:[{index, supported}]}`를 받는다. 응답에서 빠진 claim은 미지지로 본다. 키 없음/JSON 오류면 `None`(검사 불가). 통과분이 없으면 호출 자체를 하지 않는다.
5. **상태**(`verify_claims`): `rejected`(규칙 탈락), `unsupported`(LLM 미지지), `unchecked`(LLM 검사 불가, 규칙 통과), `verified`. **LLM이 규칙 탈락을 되살릴 수 없다.**
6. **처리 모드**: `drop`(기본)은 `rejected`/`unsupported`를 제거하고 `dropped`를 센다. `flag`는 남기고 `display_text` 앞에 `⚠ `를 붙인다.
7. 출처 본문은 `{S#: chunk_id}`를 받아 DB에서 읽는다(`load_sources`). 모델이 만든 출처 텍스트는 검증에 쓰이지 않는다.

## 11. API (`app/main.py`)

HTTP 계층만 담당한다. `create_app(web_dir=None)`, 모듈 수준 `app`.

| 라우트 | 동작 |
|---|---|
| `POST /books` | multipart `file`. `.pdf` 확장자 아니면 400, 첫 5바이트가 `%PDF-`가 아니면 400, `max_upload_mb` 초과 413. 저장 경로는 서버가 정함. `BackgroundTasks`로 `ingest_book` 등록, `BookOut`(처음엔 `queued`) 반환 |
| `GET /books` | `BookOut` 목록(`num_chunks` 포함) |
| `GET /books/{id}/status` | `{id, status, progress, message, num_pages}`, 없으면 404 |
| `DELETE /books/{id}` | `pipeline.purge_book`(벡터 → SQLite → 디스크). 진행 중인 인제스트가 있어도 안전. `{"deleted": id}` |
| `POST /chat` | `{book_ids(1~20), question(≤4000자, 공백만이면 422), conversation_id?(^[A-Za-z0-9_-]{1,64}$)}` → `AnswerResponse + conversation_id` |
| `GET /books/{id}/page/{n}` | 150dpi PNG. 범위(1..num_pages) 밖/없는 책이면 404 |
| `/` | `web/` 정적 서빙(마지막에 마운트, 디렉터리가 없어도 앱은 뜸) |

- `/chat` 오류: 책 없음 404, 아직 `ready`가 아닌 책 409, 키 없음 503, 모델 JSON 오류 502(이 경우 메시지는 저장하지 않음).
- 성공 시 user/assistant 메시지를 `messages`에 저장한다(assistant의 `meta`에 응답 전체). `conversation_id`가 없으면 새로 만들고, 있으면 최근 메시지 50개를 history로 쓴다.
- 인제스트는 반드시 백그라운드로 돌고, 요청을 막지 않는다. `/chat`은 동기이며 스트리밍은 없다.
- 업로드는 클라이언트 파일명을 경로에 쓰지 않는다(경로 순회 방지). 제목에는 디렉터리 부분을 뗀 stem만 쓴다.

## 12. 프런트엔드 (`web/`)

빌드 없는 HTML/CSS/JS. CDN 라이브러리는 jsdelivr에서 **정확한 버전으로 고정**한다(marked 12.0.2, dompurify 3.1.6, katex 0.16.11).

- `index.html`: 서재 패널(업로드, 진행률, 책 체크박스), 채팅 패널, 출처 패널.
- `lib.js`: DOM에 의존하지 않는 순수 함수(`extractMath`, `splitCitations`, `claimBadge`, `pageRange`). 브라우저에서는 `TLLib`, node에서는 `module.exports`. node로 단위 검사할 수 있다.
- `app.js`: 서재는 처리 중인 책만 2초마다 폴링하고, `ready`인 책만 선택할 수 있다. 채팅은 `conversation_id`를 유지한다.
- **렌더링 파이프라인**: 수식을 자리표시자로 빼냄(마크다운 변환기가 `_`, `*`, `\`를 망가뜨리므로) → `marked.parse` → `DOMPurify.sanitize`(`style/img/a/form/input/iframe` 금지) → `innerHTML` 대입(**코드 전체에서 유일한 지점**) → 텍스트 노드에서 자리표시자를 `katex.render(trust:false)`로, `[S#]`를 클릭 가능한 버튼으로 치환. `code`/`pre` 안은 치환하지 않는다.
- **출처 패널**: 서버가 준 `citations[].text`를 렌더링하고, `/books/{id}/page/{n}` 이미지를 최대 4쪽까지 지연 로딩한다.
- **답변 표시**: `not_found` 배너, `outside_book` 경고, claim 상태 배지(`verified`만 ✅, 나머지는 ⚠ + 사유), "보충(책 외부)" 구간 구분 표시.

## 13. 알려진 한계와 향후 과제

- 후속 질문 검색은 직전 사용자 질문 한 개만 쓰는 폴백이다. 단독 질문이 우연히 임계값을 넘으면 이력은 검색에 쓰이지 않는다.
- 재시작 복구는 기동 시 한 번만 돌고 책을 순차로 재개한다. 삭제 경쟁을 막는 락은 프로세스 안에서만 유효하므로, 같은 DB를 여러 프로세스(`--reload` 워커 등)가 쓰면 같은 책이 중복 재개되거나 삭제와 겹칠 수 있다. 단일 프로세스 실행을 전제로 한다.
- `/chat`은 `on_unsupported`를 넘기지 않아 항상 `drop`이다. 사용자에게 ⚠ 표시를 보여 주려면 `flag` 모드 노출이 필요하다.
- 퍼지 임계값(90)과 `min_similarity`(0.55)는 합성 테스트 기준이며 실제 책으로 튜닝하지 않았다.
- 스트리밍 응답이 없다.
- `tests/eval/run_eval.py`(검색 적중률, 인용 정확도, 거절률 평가)는 있으나 예시 질문 파일(`questions.example.json`)뿐이다. 책별 질문 파일을 만들어 기준선을 측정해야 한다.
