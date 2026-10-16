# TODO

남은 작업 목록. 규칙은 [CLAUDE.md](CLAUDE.md)의 "TODO 관리" 참고.

## Next

- [ ] 테스트 책 `data/samples/Enderton_Elements of set theory_(1977).pdf`(gitignore됨, 289쪽)용 평가 질문 파일 작성. 아래 평가 질문 항목과 같이 진행
- [ ] MinerU 호출 비용 실측: 20쪽 배치마다 `mineru-kit`가 모델을 다시 로딩한다(스모크에서 19쪽에 약 2분 30초, 첫 쪽 워밍업 약 56초). 느리면 `EXTRACT_BATCH`를 키우거나 상주 프로세스로 바꾼다
- [ ] 실제 책으로 `blocks_to_markdown` 품질 점검: 표(HTML), 각주, 목록, 쪽을 넘어가는 문단(`continues_prev`)이 `structure.py` 분할에 미치는 영향
- [ ] 책별 평가 질문 파일 20~30문항 작성 (사람이 직접, `tests/eval/`), `run_eval.py`로 기준선 측정
- [ ] 로컬 임베딩(bge-m3)에 맞춰 `min_similarity` 재튜닝 (현재 0.55는 Gemini 기준값). 평가 질문 파일이 있어야 가능
- [ ] 임계값·청크 크기 튜닝 (`MATCH_THRESHOLD=90`, `min_similarity=0.55`, 청크 2000자). 하나씩 바꾸고 재측정, `architecture.md` 같이 갱신

## Later

- [ ] 검증 실패 시 1회 재생성 (`theoremlens/rag/answer.py`)
- [ ] `/chat`에서 `flag` 모드와 `fast` 모델 선택 노출 (`app/main.py`, `web/app.js`)
- [ ] 대화 기록 조회 GET 엔드포인트와 새로고침 복원
- [ ] 추출 미리보기 화면: 원본 페이지 이미지와 변환 텍스트 비교 (`db.get_page` 재사용)
- [ ] (선택) 스트리밍 응답
- [ ] (선택) README
- [ ] (선택) 프런트 DOM 동작 테스트
- [ ] (선택) `generate_json` 통합 테스트

## Done

- [x] 추출을 Marker/PyMuPDF/Gemini 비전에서 MinerU(middle_json)로 일원화하고 파싱의 Gemini 의존 제거
- [x] 임베딩을 Gemini에서 로컬 모델(sentence-transformers, bge-m3)로 전환하고 모델별 컬렉션 + 자동 재임베딩 추가
- [x] 백엔드(ingest, index, rag, app)와 프런트엔드 기본 구현
- [x] 품질 평가 러너 `tests/eval/run_eval.py`
- [x] 멈춘 인제스트 재개와 삭제 경쟁 조건 처리
- [x] architecture.md 13장을 현재 구현에 맞게 수정
