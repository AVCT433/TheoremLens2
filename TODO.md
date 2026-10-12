# TODO

남은 작업 목록. 규칙은 [CLAUDE.md](CLAUDE.md)의 "TODO 관리" 참고.

## Next

- [ ] 책별 평가 질문 파일 20~30문항 작성 (사람이 직접, `tests/eval/`), `run_eval.py`로 기준선 측정
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

- [x] 백엔드(ingest, index, rag, app)와 프런트엔드 기본 구현
- [x] 품질 평가 러너 `tests/eval/run_eval.py`
- [x] 멈춘 인제스트 재개와 삭제 경쟁 조건 처리
- [x] architecture.md 13장을 현재 구현에 맞게 수정
