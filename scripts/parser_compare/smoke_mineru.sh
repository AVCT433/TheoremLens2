#!/usr/bin/env bash
# MinerU 스모크 테스트: 임시 격리 환경(uv run --isolated)에서 PDF 일부 쪽만 변환한다.
# 사용법: bash scripts/parser_compare/smoke_mineru.sh <pdf> [출력.md] [tier]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "사용법: $0 <pdf> [출력.md] [tier]" >&2
  exit 1
fi

PDF="$1"
OUT="${2:-scripts/parser_compare/out/mineru/mineru_smoke}"
TIER="${3:-standard}"

cd "$(dirname "$0")/../.."  # 프로젝트 루트
mkdir -p "$(dirname "$OUT")"

uv run --isolated --with "mineru>=4.0,<5" --with pymupdf \
  python scripts/parser_compare/smoke_mineru.py "$PDF" "$OUT" --tier "$TIER"
echo "출력: $OUT"
