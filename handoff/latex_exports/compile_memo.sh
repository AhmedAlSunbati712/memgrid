#!/usr/bin/env bash
# Build REPORT.pdf from REPORT.md using Pandoc + memoir + Tectonic.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
export HOME="${HOME:-/Volumes/youwhat/tmp/fakehome}"
export TMPDIR="${TMPDIR:-/Volumes/youwhat/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/Volumes/youwhat/tmp/xdg-cache}"
export TECTONIC_CACHE_DIR="${TECTONIC_CACHE_DIR:-/Volumes/youwhat/tmp/tectonic-cache}"
PANDOC="${PANDOC:-/Volumes/youwhat/anac/anaconda3/bin/pandoc}"
TECTONIC="${TECTONIC:-/Volumes/youwhat/tools/tectonic}"

mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TECTONIC_CACHE_DIR" "$HERE/pdf"

# Tectonic runs from pdf/; symlink bundled figures so \\includegraphics{results/...} resolves.
ln -sfn "../../results" "$HERE/pdf/results"

"$PANDOC" "$ROOT/REPORT.md" \
  -o "$HERE/pdf/REPORT.tex" \
  --standalone \
  --number-sections \
  --resource-path="$ROOT" \
  -V documentclass=memoir \
  -V fontsize=11pt \
  -V geometry:margin=1in \
  -H "$HERE/memo_header.tex"

python3 "$HERE/patch_memo_tex.py" "$HERE/pdf/REPORT.tex"

"$TECTONIC" --outdir "$HERE/pdf" --keep-logs "$HERE/pdf/REPORT.tex"

echo "Wrote $HERE/pdf/REPORT.pdf"
