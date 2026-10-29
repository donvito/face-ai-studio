#!/usr/bin/env bash
# InsightFace Studio: create the venv on first run, then serve on http://127.0.0.1:7860
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/uvicorn ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
echo "First start downloads the InsightFace model packs (~400 MB) to ~/.insightface/models"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port "${PORT:-7860}"
