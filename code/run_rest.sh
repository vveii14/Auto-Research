#!/usr/bin/env bash
# Wait for P2 extraction to finish, then run resolve + temporal eval.
cd "$(dirname "$0")/.."
echo "[run_rest] waiting for p2_extract to finish..."
while pgrep -f p2_extract.py >/dev/null; do sleep 30; done
echo "[run_rest] extraction done: $(python3 -c "import json;print(len(json.load(open('data/processed_ids.json'))))" 2>/dev/null) papers processed"
echo "[run_rest] === P2-resolve ==="
python3 code/p2_resolve.py
echo "[run_rest] === P4-eval (temporal replay) ==="
python3 code/p4_eval.py
echo "[run_rest] DONE"
