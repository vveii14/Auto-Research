#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "[chain] waiting for p10_reextract..."
while pgrep -f p10_reextract.py >/dev/null; do sleep 30; done
echo "[chain] re-extraction done: $(python3 -c "import json;print(len(json.load(open('data/processed_transfer_ids.json'))))" 2>/dev/null) papers"
echo "[chain] === resolve ==="; python3 code/p10b_resolve.py
echo "[chain] === re-eval (purified transfer graph) ==="; python3 code/p10c_eval.py
echo "[chain] DONE"
