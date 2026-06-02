#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "[chain] waiting for p14_fullextract..."
while pgrep -f p14_fullextract.py >/dev/null; do sleep 60; done
echo "[chain] extraction done: $(python3 -c "import json;print(len(json.load(open('data/processed_transfer_ids.json'))))" 2>/dev/null) papers"
echo "[chain] === resolve (full transfer graph) ==="; python3 code/p10b_resolve.py
echo "[chain] === hub-compounding (full graph) ==="; python3 code/p13_hub_compounding.py
echo "[chain] === graph-value (full graph) ==="; python3 code/p12_graph_value.py
echo "[chain] DONE"
