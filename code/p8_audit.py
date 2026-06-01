#!/usr/bin/env python3
"""P8: audit edge QUALITY — of existing edges, how many are genuine TRANSFER
(knowledge/representation/pretraining from src benefits dst) vs mere USAGE
(src is just applied to dst)? Batched LLM classification on evidence spans.
"""
import sys, json, random, pathlib
import llm
from graph import Graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = Graph.load(ROOT / "data" / "graph_resolved.json")
random.seed(0)
N_PER_TYPE = 40
BATCH = 8

SYSTEM = """You audit edges of a research transfer graph. For each edge "SRC --> DST" with its evidence sentence, classify it as:
- "transfer": genuine transfer/help — knowledge, representation, pretraining, multi-task signal, or features from SRC measurably benefit DST (who-helps-whom).
- "usage": SRC is merely APPLIED TO / USED FOR DST with no transfer claim (e.g., "we use U-Net for segmentation"). This is just method-task usage, not transfer.
- "unclear": evidence insufficient to tell.
Output STRICT JSON: {"labels":[{"i":0,"label":"transfer|usage|unclear"}, ...]} in input order."""


def sample(etype):
    es = [e for (s, d), e in G.edges.items()
          if f'{G.nodes[s]["type"]}->{G.nodes[d]["type"]}' == etype
          and e["state"] != "inferred" and e.get("evidence")]
    random.shuffle(es)
    return es[:N_PER_TYPE]


def classify(edges):
    labels = []
    for b in range(0, len(edges), BATCH):
        chunk = edges[b:b + BATCH]
        lines = []
        for i, e in enumerate(chunk):
            lines.append(f'{i}. SRC="{G.nodes[e["src"]]["text"]}" DST="{G.nodes[e["dst"]]["text"]}" '
                         f'EVIDENCE="{e["evidence"][:160]}"')
        usr = "Classify each edge:\n" + "\n".join(lines)
        try:
            out = llm.extract_json(llm.call(SYSTEM, usr, max_tokens=800))
            m = {d["i"]: d["label"] for d in out.get("labels", [])}
        except Exception as e:
            print("  batch failed:", e); m = {}
        for i, e in enumerate(chunk):
            labels.append((e, m.get(i, "unclear")))
    return labels


def main():
    print(f"auditing edge quality (transfer vs usage), {N_PER_TYPE}/type\n")
    for etype in ("method->task", "task->task"):
        labs = classify(sample(etype))
        c = {"transfer": 0, "usage": 0, "unclear": 0}
        for _, l in labs:
            c[l] = c.get(l, 0) + 1
        tot = len(labs)
        print(f"=== {etype}  (n={tot}) ===")
        for k in ("transfer", "usage", "unclear"):
            print(f"   {k:<9}: {c[k]:2d}  ({100*c[k]/tot:.0f}%)")
        print("   examples flagged USAGE:")
        for e, l in labs:
            if l == "usage":
                print(f"      {G.nodes[e['src']]['text']} -> {G.nodes[e['dst']]['text']}")
        print("   examples confirmed TRANSFER:")
        shown = 0
        for e, l in labs:
            if l == "transfer" and shown < 4:
                print(f"      {G.nodes[e['src']]['text']} -> {G.nodes[e['dst']]['text']}  | \"{e['evidence'][:70]}\"")
                shown += 1
        print()


if __name__ == "__main__":
    main()
