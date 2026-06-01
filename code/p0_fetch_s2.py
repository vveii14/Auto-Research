#!/usr/bin/env python3
"""P0: fetch a brain medical-imaging corpus from Semantic Scholar (bulk search).

Saves timestamped papers (title, abstract, year, venue, citations) to
data/brain_papers.jsonl. Rate-limited to <1 req/sec per the S2 key limit.
"""
import os, sys, json, time, pathlib, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
OUT = DATA / "brain_papers.jsonl"

# load key from secrets.env (never printed)
KEY = None
for line in (ROOT / "secrets.env").read_text().splitlines():
    if line.startswith("S2_API_KEY="):
        KEY = line.split("=", 1)[1].strip()
assert KEY, "S2_API_KEY not found in secrets.env"

BASE = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
# brain medical-IMAGE-ANALYSIS / ML methods, where task transfer is a real concept.
# + = AND, | = OR
QUERY = ("(brain | neuroimaging | MRI | fMRI) + "
         "(deep learning | machine learning | segmentation | classification | "
         "self-supervised | contrastive | transfer learning | pretraining | "
         "convolutional | neural network | representation | registration | reconstruction)")
FOS = "Computer Science"  # concentrate on the methods community
FIELDS = "title,abstract,year,venue,publicationVenue,citationCount,externalIds,fieldsOfStudy"
YEARS = "2015-2024"
MAX_PAPERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SLEEP = 1.2  # > 1s to respect rate limit

def get(token=None):
    params = {"query": QUERY, "year": YEARS, "fields": FIELDS, "fieldsOfStudy": FOS}
    if token:
        params["token"] = token
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-api-key": KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def main():
    seen, kept, pages = set(), 0, 0
    token = None
    with OUT.open("w") as f:
        while kept < MAX_PAPERS:
            try:
                d = get(token)
            except Exception as e:
                print("request error, retrying once:", e); time.sleep(5)
                d = get(token)
            pages += 1
            if pages == 1:
                print(f"total query matches: {d.get('total')}")
            for p in d.get("data") or []:
                pid = p.get("paperId")
                if not pid or pid in seen:
                    continue
                if not (p.get("abstract") and p.get("year")):
                    continue  # need abstract + timestamp
                seen.add(pid)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= MAX_PAPERS:
                    break
            print(f"page {pages}: kept so far {kept}")
            token = d.get("token")
            if not token:
                break
            time.sleep(SLEEP)
    print(f"\nDONE: {kept} papers with abstract -> {OUT}")

    # quick summary
    yrs, venues = {}, {}
    for line in OUT.read_text().splitlines():
        p = json.loads(line)
        yrs[p["year"]] = yrs.get(p["year"], 0) + 1
        v = (p.get("venue") or "?").strip()
        venues[v] = venues.get(v, 0) + 1
    print("\nyear distribution:")
    for y in sorted(yrs):
        print(f"  {y}: {yrs[y]}")
    print("\ntop 15 venues:")
    for v, c in sorted(venues.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:5d}  {v[:70]}")

if __name__ == "__main__":
    main()
