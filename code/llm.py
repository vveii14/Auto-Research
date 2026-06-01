"""Thin client for the Azure-hosted Anthropic endpoint (temp 0 by default)."""
import os, json, time, pathlib, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent.parent
_cfg = {}
for line in (ROOT / "secrets.env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); _cfg[k.strip()] = v.strip()

ENDPOINT = _cfg["ANTHROPIC_AZURE_ENDPOINT"]
KEY = _cfg["ANTHROPIC_AZURE_KEY"]
MODEL = _cfg.get("ANTHROPIC_MODEL", "claude-opus-4-7-3")
VERSION = _cfg.get("ANTHROPIC_VERSION", "2023-06-01")


def call(system, user, max_tokens=2000, retries=4):
    # NB: this model rejects `temperature` ("deprecated for this model"); omit it.
    body = json.dumps({
        "model": MODEL, "max_tokens": max_tokens,
        "system": system, "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Content-Type": "application/json", "x-api-key": KEY,
                 "anthropic-version": VERSION})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["content"][0]["text"]
        except urllib.error.HTTPError as e:
            wait = 2 ** attempt
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(wait); continue
            raise
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after retries")


def extract_json(text):
    """Pull the first {...} JSON object out of a model reply."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"): s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1])
