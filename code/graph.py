"""Task-transfer graph data model (proposal §5.1). JSON-backed for the pilot."""
import json, pathlib


class Graph:
    def __init__(self):
        self.nodes = {}   # id -> node dict
        self.edges = {}   # (src,dst) -> edge dict

    @staticmethod
    def nid(text, ntype):
        return f"{ntype}:{text.strip().lower()}"

    def add_node(self, text, ntype, domain=None, year=None, paper_id=None):
        nid = self.nid(text, ntype)
        n = self.nodes.get(nid)
        if n is None:
            n = {"id": nid, "text": text.strip(), "type": ntype, "domain": domain,
                 "first_year": year, "paper_refs": [], "freq": 0}
            self.nodes[nid] = n
        n["freq"] += 1
        if paper_id and paper_id not in n["paper_refs"]:
            n["paper_refs"].append(paper_id)
        if year is not None:
            n["first_year"] = year if n["first_year"] is None else min(n["first_year"], year)
        return nid

    def add_edge(self, src, dst, sign, strength, state, evidence, year=None,
                 measured=False, paper_id=None):
        key = (src, dst)
        e = self.edges.get(key)
        if e is None:
            e = {"src": src, "dst": dst, "sign": sign, "strength": strength,
                 "state": state, "confidence": strength, "measured": measured,
                 "evidence": evidence, "first_year": year, "paper_refs": [],
                 "access_count": 0, "contradicts_prior": False}
            self.edges[key] = e
        else:  # verified beats prior; record contradiction on sign flip
            if sign != e["sign"] and "verified" in (state, e["state"]):
                e["contradicts_prior"] = True
            if state == "verified":
                e.update(sign=sign, strength=strength, state="verified",
                         measured=measured, evidence=evidence)
            if year is not None:
                e["first_year"] = year if e["first_year"] is None else min(e["first_year"], year)
        if paper_id and paper_id not in e["paper_refs"]:
            e["paper_refs"].append(paper_id)
        return key

    def stats(self):
        nt = {}
        for n in self.nodes.values():
            nt[n["type"]] = nt.get(n["type"], 0) + 1
        es = {}
        for e in self.edges.values():
            es[e["sign"]] = es.get(e["sign"], 0) + 1
        return {"nodes": len(self.nodes), "by_type": nt,
                "edges": len(self.edges), "by_sign": es}

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"nodes": list(self.nodes.values()),
             "edges": list(self.edges.values())}, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path):
        g = cls(); d = json.loads(pathlib.Path(path).read_text())
        for n in d["nodes"]:
            g.nodes[n["id"]] = n
        for e in d["edges"]:
            g.edges[(e["src"], e["dst"])] = e
        return g
