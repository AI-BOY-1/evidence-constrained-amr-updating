"""Exhaustive all-query skani screen for the frozen E. coli FASTA cohort.

This is an ANI screening audit, not a SNP phylogeny and not a patient/lab
independence proof.  It queries every valid genome against one shared skani
database with a frozen approximate identity cutoff and writes resumable output.
"""
from __future__ import annotations
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib, json, os, time
from pathlib import Path
import pyskani

ROOT = ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT = ncs_paths.RESEARCH_ROOT
FASTA = ROOT / "fasta"
INTEGRITY = ROOT / "ecoli_fasta_integrity.json"
RESULT = OUT / "ecoli_full_skani_screen.json"
CHECKPOINT = OUT / "ecoli_full_skani_screen_checkpoint.json"
CODE = Path(__file__)

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load_contigs(path: Path):
    out, seq = [], []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    out.append("".join(seq).encode("ascii")); seq = []
            elif line.strip():
                seq.append(line.strip())
    if seq:
        out.append("".join(seq).encode("ascii"))
    return out

def main():
    inv = json.loads(INTEGRITY.read_text(encoding="utf-8"))
    paths = sorted(FASTA / f"{r['sequence_id']}.fna" for r in inv["inventory"] if r["status"] == "valid")
    assert len(paths) == 3516 and all(p.exists() for p in paths)
    names = [p.stem for p in paths]
    by_name = dict(zip(names, paths))
    started = time.time()
    ck = {"done": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "query_cutoff": 0.99, "database_k": 15, "database_compression": 125,
          "database_marker_compression": 1000, "code_sha256": sha(CODE),
          "pyskani_version": pyskani.SKANI_VERSION}
    if CHECKPOINT.exists():
        ck.update(json.loads(CHECKPOINT.read_text(encoding="utf-8")))
    done = ck.get("done", {})
    db = pyskani.Database(compression=125, marker_compression=1000, k=15)
    # The database is rebuilt from the frozen FASTA list on each resumable run;
    # validated per-query results are retained in the checkpoint.
    for i, p in enumerate(paths, 1):
        db.sketch(p.stem, *load_contigs(p), seed=True)
        if i % 100 == 0:
            print(json.dumps({"stage":"database","done":i,"total":len(paths),"elapsed_s":round(time.time()-started,1)}), flush=True)
    for i, p in enumerate(paths, 1):
        name = p.stem
        if name in done:
            continue
        hits = db.query(name, *load_contigs(p), seed=True, learned_ani=False, robust=True, cutoff=0.99)
        recs = []
        for h in hits:
            if h.reference_name == name:
                continue
            ident, qf, rf = float(h.identity), float(h.query_fraction), float(h.reference_fraction)
            if ident >= 0.99 and min(qf, rf) >= 0.80:
                recs.append({"reference": h.reference_name, "identity": ident,
                             "query_fraction": qf, "reference_fraction": rf})
        done[name] = recs
        if i % 25 == 0:
            ck["done"] = done; ck["last_query"] = name
            CHECKPOINT.write_text(json.dumps(ck, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"stage":"query","done":len(done),"total":len(paths),"hits":sum(len(v) for v in done.values()),"elapsed_s":round(time.time()-started,1)}), flush=True)
    # Canonicalize unordered pairs and compute reciprocal strict criteria.
    pair_map = {}
    for q, hits in done.items():
        for h in hits:
            r = h["reference"]
            key = tuple(sorted((q, r)))
            pair_map.setdefault(key, {})[q] = h
    pairs = []
    for (a, b), d in sorted(pair_map.items()):
        ab, ba = d.get(a), d.get(b)
        rec = {"a": a, "b": b, "a_to_b": ab, "b_to_a": ba,
               "reciprocal_complete": bool(ab and ba)}
        if ab and ba:
            rec["reciprocal_ani"] = min(ab["identity"], ba["identity"])
            rec["reciprocal_query_coverage"] = min(ab["query_fraction"], ba["query_fraction"])
            rec["reciprocal_reference_coverage"] = min(ab["reference_fraction"], ba["reference_fraction"])
            rec["strict_ani_0_99_cov_0_80"] = bool(rec["reciprocal_ani"] >= .99 and rec["reciprocal_query_coverage"] >= .80 and rec["reciprocal_reference_coverage"] >= .80)
            rec["ani_0_95_cov_0_80"] = bool(rec["reciprocal_ani"] >= .95 and rec["reciprocal_query_coverage"] >= .80 and rec["reciprocal_reference_coverage"] >= .80)
        pairs.append(rec)
    result = {"status":"complete", "fasta_files":len(paths), "queries_completed":len(done),
              "query_cutoff":.99, "candidate_pairs_returned":len(pairs),
              "reciprocal_complete_pairs":sum(x["reciprocal_complete"] for x in pairs),
              "reciprocal_ani_ge_0_95_cov_ge_0_80":sum(x.get("ani_0_95_cov_0_80",False) for x in pairs),
              "reciprocal_ani_ge_0_99_cov_ge_0_80":sum(x.get("strict_ani_0_99_cov_0_80",False) for x in pairs),
              "pairs":pairs,
              "method":{"pyskani_version":pyskani.SKANI_VERSION,"k":15,"compression":125,"marker_compression":1000,"robust":True,"learned_ani":False,"cutoff":.99},
              "inputs":{"fasta_integrity_sha256":sha(INTEGRITY),"code_sha256":sha(CODE)},
              "limitations":["The skani cutoff screens approximate identity; pairs below the cutoff are not reported with an exact ANI estimate.","ANI is not a SNP distance or phylogeny and cannot establish patient/laboratory independence.","No labels, thresholds or prediction files were changed."]}
    RESULT.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    ck["done"] = done; ck["status"] = "complete"; ck["result_sha256"] = sha(RESULT)
    CHECKPOINT.write_text(json.dumps(ck, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","fasta_files","queries_completed","candidate_pairs_returned","reciprocal_ani_ge_0_95_cov_ge_0_80","reciprocal_ani_ge_0_99_cov_ge_0_80")}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
