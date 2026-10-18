"""Build a conservative, auditable E. coli human-host S/R table."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv, json
from collections import defaultdict, Counter
from pathlib import Path

ROOT = ncs_paths.data_path('experiments/ecoli_bvbrc')
G = ROOT / "ecoli_genome_metadata.tsv"
A = ROOT / "ecoli_lab_amr.tsv"
OUT = ROOT / "ecoli_human_sr_pairs.tsv"
REPORT = ROOT / "ecoli_curated_dataset.json"

def main():
    with G.open(encoding="utf-8", newline="") as f: genomes = {r["genome_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    with A.open(encoding="utf-8", newline="") as f: amr = list(csv.DictReader(f, delimiter="\t"))
    def is_human(v):
        h = (v or "").strip().lower()
        return h == "human" or "homo sapiens" in h
    human = {gid for gid, r in genomes.items() if is_human(r.get("host_name"))}
    groups = defaultdict(list)
    for r in amr:
        if r["genome_id"] in human:
            groups[(r["genome_id"], r["antibiotic"])].append(r)
    kept, excluded = [], Counter()
    for (gid, drug), rows in groups.items():
        labels = {r.get("resistant_phenotype", "").strip().lower() for r in rows}
        if labels - {"resistant", "susceptible"}:
            excluded["non_SR_or_missing"] += 1; continue
        if len(labels) != 1:
            excluded["phenotype_conflict"] += 1; continue
        g = genomes[gid]
        standards = sorted({r.get("testing_standard", "").strip() for r in rows if r.get("testing_standard", "").strip()})
        methods = sorted({r.get("laboratory_typing_method", "").strip() for r in rows if r.get("laboratory_typing_method", "").strip()})
        kept.append({"genome_id": gid, "biosample_accession": g.get("biosample_accession", ""), "assembly_accession": g.get("assembly_accession", ""),
                     "sequence_id": g.get("assembly_accession", "") or 'BV_'+gid,
                     "bioproject_accession": g.get("bioproject_accession", ""), "collection_date": g.get("collection_date", ""),
                     "host_name": g.get("host_name", ""), "antibiotic": drug,
                     "label": next(iter(labels)).title(), "standards": ";".join(standards), "methods": ";".join(methods), "n_source_records": len(rows)})
    fields = list(kept[0]) if kept else []
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(kept)
    report = {"status": "complete", "human_genomes": len(human), "human_genome_drug_groups": len(groups), "curated_sr_pairs": len(kept),
              "excluded_groups": dict(excluded), "drug_counts": Counter(r["antibiotic"] for r in kept),
              "label_counts": Counter(r["label"] for r in kept), "with_assembly": sum(bool(r["assembly_accession"]) for r in kept),
              "standard_counts": Counter(r["standards"] or "__MISSING__" for r in kept), "method_counts": Counter(r["methods"] or "__MISSING__" for r in kept),
              "limitations": ["This is a conservative public-data curation; it is not a prospective or blinded cohort.", "Rows with phenotype conflicts or non-S/R labels are excluded rather than silently reconciled.", "Genome IDs are not necessarily independent isolates; BioSample/assembly conflicts and relatedness must be handled in each model split.", "Multiple source-reported standards are preserved; labels have not been breakpoint-harmonized."]}
    # Convert Counter objects for JSON.
    for key in ("drug_counts", "label_counts", "standard_counts", "method_counts"): report[key] = dict(report[key])
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "human_genomes": report["human_genomes"], "curated_sr_pairs": report["curated_sr_pairs"], "excluded": report["excluded_groups"]}, ensure_ascii=False))

if __name__ == "__main__": main()
