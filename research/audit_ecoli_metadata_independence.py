"""Audit isolate/source metadata for the frozen 3,516-genome E. coli cohort.

This is a provenance audit. BioSample/BioProject/assembly fields are useful
duplicate-source indicators, but they do not establish patient independence or
replace a SNP tree.
"""
from __future__ import annotations
import ncs_paths  # portable I/O paths; original computational rules preserved

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = ncs_paths.data_path('experiments/ecoli_bvbrc')
RESEARCH = Path(__file__).resolve().parent
FASTA_AUDIT = ROOT / "ecoli_fasta_integrity.json"
GENOME_METADATA = ROOT / "ecoli_genome_metadata.tsv"
LAB_AMR = ROOT / "ecoli_lab_amr.tsv"
OUT_JSON = ROOT / "ecoli_metadata_independence_audit.json"
OUT_MD = RESEARCH / "ECOLI_METADATA_INDEPENDENCE_AUDIT.md"


def norm(value: str | None) -> str:
    value = (value or "").strip()
    return value if value else "__MISSING__"


def duplicate_groups(rows: list[dict[str, str]], field: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = norm(row.get(field))
        if value != "__MISSING__":
            groups[value].append(row["genome_id"])
    return {key: sorted(set(values)) for key, values in groups.items() if len(set(values)) > 1}


def main() -> None:
    integrity = json.loads(FASTA_AUDIT.read_text(encoding="utf-8"))
    selected_records = [
        item for item in integrity["inventory"] if item.get("status") == "valid"
    ]
    selected = [
        item.get("provenance", {}).get("genome_id")
        or item.get("provenance", {}).get("accession")
        or item["sequence_id"]
        for item in selected_records
    ]
    selected_set = set(selected)
    if len(selected_set) != 3516:
        raise RuntimeError(f"expected 3516 unique valid genomes, found {len(selected_set)}")

    with GENOME_METADATA.open(newline="", encoding="utf-8") as handle:
        all_metadata = list(csv.DictReader(handle, delimiter="\t"))
    metadata = [
        row for row in all_metadata
        if row.get("genome_id") in selected_set or row.get("assembly_accession") in selected_set
    ]
    metadata_ids = {row.get("genome_id") for row in metadata} | {row.get("assembly_accession") for row in metadata}
    metadata_ids.discard(None)
    # A selected FASTA can match a metadata row through either BV genome_id or
    # assembly_accession. Count unique selected genomes separately from raw
    # matched rows because legacy snapshots can contain both representations.
    selected_to_metadata: dict[str, int] = Counter()
    for row in metadata:
        matched = set()
        if row.get("genome_id") in selected_set:
            matched.add(row["genome_id"])
        if row.get("assembly_accession") in selected_set:
            matched.add(row["assembly_accession"])
        for key in matched:
            selected_to_metadata[key] += 1

    lab_counts: Counter[str] = Counter()
    lab_drugs: dict[str, set[str]] = defaultdict(set)
    lab_standards: dict[str, set[str]] = defaultdict(set)
    with LAB_AMR.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            genome_id = row.get("genome_id", "")
            if genome_id in selected_set:
                lab_counts[genome_id] += 1
                if row.get("antibiotic"):
                    lab_drugs[genome_id].add(row["antibiotic"])
                if row.get("testing_standard"):
                    lab_standards[genome_id].add(row["testing_standard"])

    metadata_for_groups = [
        dict(row, genome_id=row.get("genome_id") or row.get("assembly_accession") or "__MISSING__")
        for row in metadata
    ]
    duplicate_fields = {
        field: duplicate_groups(metadata_for_groups, field)
        for field in ("biosample_accession", "assembly_accession")
    }
    project_groups: dict[str, list[str]] = defaultdict(list)
    for row in metadata_for_groups:
        project = norm(row.get("bioproject_accession"))
        if project != "__MISSING__":
            project_groups[project].append(row["genome_id"])
    project_sizes = Counter(norm(row.get("bioproject_accession")) for row in metadata_for_groups)
    host_sizes = Counter(norm(row.get("host_name")) for row in metadata_for_groups)
    years = Counter(norm(row.get("collection_date"))[:4] for row in metadata_for_groups)
    lab_multi_condition = sum(1 for count in lab_counts.values() if count > 1)

    result = {
        "status": "complete",
        "scope": "frozen 3516 valid FASTA genomes and their locally archived BV-BRC metadata",
        "selected_valid_genomes": len(selected_set),
        "metadata_rows_found": len(metadata),
        "metadata_unique_selected_genomes": len(selected_to_metadata),
        "metadata_duplicate_mapping_rows": len(metadata) - len(selected_to_metadata),
        "metadata_rows_per_selected_genome_max": max(selected_to_metadata.values(), default=0),
        "metadata_rows_missing": len(selected_set - metadata_ids),
        "bv_genome_id_records": sum(bool(item.get("provenance", {}).get("genome_id")) for item in selected_records),
        "assembly_accession_records": sum(not bool(item.get("provenance", {}).get("genome_id")) for item in selected_records),
        "genomes_with_lab_ast_rows": len(lab_counts),
        "genomes_with_multiple_lab_ast_rows": lab_multi_condition,
        "duplicate_biosample_groups_across_genomes": len(duplicate_fields["biosample_accession"]),
        "duplicate_assembly_groups_across_genomes": len(duplicate_fields["assembly_accession"]),
        "bioproject_count": len([key for key in project_sizes if key != "__MISSING__"]),
        "genomes_with_missing_biosample": sum(norm(row.get("biosample_accession")) == "__MISSING__" for row in metadata_for_groups),
        "genomes_with_missing_bioproject": sum(norm(row.get("bioproject_accession")) == "__MISSING__" for row in metadata_for_groups),
        "genomes_with_missing_collection_date": sum(norm(row.get("collection_date")) == "__MISSING__" for row in metadata_for_groups),
        "genomes_with_missing_host": sum(norm(row.get("host_name")) == "__MISSING__" for row in metadata_for_groups),
        "largest_bioproject_genome_count": max(project_sizes.values(), default=0),
        "largest_host_genome_count": max(host_sizes.values(), default=0),
        "duplicate_biosample_groups": duplicate_fields["biosample_accession"],
        "duplicate_assembly_groups": duplicate_fields["assembly_accession"],
        "top_bioprojects": project_sizes.most_common(20),
        "collection_year_counts": sorted(years.items()),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (FASTA_AUDIT, GENOME_METADATA, LAB_AMR)
        },
        "limitations": [
            "BioSample, BioProject, assembly and host fields are provenance proxies, not patient identifiers.",
            "No patient_group_id or laboratory_id is present in the archived metadata, so patient/laboratory independence cannot be established.",
            "This audit does not calculate SNP distances and does not turn ANI into a phylogeny.",
            "Repeated AST rows for one genome are assay records, not independent isolates.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# E. coli 元数据独立性审计",
        "",
        "本审计针对冻结的 3,516 条有效 FASTA，使用已归档的 BV-BRC genome metadata 和 laboratory AMR 表。它量化来源重复和一株多条药敏记录，不把代理字段当作患者独立或 SNP 证据。",
        "",
        f"- 有效基因组：{result['selected_valid_genomes']:,}；元数据命中：{result['metadata_rows_found']:,}；缺失：{result['metadata_rows_missing']:,}",
        f"- 唯一冻结基因组映射：{result['metadata_unique_selected_genomes']:,}；重复映射行：{result['metadata_duplicate_mapping_rows']:,}；单个基因组最多匹配行：{result['metadata_rows_per_selected_genome_max']:,}",
        f"- 跨基因组重复 BioSample 组：{result['duplicate_biosample_groups_across_genomes']:,}；重复 assembly 组：{result['duplicate_assembly_groups_across_genomes']:,}",
        f"- BioProject 数：{result['bioproject_count']:,}；有药敏记录基因组：{result['genomes_with_lab_ast_rows']:,}；一株多药敏记录：{result['genomes_with_multiple_lab_ast_rows']:,}",
        f"- 缺失 BioSample：{result['genomes_with_missing_biosample']:,}；缺失 BioProject：{result['genomes_with_missing_bioproject']:,}；缺失采样日期：{result['genomes_with_missing_collection_date']:,}；缺失宿主：{result['genomes_with_missing_host']:,}",
        "",
        "## 解释边界",
        "",
        "BioSample、BioProject、assembly 和宿主字段可发现重复来源，但不是患者伪名或实验室ID。当前归档表没有 `patient_group_id` 或 `laboratory_id`，因此不能声称患者/实验室独立；本审计也没有把 ANI 转成 SNP 树。重复药敏行按同一 genome 归并为记录，不计作独立菌株。",
        "",
        f"机器可读结果：`{OUT_JSON}`。输入哈希已写入结果文件。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_valid_genomes": result["selected_valid_genomes"],
        "metadata_rows_found": result["metadata_rows_found"],
        "duplicate_biosample_groups": result["duplicate_biosample_groups_across_genomes"],
        "duplicate_assembly_groups": result["duplicate_assembly_groups_across_genomes"],
        "bioproject_count": result["bioproject_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
