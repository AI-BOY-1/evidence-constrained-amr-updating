# Running the full study-script copies

These scripts require separately provisioned data. They are not launched by the
builder, demo or test suite. The complete historical experiments are NOT rerun
by this release's smoke checks. The source calculations, thresholds and fixed
budgets remain as reported; configurable I/O changes create a new code binding.

## Configuration

Install `requirements-study.txt` in a separate environment for the genomic/photo
work. For the optional published MALDI B-subset reexecution use a separate
environment with `requirements-maldi.txt`, externally obtained author weights
and dataset assets. Dependencies are the versions observed for package checks;
historical manifests may report different environments. The historical
`requirements_reported_2026_10_08.txt` is a record, not the recommended portable
installation file.

Set `NCS_DATA_ROOT` to a NEW reproducibility data/output root:

```powershell
$env:NCS_DATA_ROOT = 'E:\amr_reproduction'
```

```bash
export NCS_DATA_ROOT=/scratch/amr_reproduction
```

It defaults to `<repository>/_runtime`. Do not point it at the old live
experiment root unless doing a read-only audit and understanding existing hash
bindings. New code copies must not overwrite completed historical results.

## Required preprocessed assets

```text
<NCS_DATA_ROOT>/
  experiments/ecoli_bvbrc/
    fasta/*.fna                         # final declared inventory
    ecoli_human_sr_pairs.tsv            # curated singleton recorded S/R labels
    ecoli_fasta_integrity.json
    ecoli_assemblies_k31_scaled1000.sig  # frozen k31 inventory
    ecoli_assemblies_jaccard.csv         # required only by original sketch branch
    acquired_amr_v1/manifest.json
    acquired_amr_v1/<sequence_id>.json  # candidates from pinned BLAST HSPs
  experiments/combiant_public_v1/       # independently licensed public assets
  experiments/msdeepamr_reproduction/
    repo/                              # pinned author's repository + datasets
    ...                                # author H5 weights and locked assets
  reference/resfinder_db/genomicepidemiology-resfinder_db-eecf0aa20759/
    all.fsa, VERSION
  tools/blast/ncbi-blast-2.17.0+/bin/
    blastn.exe, makeblastdb.exe         # Windows study setup
```

Research-side source assets such as `ecoli_full_skani_screen.json`, image pairing
manifests and `msdeepamr_assets.json` must also be provisioned to `research/`.
Absolute paths embedded inside previously generated JSON/NPZ manifests do not
become portable automatically. Regenerate relevant manifests through the new
versioned pipeline; do not edit a historical `passed` status or its source hash.
The sketch inventory must follow the frozen k31/scaled1000 scheme and its model
downsampling to scaled5000. Changing sequence inventory or labels defines a new
experiment and its expected configuration counts must be updated separately.

## Main executed stages

For original sketch experiments run every fixed drug separately:

```bash
python research/run_ecoli_repair.py --full --drug meropenem
# repeat for all nine fixed drugs, preserving each result and skip
python research/analyze_ecoli_full.py
python research/run_ecoli_frozen_query_ablation.py
python research/analyze_ecoli_frozen_query_ablation.py
```

For the later ANI/input branch, start from declared skani screen and annotations:

```bash
python research/run_ecoli_ani_annotation_repair.py
python research/validate_ecoli_ani_annotation_repair.py
python research/analyze_ecoli_ani_annotation_repair.py
python research/run_repair_dynamics.py --dataset ani
python research/validate_repair_dynamics.py --dataset ani
python research/analyze_repair_dynamics.py --dataset ani
python research/run_audit_acquisition_action.py
python research/validate_audit_acquisition_action.py
python research/analyze_audit_acquisition_action.py
python research/validate_audit_acquisition_analysis.py
python research/run_reliability_controls.py
python research/validate_reliability_controls.py
python research/analyze_reliability_controls.py
```

Scripts check source/input binding and refuse mixed existing versions. Run the
independent validators before interpreting any output. Graphical validators
check exported source/hash consistency; actual final PNG/PDF layout still needs
human review. Do not copy a previous visual-review `passed` to a new image hash.

Sequence portrait/photo/spectral scripts require their actual source objects.
They must not be replaced with generated or synthetic images. CombiANT and
MALDI are unpaired secondary tasks; their methods do not establish a unified
multimodal controller. The preprocessing tools for BLAST retain the original
Windows `.exe` layout; Linux users must adapt the external tool path explicitly
and bind that code change to a new result version. No remote third-party source
code or model weights are redistributed here.
