# Blind external validation protocol

This protocol defines the evidence needed to close the new-cohort gap. The current public BVBRC and NCBI labels are retrospective and must not be called blind confirmation.

## Freeze before labels are revealed

1. Lock the exact code commit, feature definition, preprocessing, drug list, probability threshold, query policy, budget, abstention rule and directional guard.
2. Record SHA-256 hashes of the source snapshot, model artifact and protocol in a timestamped manifest.
3. The data holder supplies genome FASTA or raw reads plus a one-way sample identifier. The holder keeps AST/MIC labels encrypted or withheld until the prediction manifest is returned.
4. The cohort must contain patient or laboratory source identifiers, collection date, testing method, standard and revision, infection site, host and species confirmation. Identifiers are used only to construct prespecified disjointness and temporal strata.

## Reveal and report

The holder reveals labels only after the prediction hash is acknowledged. Report every eligible sample, every drug, missingness, exclusions and all prespecified strata. Primary metrics are Brier, AUROC/AUPRC where defined, false-susceptible rate, calibration, selective risk and coverage. Report paired bootstrap intervals by patient/lab source, and disclose every failed stratum.

## Acceptance criteria

Acceptance is a scientific decision after seeing the complete report, not a tuning step. The protocol does not guarantee NCS/Nature acceptance, stable SOTA or clinical safety. A failure is retained as a result.

## Required external input

The present repository has no label-blind holder-provided cohort, so this protocol remains ready but unexecuted.
