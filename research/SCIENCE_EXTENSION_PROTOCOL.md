# Science extension protocol, 2026-10-09

These are post-result exploratory extensions. Old sketch results, labels, thresholds, code and prediction hashes remain immutable. No new analysis is called blind confirmation.

## Sequence independence

The first full-database query at approximate skani cutoff 0.80 was stopped before writing per-query results because of runtime. The same 3,516-genome inventory was restarted with a 0.99 approximate screening cutoff. This is a computational screening change made for runtime, not a selected performance result. Query every genome, retain identity >=0.99 with query/reference fractions >=0.80. Query screening can have false negatives and ANI does not replace SNP phylogeny.

After full screening, create conservative connected components from every retained direction, shared BioSample and exact sequence group. Primary ANI edge threshold 0.99, sensitivity 0.999; both require >=0.80 query/reference coverage. The latter is a stronger identity rule, not a promise of clonality. Use the same fixed seeds 7/17/27/37/47 and hash roles source/query/audit/evaluation. Do not choose component thresholds or skip reasons by performance. If giant components make some experiments infeasible, retain all skip reasons and denominators.

## Acquired AMR annotation

The reference is ResFinder database 2.6.0, snapshot eecf0aa20759. BLASTn 2.17.0+ homology calls use identity >=90%, one-HSP reference span >=80%, E-value <=1e-10; all qualifying alleles remain in raw outputs. Predictive tokens are the sorted reference-name prefixes before the first underscore, prefixed with `homolog::`. They represent acquired-gene homology candidates, not resolved allele identities, mutation calls, expression or causal mechanisms.

Run the original four controller/baseline strategies independently on the new ANI partitions and this new feature version; source/query/audit/evaluation labels follow the original sealed-label interface. Budgets are 0/10/25/50 with the same seeds. Report every drug, both partition rules, all strategies and feasible budgets. Compare sketch and homolog inputs on exactly the same new partitions; this is a feature sensitivity experiment, not cross-species model transfer or a published ResFinder pipeline reproduction.

## Directional guard

For any frozen baseline prediction >=0.5, a repaired probability below 0.5 is raised to 0.5; all other probabilities are retained. This relative direction property holds without labels but cannot remove baseline errors or establish clinical safety. Preserve Brier and false-resistant trade-offs for all drugs and budgets.

## MIC sensitivity and new-cohort evidence

Keep original AST labels. Official year-table EUCAST MIC interpretation is a sensitivity analysis with explicit censored intervals and unresolved context. Exact revision remains unknown in many records. CLSI tables require a legitimate versioned source. New public cohorts are used for retrospective external pressure tests only after accession/sequence overlap audit. A holder-blind confirmation is executed only if the external data holder withholds labels until the frozen prediction manifest is returned.
