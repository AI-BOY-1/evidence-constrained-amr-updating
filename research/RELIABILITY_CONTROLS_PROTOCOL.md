# Reliability controls, frozen before the new runs

These are exploratory follow-up experiments after inspection of old results,
not preregistration or label-blind confirmation. Preserve all original artifacts.

## Frozen population and model

Use all nine original drugs, both sequence-sketch and acquired-homolog inputs,
ANI-screen edges at 0.99 and 0.999, query budgets 0/10/25/50, and seeds
7/17/27/37/47. Reuse the existing source-only fitted experts/controller and
random-gene strategy. Mask query, audit and evaluation labels at the trajectory
interface. Audit label cost is charged even though those labels are not used to
select or train a model. All assay labels and probabilities are retained.

Primary descriptive displays use ANI 0.99, budget 25. Both thresholds, all
budgets and seeds are reported; no hypothesis-dependent configuration selection.

## Three synthetic negative controls

Use the exact four-way ANI partitions of the real-feature experiment.
Permute only source and query-pool records, separately within each role.
Audit and evaluation labels/features stay untouched.

1. `label_global`: permute labels among all records within each training role;
   preserve role sample count and resistant count, destroy the genome-label link.
2. `label_within_component`: permute labels within each ANI component and role;
   preserve component-specific prevalence but disturb within-component label
   allocation where heterogeneous labels exist. This is a lineage-preserving
   diagnostic, not a global-null significance test. Record zero-effect cases.
3. `features_within_context`: permute whole genomic feature vectors within
   role, reported-standard category and collection-period category. Preserve
   the context/label relationship and feature multiset while disrupting the
   original feature-label pairing where a stratum permits it. Keep donor IDs,
   moved-vector count and identity permutations. This does not prove causality.

Permutation RNG is SHA256 of condition, drug, ANI threshold, seed and feature;
the first 16 hex digits seed NumPy. One permutation per fixed seed, not a large
independent null distribution and not a valid permutation p-value. Null models
receive synthetic labels where applicable; never relabel the real results.
Two strategies per condition, with their own adaptive query paths; do not
interpret raw adaptive-path contrasts as equal queried-sample contrasts.

## Temporal stress tests

Fixed windows: source <=2012, query 2013-2014, held-out >=2015; and source
<=2014, query 2015-2017, held-out >=2018. Only unambiguous, valid collection
dates are eligible. For any ANI component spanning more than one window,
exclude that entire component from this stress test; do not shift boundaries
to increase performance. Within held-out components, SHA256(seed:component)
modulo 10 assigns audit (0-2) or evaluation (3-9). This test is temporal
evaluation on reused public labels, not prospective confirmation.

## Reported-standard transport

Both directions EUCAST-to-CLSI and CLSI-to-EUCAST. Source has only the named
source standard. Components with source and target standard records are
excluded completely. Target components are hash assigned query (0-3), audit
(4-5), evaluation (6-9). Mixed/unknown reported standards are excluded with
counts. Source standard is a recorded context, not a verified laboratory
breakpoint revision. Prevalence, geography, lineages and year can confound this
comparison: it cannot isolate causal assay effects.

## Skips, metrics and validation

Keep insufficient-source-classes, fewer-than-10 query/evaluation records,
no-source-features and giant-component exclusion cases. Empty audit groups
are allowed but recorded. Only budgets <= pool size are used. Report resistant
counts and undefined AUROC/AUPRC or class-conditional error as null. Report
Brier, class-conditional loss, false-susceptible and false-resistant counts,
mean predicted risk minus prevalence (calibration-in-the-large), and fixed
ten-bin ECE with denominators. ECE is descriptive, not a safety guarantee.

Independently validate hashes, configuration coverage including skips, source
labels from raw AST, sample/component isolation, permutation donor bijections
and conserved multisets, date/standard eligibility, query label provenance,
query-weight replay, budget prefixes, predictions and metric/cost recomputation.
Independent validator checks measured invariants; it does not establish SNP
phylogeny, clinical validity or a complete retraining replay.
