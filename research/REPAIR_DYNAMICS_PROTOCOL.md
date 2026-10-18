# Repair dynamics and audit-gated checkpoint replay, 2026-10-10

This is a new exploratory analysis after inspecting older aggregate results.
The original models, query paths, labels, thresholds and live pipelines remain
unchanged. The protocol is frozen before inspecting the new quantities below.
It does not claim preregistration, clinical safety or a new trained model.

## Population and coverage

Include every validated E. coli trajectory: nine drugs, four strategies, five
seeds, both original sketch partition thresholds, budgets 0/10/25/50. A separate
ANI version uses both ANI thresholds and both feature inputs only after its
independent validator passes. All missing budgets and skipped configurations
are retained. Do not compare old and ANI splits as matched cohorts.

## Three questions

1. Does increasing query budget replace old errors with new ones? Measure all
   six ordered forward budget pairs per trajectory. Record corrected and newly
   introduced classification errors, separately false susceptible (FS) and
   false resistant (FR), and per-sample squared-loss increases/decreases.
   Adjacent pairs assess local nonmonotonicity; all pairs test error identities.
   Classification remains probability >=0.5 for resistance.
2. Do different strategies leave the same resistant genomes misclassified?
   On identical evaluation IDs/labels/components, record how many of four
   strategies miss each resistant genome, at every budget. Store all sample IDs
   and denominators. The four-strategy union is a label-informed descriptive
   ceiling, not a deployable model or evidence of irreducible biological error.
3. Can a separate audit set reject a harmful update? Replay two deterministic
   stopping rules on already generated nested query prefixes. Start at budget0,
   then attempt10,25,50 in order. Accept only strictly lower audit Brier than the
   current accepted checkpoint. The directional rule additionally requires no
   increase in audit FS count. At the first rejection, stop and output the last
   accepted checkpoint. Charge the full attempted budget, including rejected
   measurements, plus every audit label. Compare to the same strategy using its
   checkpoint at that same consumed budget. Both classes' errors and loss are
   reported. Empty audit sets are infeasible; the directional rule is infeasible
   without any audit resistant records. No minimum-R cutoff is selected later.

The replay uses no evaluation outcomes to make stopping choices, and cannot
change the query path before stopping. It evaluates checkpoint acceptance,
not an independently trained diagnosis algorithm. Repeatedly reusing audit
labels may overfit this finite audit set; evaluation remains separate. Tiny
audit/evaluation R denominators are explicit.

## Reporting and reproducibility

All strata, drug/strategy/seed configurations and both gates are retained.
Display original threshold0.99 and budget25 for continuity; show all budgets
in exported tables. Summaries are descriptive means over shared-cohort
partitions, not confidence intervals or independent patient events. No p-values,
significance stars, post-hoc threshold optimization or preferred-drug selection.
Input NPZ hashes must match the prior independent validator. Record original
sidecar/protocol/code/validator hashes and verify isolation, probabilities,
transition conservation, paired identity and exact gate cost independently.
Keep output versions separate. The ANI replication waits for real validated
artifacts; planned output must never be reported as completed evidence.
