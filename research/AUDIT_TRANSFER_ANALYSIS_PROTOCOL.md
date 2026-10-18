# Audit-to-evaluation transfer: descriptive follow-up

Frozen on 2026-10-10 after observing repair dynamics. This is post-result exploratory analysis, not preregistration or new training. Original models, probabilities, roles, queries and thresholds remain unchanged.

Use all independently validated adjacent transitions in repair_dynamics_full_v1 and repair_dynamics_ani_v1, including every drug, input, partition threshold and strategy. Keep datasets separate; partitions reuse cohorts and are not independent observations.

Fixed audit-resistant bins: 0, 1-4, 5-19, >=20. Empty audit sets are a separate row. These bins are descriptive, chosen now and never used to select an algorithm.

Two local acceptance predicates: strictly lower audit Brier; strictly lower audit Brier and nonincreasing audit false-susceptible count, requiring at least one audit-resistant record. They compare one adjacent update to its preceding checkpoint. They are not the sequential stopping interventions already reported, and do not use evaluation labels to accept an update.

For accepted updates report evaluation Brier increase/decrease/equality, any new false-susceptible call, net false-susceptible increase, and lower Brier with net false-susceptible increase. Count evaluation-R=0 separately. A new error and a net increase are different endpoints.

Describe both pooled configuration counts and the full drug/threshold/input/strategy strata, retaining missing and undefined cases. No p-values, confidence intervals, causal sample-size effect, significance or clinical guarantee follow from these counts. Audit size, class prevalence, component structure and domain shift can be jointly confounded.

Use the same budgets, labels and hashes as the existing dynamics artifacts. An independent checker recomputes bins and acceptance from the validated transition records and verifies every aggregate and the source result hashes. This adds a descriptive analysis, not a new experimental cohort.
