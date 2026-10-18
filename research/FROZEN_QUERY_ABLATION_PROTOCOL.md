# Frozen-query repair decomposition (exploratory)

Registered locally before this follow-up computation on 2026-10-09. The full
E. coli results have already been seen. This is a post-result diagnostic
ablation, not preregistered confirmation or an independent test set.

## Question and fixed coverage

On every completed E. coli configuration (nine drugs, Jaccard component
thresholds 0.99 and 0.95, seeds 7/17/27/37/47), reuse the saved diagnostic
and random query paths. Fit the existing sequence-token logistic model on
source plus each prefix of 0/10/25/50 queried labels. Do not select new queries,
change any original controller, tune hyperparameters, thresholds or splits.

The model is the unchanged ProbabilityModel('gene', seed), C=1, liblinear,
max_iter=1500. Its inputs are k31 scaled5000 sequence-sketch hash presence,
not AMR annotations. Source/evaluation sample order and component IDs come
from the validated original artifacts. Raw singleton AST labels must match
event reveals and saved evaluation labels. Context fields are neutral because
this model uses sequence features alone. Evaluation and audit labels are -1
when passed to prediction. Audit labels are never used for fitting.

Match the formal pipeline's OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS=1 before importing numerical libraries, and record actual
threadpool counts. The first reconstruction attempt lacked these settings
and failed the fixed replay tolerance (maximum difference 1.077914e-6 at
meropenem/0.99/seed7/budget0). A single-thread diagnostic reproduced that
prediction with error 0. Preserve the initial partial attempt and its code in
frozen_query_initial_replay_failure_20261009; it is not part of this analysis.
Recompute all new ablation predictions under the matching numerical settings.
This correction does not change models, labels, splits or the 1e-10 tolerance.

## Controls and estimands

Rebuild random_gene at every budget and require probability agreement with
the original saved evaluation and audit predictions within 1e-10. This is
an implementation replay check, not additional independent evidence.

Let D denote the original diagnostic action, G the logistic model retrained
on the diagnostic query prefix, and R the original random_gene prediction.
For Brier loss on identical evaluation samples:

- fixed-path query contrast = Brier(G) - Brier(R);
- fixed-path action contrast = Brier(D) - Brier(G);
- total contrast = Brier(D) - Brier(R), equal to their sum.

Negative means lower loss. This decomposition is descriptive and conditional
on the saved adaptive paths; it is not a causal mediation analysis or the
performance of a newly deployed gene-only querying policy.

Report all 360 configuration/budget contrasts, class-specific Brier losses,
false-susceptible and false-resistant counts/rates at the fixed 0.5 cutoff,
classification fixes/harms and resistant false-susceptible transitions versus
the original random_gene budget-zero prediction. Undefined class rates are
null, never zero. Assay cost retains query budget plus original audit size.

Summaries average the five seeds equally per drug/threshold/budget. Both
component thresholds and all budgets are retained. Seed splits reuse samples;
no seed-level significance tests or independent-replicate confidence claim.
No threshold/budget/drug may be selected to maximize the reported benefit.

## Integrity and outputs

Keep the completed formal batch untouched. Separate artifacts are stored in
D:/NCS_data/experiments/ecoli_bvbrc/frozen_query_ablation_v1. Bind them to the
passed original validation, original trace/prediction/input/method hashes,
this protocol and follow-up runner hash. Refuse stale resume artifacts.
Validate metric recomputation, decomposition identity, class weighting,
four-role sample/component isolation, query membership/uniqueness/reveals,
all 90 configurations and 720 fitted-model metric rows before reporting.

Figures show the full fixed-path query/action contrasts and resistant error
transitions, using actual saved predictions. Record source/export hashes and
inspect layout manually. Do not infer mechanisms or clinical safety from
these figures.

Remaining scientific gaps include MIC/breakpoint harmonization, true ANI/SNP
and patient/laboratory independence, AMR annotation, stronger AMR-specific
method replication, and genuinely new blinded external confirmation.
