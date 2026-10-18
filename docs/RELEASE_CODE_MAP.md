# Release code and formula map

The manuscript crosswalk in docs/CODE_FORMULA_CROSSWALK.md refers to historical snapshot line numbers. In this release the exact computational bodies are extracted into src/ncs_amr. The research copies have new path-binding hashes. This map gives current release locations.

| Methods blocks | Portable implementation |
|---|---|
| E01 data/interface | src/ncs_amr/records.py |
| E04–E06 logistic experts/offset | src/ncs_amr/models.py |
| E07–E12 weights, action, query | src/ncs_amr/controller.py |
| E13, E16–E18 metrics/error flows | src/ncs_amr/metrics.py |
| E19–E21 acquisition and audit rules | src/ncs_amr/acquisition.py |
| E02–E03 ANI role/graph rules | research/run_ecoli_ani_annotation_repair.py |
| E14–E15 frozen-query decomposition | research/run_ecoli_frozen_query_ablation.py; research/analyze_ecoli_frozen_query_ablation.py |
| E22 costs/offline oracle | research/run_audit_acquisition_action.py |
| E23 sequential gate | research/run_repair_dynamics.py |
| E24 conditional explanatory derivation | docs/METHODS_MATH_DERIVATIONS_EN.md; no production guarantee |

## __init__.py

SHA256: `1b283e2c3d2c415af642576da5ae4a5cd5f26764660c9fd7640526adb1b2463c`

| Definition | Line |
|---|---|

## acquisition.py

SHA256: `2b9e19236bf7cb7110ba0d2abded9ef3ae69182c80515d75044afdab9529cc5f`

| Definition | Line |
|---|---|
| acquisition_order | 6 |
| audit_evidence | 16 |
| decide_action | 23 |

## controller.py

SHA256: `e03929434718c2a73007839c71d8726cd0906492e45a4497ae7ef6b0c9fdb30c`

| Definition | Line |
|---|---|
| action_matrix | 11 |
| update_weights | 14 |
| query_scores | 21 |
| diverse_indices | 39 |
| tree_predict | 55 |
| trajectory | 65 |
| run_trajectory | 118 |

## metrics.py

SHA256: `53724f5d2de14d5c72d188056d53b82c46322a086e48d50cd33b26ef56a7ce50`

| Definition | Line |
|---|---|
| binary_metrics | 3 |
| transition_metrics | 13 |

## models.py

SHA256: `fd3dd1ca641a2e6e03d2a692ea737b87b54669370251d59aba3d371ea5869aa1`

| Definition | Line |
|---|---|
| compact_sparse | 9 |
| feature_dict | 18 |
| ProbabilityModel | 30 |
| entropy_score | 58 |
| ExpertBank | 63 |

## records.py

SHA256: `af7f0d2bb039508edbc266a38e3b5eba8780e465bbcba057a8ef274f9cc1f52c`

| Definition | Line |
|---|---|
| Record | 6 |
| validate_roles | 35 |
