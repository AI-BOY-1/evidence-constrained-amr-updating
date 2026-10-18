# Formula–code–artifact crosswalk

This map is for manuscript audit. Line numbers refer to the immutable source snapshots shipped in `methods/source_code/`; a future code edit requires a new hash binding and a new result replay.

| Formula / property | Executed function and source | Saved artifact / validation | What the manuscript may claim |
|---|---|---|---|
| E01 record, labels, feature views | `counterexample_guided_repair.py`: `Record`, `feature_dict`, `load_records` | `methods/source_code/counterexample_guided_repair.py`; ANI `results.json`; `ecoli_ani_annotation_validation.json` | Exact data fields and label interface; AST context is recorded metadata. |
| E02 hash role assignment | `run_ecoli_ani_annotation_repair.py`: local `role` lambda in `main` | `ani_annotation_repair_v1/binding.json`; per-trajectory `source/query_pool/audit/evaluation_clusters` | Observed component role isolation under the inventory; not patient/lab independence. |
| E03 approximate component union | `run_ecoli_ani_annotation_repair.py`: `components` | `screen_sha256` and ANI validation | Approximate ANI-like grouping with stated either-direction rule; not SNP phylogeny. |
| E04–E05 logistic predictor | `counterexample_guided_repair.py`: `ProbabilityModel.__init__`, `fit`, `predict` | `methods/source_code/counterexample_guided_repair.py`; frozen replay validation | Sparse logistic model and fixed solver settings. Do not claim an unpenalized intercept. |
| E06 intercept expert | `decision_repair_core.py`: `ExpertBank.update` | `.json` trajectory `evidence_weights` and events; ANI result hashes | Penalized scalar calibration-like rule fitted only to queried labels. |
| E07 likelihood weights | `decision_repair_core.py`: `update_weights` | trajectory events `pre_reveal_predictions`, `revealed_labels`, `post_reveal_evidence_weights`; replay validators | Predictive-rule evidence update with clipping/protected weights, not biological posterior. |
| E08–E09 action rule | `decision_repair_core.py`: `action_matrix`; `run_decision_repair.py`: `trajectory` action selection | trajectory rows `action`; dynamics validation | Chosen expert is closest to the weighted predictive mixture on the remaining pool. |
| E10–E11 one-step query value | `decision_repair_core.py`: `query_scores` (`objective='decision'`) | query events and trajectory JSON; numerical formula check | Nonnegative fixed-surrogate disagreement value; not target Brier/FS/clinical-risk value. |
| E12 information identity | `decision_repair_core.py`: `query_scores` (`objective='information'`) | Optional strategy code path; synthetic formula check only | Standard entropy identity; not the primary four-strategy result. |
| E13 metrics | `run_repair_dynamics.py`: `measure`; `run_audit_acquisition_action.py`: `score` | dynamics/acquisition results and validation JSON | Exact Brier, FS, FR with explicit denominators and unavailable R=0. |
| E14–E15 query/action decomposition | `run_ecoli_frozen_query_ablation.py`: saved-path refits and deltas; `analyze_ecoli_frozen_query_ablation.py`: independent replay and identity checks | `ecoli_frozen_query_analysis.json`; ablation validation | Exact matched algebraic decomposition; not causal mediation. |
| E16–E18 error flow and Brier change | `run_repair_dynamics.py`: `transition`; acquisition `change` | repair dynamics results, source JSON, validation | Counts of corrected/introduced errors and probability-loss change on same records. |
| E19 acquisition invariance/nesting | `run_audit_acquisition_action.py`: `acquisition_order` | per-condition `audit_samples`, source data; acquisition validation | Acquisition ordering uses IDs/components/p0 without purchased labels; original cohort exclusions remain label-aware. |
| E20 acceptance rules | `run_audit_acquisition_action.py`: `decide` | action rows `decision_status`, `accepted`; validator | Strict fixed rules, R=0 direction fallback, no evaluation selection. |
| E21 component-direction aggregate | `run_audit_acquisition_action.py`: `audit_evidence` | `audit_evidence` fields in condition JSON | Equal-component aggregate criterion, not every-component guarantee. |
| E22 cost and oracle comparator | `run_audit_acquisition_action.py`: `replay` | action rows `total_new_label_cost`, `oracle_brier_regret` | All consumed labels charged; oracle is offline diagnostic only. |
| E23 sequential gate | `run_repair_dynamics.py`: `audit_gate` | `gates` and dynamics summaries | Earlier gate replay, separate denominators from acquisition study. |
| E24 transfer bound | Analytic explanatory derivation only; no production function | No empirical bound artifact | Conditional mathematical illustration; no observed transport guarantee. |
| P1–P8 | Derivations in `METHODS_MATH_DERIVATIONS_EN.md`; synthetic check `verify_manuscript_math_2026_10_12.py` | `METHODS_MATH_IMPLEMENTATION_CHECK.json` | Standard identities/properties tied to implementation; not new empirical evidence or safety proof. |

## Code and result identity rules

The five core ANI source files are bound in `methods/source_code/ANI_SOURCE_BINDING.json` to the original `ani_annotation_repair_v1/binding.json`. The crosswalk is explanatory and does not replace the validators. Saved JSON/NPZ predictions and independent validators remain the authority for all numerical claims. The handoff package intentionally does not claim to be a portable runnable release because source files use local paths and external data assets.

## Exact snapshot locations
Locations were parsed from the actual shipped Python snapshots, not inferred from function names. Nested definitions are included. Hashes identify the full source file.

### analyze_ecoli_frozen_query_ablation.py

SHA256: `0b18668e8a889171fe1fbf3f4690062d03359d943c85e57dfac28c577993d681`

| Function or class | Snapshot lines |
|---|---|
| `independent_metrics` | 12–28 |
| `main` | 31–242 |
| `compare` | 70–80 |
| `difference` | 212–213 |
| `false_sensitive` | 214–216 |

### counterexample_guided_repair.py

SHA256: `ee5e22bf56e787f66b1c398539b38abfa08b789dd359b4e26535abd70856c000`

| Function or class | Snapshot lines |
|---|---|
| `Record` | 46–53 |
| `compact_sparse` | 56–63 |
| `clean` | 66–67 |
| `parse_year` | 70–79 |
| `parse_ast` | 82–88 |
| `is_mechanism_token` | 91–104 |
| `load_records` | 107–139 |
| `feature_dict` | 142–152 |
| `ProbabilityModel` | 155–181 |
| `__init__` | 156–162 |
| `fit` | 164–172 |
| `predict` | 174–181 |
| `entropy_score` | 184–187 |
| `novelty_score` | 190–193 |
| `collision_score` | 196–201 |
| `train_calibrator` | 204–215 |
| `calibrated` | 218–221 |
| `choose_action` | 224–246 |
| `risk_coverage` | 249–266 |
| `metrics` | 269–282 |
| `select_indices` | 285–322 |
| `run_strategy` | 325–391 |
| `main` | 394–450 |

### decision_repair_core.py

SHA256: `351bd61f1aa8dce1ecb624ca87ac48047824127b4b00f72ad761d1ce4553e808`

| Function or class | Snapshot lines |
|---|---|
| `action_matrix` | 15–16 |
| `update_weights` | 19–24 |
| `query_scores` | 27–43 |
| `entropy` | 40–41 |
| `ExpertBank` | 46–71 |
| `__init__` | 47–53 |
| `update` | 55–66 |
| `loss` | 63–65 |
| `predict` | 68–71 |
| `diverse_indices` | 74–88 |

### run_audit_acquisition_action.py

SHA256: `4a0f13b7714e87148ace0d59dc568976b2bd285aa81aac0c38c9c03b18c3268e`

| Function or class | Snapshot lines |
|---|---|
| `read` | 22–22 |
| `sha` | 23–27 |
| `write` | 28–30 |
| `utc` | 31–31 |
| `acquisition_order` | 33–43 |
| `score` | 45–50 |
| `change` | 52–58 |
| `audit_evidence` | 60–68 |
| `decide` | 70–76 |
| `replay` | 78–114 |
| `main` | 116–172 |

### run_combiant_own_vision.py

SHA256: `7eec1c886960f844997f82f092042724a88ec0d8292808ce694fc8bfe617c38a`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 22–23 |
| `image_features` | 26–43 |
| `metric_rows` | 46–65 |
| `main` | 68–205 |

### run_decision_repair.py

SHA256: `01e524fb4e3c83802434c715150b5c5932855fa4520eb9d4311daec9cbac6d39`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 37–38 |
| `clean_json` | 41–48 |
| `write_json` | 51–52 |
| `mask` | 55–56 |
| `audit_partition` | 59–62 |
| `tree_predict` | 65–73 |
| `predict` | 69–72 |
| `trajectory` | 76–127 |
| `selective_metrics` | 130–142 |
| `main` | 145–221 |
| `reveal` | 179–182 |

Important operation at line 89: `action = int(np.argmin(weights @ matrix)) if repair == 'diagnostic' else 2`

Important operation at line 119: `weights = update_weights(weights, prior, labels)`

### run_ecoli_ani_annotation_repair.py

SHA256: `e4dd344a73fa7401c7876a12123ba21a4aa7ee6100f941f36cb464a7a9ff6fb3`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 13–17 |
| `read` | 18–18 |
| `components` | 19–38 |
| `find` | 21–23 |
| `union` | 24–26 |
| `main` | 39–112 |
| `reveal` | 97–97 |

Important operation at line 79: `role=lambda c:int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10`

### run_ecoli_frozen_query_ablation.py

SHA256: `992fc4e169a66138840b3fcb9261cff6a7d837bc7e40bbb47f101ae26a429e07`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 30–35 |
| `read` | 38–39 |
| `write` | 42–47 |
| `stem` | 50–51 |
| `original` | 54–55 |
| `raw_labels` | 58–64 |
| `metrics` | 67–85 |
| `check_roles` | 88–102 |
| `check_trace` | 105–119 |
| `provenance` | 122–147 |
| `main` | 150–260 |
| `records` | 204–206 |

### run_ecoli_repair.py

SHA256: `f5734cf9a435e1db8e5c3fcf9cb61810ea41cee838077b42afcf3210caab976c`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 23–23 |
| `measured_metrics` | 25–29 |
| `main` | 31–130 |
| `role` | 82–82 |
| `reveal` | 102–104 |

### run_msdeepamr_reexecution.py

SHA256: `2d9052549c9d117653dbad975692888baf4bffc9024dc4b5366a3866e0c95bad`

| Function or class | Snapshot lines |
|---|---|
| `Progress` | 35–43 |
| `__init__` | 36–38 |
| `on_epoch_end` | 40–43 |
| `sha` | 46–50 |
| `write` | 53–56 |
| `numpy_scalar` | 59–62 |
| `measure` | 65–75 |
| `get_data` | 78–89 |
| `load_author` | 92–93 |
| `predict` | 96–98 |
| `frozen_head` | 101–110 |
| `main` | 113–227 |

### run_reliability_controls.py

SHA256: `ac7fc20106a5e9c36bfc90457e5158503fb9f417a793b077a8fde3e3398a26dc`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 27–31 |
| `read` | 34–34 |
| `category` | 37–39 |
| `period` | 42–42 |
| `extra_metrics` | 45–53 |
| `population` | 56–80 |
| `partition` | 83–110 |
| `transform` | 113–134 |
| `main` | 137–202 |
| `reveal` | 181–183 |

### run_repair_dynamics.py

SHA256: `a9aca6b4148254e4f97a803f61921812ccad98944391e152f3c45fd4b8f1a384`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 21–26 |
| `read` | 29–30 |
| `write` | 33–34 |
| `measure` | 37–40 |
| `transition` | 43–58 |
| `audit_gate` | 61–79 |
| `summarize` | 82–93 |
| `main` | 96–222 |

### validate_audit_acquisition_action.py

SHA256: `2b0b99e40ede76401a3dc7bf4f9d8c87a62e6c259199b512b9796ae4c9868405`

| Function or class | Snapshot lines |
|---|---|
| `read` | 10–10 |
| `sha` | 11–11 |
| `equal` | 12–20 |
| `metrics` | 21–28 |
| `flips` | 29–34 |
| `independent_order` | 35–47 |
| `main` | 48–129 |

### validate_ecoli_ani_annotation_repair.py

SHA256: `b974321231a96ee6e0af238ab4e41adbcd11a574822baeb53b64da94414d5c49`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 18–23 |
| `read` | 26–27 |
| `groups` | 30–64 |
| `connect` | 32–34 |
| `main` | 67–201 |

### validate_repair_dynamics.py

SHA256: `f0b44f96f0aebe843bdc7249d988df52feb344ca808f049327b5b63b76d4524a`

| Function or class | Snapshot lines |
|---|---|
| `digest` | 15–16 |
| `load` | 19–20 |
| `check_metrics` | 23–27 |
| `main` | 30–163 |

### verify_manuscript_math_2026_10_12.py

SHA256: `5ef454619cd19c95eec6a5adfcdf58a72690f1d0c1b8333407171166dc0d3a76`

| Function or class | Snapshot lines |
|---|---|
| `sha` | 15–15 |
| `main` | 17–102 |
| `close` | 20–24 |
