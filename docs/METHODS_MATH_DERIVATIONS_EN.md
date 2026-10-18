# Mathematical Methods and derivations tied to the executed implementation

Writing resource prepared on 2026-10-12. Equations below formalize already executed code; proofs are supplied now for mathematical interpretation, not retrospectively declared preregistered theory. Standard identities and value-of-information properties are not claimed as new theorems. Synthetic counterexamples are mathematical examples, not study observations. Use CODE_FORMULA_CROSSWALK.md for exact function locations and immutable snapshot hashes.

## Suggested placement

Main Methods: E01–E03 (data/roles), E04–E06 (predictors), E07–E11 (controller), E13–E18 (metrics/decomposition), E19–E23 (audit decisions and costs). A compact algorithm belongs in Methods. Put full proofs P1–P8, counterexamples and implementation caveats in Supplementary Methods. Explain each equation's scientific purpose; do not fill the manuscript with equations unrelated to executed experiments. Eq. E12 is an optional explanatory comparator not among the four formal E. coli strategies; E24 is a boundary identity, not a fitted guarantee. Adapt numbering after final editing.

## 1. Data objects, labels and isolated roles

For one drug, let a record be

\[
z_i=(s_i,c_i,\phi_i,h_i,t_i,y_i),\qquad y_i\in\{0,1\}. \tag{E01}
\]

Here s_i is a sequence/sample identifier, c_i its grouping component, phi_i input-token presence, h_i the recorded AST standards and t_i the collection year; y_i=1 denotes recorded resistance. Each drug has its own eligible cohort; identifiers can appear under multiple drugs. The labels are retained public AST labels, not new clinical confirmations. Unknown/ambiguous years map to 0. The E. coli context variable is standards although its inherited code field is named country.

For seed s, the role hash is

\[
u_s(c)=\operatorname{int}_{16}\!\left(\operatorname{SHA256}(s\!:\!c)_{1:8}\right)\bmod10,
\quad
\rho_s(c)=
\begin{cases}
\mathsf{source},&u_s(c)<4,\\
\mathsf{query},&4\le u_s(c)<7,\\
\mathsf{audit},&u_s(c)=7,\\
\mathsf{evaluation},&u_s(c)\ge8.
\end{cases} \tag{E02}
\]

Denote the resulting sets by S,Q,A,E. Hash buckets do not guarantee exact 40/30/10/20 sample fractions. All samples in an observed connected component receive the same role. This gives component isolation within this inventory, not independence of patients/laboratories or unidentified near relatives.

The new ANI-screen graph uses

\[
(i,j)\in\mathcal G_\tau
\iff \left[\exists d\in\{i\to j,j\to i\}_{\rm available}:
 a_d\ge\tau\ \land\ \min(f_{q,d},f_{r,d})\ge0.80\right]
\lor B_{ij}\lor X_{ij}. \tag{E03}
\]

a_d is the identity value of the available screened direction; B_ij means a shared nonempty BioSample accession and X_ij membership in the same exact normalized-sequence group. Connected components of this union graph define c_i. Main tau=0.99; sensitivity tau=0.999. E03 requires either available direction to qualify, not both. Approximate screening cutoff 0.99 can omit genuine edges. Original sketch/Jaccard components used by Fig. 2 are a distinct graph, not Eq. E03.

**P1 — Observed component-role isolation.** If c_i=c_j, then rho_s(c_i)=rho_s(c_j) by the determinism of E02. Distinct role sets therefore cannot share an observed component. Transitivity follows from taking connected components, so an observed connecting chain cannot straddle roles. This proof has no implication for unobserved SNP links, patients or laboratories.

## 2. Base predictors and five candidate rules

For gene view, phi_g records binary sketch/hash or homolog-token presence. Context view concatenates gene features, a standards indicator and a three-year-bin indicator. DictVectorizer fitted on the current training records fixes its sparse vocabulary. Unseen token columns are absent at prediction. Logistic predictions are

\[
p_\theta(x)=\sigma(\tilde\theta^\top\tilde\phi(x)),\qquad
\sigma(z)=(1+e^{-z})^{-1}. \tag{E04}
\]

For the implemented liblinear L2 logistic solver, its objective can be written, up to positive scaling, as

\[
\min_{\tilde\theta}\ \frac12\|\tilde\theta\|_2^2+
C\sum_{i\in T}\left[\log(1+e^{\tilde\theta^\top\tilde\phi_i})-y_i\tilde\theta^\top\tilde\phi_i\right],
\quad C=1. \tag{E05}
\]

The default liblinear synthetic intercept feature has intercept_scaling=1 and is penalized too; do not write an unpenalized intercept objective as an exact match. Code specifies solver=liblinear, max_iter=1500, C=1 and seed; numerical stopping is the library's default, not exact analytic convergence. One-class subsets return a constant probability instead of solving E05.

The ordered expert bank is (frozen_gene, frozen_context, updated_gene, updated_context, intercept_update). Frozen experts are fitted on S. Updated experts refit on S union L_t, where L_t contains queried labels so far. At t=0 the updated and frozen experts coincide. The offset expert is

\[
p_{5,t}(x)=\sigma\!\left(\operatorname{logit}(\operatorname{clip}_{\epsilon}(p_{1}(x)))+a_t\right),
\quad
a_t=\arg\min_{a\in[-6,6]}\left\{
\sum_{i\in L_t}[\log(1+e^{\ell_i+a})-y_i(\ell_i+a)]+\frac12a^2\right\},
\quad \epsilon=10^{-6}. \tag{E06}
\]

ell_i is the clipped frozen-gene logit. At zero queried labels the code sets a_0=0.

**P2 — Unique offset solution.** For the objective F(a) in E06,

\[
F'(a)=\sum_{i\in L_t}[\sigma(\ell_i+a)-y_i]+a,\qquad
F''(a)=\sum_{i\in L_t}\sigma(\ell_i+a)[1-\sigma(\ell_i+a)]+1\ge1.
\]

F is strictly convex, continuous on the compact interval, and has a unique minimizer. An interior minimizer solves F'(a)=0; otherwise the constrained optimum is a boundary. scipy minimize_scalar(method='bounded') approximates this optimum subject to numerical tolerance. Uniqueness of this one-dimensional fit does not establish calibration or target benefit.

## 3. Evidence weights, action choice and approximate query value

Let M=5 and w_0=(1/M,...,1/M). For a chosen batch B_t, cache the M probabilities before revealing its labels. Let q_ji=clip_epsilon(p_j,t(x_i)). The intended multiplicative update is

\[
w_{j,t+1}=\frac{w_{j,t}\prod_{i\in B_t}q_{ji}^{y_i}(1-q_{ji})^{1-y_i}}
{\sum_{k=1}^{M}w_{k,t}\prod_{i\in B_t}q_{ki}^{y_i}(1-q_{ki})^{1-y_i}}. \tag{E07}
\]

The exact code protects prior weights by max(w_j,1e-300), computes log weights, subtracts their maximum, exponentiates and normalizes. To express the safeguard exactly, replace w_j in E07 by max(w_j,1e-300); its normalization cancels. This may rescue an underflowed zero weight. Proofs using w as an exact hypothetical prior below assume a normalized nonnegative weight vector and the stated one-step updates; they do not ignore numerical safeguards in actual replay.

**P3 — Simplex preservation and prequential interpretation.** With finite clipped probabilities and positive protected prior weights, the log-normalized update yields nonnegative weights summing to one; at least a maximal log-weight yields exp(0)=1 before normalization. In exact arithmetic, weights remain strictly positive. Ignoring the underflow safeguard, repeated E07 updates unroll into products of each expert's *pre-reveal, time-varying* predictive likelihoods. As experts are refitted between batches, these are prequential evidence weights over rules, not a posterior over five fixed biological hypotheses or a globally exact Bayesian generative model. No weight concentration or regret rate has been proved for the deployed data.

For remaining query pool U_t, define the raw-prediction disagreement matrix and action criterion

\[
A^{\rm raw}_{jk}=|U_t|^{-1}\sum_{x\in U_t}[p_{j,t}(x)-p_{k,t}(x)]^2,
\qquad k_t=\arg\min_k\sum_jw_{j,t}A^{\rm raw}_{jk}. \tag{E08}
\]

trajectory selects an expert, rather than returning its probability mixture. Empty-pool snapshot handling sets A=0, and numpy argmin selects the first expert in a tie. This criterion is predictive disagreement, not empirical target loss or clinical harm.

**P4 — Action choice is the closest available expert to the weighted mixture.** Define bar p(x)=sum_j w_j p_j(x). At each x,

\[
\sum_jw_j[p_j(x)-p_k(x)]^2
=\sum_jw_j[p_j(x)-\bar p(x)]^2+[\bar p(x)-p_k(x)]^2. \tag{E09}
\]

Proof: write p_j-p_k=(p_j-bar p)+(bar p-p_k), expand, and use sum_j w_j(p_j-bar p)=0. Average over U_t; the first term is independent of k. Thus E08 selects the available expert closest to the weighted mixture in pool squared distance. This is an exact characterization of the selection rule, not a proof that its selected expert is closest to true labels.

query_scores first clips all pool probabilities, and constructs A^epsilon from those clipped probabilities. This differs slightly from the raw matrix in E08; do not suppress that distinction. For a candidate x, let

\[
m(x)=\sum_jw_jq_j(x),\quad
w_j^{(1)}(x)=\frac{w_jq_j(x)}{m(x)},\quad
w_j^{(0)}(x)=\frac{w_j[1-q_j(x)]}{1-m(x)},
\quad r_A(w)=\min_k w^\top A_{:k}. \tag{E10}
\]

The executed decision query score is

\[
v_t(x)=\max\!\left\{0,\ r_{A^\epsilon}(w_t)-
m(x)r_{A^\epsilon}(w^{(1)}(x))-[1-m(x)]r_{A^\epsilon}(w^{(0)}(x))\right\}. \tag{E11}
\]

All hypothetical outcomes use a fixed current A^epsilon and current expert predictions. Real future refitting, the removal of a queried point and joint batch outcomes are not included in this single-label lookahead.

**P5 — Nonnegative one-step value under the fixed surrogate.** E10 gives m w^(1)+(1-m)w^(0)=w coordinatewise. The minimum of linear functions r_A is concave. Therefore

\[
r_A(w)\ge m r_A(w^{(1)})+(1-m)r_A(w^{(0)}).
\]

Alternatively, let k* minimize w^T A: expected posterior optimal cost is at most expected posterior cost of keeping k*, which equals w^T A_:k*. Since squared disagreement is nonnegative, 0<=v(x)<=r_A(w)<=1 in exact arithmetic. If one action minimizes both posterior objectives, equality gives v(x)=0. Thus label uncertainty alone need not have positive decision value. The max(0,.) operation handles roundoff. This standard one-step value property does **not** prove that querying lowers actual Brier, FS or clinical risk; nor does it prove monotonicity after real expert refits or optimality of the five-point static batch.

The optional information objective in the core is

\[
v_H(x)=H(w)-mH(w^{(1)})-(1-m)H(w^{(0)})=I(J;Y_x),
\quad H(w)=-\sum_jw_j\log w_j. \tag{E12}
\]

This equality is conditional on the hypothetical model P(J=j)=w_j, P(Y_x=1|J=j)=q_j(x), and natural logarithms. It follows from H(J)-H(J|Y). It is a standard identity, not the formal four-strategy study's additional tested comparator. The main controller uses E11. Entropy_gene instead scores the Bernoulli entropy of the updated gene expert; these objectives should not be confused.

The executed controller ranks scores plus seeded tie jitter (0 to 1e-12), chooses up to five records without replacement, caches probabilities, reveals only query labels, updates E07, removes these records and refits the bank. Formal baselines are random_gene, uncertainty_gene and random_tree; their details are in METHODS_IMPLEMENTATION_NOTES.md.

## 4. Evaluation metrics and matched decomposition

For any fixed labelled set D,

\[
\mathcal B_D(p)=\frac1{|D|}\sum_{i\in D}(p_i-y_i)^2,
\quad \hat y_i=\mathbf1\{p_i\ge0.5\},
\quad \mathrm{FS}_D(p)=\sum_{i\in D}y_i\mathbf1\{p_i<0.5\},
\quad \mathrm{FR}_D(p)=\sum_{i\in D}(1-y_i)\mathbf1\{p_i\ge0.5\}. \tag{E13}
\]

With R_D=sum y_i, S_D=|D|-R_D, define FSR=FS/R_D only when R_D>0 and FRR=FR/S_D only when S_D>0. Undefined values remain unavailable. Source scripts for MALDI use p>0.5 for a resistant call instead; it is a separate convention.

For fixed evaluation identities, let R be random-path logistic refitting, G diagnostic-path logistic refitting, D the original diagnostic action. Define

\[
\Delta_{\rm query}=\mathcal B_E(G)-\mathcal B_E(R),\quad
\Delta_{\rm action}=\mathcal B_E(D)-\mathcal B_E(G),\quad
\Delta_{\rm total}=\mathcal B_E(D)-\mathcal B_E(R). \tag{E14}
\]

**P6 — Exact additive matched decomposition.** Addition cancels B_E(G), so

\[
\Delta_{\rm total}=\Delta_{\rm query}+\Delta_{\rm action}. \tag{E15}
\]

This holds per matched configuration and after taking the same weighted average over all three terms. It need not hold after averaging different eligible subsets. It is algebra, not identification of causal mediation. Its inputs are original sketch trajectories, not the later ANI partitions.

For old and new probabilities on the same D, define error indicators e_i^o=1{hat y_i^o!=y_i}, e_i^n similarly, and

\[
C=\sum_i e_i^o(1-e_i^n),\quad I=\sum_i(1-e_i^o)e_i^n,\quad
N_{\rm err}^{n}-N_{\rm err}^{o}=I-C. \tag{E16}
\]

Restricting to y_i=1 gives

\[
\mathrm{FS}^{n}-\mathrm{FS}^{o}=I_R-C_R,
\quad \Delta\mathrm{FSR}=(I_R-C_R)/R_D\quad(R_D>0). \tag{E17}
\]

**P7 — Error-flow accounting.** Per sample e_i^n-e_i^o=(1-e_i^o)e_i^n-e_i^o(1-e_i^n); summing proves E16, and restricting the sum proves E17. A net reduction can coexist with individual newly introduced errors. Neither the net count nor the rate should replace the separate C and I counts.

The probability-loss change is

\[
\Delta\mathcal B_D=|D|^{-1}\sum_i(p_i^n-p_i^o)(p_i^n+p_i^o-2y_i). \tag{E18}
\]

This is the difference-of-squares identity. Its sign has no universal equivalence to the thresholded signs in E17.

**Counterexample C1 — Lower Brier with a newly false-susceptible call.** For two mathematical records y=(1,0), p^o=(0.51,0.40) and p^n=(0.49,0.10), B^o=0.20005, B^n=0.13505 and Delta B=-0.065; FS changes from 0 to 1. These invented numbers illustrate E18/E17, not actual isolates. In the actual study, 82/2,040 adjacent ANI comparisons had lower Brier with higher net FS; do not merge the mathematical example with that empirical numerator.

## 5. Audit acquisition and action acceptance

For frozen trajectory and budget b, let p^0,p^b be budget-zero and updated probabilities on the fixed roles. Let H_{a,n} be the first n audit identities from acquisition order a. Random acquisition uses seeded SHA256 ranks; predicted_R ranks p^0 descending with those ranks breaking ties; component_coverage round-robins fixed component buckets. Each order is built without audit labels or evaluation outcomes. Consequently, for fixed metadata and p^0,

\[
H_{a,n}(y_A)=H_{a,n}(y'_A),\qquad
H_{a,n}\subseteq H_{a,n'}\ (n\le n'). \tag{E19}
\]

This is a conditional label-invariance/nesting property of the acquisition routine, not of all cohort preprocessing: original discordance exclusions do use recorded labels.

Define d_B=B_H(p^b)-B_H(p^0), d_F=FS_H(p^b)-FS_H(p^0), and R_H=sum_H y. The acceptance indicators are

\[
\alpha_{\rm retain}=0,\quad\alpha_{\rm update}=1,\quad
\alpha_{\rm loss}=\mathbf1\{d_B<0\},\quad
\alpha_{\rm direction}=\mathbf1\{R_H>0,\ d_B<0,\ d_F\le0\}. \tag{E20}
\]

For components represented in H, let K_H their number and K_R the number with positive recorded resistance; define

\[
d_{B,c}=\frac1{K_H}\sum_{c}\frac1{|H_c|}\sum_{i\in H_c}
\{(p_i^b-y_i)^2-(p_i^0-y_i)^2\},
\quad
d_{F,c}=\frac1{K_R}\sum_{c:R_c>0}\frac1{R_c}
\sum_{i\in H_c}y_i[\mathbf1\{p_i^b<0.5\}-\mathbf1\{p_i^0<0.5\}],
\quad
\alpha_{\rm component}=\mathbf1\{R_H>0,\ d_{B,c}<0,\ d_{F,c}\le0\}. \tag{E21}
\]

E21 is a component-weighted *aggregate* criterion, not a requirement that every component improve. Direction rules at R_H=0 are marked infeasible and retain as a fallback, not eligible successes. Strict ties retain; do not introduce tolerance-based acceptance or post-hoc threshold tuning. Exact replay retains NumPy reduction order because floating differences near zero can change a strict comparison.

The selected candidate, accounting and offline oracle comparator are

\[
p^\alpha=(1-\alpha)p^0+\alpha p^b,\quad
C_{\rm new}=b+n,\quad C_{\rm source}=|S|,
\quad G_E=\mathcal B_E(p^\alpha)-\min\{\mathcal B_E(p^0),\mathcal B_E(p^b)\}\ge0. \tag{E22}
\]

alpha is 0 or 1, so E22 does not blend two models. All rules incur b+n, including rejection and retain; source labels are shared sunk cost. Evaluation-label oracle G_E is an offline diagnostic, not an available deployment strategy. Code clips its arithmetic at zero for roundoff. Acquisition-size comparisons do not retrain after reallocating a fixed total budget; they compare frozen candidates with fully charged labels.

The earlier *sequential gate replay* is different. At current accepted checkpoint c and next budget b, accept if B_A(p^b)<B_A(p^c), optionally also FS_A(p^b)<=FS_A(p^c). On first rejection stop, keep c and charge consumed b:

\[
c\leftarrow b\text{ on acceptance};\qquad
\text{on rejection: }c\text{ unchanged, stop};\qquad
C_{\rm gate}=b_{\rm consumed}+|A|. \tag{E23}
\]

It compares against updating at that consumed budget. The later E20–E22 acquisition study always compares budget b to zero, evaluates each fixed condition, and does not enact E23. Do not combine the two experiments' denominators or costs.

**P8 — An empirical audit condition does not entail a target condition.** E20/E21 restrict values on H only. They contain no inequality constraining the evaluation probabilities/labels. Explicitly, for one mathematical audit record y_A=1, old/new=0.4/0.6, audit Brier decreases 0.36 to 0.16 and FS decreases 1 to 0, satisfying both direction rules with one component. For a separate evaluation record y_E=1, old/new=0.6/0.4, target Brier increases 0.16 to 0.36 and FS increases 0 to 1. Thus a universal implication from audit acceptance to target improvement is false without additional assumptions. These are invented explanatory probabilities, not a new biological result, a clinical failure rate or proof that every audit fails.

For a fixed old/new pair and an illustrative partition of outcomes into strata c, assume the conditional loss-change means mu_c are *the same* for audit and target. If a_c,e_c are their stratum weights, then

\[
\Delta\mathcal B_E-\Delta\mathcal B_A=\sum_c(e_c-a_c)\mu_c,
\quad |\Delta\mathcal B_E-\Delta\mathcal B_A|
\le\sum_c|e_c-a_c|=2\operatorname{TV}(e,a),\quad |\mu_c|\le1. \tag{E24}
\]

Under that restrictive invariance assumption the bound follows by the triangle inequality; differences in conditional means add further terms. In the executed ANI design roles have disjoint component IDs, so audit and evaluation do not share component-specific estimates. E24 supplies **no** observed nontrivial transfer bound: on disjoint component support TV=1 and the displayed upper bound is only trivial. Do not replace the missing assumption with a claim that component_coverage guarantees transport or apply a concentration inequality to correlated configurations.

## 6. Pseudocode matching execution order

```text
Input: source labels S; masked query/audit/evaluation records; query-only reveal;
       fixed strategy, seed and feasible budgets; five expert bank
Initialize uniform weights, no queries, remaining query pool U
Repeat:
    Predict expert probabilities on remaining U
    If queried count is a checkpoint:
        diagnostic: choose expert minimizing raw pool disagreement criterion
        gene baseline: use updated_gene; tree baseline: fit source+queried tree
        Save evaluation/audit probabilities, action, weights and queried IDs
    If maximum feasible budget reached: stop
    Score current U using strategy; diagnostic score uses clipped E10–E11
    Select up to five records by score plus fixed seeded tie jitter
    Cache their current expert probabilities BEFORE revealing their labels
    Reveal only chosen query IDs; update likelihood weights
    Append queried records; remove them from U; refit updated experts and offset
Output: frozen snapshots and pre/post-reveal events

Separate acquisition replay:
    Build each audit order from fixed IDs/components/budget-zero probabilities
    For each budget b, size n, acquisition a:
        If budget missing or audit pool smaller than n: record unavailable
        Otherwise reveal selected audit references for offline rule scoring
        Compute E20–E21; choose old or updated frozen candidate for each rule
        Charge b+n; independently score evaluation without using it for choice
```

For M experts and N remaining query records, disagreement construction and all hypothetical posterior action costs take O(M^2 N) arithmetic per query-scoring snapshot, excluding model fitting. The vectorized squared-difference temporary has O(M^2 N) memory. With M=5 this is a fixed-bank surrogate, not exact joint batch optimization or a complexity claim for all training.

## 7. Instructions for Gemini when using these derivations

Write a cohesive Methods section beginning with the information boundaries, then features/partitions, expert fitting, queries/actions, evaluation, audit replay, controls and validation. Explain how each mathematical object appears in the code and saved artifacts. State P1–P8 with their assumptions if used; retain all implementation caveats. Put standard proofs in supplementary material and describe them as characterizations or identities, not groundbreaking new theory. Numerical checks of these identities are checks, not a replacement for proofs and not new scientific results.

Do not add invented convergence rates, consistency, regret bounds, sample complexity, confidence-based stopping, clinical-risk objectives, distribution-free safety, causal identification, joint batch optimality or guarantees of ANI/patient isolation. Such extensions would describe a different unexecuted method. Any proposed extension must be labelled future work. The manuscript's empirical contribution remains the evidence chain and complete matched comparisons.
