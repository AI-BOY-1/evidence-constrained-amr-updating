"""Counterexample-guided auditing and repair for AMR prediction.

This is a reproducible research implementation of the proposed NCS method.
It treats a queried AST label as evidence for both prediction and diagnosis:
the controller searches for samples on which independent feature views disagree,
then chooses a repair action (retrain, recalibrate, or abstain) under a fixed
label budget. Evaluation labels are never used for selection or repair.

The default experiment is deliberately conservative:
  development: collection year <= 2019
  hidden query pool: 2020-2021
  untouched evaluation: collection year >= 2022

This is a computational study, not a clinical decision system. Public AST
records have heterogeneous breakpoints and source processes; the report keeps
those limitations visible instead of treating the data as one harmonized assay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DRUG = "meropenem"
MISSING = {"", "null", "na", "n/a", "-", "not collected", "not provided", "missing"}


@dataclass(frozen=True)
class Record:
    sample: str
    project: str
    year: int
    country: str
    features: tuple[str, ...]
    mechanism_features: tuple[str, ...]
    label: int


def compact_sparse(matrix):
    """Make DictVectorizer output compatible with current scikit-learn."""
    matrix = matrix.tocsr()
    if matrix.nnz >= np.iinfo(np.int32).max or matrix.shape[1] >= np.iinfo(np.int32).max:
        raise ValueError("sparse design matrix is too large for 32-bit sklearn indices")
    matrix.indices = np.asarray(matrix.indices, dtype=np.int32)
    matrix.indptr = np.asarray(matrix.indptr, dtype=np.int32)
    return matrix


def clean(value: str | None) -> str:
    return (value or "").strip()


def parse_year(value: str | None) -> int | None:
    value = clean(value)
    if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        return None
    try:
        year = int(value[:4])
        date(year, 1, 1)
        return year if 1900 <= year <= 2026 else None
    except ValueError:
        return None


def parse_ast(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        drug, separator, label = item.partition("=")
        if separator and drug and label in {"S", "R"}:
            result[drug] = label
    return result


def is_mechanism_token(token: str) -> bool:
    """Keep a deliberately interpretable AMR view.

    Carbapenemases, porin changes and quinolone/efflux markers are retained;
    partial calls are retained as a separate token rather than silently being
    promoted to a functional resistance gene.
    """
    low = token.lower()
    prefixes = (
        "blakpc", "blandm", "blavim", "blaimp", "blaoxa-48", "blaoxa-181",
        "blaoxa-232", "ompk35", "ompk36", "ramr", "mar", "acr", "oqxa",
        "oqxb", "gyr", "parc", "crrb",
    )
    return low.startswith(prefixes) or "partial" in low or "mistranslation" in low


def load_records(path: Path, drug: str) -> list[Record]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            host = clean(row.get("host"))
            if host not in {"Homo sapiens", "Homo sapiens sapiens"}:
                continue
            ast = parse_ast(row.get("AST_phenotypes", ""))
            year = parse_year(row.get("year"))
            if year is None or ast.get(drug) not in {"S", "R"}:
                continue
            sample = clean(row.get("biosample_acc")) or clean(row.get("target_acc"))
            if not sample:
                continue
            by_sample[sample].append(row)

    records: list[Record] = []
    for sample, rows in by_sample.items():
        # If a BioSample has incompatible public records, exclude it rather
        # than letting a duplicate label leak into the query/evaluation split.
        signatures = {(clean(r.get("AST_phenotypes")), clean(r.get("AMR_genotypes_core"))) for r in rows}
        if len(signatures) != 1:
            continue
        row = sorted(rows, key=lambda x: clean(x.get("target_acc")))[0]
        tokens = tuple(sorted({clean(x) for x in clean(row.get("AMR_genotypes_core")).split(",") if clean(x)}))
        if not tokens:
            continue
        mechanisms = tuple(sorted({x for x in tokens if is_mechanism_token(x)}))
        country = clean(row.get("country"))
        label = int(parse_ast(row.get("AST_phenotypes", ""))[drug] == "R")
        records.append(Record(sample, clean(row.get("project")), int(clean(row.get("year"))),
                              country, tokens, mechanisms or ("__NO_MECHANISM_CALL__",), label))
    return records


def feature_dict(record: Record, view: str) -> dict[str, float]:
    if view == "gene":
        return {f"gene::{token}": 1.0 for token in record.features}
    if view == "mechanism":
        return {f"mechanism::{token}": 1.0 for token in record.mechanism_features}
    if view == "context":
        result = {f"gene::{token}": 1.0 for token in record.features}
        result[f"country::{record.country or '__MISSING__'}"] = 1.0
        result[f"year_bin::{record.year // 3 * 3}"] = 1.0
        return result
    raise ValueError(f"unknown feature view: {view}")


class ProbabilityModel:
    def __init__(self, view: str, random_state: int):
        self.view = view
        self.random_state = random_state
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1500,
                                        random_state=random_state)
        self.constant: float | None = None

    def fit(self, records: Sequence[Record]) -> "ProbabilityModel":
        labels = [r.label for r in records]
        self.constant = None
        if len(set(labels)) < 2:
            self.constant = float(labels[0]) if labels else 0.5
            return self
        x = compact_sparse(self.vectorizer.fit_transform([feature_dict(r, self.view) for r in records]))
        self.model.fit(x, labels)
        return self

    def predict(self, records: Sequence[Record]) -> np.ndarray:
        if not records:
            return np.array([], dtype=float)
        if self.constant is not None:
            return np.full(len(records), self.constant, dtype=float)
        x = self.vectorizer.transform([feature_dict(r, self.view) for r in records])
        x = compact_sparse(x)
        return self.model.predict_proba(x)[:, 1]


def entropy_score(probability: np.ndarray) -> np.ndarray:
    eps = 1e-8
    p = np.clip(probability, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def novelty_score(training: Sequence[Record], candidates: Sequence[Record]) -> np.ndarray:
    known = set(token for row in training for token in row.features)
    return np.array([sum(token not in known for token in row.features) / max(len(row.features), 1)
                     for row in candidates], dtype=float)


def collision_score(training: Sequence[Record], candidates: Sequence[Record]) -> np.ndarray:
    labels_by_signature: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for row in training:
        labels_by_signature[row.features].add(row.label)
    return np.array([float(len(labels_by_signature.get(row.features, set())) > 1)
                     for row in candidates], dtype=float)


def train_calibrator(training: Sequence[Record], random_state: int) -> tuple[ProbabilityModel, IsotonicRegression | None]:
    """Fit a fixed gene model and a calibration map using only development data."""
    if len(training) < 30 or len({r.label for r in training}) < 2:
        return ProbabilityModel("gene", random_state).fit(training), None
    train, validation = train_test_split(training, test_size=0.25, random_state=random_state,
                                         stratify=[r.label for r in training])
    model = ProbabilityModel("gene", random_state).fit(train)
    p = model.predict(validation)
    if len(set(p)) < 2:
        return ProbabilityModel("gene", random_state).fit(training), None
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(p, [r.label for r in validation])
    return ProbabilityModel("gene", random_state).fit(training), calibrator


def calibrated(model: ProbabilityModel, calibrator: IsotonicRegression | None,
               records: Sequence[Record]) -> np.ndarray:
    p = model.predict(records)
    return calibrator.predict(p) if calibrator is not None and len(p) else p


def choose_action(training: Sequence[Record], query: Sequence[Record], p_gene: np.ndarray,
                  p_context: np.ndarray, budget_left: int) -> tuple[str, dict[str, float]]:
    """Choose the repair from observable diagnostics only.

    The controller never reads query/evaluation labels here. A large view gap
    means the gene-only representation and contextual representation disagree;
    a large shift score with small disagreement favours recalibration. With
    little evidence left, abstention is safer than claiming a repair.
    """
    if not len(query):
        return "abstain", {"view_gap": 0.0, "entropy": 0.0, "shift": 0.0}
    gap = float(np.mean(np.abs(p_gene - p_context)))
    uncertainty = float(np.mean(entropy_score(p_gene)))
    shift = float(np.mean(novelty_score(training, query)))
    if budget_left <= 0:
        action = "abstain"
    elif gap >= 0.10 and shift >= 0.04:
        action = "information_repair"
    elif shift >= 0.08:
        action = "recalibrate"
    else:
        action = "retrain"
    return action, {"view_gap": gap, "entropy": uncertainty, "shift": shift}


def risk_coverage(y: np.ndarray, p: np.ndarray, abstain_score: np.ndarray,
                  source_score: np.ndarray, target_coverage: float = 0.90) -> dict[str, float]:
    """Evaluate selective prediction with the abstention threshold fixed on source data.

    Labels from the future evaluation queue are used only to report realized risk;
    they never choose the threshold or a repair action.
    """
    if not len(y):
        return {"coverage": 0.0, "false_susceptible_rate": 1.0, "threshold": 1.0}
    keep_fraction = min(max(target_coverage, 0.0), 1.0)
    threshold = float(np.quantile(source_score, keep_fraction)) if len(source_score) else 1.0
    keep = abstain_score <= threshold
    if not keep.any():
        keep[np.argmin(abstain_score)] = True
    actual_r = y[keep] == 1
    false_s = ((p[keep] < 0.5) & actual_r).sum() / max(actual_r.sum(), 1)
    return {"coverage": float(keep.mean()), "false_susceptible_rate": float(false_s),
            "threshold": threshold}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    actual_r = y == 1
    actual_s = y == 0
    predicted_r = p >= 0.5
    return {
        "n": int(len(y)),
        "resistant": int(actual_r.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "false_susceptible": int(((~predicted_r) & actual_r).sum()),
        "false_susceptible_rate": float(((~predicted_r) & actual_r).sum() / max(actual_r.sum(), 1)),
        "false_resistant_rate": float((predicted_r & actual_s).sum() / max(actual_s.sum(), 1)),
    }


def select_indices(strategy: str, training: Sequence[Record], candidates: Sequence[Record],
                   gene: np.ndarray, mechanism: np.ndarray, context: np.ndarray,
                   rng: np.random.Generator, batch_size: int) -> np.ndarray:
    if not len(candidates):
        return np.array([], dtype=int)
    uncertainty = entropy_score(gene)
    disagreement = np.abs(gene - mechanism)
    context_gap = np.abs(gene - context)
    novelty = novelty_score(training, candidates)
    collisions = collision_score(training, candidates)
    if strategy == "random":
        score = rng.random(len(candidates))
    elif strategy == "uncertainty":
        score = uncertainty
    elif strategy == "disagreement":
        score = disagreement + 0.25 * uncertainty
    elif strategy.startswith("counterexample"):
        # Diagnostic value: choose samples that could simultaneously expose
        # a representation blind spot, a source shift and an exact-signature
        # collision. Ablation variants remove one component without changing
        # the label budget or the downstream repair controller.
        weights = {"disagreement": 0.40, "context": 0.25, "novelty": 0.20,
                   "collision": 0.10, "uncertainty": 0.05}
        if strategy == "counterexample_no_context":
            weights["context"] = 0.0
        elif strategy == "counterexample_no_novelty":
            weights["novelty"] = 0.0
        elif strategy == "counterexample_no_collision":
            weights["collision"] = 0.0
        elif strategy == "counterexample_no_uncertainty":
            weights["uncertainty"] = 0.0
        score = (weights["disagreement"] * disagreement + weights["context"] * context_gap
                 + weights["novelty"] * novelty + weights["collision"] * collisions
                 + weights["uncertainty"] * uncertainty)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    tie = rng.uniform(0.0, 1e-10, len(score))
    return np.argsort(-(score + tie))[:min(batch_size, len(candidates))]


def run_strategy(strategy: str, source: list[Record], pool: list[Record], evaluation: list[Record],
                 budgets: Sequence[int], seed: int, batch_size: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    training = list(source)
    available = list(pool)
    results: list[dict] = []
    events: list[dict] = []
    max_budget = max(budgets)
    queried = 0
    repair_mode = "gene"
    calibrator: IsotonicRegression | None = None
    while True:
        gene_model = ProbabilityModel("gene", seed).fit(training)
        mech_model = ProbabilityModel("mechanism", seed).fit(training)
        context_model = ProbabilityModel("context", seed).fit(training)
        p_source_gene = gene_model.predict(source)
        p_source_mech = mech_model.predict(source)
        p_source_context = context_model.predict(source)
        p_eval_gene = gene_model.predict(evaluation)
        p_mech_eval = mech_model.predict(evaluation)
        p_context_eval = context_model.predict(evaluation)
        p_eval = (calibrated(gene_model, calibrator, evaluation) if repair_mode == "recalibrated"
                  else p_context_eval if repair_mode == "information_repair"
                  else p_eval_gene)
        source_abstain = np.abs(p_source_gene - p_source_mech) + np.abs(p_source_gene - p_source_context)
        eval_abstain = np.abs(p_eval_gene - p_mech_eval) + np.abs(p_eval_gene - p_context_eval)
        audit = risk_coverage(np.array([r.label for r in evaluation]), p_eval, eval_abstain,
                              source_abstain)
        if queried in budgets:
            row = {"strategy": strategy, "seed": seed, "budget": queried, **metrics(np.array([r.label for r in evaluation]), p_eval),
                   "audit_coverage_at_source_90pct": audit["coverage"],
                   "audit_false_S_rate": audit["false_susceptible_rate"],
                   "audit_threshold_from_source": audit["threshold"],
                   "repair_mode": repair_mode,
                   "action_last": events[-1]["action"] if events else "none"}
            results.append(row)
        if queried >= max_budget or not available:
            break

        p_pool_gene = gene_model.predict(available)
        p_pool_mech = mech_model.predict(available)
        p_pool_context = context_model.predict(available)
        idx = select_indices(strategy, training, available, p_pool_gene, p_pool_mech,
                             p_pool_context, rng, batch_size)
        chosen = [available[i] for i in idx]
        action, diagnostics = choose_action(training, chosen, p_pool_gene[idx],
                                            p_pool_context[idx], max_budget - queried)
        # The action is chosen before labels are revealed. After labels are
        # added, information_repair changes the evaluated predictor to the
        # context view, while recalibrate applies a source-only calibration map.
        events.append({"strategy": strategy, "seed": seed, "query_start": queried,
                       "query_n": len(chosen), "action": action, **diagnostics,
                       "samples": [r.sample for r in chosen]})
        training.extend(chosen)
        if action == "information_repair":
            repair_mode = "information_repair"
            calibrator = None
        elif action == "recalibrate":
            repair_mode = "recalibrated"
            _, calibrator = train_calibrator(training, seed)
        elif action == "retrain":
            repair_mode = "gene"
            calibrator = None
        used = {r.sample for r in chosen}
        available = [r for r in available if r.sample not in used]
        queried += len(chosen)
    return results, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cleaned_human_kp.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "counterexample_repair_results.json")
    parser.add_argument("--drug", default=DRUG)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--budgets", default="0,10,25,50,100")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    args = parser.parse_args()
    budgets = sorted({int(x) for x in args.budgets.split(",")})
    seeds = [int(x) for x in args.seeds.split(",")]
    records = load_records(args.input, args.drug)
    source = [r for r in records if r.year <= 2019]
    pool = [r for r in records if 2020 <= r.year <= 2021]
    evaluation = [r for r in records if r.year >= 2022]
    if not source or not pool or not evaluation:
        raise RuntimeError(f"split is empty: source={len(source)}, pool={len(pool)}, evaluation={len(evaluation)}")
    strategies = ["random", "uncertainty", "disagreement", "counterexample"]
    all_results, all_events = [], []
    for strategy in strategies:
        for seed in seeds:
            result, events = run_strategy(strategy, source, pool, evaluation, budgets, seed, args.batch_size)
            all_results.extend(result)
            all_events.extend(events)
    summary = []
    for strategy in strategies:
        for budget in budgets:
            rows = [r for r in all_results if r["strategy"] == strategy and r["budget"] == budget]
            summary.append({"strategy": strategy, "budget": budget, "n_runs": len(rows),
                            **{f"{key}_mean": float(np.nanmean([r[key] for r in rows]))
                               for key in ("auroc", "auprc", "brier", "false_susceptible_rate",
                                           "audit_coverage_at_source_90pct")}})
    report = {
        "method": "counterexample-guided auditing and repair",
        "drug": args.drug,
        "split": {"source": "year<=2019", "query_pool": "2020<=year<=2021", "evaluation": "year>=2022"},
        "n": {"all": len(records), "source": len(source), "pool": len(pool), "evaluation": len(evaluation)},
        "budgets": budgets, "seeds": seeds, "batch_size": args.batch_size,
        "action_policy": {
            "information_repair_if": "mean_view_gap >= 0.10 and mean_novelty >= 0.04",
            "recalibrate_if": "mean_novelty >= 0.08",
            "otherwise": "retrain",
            "abstain_if": "budget_left <= 0",
        },
        "results": all_results, "summary": summary, "events": all_events,
        "action_counts": dict(Counter(event["action"] for event in all_events)),
        "limitations": [
            "Retrospective label-reveal simulation, not new AST experiments.",
            "AST breakpoints and laboratory workflows are not harmonized across public sources.",
            "The query pool is from the same public cohort family; external BV-BRC validation is a separate next step.",
            "Feature-view disagreement is a diagnostic hypothesis, not a causal biological explanation.",
            "Risk-coverage auditing is empirical; it is not a distribution-free guarantee under arbitrary shift.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n": report["n"], "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
