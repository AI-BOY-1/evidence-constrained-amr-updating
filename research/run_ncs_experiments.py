"""Run the multi-drug, temporal and ablation experiment suite.

This driver keeps the scientific claim narrow: it compares the proposed
counterexample-guided controller against same-budget baselines across drugs,
time windows and component ablations. It does not use future evaluation labels
to choose queries, actions or thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from counterexample_guided_repair import load_records, run_strategy  # noqa: E402


DEFAULT_DRUGS = [
    "meropenem", "imipenem", "ertapenem", "ceftazidime", "ciprofloxacin",
    "trimethoprim-sulfamethoxazole", "amikacin", "gentamicin", "cefepime",
]
DEFAULT_STRATEGIES = [
    "random", "uncertainty", "disagreement", "counterexample",
    "counterexample_no_context", "counterexample_no_novelty",
    "counterexample_no_collision", "counterexample_no_uncertainty",
]


def cohorts(records, protocol):
    if protocol == "forward_2019":
        source = [r for r in records if r.year <= 2019]
        pool = [r for r in records if 2020 <= r.year <= 2021]
        evaluation = [r for r in records if r.year >= 2022]
    elif protocol == "forward_2017":
        source = [r for r in records if r.year <= 2017]
        pool = [r for r in records if 2018 <= r.year <= 2019]
        evaluation = [r for r in records if r.year >= 2020]
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    if not source or not pool or not evaluation:
        raise ValueError(f"empty cohort for {protocol}: {len(source)}, {len(pool)}, {len(evaluation)}")
    if len({r.label for r in source}) < 2 or len({r.label for r in pool}) < 2 or len({r.label for r in evaluation}) < 2:
        raise ValueError(f"single-class cohort for {protocol}: {len(source)}, {len(pool)}, {len(evaluation)}")
    return source, pool, evaluation


def bootstrap_interval(values, rng, repetitions=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    sample = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def summarize(rows, bootstrap_repetitions, seed):
    rng = np.random.default_rng(seed)
    metrics = ("auroc", "auprc", "brier", "false_susceptible_rate",
               "false_resistant_rate", "audit_coverage_at_source_90pct")
    groups = {}
    for row in rows:
        key = (row["protocol"], row["drug"], row["strategy"], row["budget"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        protocol, drug, strategy, budget = key
        out = {"protocol": protocol, "drug": drug, "strategy": strategy,
               "budget": budget, "n_runs": len(group)}
        for metric in metrics:
            values = [row[metric] for row in group]
            finite = [float(v) for v in values if np.isfinite(v)]
            out[f"{metric}_mean"] = float(np.mean(finite)) if finite else None
            out[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            low, high = bootstrap_interval(values, rng, bootstrap_repetitions)
            out[f"{metric}_ci95_low"] = low
            out[f"{metric}_ci95_high"] = high
        summary.append(out)
    random_brier = {
        (r["protocol"], r["drug"], r["budget"]): r["brier_mean"]
        for r in summary if r["strategy"] == "random"
    }
    for row in summary:
        baseline = random_brier.get((row["protocol"], row["drug"], row["budget"]))
        row["brier_delta_vs_random"] = (row["brier_mean"] - baseline
                                         if baseline is not None and row["brier_mean"] is not None else None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cleaned_human_kp.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "ncs_experiment_suite.json")
    parser.add_argument("--drugs", default=",".join(DEFAULT_DRUGS))
    parser.add_argument("--protocols", default="forward_2019,forward_2017")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--budgets", default="0,10,25,50,100")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    drugs = [x.strip() for x in args.drugs.split(",") if x.strip()]
    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    budgets = sorted({int(x) for x in args.budgets.split(",")})
    seeds = [int(x) for x in args.seeds.split(",")]
    rows, events, audits, skipped = [], [], [], []

    for drug in drugs:
        records = load_records(args.input, drug)
        for protocol in protocols:
            try:
                source, pool, evaluation = cohorts(records, protocol)
            except ValueError as exc:
                skipped.append({"drug": drug, "protocol": protocol, "reason": str(exc)})
                continue
            audits.append({"drug": drug, "protocol": protocol, "all": len(records),
                           "source": len(source), "query_pool": len(pool),
                           "evaluation": len(evaluation),
                           "eligible_budgets": [b for b in budgets if b <= len(pool)],
                           "source_resistant": int(sum(r.label for r in source)),
                           "pool_resistant": int(sum(r.label for r in pool)),
                           "evaluation_resistant": int(sum(r.label for r in evaluation))})
            cohort_budgets = [b for b in budgets if b <= len(pool)]
            for strategy in strategies:
                for seed in seeds:
                    result, event = run_strategy(strategy, source, pool, evaluation,
                                                 cohort_budgets, seed, args.batch_size)
                    for item in result:
                        rows.append({"protocol": protocol, "drug": drug, **item})
                    for item in event:
                        events.append({"protocol": protocol, "drug": drug, **item})
            print(json.dumps({"drug": drug, "protocol": protocol,
                              "source": len(source), "pool": len(pool),
                              "evaluation": len(evaluation)}, ensure_ascii=False))

    report = {
        "method": "counterexample-guided auditing and repair",
        "drugs": drugs, "protocols": protocols, "strategies": strategies,
        "budgets": budgets, "seeds": seeds, "batch_size": args.batch_size,
        "input": str(args.input), "cohort_audits": audits, "skipped": skipped,
        "results": rows, "events": events,
        "summary": summarize(rows, args.bootstrap, 20261007),
        "action_counts": dict(Counter(item["action"] for item in events)),
        "limitations": [
            "All labels are retrospective public AST labels, not new experiments.",
            "The time protocols test temporal transport but do not prove arbitrary-shift guarantees.",
            "BV-BRC is reserved for an independent validation adapter because its public AMR table does not contain the same genotype-token view.",
            "Bootstrap intervals describe seed variability and are not patient-level confidence intervals.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "result_rows": len(rows),
                      "event_rows": len(events), "summary_rows": len(report["summary"]),
                      "skipped": skipped, "action_counts": report["action_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

