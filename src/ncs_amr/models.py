"""Exact computational definitions extracted from frozen source functions."""
from typing import Sequence
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from .records import Record
def compact_sparse(matrix):
    """Make DictVectorizer output compatible with current scikit-learn."""
    matrix = matrix.tocsr()
    if matrix.nnz >= np.iinfo(np.int32).max or matrix.shape[1] >= np.iinfo(np.int32).max:
        raise ValueError("sparse design matrix is too large for 32-bit sklearn indices")
    matrix.indices = np.asarray(matrix.indices, dtype=np.int32)
    matrix.indptr = np.asarray(matrix.indptr, dtype=np.int32)
    return matrix

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

class ExpertBank:
    def __init__(self, source, seed):
        self.source = list(source)
        self.seed = seed
        self.frozen_gene = ProbabilityModel('gene', seed).fit(source)
        self.frozen_context = ProbabilityModel('context', seed).fit(source)
        self.updated_gene, self.updated_context = self.frozen_gene, self.frozen_context
        self.offset = 0.0

    def update(self, queried):
        if not queried:
            return
        self.updated_gene = ProbabilityModel('gene', self.seed).fit(self.source + list(queried))
        self.updated_context = ProbabilityModel('context', self.seed).fit(self.source + list(queried))
        logits = logit(np.clip(self.frozen_gene.predict(queried), 1e-6, 1 - 1e-6))
        labels = np.asarray([r.label for r in queried])
        # Fixed Gaussian offset penalty; no target/evaluation tuning.
        def loss(a):
            z = logits + a
            return float(np.sum(np.logaddexp(0, z) - labels * z) + 0.5 * a * a)
        self.offset = float(minimize_scalar(loss, bounds=(-6, 6), method='bounded').x)

    def predict(self, records):
        pg = self.frozen_gene.predict(records)
        return np.vstack((pg, self.frozen_context.predict(records), self.updated_gene.predict(records),
            self.updated_context.predict(records), expit(logit(np.clip(pg, 1e-6, 1 - 1e-6)) + self.offset)))
