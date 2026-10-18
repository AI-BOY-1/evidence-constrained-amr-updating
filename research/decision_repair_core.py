"""Predictive expert evidence and approximate single-query action value.

Evidence weights describe prediction rules, not identified biological causes.
Query scoring receives probabilities/features only; reveal labels afterwards.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from counterexample_guided_repair import ProbabilityModel

EXPERTS = ('frozen_gene', 'frozen_context', 'updated_gene', 'updated_context', 'intercept_update')


def action_matrix(probabilities):
    return np.mean((probabilities[:, None, :] - probabilities[None, :, :]) ** 2, axis=2)


def update_weights(weights, predictions, labels):
    p = np.clip(predictions, 1e-6, 1 - 1e-6)
    logw = np.log(np.maximum(weights, 1e-300)) + np.sum(
        labels[None, :] * np.log(p) + (1 - labels[None, :]) * np.log1p(-p), axis=1)
    result = np.exp(logw - logw.max())
    return result / result.sum()


def query_scores(probabilities, weights, objective):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    positive = weights[:, None] * p
    negative = weights[:, None] * (1 - p)
    marginal = positive.sum(axis=0)
    post1 = positive / marginal
    post0 = negative / (1 - marginal)
    if objective == 'decision':
        matrix = action_matrix(p)
        current = np.min(weights @ matrix)
        expected = marginal * np.min(post1.T @ matrix, axis=1) + (1 - marginal) * np.min(post0.T @ matrix, axis=1)
        return np.maximum(current - expected, 0)
    if objective == 'information':
        def entropy(q):
            return -np.sum(q * np.log(np.maximum(q, 1e-300)), axis=0)
        return entropy(weights) - marginal * entropy(post1) - (1 - marginal) * entropy(post0)
    raise ValueError(objective)


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


def diverse_indices(training, available, batch_size):
    """Farthest-first Jaccard distance in observed AMR token sets."""
    candidates = [set(r.features) for r in available]
    references = list({r.features for r in training})
    references = [set(x) for x in references]
    nearest = np.asarray([min(1 - len(x & ref) / max(len(x | ref), 1) for ref in references)
        for x in candidates])
    selected = []
    for _ in range(min(batch_size, len(available))):
        j = int(np.argmax(nearest))
        selected.append(j)
        distances = np.asarray([1 - len(x & candidates[j]) / max(len(x | candidates[j]), 1) for x in candidates])
        nearest = np.minimum(nearest, distances)
        nearest[selected] = -1
    return np.asarray(selected, dtype=int)
