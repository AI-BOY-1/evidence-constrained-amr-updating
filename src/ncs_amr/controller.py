"""Portable facade over unchanged computational function bodies."""
from typing import Callable, Sequence
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from .records import Record, validate_roles
from .models import ExpertBank, ProbabilityModel, compact_sparse, feature_dict, entropy_score
EXPERTS = ('frozen_gene', 'frozen_context', 'updated_gene', 'updated_context', 'intercept_update')
STRATEGIES = {'random_gene':('random','gene'), 'uncertainty_gene':('uncertainty','gene'),
              'decision_diagnostic':('decision','diagnostic'), 'random_tree':('random','tree')}
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

def tree_predict(training, evaluation, audit, seed):
    vec = DictVectorizer(sparse=True)
    x = compact_sparse(vec.fit_transform([feature_dict(r, 'gene') for r in training]))
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2).fit(x, [r.label for r in training])
    def predict(rows):
        if not rows:
            return np.asarray([], dtype=float)
        return model.predict_proba(compact_sparse(vec.transform([feature_dict(r, 'gene') for r in rows])))[:, list(model.classes_).index(1)]
    return predict(evaluation), predict(audit)

def trajectory(source, feature_pool, evaluation_features, audit_features, reveal, strategy, seed, budgets):
    """No audit/evaluation labels are accepted by this function."""
    assert all(r.label == -1 for r in feature_pool + evaluation_features + audit_features)
    objective, repair = STRATEGIES[strategy]
    rng = np.random.default_rng(seed)
    available, queried, events, snapshots = list(feature_pool), [], [], []
    bank = ExpertBank(source, seed)
    weights = np.full(len(EXPERTS), 1 / len(EXPERTS))
    while True:
        count = len(queried)
        pool_p = bank.predict(available)
        if count in budgets:
            matrix = action_matrix(pool_p) if len(available) else np.zeros((len(EXPERTS), len(EXPERTS)))
            action = int(np.argmin(weights @ matrix)) if repair == 'diagnostic' else 2
            if repair == 'tree':
                pe, pa = tree_predict(source + queried, evaluation_features, audit_features, seed)
                action_name = 'extra_trees'
            else:
                pe, pa = bank.predict(evaluation_features)[action], bank.predict(audit_features)[action]
                action_name = EXPERTS[action]
            snapshots.append({'budget': count, 'evaluation_p': pe, 'audit_p': pa,
                'action': action_name, 'evidence_weights': weights.tolist(),
                'queried_samples': [r.sample for r in queried]})
        if count >= max(budgets):
            break
        size = min(5, max(budgets) - count, len(available))
        if objective == 'diversity':
            selected = diverse_indices(source + queried, available, size)
        else:
            if objective == 'random':
                scores = rng.random(len(available))
            elif objective == 'uncertainty':
                scores = entropy_score(pool_p[2])
            elif objective == 'disagreement':
                scores = np.abs(pool_p[2] - pool_p[3])
            else:
                scores = query_scores(pool_p, weights, objective)
            selected = np.argsort(-(scores + rng.uniform(0, 1e-12, len(scores))))[:size]
        chosen = [available[i] for i in selected]
        # Cache predictions before accessing any new labels.
        prior = pool_p[:, selected].copy()
        observed = [reveal(r.sample) for r in chosen]
        labels = np.asarray([r.label for r in observed])
        weights = update_weights(weights, prior, labels)
        queried.extend(observed)
        events.append({'query_start': count, 'query_n': size, 'samples': [r.sample for r in chosen],
            'pre_reveal_predictions': prior.tolist(), 'revealed_labels': labels.tolist(),
            'post_reveal_evidence_weights': weights.tolist()})
        used = {r.sample for r in chosen}
        available = [r for r in available if r.sample not in used]
        bank.update(queried)
    return snapshots, events

def run_trajectory(groups, reveal, seed, budgets=(0,10,25,50), batch_size=5, strategy='decision_diagnostic'):
    groups={k:list(v) for k,v in groups.items()}
    validate_roles(groups)
    if strategy not in STRATEGIES: raise ValueError(f'unknown strategy: {strategy}')
    if batch_size!=5: raise ValueError('The frozen implementation uses static batches of five')
    if not budgets or 0 not in budgets or len(set(budgets))!=len(budgets):
        raise ValueError('budgets must be unique and include zero')
    if any(not isinstance(b,int) or b<0 or b%5!=0 for b in budgets):
        raise ValueError('budgets must be nonnegative multiples of five')
    if max(budgets)>len(groups['query_pool']): raise ValueError('budget exceeds query pool')
    pool={r.sample:r for r in groups['query_pool']};seen=set()
    def checked_reveal(sample):
        if sample not in pool or sample in seen: raise ValueError('query outside pool or duplicate')
        record=reveal(sample); masked=pool[sample]
        if record.sample!=sample or record.masked()!=masked or record.label not in (0,1):
            raise ValueError('revealed record metadata or label does not match selected query')
        seen.add(sample);return record
    return trajectory(groups['source'],groups['query_pool'],groups['evaluation'],groups['audit'],
                      checked_reveal,strategy,seed,tuple(sorted(budgets)))
