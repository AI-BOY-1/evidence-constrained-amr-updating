"""Prespecified B-cohort reexecution using verified author weights.

This is a migrated-runtime partial study, never exact full-paper reproduction.
"""
import ncs_paths  # portable I/O paths; original computational rules preserved
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='1',
                  OMP_NUM_THREADS='4', TF_ENABLE_ONEDNN_OPTS='0', TF_USE_LEGACY_KERAS='1')
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, average_precision_score,
                             f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss)
from msdeepamr_model_compatibility import load_spectrum_model

ROOT = Path(__file__).resolve().parent
BASE = ncs_paths.data_path('experiments/msdeepamr_reproduction')
REPO = BASE / 'repo'
RUN = BASE / 'reexecution_B_v3'
TASKS = [('e_coli', 'Ceftriaxone'), ('k_pneumoniae', 'Ceftriaxone'), ('s_aureus', 'Oxacillin')]
STRATEGIES = ['source_only', 'local_only', 'frozen_features', 'all_layers']


class Progress(keras.callbacks.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps({'training_config': self.config, 'epoch': epoch + 1,
                              'logs': {k: float(v) for k, v in (logs or {}).items()}}), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, result):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False, default=numpy_scalar), encoding='utf-8')
    temp.replace(path)


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def measure(y, prob):
    pred = (prob > .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, prob)
    return {'n': len(y), 'R': int(y.sum()), 'S': int((y == 0).sum()),
            'AUROC': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
            'AUPRC_trapezoid': float(auc(recall, precision)),
            'average_precision': float(average_precision_score(y, prob)),
            'Brier': float(brier_score_loss(y, prob)), 'F1': float(f1_score(y, pred, zero_division=0)),
            'MCC': float(matthews_corrcoef(y, pred)),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}


def get_data(species, drug):
    p = REPO / 'datasets/driams_b' / f'{species}_driams_b_bin3_2000_20000Da.csv'
    df = pd.read_csv(p)
    feature_columns = [str(i) for i in range(2000, 20000, 3)]
    mask = df[drug].notna() & df[feature_columns].notna().all(axis=1)
    frame = df[mask].copy()
    x = frame[feature_columns].to_numpy(np.float32)
    y = frame[drug].to_numpy(np.float32)
    assert np.isfinite(x).all() and np.isin(y, [0, 1]).all()
    assert (x >= 0).all()
    x = Normalizer(norm='max').fit_transform(x).reshape(len(x), 6000, 1)
    return p, frame, x, y, df.index[~mask].tolist()


def load_author(path):
    return load_spectrum_model(path)


def predict(model, x):
    return np.concatenate([model(x[i:i+16], training=False).numpy().reshape(-1)
                           for i in range(0, len(x), 16)]).astype(float)


def frozen_head(author):
    trunk = keras.Sequential(author.layers[:-4], name='frozen_author_trunk')
    for layer in trunk.layers: layer.trainable = False
    init = 'normal'
    reg = keras.regularizers.l2(.0001)
    return keras.Sequential([trunk,
         keras.layers.Dense(256, activation='relu', kernel_initializer=init, kernel_regularizer=reg),
         keras.layers.Dense(64, activation='relu', kernel_initializer=init, kernel_regularizer=reg),
         keras.layers.Dense(64, activation='relu', kernel_initializer=init, kernel_regularizer=reg),
         keras.layers.Dense(1, activation='sigmoid')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    lock = RUN / 'execution.lock'
    try: fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError: raise RuntimeError('Existing execution lock: inspect PID; do not duplicate.')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    manifest_path = RUN / 'manifest.json'
    manifest = {'status': 'preparing', 'pid': os.getpid(), 'expected_configs': 120, 'completed': 0, 'failed': 0,
                'exact_paper_reproduction': False, 'runtime': {'python': sys.version, 'tensorflow': tf.__version__,
                  'tf_keras': keras.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
                'scope': 'three_B_tasks_ten_local_stratified_splits_four_strategies',
                'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        assets = json.loads((ROOT / 'msdeepamr_assets.json').read_text())
        assert assets['status'] == 'assets_locked_models_ready'
        input_paths = [Path(__file__).resolve(), ROOT / 'MSDEEPAMR_REEXECUTION_PROTOCOL.md',
                       ROOT / 'msdeepamr_assets.json', ROOT / 'msdeepamr_model_compatibility.py',
                       BASE / 'runtime_preflight.json', BASE / 'runtime_patch.json',
                       BASE / 'env/Lib/site-packages/tf_keras/src/backend.py',
                       BASE / 'env/Lib/site-packages/tf_keras/src/utils/tf_utils.py']
        input_paths += [Path(m['materialized_path']) for m in assets['models']]
        input_paths += [Path(d['path']) for d in assets['datasets'] if d['cohort'] == 'driams_b']
        binding = {str(p): sha(p) for p in input_paths}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            assert previous['binding'] == binding, 'Input/code hash change: refuse mixed-version resume'
        saved_rows = [json.loads(p.read_text()) for p in RUN.glob('*_seed*_*.json')]
        assert all(r['binding'] == binding for r in saved_rows)
        manifest['completed'] = sum(r['status'] == 'complete' for r in saved_rows)
        manifest['failed'] = sum(r['status'] == 'failed' for r in saved_rows)
        manifest.update(binding=binding, status='running')
        write(manifest_path, manifest)
        processed = 0
        for species, drug in TASKS:
            p, frame, x, y, excluded = get_data(species, drug)
            model_path = BASE / 'models' / (species + '_' + drug.lower() + '.h5')
            # Full source predictions are descriptive, not target-label tuned.
            whole_path = RUN / f'{species}_source_full.json'
            if not whole_path.exists():
                keras.backend.clear_session()
                source = load_author(model_path)
                prob = predict(source, x)
                write(whole_path, {'binding': binding, 'species': species, 'drug': drug,
                                  'codes': frame.code.astype(str).tolist(), 'original_rows': frame.index.tolist(),
                                  'y': y.astype(int).tolist(), 'probability': prob.tolist(),
                                  'excluded_missing_rows': excluded, 'metrics': measure(y, prob)})
                del source; gc.collect()
            for seed in range(10):
                tr, ts = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
                assert not set(tr) & set(ts)
                for strategy in STRATEGIES:
                    out = RUN / f'{species}_seed{seed}_{strategy}.json'
                    if out.exists():
                        saved = json.loads(out.read_text())
                        assert saved['binding'] == binding
                        continue
                    if args.limit is not None and processed >= args.limit:
                        manifest['status'] = 'partial_execution_limit'
                        return
                    keras.backend.clear_session(); gc.collect()
                    keras.utils.set_random_seed(20261011 + seed)
                    row = {'binding': binding, 'species': species, 'drug': drug, 'seed': seed,
                           'strategy': strategy, 'train_rows': frame.index[tr].tolist(),
                           'test_rows': frame.index[ts].tolist(), 'train_codes': frame.code.iloc[tr].astype(str).tolist(),
                           'test_codes': frame.code.iloc[ts].astype(str).tolist(),
                           'train_y': y[tr].astype(int).tolist(), 'test_y': y[ts].astype(int).tolist(),
                           'excluded_missing_rows': excluded, 'started_utc': datetime.now(timezone.utc).isoformat()}
                    try:
                        author = load_author(model_path)
                        if strategy == 'source_only':
                            model = author; history = None
                        elif strategy == 'local_only':
                            model = keras.models.clone_model(author)
                            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy')
                            callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=3, min_lr=1e-6),
                                         keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                                         Progress(out.name)]
                            history = model.fit(x[tr], y[tr], epochs=100, batch_size=10, validation_split=.1,
                                                callbacks=callbacks, verbose=0)
                        else:
                            model = frozen_head(author) if strategy == 'frozen_features' else author
                            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7), loss='binary_crossentropy')
                            history = model.fit(x[tr], y[tr], epochs=10, batch_size=32, verbose=0,
                                                callbacks=[Progress(out.name)])
                        prob = predict(model, x[ts])
                        row.update(status='complete', probability=prob.tolist(), metrics=measure(y[ts], prob),
                                   history=history.history if history else {}, epochs_run=len(history.epoch) if history else 0,
                                   trainable_parameters=int(sum(np.prod(v.shape) for v in model.trainable_weights)))
                        del model, author, history
                    except Exception:
                        row.update(status='failed', error=traceback.format_exc())
                        print(row['error'], flush=True)
                    row['ended_utc'] = datetime.now(timezone.utc).isoformat()
                    write(out, row)
                    processed += 1
                    saved_rows = [json.loads(p.read_text()) for p in RUN.glob('*_seed*_*.json')]
                    manifest['completed'] = sum(r['status'] == 'complete' for r in saved_rows)
                    manifest['failed'] = sum(r['status'] == 'failed' for r in saved_rows)
                    manifest['last_config'] = out.name
                    write(manifest_path, manifest)
                    print(json.dumps({'config': out.name, 'status': row['status'], 'metrics': row.get('metrics'),
                                      'completed': manifest['completed'], 'failed': manifest['failed']}), flush=True)
                    if row['status'] == 'failed':
                        raise RuntimeError('Stopped at the first failed configuration; preserve it and diagnose before proceeding.')
        manifest['status'] = 'complete_pending_validation' if manifest['failed'] == 0 else 'completed_with_failed_configs'
    except Exception:
        manifest.update(status='pipeline_failed', error=traceback.format_exc())
        raise
    finally:
        manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(manifest_path, manifest)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
