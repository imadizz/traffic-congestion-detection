"""Main experiment script. Reproduces every result reported in the
dissertation: classifier accuracies, per-class metrics, the ablation
study, the weather breakdown, the McNemar significance test, feature
importances and the KITTI cross-dataset transfer.

Usage (from the repository root, takes roughly 20-30 minutes):
    python run_experiments.py

Outputs:
    results/summary.txt            human-readable summary of everything
    results/ablation.csv           Table E1 in the dissertation
    results/per_class.csv          Table 4.2
    results/weather.csv            Table 4.3
    results/feature_importance.csv Table 4.7 / Figure 4.6
    results/mcnemar.json           significance test in Section 4.2
    results/cache.joblib           fitted models + predictions for make_figures.py
"""

import json
import os
from math import comb

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from build_features import (CAMERA_COLS, GPS_COLS, KALMAN_COLS,
                            FEATURE_NAMES, build_bdd, build_kitti)


def make_pipeline(clf):
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def gb():
    return make_pipeline(GradientBoostingClassifier(
        random_state=config.RANDOM_SEED, **config.GB_PARAMS))


def mcnemar_test(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness vectors."""
    b = int(np.sum(~correct_a & correct_b))   # only B right
    c = int(np.sum(correct_a & ~correct_b))   # only A right
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'p_value': 1.0}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return {'b': b, 'c': c, 'p_value': p}


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report = []

    def log(line=''):
        print(line)
        report.append(line)

    # ---------- data ----------
    log('Building BDD100K features...')
    X, y_raw, rows = build_bdd()
    le = LabelEncoder()
    le.fit(config.CLASS_ORDER)
    y = le.transform(y_raw)
    class_ids = [le.transform([c])[0] for c in config.CLASS_ORDER]
    log(f'  frames: {len(y)}, features: {X.shape[1]}')

    idx = np.arange(len(y))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y)
    log(f'  train: {len(ytr)}, test: {len(yte)}')

    weather_te = np.array([rows[i]['weather'] for i in ite])
    tod_te = np.array([rows[i]['timeofday'] for i in ite])

    # ---------- three classifiers, full 27 features ----------
    log('\nTraining classifiers (full fusion, 27 features)...')
    models = {
        'GradientBoosting': gb(),
        'RandomForest': make_pipeline(RandomForestClassifier(
            random_state=config.RANDOM_SEED, n_jobs=-1, **config.RF_PARAMS)),
        'MLP': make_pipeline(MLPClassifier(
            random_state=config.RANDOM_SEED, **config.MLP_PARAMS)),
    }
    preds = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds[name] = model.predict(Xte)
        a = accuracy_score(yte, preds[name])
        P, R, F, _ = precision_recall_fscore_support(
            yte, preds[name], labels=class_ids, average='macro', zero_division=0)
        log(f'  {name:16s} acc={a*100:.2f}%  macroP={P*100:.2f}  '
            f'macroR={R*100:.2f}  macroF1={F*100:.2f}')

    p_full = preds['GradientBoosting']

    # ---------- ablation: feature groups (Table E1) ----------
    log('\nAblation study (Gradient Boosting)...')
    subsets = {
        'camera_only':   CAMERA_COLS,
        'camera_kalman': CAMERA_COLS + KALMAN_COLS,
        'camera_gps':    CAMERA_COLS + GPS_COLS,
        'full_fusion':   list(range(27)),
    }
    abl_preds = {'full_fusion': p_full}
    with open(os.path.join(config.RESULTS_DIR, 'ablation.csv'), 'w') as f:
        f.write('configuration,n_features,accuracy,macro_precision,macro_recall,macro_f1\n')
        for name, cols in subsets.items():
            if name != 'full_fusion':
                m = gb()
                m.fit(Xtr[:, cols], ytr)
                abl_preds[name] = m.predict(Xte[:, cols])
            pp = abl_preds[name]
            a = accuracy_score(yte, pp)
            P, R, F, _ = precision_recall_fscore_support(
                yte, pp, labels=class_ids, average='macro', zero_division=0)
            f.write(f'{name},{len(cols)},{a*100:.2f},{P*100:.2f},{R*100:.2f},{F*100:.2f}\n')
            log(f'  {name:14s} ({len(cols):2d} feats) acc={a*100:.2f}%  macroF1={F*100:.2f}')

    p_cam = abl_preds['camera_only']

    # ---------- McNemar: camera-only vs full fusion ----------
    mc = mcnemar_test(p_cam == yte, p_full == yte)
    log(f'\nMcNemar (camera-only vs full fusion): only-fusion-right={mc["b"]}, '
        f'only-camera-right={mc["c"]}, p={mc["p_value"]:.4f}')
    with open(os.path.join(config.RESULTS_DIR, 'mcnemar.json'), 'w') as f:
        json.dump(mc, f, indent=2)

    # ---------- per-class results (Table 4.2) ----------
    P, R, F, S = precision_recall_fscore_support(
        yte, p_full, labels=class_ids, zero_division=0)
    with open(os.path.join(config.RESULTS_DIR, 'per_class.csv'), 'w') as f:
        f.write('class,precision,recall,f1,support\n')
        for i, c in enumerate(config.CLASS_ORDER):
            f.write(f'{c},{P[i]*100:.1f},{R[i]*100:.1f},{F[i]*100:.1f},{int(S[i])}\n')
    log('\nPer-class (GB full fusion):')
    for i, c in enumerate(config.CLASS_ORDER):
        log(f'  {c:10s} P={P[i]*100:.1f} R={R[i]*100:.1f} '
            f'F1={F[i]*100:.1f} n={int(S[i])}')

    # ---------- accuracy by weather (Table 4.3) ----------
    log('\nAccuracy by weather (GB full fusion):')
    weather_all = np.array([r['weather'] for r in rows])
    with open(os.path.join(config.RESULTS_DIR, 'weather.csv'), 'w') as f:
        f.write('weather,accuracy,test_frames,total_frames\n')
        for w in config.WEATHER_CATS + ['undefined']:
            mask = weather_te == w
            if mask.sum() == 0:
                continue
            a = accuracy_score(yte[mask], p_full[mask])
            f.write(f'{w},{a*100:.1f},{int(mask.sum())},{int((weather_all == w).sum())}\n')
            log(f'  {w:14s} {a*100:5.1f}%  (test n={int(mask.sum())})')

    # ---------- feature importances (Table 4.7) ----------
    imp = models['GradientBoosting'].named_steps['clf'].feature_importances_
    order = np.argsort(imp)[::-1]
    with open(os.path.join(config.RESULTS_DIR, 'feature_importance.csv'), 'w') as f:
        f.write('rank,feature,importance\n')
        for rank, i in enumerate(order, 1):
            f.write(f'{rank},{FEATURE_NAMES[i]},{imp[i]:.6f}\n')
    groups = {'camera': imp[0:9].sum(), 'gps': imp[9:14].sum(),
              'kalman': imp[14:18].sum(), 'context': imp[18:27].sum()}
    log('\nTop 8 features (GB):')
    for i in order[:8]:
        log(f'  {FEATURE_NAMES[i]:18s} {imp[i]*100:5.1f}%')
    log('Group totals: ' + ', '.join(f'{k}={v*100:.1f}%' for k, v in groups.items()))

    # ---------- KITTI cross-dataset transfer ----------
    log('\nKITTI zero-shot transfer...')
    Xk, yk_raw, _ = build_kitti()
    yk = le.transform(yk_raw)
    pk = models['GradientBoosting'].predict(Xk)
    acc_k = accuracy_score(yk, pk)
    rec_k = recall_score(yk, pk, labels=class_ids, average=None, zero_division=0)
    log(f'  KITTI accuracy: {acc_k*100:.2f}%  (n={len(yk)})')
    log('  per-class recall: ' + ', '.join(
        f'{c}={r*100:.1f}' for c, r in zip(config.CLASS_ORDER, rec_k)))

    # ---------- cache for make_figures.py ----------
    joblib.dump({
        'le': le, 'class_ids': class_ids,
        'yte': yte, 'preds': preds, 'p_cam': p_cam,
        'weather_te': weather_te, 'tod_te': tod_te,
        'imp': imp, 'yk': yk, 'pk': pk,
    }, os.path.join(config.RESULTS_DIR, 'cache.joblib'))

    with open(os.path.join(config.RESULTS_DIR, 'summary.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    log('\nDone. Outputs written to results/. '
        'Run make_figures.py next to generate the figures.')


if __name__ == '__main__':
    main()
