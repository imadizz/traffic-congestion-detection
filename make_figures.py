"""Generate the dissertation result figures from the cached experiment
outputs. Run run_experiments.py first.

    python make_figures.py

Figure files map to the dissertation as follows:
    fig14_confusion_matrix_best_model.png   Figure 4.1
    fig3_accuracy_by_weather.png            Figure 4.2
    fig7_fusion_improvement_heatmap.png     Figure 4.3
    fig13_feature_importance.png            Figure 4.6
    fig32_accuracy_by_condition.png         Figure 4.7
    fig18_ml_kitti_confusion_matrix.png     Figure 4.8
    fig2_accuracy_comparison.png            Figure A1
    fig29_overall_confusion.png             Figure A2
    fig4_accuracy_by_time.png               Figure A3
"""

import os

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

import config
from build_features import FEATURE_NAMES

cache = joblib.load(os.path.join(config.RESULTS_DIR, 'cache.joblib'))
le, class_ids = cache['le'], cache['class_ids']
yte, preds, p_cam = cache['yte'], cache['preds'], cache['p_cam']
weather_te, tod_te = cache['weather_te'], cache['tod_te']
imp, yk, pk = cache['imp'], cache['yk'], cache['pk']

p_full = preds['GradientBoosting']
ORDER = config.CLASS_ORDER
os.makedirs(config.FIGURES_DIR, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, name), dpi=150,
                bbox_inches='tight')
    plt.close()
    print('saved', name)


def confusion_fig(y_true, y_pred, title, cmap, fname):
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_n = cm / cm.sum()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_n, cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            colour = 'white' if cm_n[i, j] > 0.25 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_n[i, j]*100:.1f}%)',
                    ha='center', va='center', fontsize=11, color=colour,
                    fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(4))
    ax.set_xticklabels([c.replace('_', '\n') for c in ORDER])
    ax.set_yticks(range(4))
    ax.set_yticklabels([c.replace('_', '\n') for c in ORDER])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label (PCU rule-based)')
    ax.set_title(title)
    save(fname)


# Figure 4.1 and A2: BDD100K confusion matrices (GB full fusion)
acc_gb = accuracy_score(yte, p_full)
confusion_fig(yte, p_full,
              f'Gradient Boosting (Full Fusion) — Accuracy: {acc_gb*100:.2f}%\n'
              f'BDD100K Test Set (n={len(yte):,})',
              'Blues', 'fig14_confusion_matrix_best_model.png')
confusion_fig(yte, p_full,
              f'Gradient Boosting Full-Fusion Confusion Matrix\n'
              f'BDD100K test set — Accuracy {acc_gb*100:.2f}%',
              'Blues', 'fig29_overall_confusion.png')

# Figure 4.8: KITTI confusion matrix
acc_k = accuracy_score(yk, pk)
confusion_fig(yk, pk,
              f'Gradient Boosting on KITTI Data — Accuracy: {acc_k*100:.2f}%\n'
              f'(Model trained on BDD100K, tested on KITTI)',
              'Oranges', 'fig18_ml_kitti_confusion_matrix.png')

# Figure 4.2: accuracy by weather, all classifiers
weathers = [w for w in config.WEATHER_CATS if (weather_te == w).sum() > 0]
fig, ax = plt.subplots(figsize=(11, 6))
series = [('Gradient Boosting', p_full, '#2E75B6'),
          ('Random Forest', preds['RandomForest'], '#70AD47'),
          ('MLP', preds['MLP'], '#ED7D31')]
for off, (name, pp, colour) in enumerate(series):
    vals = [accuracy_score(yte[weather_te == w], pp[weather_te == w]) * 100
            for w in weathers]
    ax.bar(np.arange(len(weathers)) + (off - 1) * 0.25, vals, 0.25,
           label=name, color=colour)
ax.set_xticks(range(len(weathers)))
ax.set_xticklabels([w.title() for w in weathers])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(90, 100.5)
ax.set_title('Classification Accuracy by Weather Condition (BDD100K test set)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save('fig3_accuracy_by_weather.png')

# Figure A3: accuracy by time of day
tods = ['daytime', 'dawn/dusk', 'night']
fig, ax = plt.subplots(figsize=(9, 6))
for off, (name, pp, colour) in enumerate(series):
    vals = [accuracy_score(yte[tod_te == t], pp[tod_te == t]) * 100
            if (tod_te == t).sum() else 0 for t in tods]
    ax.bar(np.arange(len(tods)) + (off - 1) * 0.25, vals, 0.25,
           label=name, color=colour)
ax.set_xticks(range(len(tods)))
ax.set_xticklabels([t.title() for t in tods])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(90, 100.5)
ax.set_title('Classification Accuracy by Time of Day (BDD100K test set)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save('fig4_accuracy_by_time.png')

# Figure 4.3: fusion minus camera-only recall, per class and weather
H = np.zeros((4, len(weathers)))
for ci, cname in enumerate(ORDER):
    cid = le.transform([cname])[0]
    for wi, w in enumerate(weathers):
        mask = (weather_te == w) & (yte == cid)
        if mask.sum() == 0:
            continue
        H[ci, wi] = ((p_full[mask] == yte[mask]).mean()
                     - (p_cam[mask] == yte[mask]).mean()) * 100
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(H, cmap='RdBu', vmin=-2, vmax=2)
plt.colorbar(im, ax=ax, label='Fusion - Camera-only (pp)')
for i in range(4):
    for j in range(len(weathers)):
        ax.text(j, i, f'{H[i, j]:+.2f}', ha='center', va='center', fontsize=9)
ax.set_xticks(range(len(weathers)))
ax.set_xticklabels([w.title() for w in weathers], rotation=30)
ax.set_yticks(range(4))
ax.set_yticklabels(ORDER)
ax.set_title('Full Fusion vs Camera-Only: Per-Class Recall Difference\n'
             '(percentage points, Gradient Boosting)')
save('fig7_fusion_improvement_heatmap.png')

# Figure 4.6: feature importances
order = np.argsort(imp)[::-1][:20]
names = [FEATURE_NAMES[i] for i in order]
vals = [imp[i] for i in order]


def colour_for(name):
    if name in FEATURE_NAMES[:9]:
        return 'tomato'
    if name in FEATURE_NAMES[9:14]:
        return 'steelblue'
    if name.startswith('kalman'):
        return 'mediumseagreen'
    return 'silver'


fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(names)), vals[::-1], color=[colour_for(n) for n in names[::-1]])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names[::-1], fontsize=10)
ax.set_xlabel('Feature Importance')
ax.set_title('Top 20 Feature Importances\n(Gradient Boosting)')
ax.legend(handles=[
    mpatches.Patch(color='tomato', label='Camera (score + counts)'),
    mpatches.Patch(color='steelblue', label='GPS (independent)'),
    mpatches.Patch(color='mediumseagreen', label='Kalman fusion'),
    mpatches.Patch(color='silver', label='Context')], loc='lower right')
save('fig13_feature_importance.png')

# Figure A1: accuracy comparison bar chart
acc_cam = accuracy_score(yte, p_cam)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Gradient\nBoosting', 'Random\nForest', 'MLP',
          'Camera-only\n(GB, 9 feat.)']
vals = [acc_gb * 100,
        accuracy_score(yte, preds['RandomForest']) * 100,
        accuracy_score(yte, preds['MLP']) * 100,
        acc_cam * 100]
bars = ax.bar(labels, vals, color=['#2E75B6', '#70AD47', '#ED7D31', '#FFC000'])
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f'{v:.2f}%',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(90, 100.5)
ax.axhline(34.7, color='gray', ls='--', label='Majority baseline (34.7%)')
ax.set_title('Classifier Accuracy Comparison (BDD100K test set, n=17,500)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save('fig2_accuracy_comparison.png')

# Figure 4.7: per-class recall, BDD100K vs KITTI
rec_b = recall_score(yte, p_full, labels=class_ids, average=None, zero_division=0)
rec_k = recall_score(yk, pk, labels=class_ids, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(4)
ax.bar(x - 0.2, rec_b * 100, 0.4,
       label=f'BDD100K test ({acc_gb*100:.1f}%)', color='#2E75B6')
ax.bar(x + 0.2, rec_k * 100, 0.4,
       label=f'KITTI zero-shot ({acc_k*100:.1f}%)', color='#ED7D31')
for i in range(4):
    ax.text(x[i] - 0.2, rec_b[i] * 100 + 0.3, f'{rec_b[i]*100:.1f}',
            ha='center', fontsize=9)
    ax.text(x[i] + 0.2, rec_k[i] * 100 + 0.3, f'{rec_k[i]*100:.1f}',
            ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(ORDER)
ax.set_ylabel('Per-class Recall (%)')
ax.set_ylim(80, 102)
ax.set_title('Per-Class Recall: BDD100K Test Set vs KITTI Zero-Shot Transfer\n'
             '(Gradient Boosting, full fusion)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save('fig32_accuracy_by_condition.png')

print('All figures written to figures/')
