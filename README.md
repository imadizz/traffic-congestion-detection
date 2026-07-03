# Improving Traffic Congestion Detection through Multi-Source Fusion of Camera and GPS Data

MSc Artificial Intelligence dissertation project
BSBI Berlin / University for the Creative Arts
Aditya Lokhande (Q1093411) — Supervisor: Dr. Vincent English — July 2026

## What this project does

The project investigates whether fusing camera-based traffic observations
with GPS movement features improves congestion classification, and whether
better detection translates into journey time savings through V2V
cooperative rerouting. It uses two public driving datasets:

- **BDD100K** — 70,000 US dashcam frames (Berkeley)
- **KITTI** — 7,481 German urban frames (Karlsruhe)

Per-frame object counts come from the official dataset annotations. Counts
are converted to a congestion percentage with Passenger Car Unit weighting
(Highway Capacity Manual, 2010) and discretised into four classes
(FREE_FLOW, MODERATE, HEAVY, GRIDLOCK). The camera congestion score used
as a classifier feature carries Gaussian detector noise (std 6pp). GPS
features are synthesised from scene type, weather and time of day only,
deliberately independent of the congestion label, so they cannot leak
label information into the classifier. An adaptive scalar Kalman filter
(weather-dependent measurement noise) smooths the camera score.

## Research questions

1. How effectively can congestion be detected from camera-based driving
   datasets?
2. Does combining camera indicators with GPS data improve accuracy?
3. Can improved detection support more efficient route replanning in a
   simulated V2V setting?

## Key results

| Metric | Value |
|--------|-------|
| Gradient Boosting, full fusion (27 features) | 99.4% |
| MLP full fusion | 99.5% |
| Random Forest full fusion | 96.9% |
| Camera-only baseline (GB, 9 features) | 99.5% |
| Fusion difference | -0.15 pp (McNemar p = 0.001: the small decrease from adding non-camera features is significant) |
| Top feature | car count (47.0%), camera score (34.2%) |
| KITTI zero-shot transfer | 99.8% (shared PCU labelling rule, see dissertation 4.3.1) |
| V2V journey time saving, 0% packet loss | 18.6% |
| V2V journey time saving, 60% packet loss | 7.6% |

The honest answer to RQ2 is negative: label-independent GPS synthesis adds
no measurable accuracy on top of camera evidence. Because the labels are
derived from a PCU rule over annotated counts that the classifier partly
observes, the high accuracies mainly show that the rule is recoverable;
the dissertation discusses this limitation explicitly (Section 4.3.3).

## Repository layout

```
config.py            all settings in one place (seed, split, hyperparameters)
pcu_scoring.py       PCU weighting and the labelling rule
kalman_filter.py     adaptive scalar Kalman filter
build_features.py    27-feature matrices for BDD100K and KITTI
run_experiments.py   trains everything, writes all result tables
make_figures.py      generates the dissertation figures
yolo_validation.py   YOLOv8m detection validation on sample frames
v2v_simulation.py    probabilistic V2V rerouting simulation (no SUMO needed)
data/                per-frame CSVs for both datasets (included)
results/             outputs of run_experiments.py
figures/             outputs of make_figures.py
```

## Reproducing the dissertation results

```bash
pip install -r requirements.txt
python run_experiments.py     # about 20-30 minutes on a laptop CPU
python make_figures.py        # a few seconds, uses cached models
python v2v_simulation.py --packet_loss 0 0.2 0.4 0.6 --n_trips 500
```

Everything is seeded (seed 42 in `config.py`), so repeated runs give
identical numbers. `results/summary.txt` lists every value reported in
Chapter Four of the dissertation.

## Known issue

scikit-learn 1.7.2 raises a `TypeError` when `MLPClassifier` is used with
`early_stopping=True` on float64 arrays. The workaround (used here) is
`early_stopping=False` with explicit float64 casting.

## Citation

Lokhande, A. (2026). Improving Traffic Congestion Detection through
Multi-Source Fusion of Camera and GPS Data. MSc Dissertation,
BSBI Berlin / UCA.
