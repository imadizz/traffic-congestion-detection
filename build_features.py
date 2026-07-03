"""Build the 27-feature matrices for BDD100K and KITTI.

Feature layout (27 per frame):
    Camera  (9): noisy congestion score + 8 per-category object counts
    GPS     (5): speed, acceleration, speed variance, stop duration,
                 normalised speed  (synthesised, independent of the label)
    Fusion  (4): Kalman state, gain, innovation, prior uncertainty
    Context (9): weather one-hot (6) + scene one-hot (3)

The GPS features are generated only from scene type, weather and time of
day plus Gaussian noise. They deliberately carry no information about the
congestion label, so any accuracy they add reflects genuine extra signal.
A single random generator (seed 42) is consumed in row order, so the
matrices are exactly reproducible: build the BDD100K matrix first, then
the KITTI matrix, as run_experiments.py does.
"""

import csv
import numpy as np

import config
from kalman_filter import AdaptiveKalman

FEATURE_NAMES = (
    ['cam_score', 'person', 'car', 'bike', 'bus', 'train', 'motor', 'rider', 'truck'] +
    ['gps_speed', 'gps_accel', 'gps_variance', 'stop_duration', 'norm_speed'] +
    ['kalman_state', 'kalman_gain', 'kalman_innovation', 'kalman_P'] +
    [f'weather_{w.replace(" ", "_")}' for w in config.WEATHER_CATS] +
    [f'scene_{s.replace(" ", "_")}' for s in config.SCENE_CATS]
)

CAMERA_COLS = list(range(9))
GPS_COLS    = list(range(9, 14))
KALMAN_COLS = list(range(14, 18))

# one generator shared by both datasets, consumed in row order
_rng = np.random.default_rng(config.RANDOM_SEED)


def _gps_features(scene, weather, timeofday):
    """Synthesise GPS features from scene/weather/time only."""
    base = config.SCENE_SPEED.get(scene, 45.0)
    factor = config.WEATHER_FACTOR.get(weather, 0.95)
    if timeofday == 'night':
        factor *= 0.95
    speed = max(3.0, _rng.normal(base * factor, 8.0))
    accel = _rng.normal(0.0, 1.5)
    variance = abs(_rng.normal(0.0, 4.0))
    stop_dur = max(0.0, _rng.normal(6.0, 4.0))
    return [speed, accel, variance, stop_dur, speed / 120.0]


def build_bdd():
    """Feature matrix and labels for the 70,000 BDD100K frames."""
    rows = list(csv.DictReader(open(config.BDD_CSV)))
    kalman = AdaptiveKalman()
    X, y = [], []
    for r in rows:
        cam_score = float(r['cam_score'])
        weather, scene, tod = r['weather'], r['scene'], r['timeofday']

        cam = [cam_score,
               float(r['person']), float(r['car']), float(r['bike']),
               float(r['bus']), float(r['train']), float(r['motor']),
               float(r['rider']), float(r['truck'])]
        gps = _gps_features(scene, weather, tod)
        k = kalman.update(cam_score / 100.0, weather=weather, timeofday=tod)
        fus = [k['state'], k['gain'], k['innovation'], k['P_prior']]
        ctx = [1.0 if weather == w else 0.0 for w in config.WEATHER_CATS] + \
              [1.0 if scene == s else 0.0 for s in config.SCENE_CATS]

        X.append(cam + gps + fus + ctx)
        y.append(r['level'])
    return np.array(X, dtype=np.float64), y, rows


def build_kitti():
    """Feature matrix and labels for the 7,481 KITTI frames.

    KITTI has no weather/scene/time metadata, so clear daytime city-street
    conditions are assumed and the context block is zero. The camera score
    is the annotation-derived congestion percentage plus the same detector
    noise used for BDD100K.
    """
    rows = list(csv.DictReader(open(config.KITTI_CSV)))
    kalman = AdaptiveKalman()
    X, y = [], []
    for r in rows:
        cpct = float(r['congestion_pct'])
        cam_score = float(np.clip(cpct + _rng.normal(0, config.CAM_NOISE_STD), 0, 100))

        cam = [cam_score,
               float(r['person']), float(r['car']), float(r['bike']),
               float(r['bus']), float(r['train']), float(r['motor']),
               float(r['rider']), float(r['truck'])]
        gps = _gps_features('city street', 'clear', 'daytime')
        k = kalman.update(cam_score / 100.0, weather='clear', timeofday='daytime')
        fus = [k['state'], k['gain'], k['innovation'], k['P_prior']]
        ctx = [0.0] * 9

        X.append(cam + gps + fus + ctx)
        y.append(r['level'])
    return np.array(X, dtype=np.float64), y, rows
