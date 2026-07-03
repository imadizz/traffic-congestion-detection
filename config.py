"""Shared configuration for all experiments.

Everything that affects reproducibility lives here: the random seed,
the train/test split, classifier hyperparameters and the Kalman filter
noise settings. All scripts import from this file.
"""

RANDOM_SEED = 42
TEST_SPLIT  = 0.25   # 75/25 stratified split -> 52,500 train / 17,500 test

# Data files (paths relative to the repository root)
BDD_CSV   = 'data/bdd100k_fused_results.csv'
KITTI_CSV = 'data/kitti_congestion_results.csv'

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# Congestion classes in severity order
CLASS_ORDER = ['FREE_FLOW', 'MODERATE', 'HEAVY', 'GRIDLOCK']

# Categorical metadata in BDD100K
WEATHER_CATS = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy']
SCENE_CATS   = ['city street', 'highway', 'residential']

# PCU weights (Highway Capacity Manual, 2010)
PCU_WEIGHTS = {
    'car': 1.0, 'truck': 2.5, 'bus': 3.0,
    'motorcycle': 0.5, 'bicycle': 0.5, 'person': 0.3,
}

# Congestion percentage = PCU score * 5, capped at 100.
# Class thresholds on that percentage:
CONGESTION_THRESHOLDS = {
    'FREE_FLOW': (0, 30),
    'MODERATE':  (30, 60),
    'HEAVY':     (60, 85),
    'GRIDLOCK':  (85, 101),
}

# Detector noise applied to the camera congestion score (std, percentage points)
CAM_NOISE_STD = 6.0

# GPS synthesis: free-flow baseline speed per scene (km/h) and weather factors.
# These depend only on scene / weather / time of day, never on the label.
SCENE_SPEED = {'highway': 100.0, 'city street': 45.0, 'residential': 35.0}
WEATHER_FACTOR = {
    'clear': 1.0, 'partly cloudy': 0.98, 'overcast': 0.95,
    'rainy': 0.88, 'snowy': 0.82, 'foggy': 0.80,
}

# Classifier hyperparameters
RF_PARAMS  = dict(n_estimators=200, max_depth=20, max_features='sqrt')
GB_PARAMS  = dict(n_estimators=200, max_depth=6, learning_rate=0.1)
MLP_PARAMS = dict(hidden_layer_sizes=(128, 64, 32), activation='relu',
                  max_iter=300, early_stopping=False)
# Note: early_stopping=False works around a scikit-learn 1.7.2 bug where
# early_stopping=True raises a TypeError with float64 input arrays.
