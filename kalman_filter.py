"""Adaptive scalar Kalman filter for the camera congestion score.

The measurement noise R changes with weather and time of day, so the
filter trusts the camera less when visibility is poor.
"""


class AdaptiveKalman:
    """Scalar Kalman filter with weather-adaptive measurement noise.

    State x is the estimated normalised congestion score (0 to 1).
    The measurement z is the noisy camera score, also 0 to 1.
    """

    R_TABLE = {
        ('clear',    'daytime'): 0.05,
        ('overcast', 'daytime'): 0.15,
        ('rainy',    'daytime'): 0.15,
        ('foggy',    'daytime'): 0.20,
        ('snowy',    'daytime'): 0.15,
        ('clear',    'night'):   0.20,
        ('overcast', 'night'):   0.20,
        ('rainy',    'night'):   0.20,
        ('foggy',    'night'):   0.20,
    }
    R_DEFAULT = 0.10
    Q = 0.01   # process noise: how quickly congestion can change between frames

    def __init__(self):
        self.x = 0.0
        self.P = 1.0

    def update(self, z, weather='clear', timeofday='daytime'):
        """One predict + update step. Returns the filter internals used
        as features: state, gain, innovation and prior uncertainty."""
        R = self.R_TABLE.get((weather, timeofday), self.R_DEFAULT)

        # predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # update
        K = P_pred / (P_pred + R)
        innovation = z - x_pred
        self.x = x_pred + K * innovation
        self.P = (1.0 - K) * P_pred

        return {'state': self.x, 'gain': K,
                'innovation': innovation, 'P_prior': P_pred}

    def reset(self):
        self.x = 0.0
        self.P = 1.0
