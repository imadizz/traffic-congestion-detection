"""PCU weighting and congestion class assignment.

Object counts are converted to a Passenger Car Unit (PCU) load using the
Highway Capacity Manual (2010) weights, scaled to a 0-100 congestion
percentage and discretised into four classes. This is the rule that
produced the labels stored in the data CSVs.
"""

import config


def pcu_score(counts):
    """PCU-weighted load from a {category: count} dict."""
    return sum(counts.get(cat, 0) * w for cat, w in config.PCU_WEIGHTS.items())


def congestion_percentage(score):
    """Linear scaling: 20 PCU or more saturates at 100 percent."""
    return min(100.0, score * 5.0)


def congestion_class(percentage):
    for label, (lo, hi) in config.CONGESTION_THRESHOLDS.items():
        if lo <= percentage < hi:
            return label
    return 'GRIDLOCK'
