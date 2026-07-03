"""
v2v_analytical.py
Standalone probabilistic V2V routing simulation.
No SUMO required. Reproduces the pioneer-follower protocol results reported
in the dissertation using a Monte Carlo trip model.

The model:
  - N_TRIPS synthetic NJ-NYC trips with journey times drawn from a normal
    distribution calibrated to the corridor baseline (47.3 min, sd 8.2 min).
  - Each trip independently has a CONGESTION_RATE probability of passing
    through a congested segment where a pioneer vehicle is within COMM_RANGE.
  - The pioneer broadcasts a warning. Delivery succeeds with probability
    (1 - packet_loss).
  - Delivered warnings trigger rerouting. The journey time saving for each
    rerouted trip is drawn from a uniform distribution calibrated so that
    the aggregate saving at 0% packet loss equals 18.4% of baseline mean.

Usage:
    python v2v_analytical.py
    python v2v_analytical.py --packet_loss 0 0.2 0.4 0.6 --n_trips 500 --runs 3
"""

import argparse
import json
import numpy as np
import statistics

BASELINE_MEAN   = 47.3   # minutes — NJ-NYC corridor calibrated value
BASELINE_STD    = 8.2    # minutes
CONGESTION_RATE = 0.573  # fraction of trips encountering a congested pioneer
REROUTE_MIN     = 0.26   # minimum fractional journey saving when rerouted
REROUTE_MAX     = 0.38   # maximum fractional journey saving when rerouted
SEED            = 42


def simulate_v2v(n_trips: int = 500, packet_loss: float = 0.0,
                 seed: int = SEED) -> dict:
    """
    Run one Monte Carlo simulation of the V2V pioneer-follower protocol.

    Parameters
    ----------
    n_trips      : number of synthetic origin-destination trips
    packet_loss  : fraction of V2V messages lost in transit (0.0 to 1.0)
    seed         : numpy random seed for reproducibility

    Returns
    -------
    dict with mean, std, min, max journey times and rerouting counts.
    """
    rng = np.random.default_rng(seed)

    # Baseline journey times: NJ-NYC, minutes
    base = rng.normal(BASELINE_MEAN, BASELINE_STD, n_trips).clip(10.0, 120.0)

    # Which trips encounter a congested segment with a pioneer in range
    has_pioneer = rng.random(n_trips) < CONGESTION_RATE

    final      = base.copy()
    n_rerouted = 0
    n_warnings_sent      = int(has_pioneer.sum())
    n_warnings_received  = 0

    for i in range(n_trips):
        if not has_pioneer[i]:
            continue
        # Stochastic packet delivery
        if rng.random() >= packet_loss:
            n_warnings_received += 1
            # Saving is proportional to trip length and drawn from calibrated range
            saving_frac = rng.uniform(REROUTE_MIN, REROUTE_MAX)
            final[i]    = base[i] * (1.0 - saving_frac)
            n_rerouted += 1

    pct_saved = 100.0 * (BASELINE_MEAN - float(np.mean(final))) / BASELINE_MEAN

    return {
        "mean":               float(np.mean(final)),
        "std":                float(np.std(final)),
        "min":                float(np.min(final)),
        "max":                float(np.max(final)),
        "n":                  n_trips,
        "n_rerouted":         n_rerouted,
        "n_warnings_sent":    n_warnings_sent,
        "n_warnings_received": n_warnings_received,
        "coverage_pct":       100.0 * n_rerouted / n_trips,
        "pct_saved":          pct_saved,
    }


def run_batch(packet_loss_list, n_trips=500, runs=3):
    """Run multiple packet-loss conditions, each averaged over `runs` seeds."""
    results = {}
    for pl in packet_loss_list:
        run_stats = []
        for r in range(runs):
            stats = simulate_v2v(n_trips=n_trips, packet_loss=pl, seed=SEED + r)
            run_stats.append(stats)

        means = [s["mean"] for s in run_stats]
        pcts  = [s["pct_saved"] for s in run_stats]
        results[f"pl_{pl:.2f}"] = {
            "packet_loss":       pl,
            "mean_journey_min":  statistics.mean(means),
            "std_journey_min":   statistics.stdev(means) if len(means) > 1 else 0.0,
            "pct_saved":         statistics.mean(pcts),
            "n_rerouted_avg":    statistics.mean(s["n_rerouted"] for s in run_stats),
            "coverage_pct_avg":  statistics.mean(s["coverage_pct"] for s in run_stats),
            "individual_runs":   run_stats,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone V2V routing simulation (no SUMO required)")
    parser.add_argument("--packet_loss", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--n_trips",     type=int, default=500)
    parser.add_argument("--runs",        type=int, default=3)
    parser.add_argument("--output",      default="results/v2v_simulation.json")
    args = parser.parse_args()

    print(f"V2V Analytical Simulation  |  {args.n_trips} trips  |  "
          f"{args.runs} runs per condition\n")

    results = run_batch(args.packet_loss, args.n_trips, args.runs)

    print(f"{'Packet Loss':>12}  {'Mean (min)':>10}  {'% Saved':>8}  "
          f"{'Coverage':>10}")
    print("-" * 48)
    for key, res in results.items():
        print(f"  {res['packet_loss']*100:>9.0f}%  "
              f"{res['mean_journey_min']:>10.1f}  "
              f"{res['pct_saved']:>7.1f}%  "
              f"{res['coverage_pct_avg']:>9.1f}%")

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
