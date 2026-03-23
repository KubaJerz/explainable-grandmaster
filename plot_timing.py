"""Compare iteration timing between sequential (run.py) and parallel (run_parallel.py) runs.

Usage:
    python plot_timing.py                                          # both logs from results/
    python plot_timing.py --sequential results_seq/training_log.json --parallel results_par/training_log.json
    python plot_timing.py --parallel results/training_log.json     # just one log
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_timing(path):
    with open(path) as f:
        log = json.load(f)
    iterations = []
    sp_times = []
    train_times = []
    iter_times = []
    for entry in log:
        t = entry.get("timing")
        if t is None:
            continue
        iterations.append(entry["iteration"])
        # sequential logs "self_play_s", parallel logs "self_play_wait_s"
        sp_times.append(t.get("self_play_s", t.get("self_play_wait_s", 0)))
        train_times.append(t["training_s"])
        iter_times.append(t["iteration_s"])
    return iterations, sp_times, train_times, iter_times


def main():
    parser = argparse.ArgumentParser(description="Plot timing comparison")
    parser.add_argument("--sequential", type=str, default=None,
                        help="Path to sequential training_log.json")
    parser.add_argument("--parallel", type=str, default=None,
                        help="Path to parallel training_log.json")
    parser.add_argument("-o", "--output", type=str, default="results/timing_comparison.png")
    args = parser.parse_args()

    has_seq = args.sequential is not None
    has_par = args.parallel is not None

    if not has_seq and not has_par:
        print("No log files specified. Use --sequential and/or --parallel.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_sp, ax_train, ax_iter = axes

    for path, label, color in [
        (args.sequential, "Sequential (run.py)", "tab:blue"),
        (args.parallel, "Parallel (run_parallel.py)", "tab:orange"),
    ]:
        if path is None:
            continue
        iters, sp, tr, total = load_timing(path)
        if not iters:
            print(f"No timing data found in {path}")
            continue
        ax_sp.plot(iters, sp, "o-", label=label, color=color, markersize=4)
        ax_train.plot(iters, tr, "o-", label=label, color=color, markersize=4)
        ax_iter.plot(iters, total, "o-", label=label, color=color, markersize=4)

        avg_total = np.mean(total)
        print(f"{label}:")
        print(f"  Avg self-play: {np.mean(sp):.1f}s | Avg train: {np.mean(tr):.1f}s | Avg iter: {avg_total:.1f}s")

    ax_sp.set_title("Self-Play Time")
    ax_sp.set_xlabel("Iteration")
    ax_sp.set_ylabel("Seconds")
    ax_sp.legend()
    ax_sp.grid(True, alpha=0.3)

    ax_train.set_title("Training Time")
    ax_train.set_xlabel("Iteration")
    ax_train.set_ylabel("Seconds")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_iter.set_title("Total Iteration Time")
    ax_iter.set_xlabel("Iteration")
    ax_iter.set_ylabel("Seconds")
    ax_iter.legend()
    ax_iter.grid(True, alpha=0.3)

    fig.suptitle("Training Pipeline Timing Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.show()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
