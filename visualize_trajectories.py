"""
Visualize trajectory data from a trajectories JSON file.

Usage:
    python visualize_trajectories.py [path/to/trajectories.json]

Defaults to the most recent trajectories_*.json in pilot_results/ if no path given.

Produces two figures:
  1. S call-timing distribution by orchestrator temperature
  2. Heatmap of bigram motif frequencies vs orchestrator temperature (mean count)
"""

import math
import sys
import json
import glob
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── data loading ─────────────────────────────────────────────────────────────

def load_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def find_latest_trajectories() -> str:
    files = sorted(glob.glob("pilot_results/trajectories_*.json"))
    if not files:
        raise FileNotFoundError("No trajectories_*.json found in pilot_results/")
    return files[-1]


# ── feature extraction ───────────────────────────────────────────────────────

def s_call_round(trajectory: dict) -> int | None:
    """Return the 0-indexed round at which S (Synthesizer) first appears."""
    seq = trajectory["sequence"]
    for i, token in enumerate(seq):
        if token == "S":
            return i
    return None


MOTIFS = [
    # Expansion-contraction cycles
    "RC",
    "RCRC",
    "CRCR",
    # Healthy three-phase rhythm and its building block
    "RCS",
    "RCSRCS",
    # Premature compression
    "RS",
    "CS",
    "RSRS",
    # Post-synthesis behavior
    "SR",
    "SC",
    # Synthesizer runs
    "SS",
    "SSSS",
    # Single-worker monocultures
    "RR",
    "RRRR",
    "CC",
    "CCCC",
]


def motif_counts(trajectory: dict) -> Counter:
    """Count overlapping occurrences of each predefined motif in the sequence."""
    seq_str = "".join(trajectory["sequence"])
    counts = Counter()
    for motif in MOTIFS:
        start = 0
        while True:
            idx = seq_str.find(motif, start)
            if idx == -1:
                break
            counts[motif] += 1
            start = idx + 1  # overlapping, matches main.py
    return counts


# ── plot 1: S call-timing distribution ───────────────────────────────────────

def plot_s_timing(trajectories: list[dict], ax: plt.Axes) -> None:
    by_temp: dict[float, list[int]] = defaultdict(list)
    for traj in trajectories:
        round_idx = s_call_round(traj)
        if round_idx is not None:
            by_temp[traj["orchestrator_temp"]].append(round_idx)

    temps = sorted(by_temp.keys())
    all_rounds = sorted({r for rounds in by_temp.values() for r in rounds})
    if not all_rounds:
        ax.set_title("S Call-Timing (no data)")
        return

    x = np.arange(len(all_rounds))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(temps)))

    for (temp, color) in zip(temps, colors):
        counts = [by_temp[temp].count(r) for r in all_rounds]
        ax.bar(x, counts, width=0.6, label=f"temp={temp}",
               color=color, alpha=0.5, edgecolor=color, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in all_rounds])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Round index")
    ax.set_ylabel("Count of trajectories")
    ax.set_title("Synthesizer (S) Call-Timing by Orchestrator Temperature")
    ax.legend(title="Orchestrator temp", framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot 2: motif frequency heatmap ──────────────────────────────────────────

def plot_motif_heatmap(trajectories: list[dict], ax: plt.Axes) -> None:
    # Collect (temp -> list of Counter)
    by_temp: dict[float, list[Counter]] = defaultdict(list)
    for traj in trajectories:
        by_temp[traj["orchestrator_temp"]].append(motif_counts(traj))

    temps = sorted(by_temp.keys())
    all_motifs = MOTIFS

    if not temps:
        ax.set_title("Motif Heatmap (no data)")
        return

    # Build matrix: rows = motifs, cols = temps
    matrix = np.zeros((len(all_motifs), len(temps)))
    for j, temp in enumerate(temps):
        counters = by_temp[temp]
        n = len(counters)
        for i, motif in enumerate(all_motifs):
            matrix[i, j] = sum(c[motif] for c in counters) / n

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([str(t) for t in temps])
    ax.set_yticks(range(len(all_motifs)))
    ax.set_yticklabels(all_motifs, fontfamily="monospace")
    ax.set_xlabel("Orchestrator temperature")
    ax.set_ylabel("Motif")
    ax.set_title("Mean Motif Frequency vs Orchestrator Temperature")

    # Annotate cells
    for i in range(len(all_motifs)):
        for j in range(len(temps)):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Mean count per trajectory")


# ── plot 3: recurrence plots ──────────────────────────────────────────────────

def plot_recurrence_grid(trajectories: list[dict], src_path: str) -> None:
    trajs = [t for t in trajectories if t.get("workspace_embeddings")]
    if not trajs:
        print("  No embeddings found; skipping recurrence plots.")
        return

    n = len(trajs)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    flat_axes = axes.flat

    for ax, traj in zip(flat_axes, trajs):
        embs = np.array(traj["workspace_embeddings"])   # (k, d)
        dist = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)  # (k, k)

        im = ax.imshow(dist, cmap="viridis", aspect="equal",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

        seq = "".join(traj["sequence"])
        ax.set_title(
            f"{traj['routing_strategy']} | {traj['placement']}\n"
            f"T={traj['orchestrator_temp']}  {seq}",
            fontsize=6,
        )
        k = len(embs)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.tick_params(labelsize=5)

    for ax in list(flat_axes)[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Recurrence plots — Euclidean distance between workspace embeddings\n"
        f"{os.path.basename(src_path)}",
        fontsize=9,
    )
    plt.tight_layout()

    out = src_path.replace(".json", "_recurrence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def make_viz(path: str) -> None:
    print(f"Loading: {path}")
    trajectories = load_trajectories(path)
    print(f"  {len(trajectories)} trajectories, "
          f"temps={sorted(set(t['orchestrator_temp'] for t in trajectories))}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(os.path.basename(path), fontsize=10, color="gray", y=1.01)

    plot_s_timing(trajectories, axes[0])
    plot_motif_heatmap(trajectories, axes[1])

    plt.tight_layout()

    out_path = path.replace(".json", "_viz.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    plot_recurrence_grid(trajectories, path)


def main() -> None:
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("pilot_results/trajectories_*.json"))
        if not paths:
            raise FileNotFoundError("No trajectories_*.json found in pilot_results/")

    for path in paths:
        make_viz(path)


if __name__ == "__main__":
    main()
