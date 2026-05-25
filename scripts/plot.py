"""
Plots for esm2_mechanism experiment results.
"""
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(run_dir):
    fi_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(fi_path):
        return None
    with open(fi_path) as f:
        return json.load(f)


def plot_auroc_bars(run_dirs, labels, out_path):
    """Bar chart of per-class AUROC across runs."""
    classes = ["GOF", "DN", "LOF"]
    run_colors = ["#2c3e50", "#7f8c8d", "#95a5a6", "#bdc3c7"]
    hatches = ["", "//", "xx", ".."]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    width = 0.8 / max(len(run_dirs), 1)

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        results = load_results(run_dir)
        if results is None:
            continue
        means = results.get("means", results)
        aurocs = [means.get(f"headline_auroc_{cls}", float("nan")) for cls in classes]
        offset = (i - len(run_dirs) / 2 + 0.5) * width
        color = run_colors[i % len(run_colors)]
        bars = ax.bar(x + offset, aurocs, width * 0.9, label=label,
                      color=color, hatch=hatches[i % len(hatches)], alpha=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("AUROC (one-vs-rest)", fontsize=12)
    ax.set_title("ESM-2 Delta-Embedding Mechanism Probe: Per-Class AUROC", fontsize=13)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cosine_matrix(run_dir, out_path):
    """Heatmap of probe direction cosine similarity matrix."""
    detail_path = os.path.join(run_dir, "detailed_results_seed0.json")
    if not os.path.exists(detail_path):
        print(f"No detailed results at {detail_path}")
        return

    with open(detail_path) as f:
        detail = json.load(f)

    cosines = detail.get("orthogonality", {}).get("cosine_matrix", {})
    if not cosines:
        return

    classes = ["GOF", "DN", "LOF"]
    mat = np.zeros((3, 3))
    np.fill_diagonal(mat, 1.0)
    pair_map = {
        ("DN", "GOF"): (0, 1), ("GOF", "DN"): (0, 1),
        ("GOF", "LOF"): (0, 2), ("LOF", "GOF"): (0, 2),
        ("DN", "LOF"): (1, 2), ("LOF", "DN"): (1, 2),
    }
    for pair_str, val in cosines.items():
        parts = pair_str.split("_vs_")
        if len(parts) == 2:
            key = tuple(parts)
            if key in pair_map:
                i, j = pair_map[key]
                mat[i, j] = val
                mat[j, i] = val

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Probe Direction Cosine Similarity\n(GOF / DN / LOF)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_variance_explained(run_dir, out_path):
    """Bar chart of variance explained by stability subspace per class."""
    detail_path = os.path.join(run_dir, "detailed_results_seed0.json")
    if not os.path.exists(detail_path):
        return

    with open(detail_path) as f:
        detail = json.load(f)

    var_exp = detail.get("variance_explained", {})
    classes = [c for c in ["GOF", "DN", "LOF"] if c in var_exp]
    if not classes:
        return

    vals = [var_exp[c] for c in classes]
    colors = {"GOF": "#e74c3c", "DN": "#3498db", "LOF": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(classes, vals, color=[colors[c] for c in classes], alpha=0.8)
    ax.set_ylabel("Fraction of variance explained\nby stability subspace", fontsize=11)
    ax.set_title("Stability Subspace Variance Explained\nper Mechanism Class", fontsize=12)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "run_0"

    plot_auroc_bars([run_dir], ["ESM-2 650M"],
                    os.path.join(run_dir, "auroc_bars.png"))
    plot_cosine_matrix(run_dir, os.path.join(run_dir, "cosine_matrix.png"))
    plot_variance_explained(run_dir, os.path.join(run_dir, "variance_explained.png"))
