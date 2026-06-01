"""
Generate the run6 portfolio figures from the existing result JSONs.

Reads only files already written by the experiments — no model training, no
re-computation — and renders four figures to reports/run6/figures/:

  fig1_dissociation.png   pathogenicity vs mechanism, gene- vs family-split
                          (the headline: same delta, signal on one task, floor
                          on the other). Source: pathogenicity_control.json,
                          aggregate.json, naive_baseline.json.
  fig2_family_split.png   per-feature gene- vs family-split mechanism macro-F1
                          with the drop annotated (the family-recognition /
                          leakage story). Source: aggregate.json, naive_baseline.json.
  fig3_probe_ranking.png  per-feature mechanism macro-F1 vs the chance floor,
                          5-seed error bars. Source: aggregate.json, naive_baseline.json.
  fig4_within_family.png  per-family delta macro-F1 minus that family's own
                          majority baseline, 5-seed error bars, zero line (the
                          within-family null). Source: within_family_mechanism.json.

Every value plotted traces to a result file under results/run6/; nothing is
hardcoded or imputed. Features/families with no scorable result (NaN) are
omitted rather than drawn as zero.

Usage:
    python -m esm2_mech.experiments.mechanism.make_figures
"""

from __future__ import annotations

import functools
import json
import math

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import numpy as np

from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.paths import (
    FAMILY_CLUSTERING_JSON,
    FIGURES_DIR,
    MECHANISM_AGGREGATE_JSON,
    NAIVE_BASELINE_JSON,
    PATHOGENICITY_CONTROL_JSON,
    WITHIN_FAMILY_MECHANISM_JSON,
)

print = functools.partial(print, flush=True)

# Shared colours for the two cross-validation schemes, used across all figures.
GENE_COLOR = "#4C72B0"    # gene-split
FAMILY_COLOR = "#DD8452"  # family-split

# Per-class colours for the one-vs-rest AUROC figures.
CLASS_COLORS = {"GOF": "#4C72B0", "DN": "#55A868", "LOF": "#C44E52"}

# Mechanism features in the order they should appear, with display labels.
MECH_FEATURES = [
    ("wt_concat_mut", "wt_concat_mut"),
    ("mut_only_mean", "mut_only"),
    ("wt_only_mean", "wt_only"),
    ("delta_per_residue", "delta_per_residue"),
    ("delta_mean", "delta_mean"),
    ("onehot_aa", "onehot_aa"),
    ("foldx_ddg", "foldx_ddg"),
    ("alphamissense", "alphamissense"),
]


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _is_nan(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _mechanism_chance():
    """Measured majority-class macro-F1 floor (gene-split) for mechanism."""
    nb = _load_json(NAIVE_BASELINE_JSON)
    return float(nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"])


# ── Figure 1: the dissociation ────────────────────────────────────────────────
def fig_dissociation():
    """Two panels: pathogenicity AUROC and mechanism macro-F1, both split-paired.

    The same delta_mean feature carries strong, family-stable signal for
    pathogenicity but sits on the floor for mechanism. wt_only is shown alongside
    as the contrast (gene-level identity feature).
    """
    path = _load_json(PATHOGENICITY_CONTROL_JSON)
    mech = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
    mech_chance = _mechanism_chance()

    fig, (ax_path, ax_mech) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: pathogenicity AUROC (MLP probe), delta_mean vs wt_only.
    path_feats = [("delta_mean", "delta_mean"), ("wt_only", "wt_only")]
    _grouped_split_bars(
        ax_path,
        labels=[lab for _, lab in path_feats],
        gene_vals=[path["by_feature"][key]["mlp_gene"]["auroc_mean"] for key, _ in path_feats],
        gene_err=[path["by_feature"][key]["mlp_gene"]["auroc_std"] for key, _ in path_feats],
        family_vals=[path["by_feature"][key]["mlp_family"]["auroc_mean"] for key, _ in path_feats],
        family_err=[path["by_feature"][key]["mlp_family"]["auroc_std"] for key, _ in path_feats],
    )
    ax_path.axhline(0.5, ls="--", c="grey", lw=1)
    ax_path.text(0.02, 0.5, "no-skill 0.50", transform=ax_path.get_yaxis_transform(),
                 va="bottom", ha="left", fontsize=8, color="grey")
    ax_path.set_ylim(0.0, 1.0)
    ax_path.set_ylabel("Pathogenicity AUROC")
    ax_path.set_title("Pathogenicity (known-answer control)")

    # Right: mechanism macro-F1 (best nonlinear probe, mlp_delta_mean / wt via MLP).
    # aggregate.json stores per-feature macro_f1; use the same two features.
    mech_feats = [("delta_mean", "delta_mean"), ("wt_only_mean", "wt_only")]
    _grouped_split_bars(
        ax_mech,
        labels=[lab for _, lab in mech_feats],
        gene_vals=[mech["gene_split"][key]["macro_f1_seed_mean"] for key, _ in mech_feats],
        gene_err=[mech["gene_split"][key]["macro_f1_seed_std"] for key, _ in mech_feats],
        family_vals=[mech["family_split"][key]["macro_f1_seed_mean"] for key, _ in mech_feats],
        family_err=[mech["family_split"][key]["macro_f1_seed_std"] for key, _ in mech_feats],
    )
    ax_mech.axhline(mech_chance, ls="--", c="grey", lw=1)
    ax_mech.text(0.02, mech_chance, f"chance {mech_chance:.2f}",
                 transform=ax_mech.get_yaxis_transform(),
                 va="bottom", ha="left", fontsize=8, color="grey")
    ax_mech.set_ylim(0.0, 1.0)
    ax_mech.set_ylabel("Mechanism macro-F1")
    ax_mech.set_title("Mechanism (GOF/DN/LOF)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GENE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR),
    ]
    fig.legend(handles, ["Gene-split", "Family-split"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Same ESM-2 delta: predicts whether, not how", y=1.08, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig1_dissociation.png")


# ── Figure 2: family-split stability / leakage ───────────────────────────────
def fig_family_split():
    """Per-feature gene- vs family-split mechanism macro-F1, drop annotated."""
    mech = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
    mech_chance = _mechanism_chance()

    labels, gene_vals, gene_err, fam_vals, fam_err = [], [], [], [], []
    for key, lab in MECH_FEATURES:
        gcell = mech["gene_split"][key]
        fcell = mech["family_split"][key]
        labels.append(lab)
        gene_vals.append(gcell["macro_f1_seed_mean"])
        gene_err.append(gcell["macro_f1_seed_std"])
        fam_vals.append(fcell["macro_f1_seed_mean"])
        fam_err.append(fcell["macro_f1_seed_std"])

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_split_bars(ax, labels, gene_vals, gene_err, fam_vals, fam_err)
    ax.axhline(mech_chance, ls="--", c="grey", lw=1)
    ax.text(0.99, mech_chance, f"chance {mech_chance:.2f}",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=8, color="grey")

    # Annotate the gene→family drop only where it is non-trivial.
    x = np.arange(len(labels))
    for idx, (gval, fval) in enumerate(zip(gene_vals, fam_vals)):
        drop = gval - fval
        if drop > 0.02:
            ax.annotate(f"−{drop:.02f}", xy=(x[idx], gval), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#333333")

    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("Mechanism macro-F1")
    ax.set_title("Gene-split vs family-split: the drop is family recognition")
    # Legend from the two bar series only (labels set in _grouped_split_bars);
    # the chance line is left unlabelled so it never enters the legend.
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_family_split.png")


# ── Figure 3: probe ranking against the floor ────────────────────────────────
def fig_probe_ranking():
    """Horizontal per-feature gene-split macro-F1 with the chance floor line."""
    mech = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]["gene_split"]
    mech_chance = _mechanism_chance()

    rows = [(lab, mech[key]["macro_f1_seed_mean"], mech[key]["macro_f1_seed_std"])
            for key, lab in MECH_FEATURES]
    rows.sort(key=lambda r: r[1])  # ascending so the best is at the top
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, vals, xerr=errs, color=GENE_COLOR, capsize=3)
    ax.axvline(mech_chance, ls="--", c="grey", lw=1)
    ax.text(mech_chance + 0.005, -0.45, f"chance {mech_chance:.2f}",
            ha="left", va="center", fontsize=8, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 0.75)
    ax.set_xlabel("Gene-split mechanism macro-F1 (5-seed mean ± std)")
    ax.set_title("Most features sit on the chance floor")
    fig.tight_layout()
    _save(fig, "fig3_probe_ranking.png")


# ── Figure 4: within-family null ─────────────────────────────────────────────
def fig_within_family():
    """Per-family delta macro-F1 minus that family's own majority baseline.

    The few families where the delta sits above baseline are the smallest and
    most class-skewed, and several contain a mechanism class represented by a
    single gene (marked) — which cannot be held out and scored, so the
    macro-F1 there is degenerate rather than evidence of within-family signal.
    These are flagged on the figure so it carries the same caveat as the prose
    in report_within_family.md; nothing is filtered out.
    """
    data = _load_json(WITHIN_FAMILY_MECHANISM_JSON)["by_family"]

    rows = []
    for fam, cell in data.items():
        delta_mlp = cell["delta"]["mlp"]["macro_f1"]
        mean = delta_mlp["mean"]
        std = delta_mlp["std"]
        base = cell["majority_baseline_f1"]
        if _is_nan(mean) or _is_nan(base):
            continue  # no scorable fold across seeds — omit, never draw as zero
        # A class held by a single gene cannot be cross-validated: the macro-F1
        # is degenerate. Derived from the data, not a hardcoded family list.
        singleton_class = min(cell["gene_class_counts"].values()) <= 1
        rows.append((fam, cell["n_genes"], mean - base, std, singleton_class))

    rows.sort(key=lambda r: r[2])  # ascending by delta-minus-baseline
    labels = [f"{fam} (n={ng})" for fam, ng, _, _, _ in rows]
    deltas = [d for _, _, d, _, _ in rows]
    errs = [s for _, _, _, s, _ in rows]
    singleton = [sc for _, _, _, _, sc in rows]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(rows))))
    y = np.arange(len(rows))
    colors = [GENE_COLOR if d <= 0 else FAMILY_COLOR for d in deltas]
    hatches = ["////" if sc else "" for sc in singleton]
    ax.barh(y, deltas, xerr=errs, color=colors, capsize=2, hatch=hatches,
            edgecolor="white")
    ax.axvline(0.0, c="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("delta macro-F1 − family majority baseline (5-seed mean ± std)")
    ax.set_title("Within-family delta vs each family's own majority baseline")
    # Legend marker for the degenerate (singleton-class) families.
    hatched = plt.Rectangle((0, 0), 1, 1, facecolor="lightgrey",
                            hatch="////", edgecolor="white")
    ax.legend([hatched], ["family has a single-gene mechanism class (degenerate)"],
              loc="lower right", fontsize=8, frameon=False)
    fig.text(0.5, -0.01,
             "Families are small (6-33 genes), so per-family scores are dominated by "
             "fold assignment (bars are 5-seed std) and are not individually reliable. "
             "Hatched families have a single-gene mechanism class (degenerate score). "
             "See report_within_family.md.",
             ha="center", va="top", fontsize=8, color="#555555")
    fig.tight_layout()
    _save(fig, "fig4_within_family.png")


# ── Figure 5: family clustering collapses under the delta ────────────────────
def fig_family_clustering():
    """Two panels: k=5 family purity and family-probe accuracy, wt/mut/delta.

    Both metrics are label-free family-recognition measures. wt and mut cluster
    strongly by family; the delta collapses to the chance reference (the purity
    null and the majority-family probe baseline), showing the subtraction
    removes the family signal.
    """
    views = [("wt_mean", "wt_mean"), ("mut_mean", "mut_mean"), ("delta_mean", "delta_mean")]
    bv = _load_json(FAMILY_CLUSTERING_JSON)["by_view"]

    fig, (ax_pur, ax_probe) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(views))

    # Left: k=5 family purity vs the shuffled-label null.
    purity = [bv[key]["knn5_purity"] for key, _ in views]
    purity_null = float(np.mean([bv[key]["knn5_purity_null"] for key, _ in views]))
    ax_pur.bar(x, purity, color=GENE_COLOR, width=0.6)
    ax_pur.axhline(purity_null, ls="--", c="grey", lw=1)
    ax_pur.text(0.02, purity_null, f"shuffled-label null {purity_null:.3f}",
                transform=ax_pur.get_yaxis_transform(), va="bottom", ha="left",
                fontsize=8, color="grey")
    ax_pur.set_xticks(x)
    ax_pur.set_xticklabels([lab for _, lab in views], rotation=15, ha="right")
    ax_pur.set_ylabel("k=5 family purity")
    ax_pur.set_title("Do nearest neighbours share the family?")

    # Right: family-probe accuracy vs the majority-family baseline.
    probe = [bv[key]["family_probe"]["accuracy"] for key, _ in views]
    probe_err = [bv[key]["family_probe"]["accuracy_std"] for key, _ in views]
    probe_base = float(np.mean([bv[key]["family_probe"]["majority_baseline_acc"]
                                for key, _ in views]))
    ax_probe.bar(x, probe, yerr=probe_err, capsize=3, color=FAMILY_COLOR, width=0.6)
    ax_probe.axhline(probe_base, ls="--", c="grey", lw=1)
    ax_probe.text(0.02, probe_base, f"majority-family baseline {probe_base:.3f}",
                  transform=ax_probe.get_yaxis_transform(), va="bottom", ha="left",
                  fontsize=8, color="grey")
    ax_probe.set_xticks(x)
    ax_probe.set_xticklabels([lab for _, lab in views], rotation=15, ha="right")
    ax_probe.set_ylabel("Family-probe accuracy")
    ax_probe.set_title("Can a probe name the family from the embedding?")

    fig.suptitle("ESM-2 clusters by family; subtracting the wildtype removes it",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5_family_clustering.png")


# ── Figure 6: per-class AUROC, gene-split vs family-split ─────────────────────
def fig_auroc_split_bars():
    """Per-class one-vs-rest AUROC, gene- vs family-split, for wt_only and delta.

    The signal-carrying wildtype embedding loses AUROC on every class when whole
    families are held out (the family-recognition portion); the delta sits near
    the 0.5 chance line on both splits.
    """
    agg = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
    panels = [("wt_only_mean", "wt_only"), ("delta_mean", "delta_mean")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(MECHANISM_CLASSES))
    width = 0.38
    for ax, (key, title) in zip(axes, panels):
        gene = agg["gene_split"][key]
        fam = agg["family_split"][key]
        gene_vals = [gene[f"auroc_{cls}_seed_mean"] for cls in MECHANISM_CLASSES]
        gene_err = [gene[f"auroc_{cls}_seed_std"] for cls in MECHANISM_CLASSES]
        fam_vals = [fam[f"auroc_{cls}_seed_mean"] for cls in MECHANISM_CLASSES]
        fam_err = [fam[f"auroc_{cls}_seed_std"] for cls in MECHANISM_CLASSES]
        ax.bar(x - width / 2, gene_vals, width, yerr=gene_err, capsize=3,
               color=GENE_COLOR, label="Gene-split")
        ax.bar(x + width / 2, fam_vals, width, yerr=fam_err, capsize=3,
               color=FAMILY_COLOR, label="Family-split")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.text(0.02, 0.5, "chance 0.50", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(MECHANISM_CLASSES)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
    axes[0].set_ylabel("One-vs-rest AUROC")
    axes[0].legend(frameon=False)
    fig.suptitle("Per-class AUROC drops when whole families are held out", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig6_auroc_split_bars.png")


# ── Figure 7: the before→after change, as a slopegraph ───────────────────────
def fig_auroc_split_slope():
    """Per-class AUROC, gene-split (before) → family-split (after), wt_only + delta.

    A slopegraph makes the homology-leakage drop the visual subject: each class
    is a line falling from its gene-split AUROC to its family-split AUROC. The
    wildtype embedding (solid) starts high and drops; the delta (dashed) sits
    near the 0.5 chance line on both splits, with little to lose.
    """
    agg = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
    # (feature key, line style, which end to label the class name on).
    features = [("wt_only_mean", "-", "left"), ("delta_mean", "--", "right")]
    x_gene, x_fam = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for key, style, name_side in features:
        gene = agg["gene_split"][key]
        fam = agg["family_split"][key]
        for cls in MECHANISM_CLASSES:
            g = gene[f"auroc_{cls}_seed_mean"]
            f = fam[f"auroc_{cls}_seed_mean"]
            color = CLASS_COLORS[cls]
            ax.plot([x_gene, x_fam], [g, f], style, marker="o", color=color, lw=2)
            if name_side == "left":
                ax.text(x_gene - 0.04, g, f"{cls} {g:.2f}", ha="right", va="center",
                        fontsize=9, color=color)
                ax.text(x_fam + 0.04, f, f"{f:.2f}", ha="left", va="center",
                        fontsize=9, color=color)
            else:  # delta: name on the right end, value on the left end
                ax.text(x_fam + 0.04, f, f"{cls} {f:.2f}", ha="left", va="center",
                        fontsize=9, color=color)
                ax.text(x_gene - 0.04, g, f"{g:.2f}", ha="right", va="center",
                        fontsize=9, color=color)

    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.text(x_gene - 0.04, 0.5, "chance 0.50", ha="right", va="bottom", fontsize=8,
            color="grey")
    # Line-style legend: solid = wt_only, dashed = delta.
    style_handles = [
        plt.Line2D([0], [0], color="grey", lw=2, ls="-"),
        plt.Line2D([0], [0], color="grey", lw=2, ls="--"),
    ]
    ax.legend(style_handles, ["wt_only", "delta_mean"], frameon=False, loc="center left")
    ax.set_xticks([x_gene, x_fam])
    ax.set_xticklabels(["Gene-split\n(related genes leak)", "Family-split\n(families held out)"])
    ax.set_xlim(-0.45, 1.55)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("One-vs-rest AUROC")
    ax.set_title("The clean change: gene-split → family-split")
    fig.tight_layout()
    _save(fig, "fig7_auroc_split_slope.png")


# ── Shared helpers ────────────────────────────────────────────────────────────
def _grouped_split_bars(ax, labels, gene_vals, gene_err, family_vals, family_err):
    """Draw paired gene-split / family-split bars for each label on ax."""
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, gene_vals, width, yerr=gene_err, capsize=3,
           color=GENE_COLOR, label="Gene-split")
    ax.bar(x + width / 2, family_vals, width, yerr=family_err, capsize=3,
           color=FAMILY_COLOR, label="Family-split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig_dissociation()
    fig_family_split()
    fig_probe_ranking()
    fig_within_family()
    fig_family_clustering()
    fig_auroc_split_bars()
    fig_auroc_split_slope()
    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
