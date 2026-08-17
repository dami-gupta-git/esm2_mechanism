"""Generate portfolio figures from existing result JSONs."""

from __future__ import annotations

import functools
import json
import math

import matplotlib

matplotlib.use("Agg")
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

GENE_COLOR = "#4C72B0"
FAMILY_COLOR = "#DD8452"

CLASS_COLORS = {"GOF": "#4C72B0", "DN": "#55A868", "LOF": "#C44E52"}

FEATURE_GROUPS = {
    "wt_only_mean": "ESM-2 protein embedding",
    "mut_only_mean": "ESM-2 protein embedding",
    "wt_concat_mut": "ESM-2 protein embedding",
    "delta_mean": "ESM-2 delta (mutation)",
    "delta_per_residue": "ESM-2 delta (mutation)",
    "onehot_aa": "non-ESM-2 baseline",
    "foldx_ddg": "non-ESM-2 baseline",
    "alphamissense": "non-ESM-2 baseline",
}
GROUP_COLORS = {
    "ESM-2 protein embedding": "#4C72B0",  # blue
    "ESM-2 delta (mutation)": "#55A868",   # green
    "non-ESM-2 baseline": "#8172B3",       # purple
}

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


def fig_dissociation():
    """Pathogenicity AUROC vs mechanism macro-F1, gene- and family-split."""
    path = _load_json(PATHOGENICITY_CONTROL_JSON)
    mech = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
    mech_chance = _mechanism_chance()

    fig, (ax_path, ax_mech) = plt.subplots(1, 2, figsize=(11, 4.5))

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
    fig.suptitle("Same ESM-2 delta embedding: predicts whether a mutation is harmful "
                 "(AUROC ~0.90), not how it acts (macro-F1 at chance)",
                 y=1.08, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig1_dissociation.png")


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

    x = np.arange(len(labels))
    for idx, (gval, fval) in enumerate(zip(gene_vals, fam_vals)):
        drop = gval - fval
        if drop > 0.02:
            ax.annotate(f"−{drop:.02f}", xy=(x[idx], gval), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#333333")

    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("Mechanism macro-F1")
    ax.set_title("Holding out whole families removes ~0.10 macro-F1 from the protein\n"
                 "embeddings — that drop is family recognition, not mechanism")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_family_split.png")


def fig_probe_ranking():
    """Horizontal per-feature gene-split macro-F1 with the chance floor line."""
    mech = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]["gene_split"]
    mech_chance = _mechanism_chance()

    rows = [(key, lab, mech[key]["macro_f1_seed_mean"], mech[key]["macro_f1_seed_std"])
            for key, lab in MECH_FEATURES]
    rows.sort(key=lambda r: r[2])
    keys = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    vals = [r[2] for r in rows]
    errs = [r[3] for r in rows]
    colors = [GROUP_COLORS[FEATURE_GROUPS[key]] for key in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, vals, xerr=errs, color=colors, capsize=3)
    ax.axvline(mech_chance, ls="--", c="grey", lw=1)
    ax.text(mech_chance + 0.005, -0.45, f"chance floor {mech_chance:.2f}",
            ha="left", va="center", fontsize=8, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 0.75)
    ax.set_xlabel("Gene-split mechanism macro-F1 (5-seed mean ± std)")
    ax.set_title("Only the ESM-2 protein embeddings beat the chance floor;\n"
                 "the mutation delta and external baselines sit on it")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in GROUP_COLORS.values()]
    ax.legend(handles, list(GROUP_COLORS.keys()), frameon=False, loc="lower right",
              fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_probe_ranking.png")


def fig_within_family():
    """Per-family delta macro-F1 minus that family's majority baseline."""
    data = _load_json(WITHIN_FAMILY_MECHANISM_JSON)["by_family"]

    rows = []
    for fam, cell in data.items():
        delta_mlp = cell["delta"]["mlp"]["macro_f1"]
        mean = delta_mlp["mean"]
        std = delta_mlp["std"]
        base = cell["majority_baseline_f1"]
        if _is_nan(mean) or _is_nan(base):
            continue
        singleton_class = min(cell["gene_class_counts"].values()) <= 1
        rows.append((fam, cell["n_genes"], mean - base, std, singleton_class))

    rows.sort(key=lambda r: r[2])
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
    ax.set_title("Within each family, the delta does not reliably beat the family's\n"
                 "own majority baseline (small, noisy families)")
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


def fig_family_clustering():
    """k=5 family purity and family-probe accuracy for wt/mut/delta."""
    views = [("wt_mean", "wt_mean"), ("mut_mean", "mut_mean"), ("delta_mean", "delta_mean")]
    bv = _load_json(FAMILY_CLUSTERING_JSON)["by_view"]

    fig, (ax_pur, ax_probe) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(views))

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

    fig.suptitle("ESM-2 embeddings cluster strongly by protein family; subtracting the\n"
                 "wildtype (the delta) drops both metrics to their chance reference",
                 y=1.05, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig5_family_clustering.png")


def fig_auroc_split_bars():
    """Per-class one-vs-rest AUROC, gene- vs family-split, for wt_only and delta."""
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
    fig.suptitle("wt_only AUROC falls on every class under family-split; the delta\n"
                 "stays near chance (0.50) on both splits", y=1.05, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig6_auroc_split_bars.png")


def fig_auroc_split_slope():
    """Per-class AUROC slopegraph: gene-split to family-split, wt_only + delta."""
    agg = _load_json(MECHANISM_AGGREGATE_JSON)["across_seed"]
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
            else:
                ax.text(x_fam + 0.04, f, f"{cls} {f:.2f}", ha="left", va="center",
                        fontsize=9, color=color)
                ax.text(x_gene - 0.04, g, f"{g:.2f}", ha="right", va="center",
                        fontsize=9, color=color)

    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.text(x_gene - 0.04, 0.5, "chance 0.50", ha="right", va="bottom", fontsize=8,
            color="grey")
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
    ax.set_title("Mechanism AUROC, gene-split → family-split:\n"
                 "wt_only is high and drops; the delta sits near chance")
    fig.tight_layout()
    _save(fig, "fig7_auroc_split_slope.png")


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
