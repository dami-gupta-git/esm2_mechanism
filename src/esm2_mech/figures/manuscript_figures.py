"""Generate manuscript figures from the verified ``run_biorxiv`` outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import functools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from esm2_mech.experiments.mechanism.seed_results import read_feature_metric
from esm2_mech.utils.constants import (
    DN,
    ESM2_MODEL,
    GOF,
    LABEL_3CLASS_FIELD,
    LOF,
    MAX_SEQ_LEN,
    MECHANISM_CLASSES,
    SOURCE_CLINVAR_G2P,
    SOURCE_GERASIMAVICIUS,
)
from esm2_mech.utils.paths import (
    CONSERVATION_AXIS_JSON,
    ENZYME_CLASSIFICATION_JSON,
    FAMILY_CLUSTERING_JSON,
    FIGURES_DIR,
    LEAKAGE_FRACTION_JSON,
    MAGNITUDE_DIRECTION_JSON,
    MECHANISM_AGGREGATE_JSON,
    PATHOGENICITY_CONTROL_JSON,
    SINGLE_SOURCE_AGGREGATE_JSON,
    STABILITY_MLP_SUMMARY_JSON,
    STABILITY_PER_PROTEIN_JSON,
    STABILITY_PROJECTION_JSON,
    STABILITY_SUMMARY_JSON,
    STABILITY_XGB_SUMMARY_JSON,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

BLUE = "#4477AA"
CYAN = "#66CCEE"
GREEN = "#228833"
YELLOW = "#CCBB44"
RED = "#EE6677"
PURPLE = "#AA3377"
GREY = "#777777"
LIGHT_GREY = "#E6E6E6"
DARK = "#222222"
WHITE = "#FFFFFF"

GENE_COLOR = BLUE
FAMILY_COLOR = RED
AUROC_COLOR = PURPLE
CLASS_COLORS = {LOF: BLUE, GOF: RED, DN: PURPLE}
DISPLAY_CLASS_ORDER = (GOF, LOF, DN)
ENZYME_CLASS_ORDER = ("kinase", "oxidoreductase", "non-enzyme", "protease")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Required figure input is missing: {path}")
    with path.open() as handle:
        return json.load(handle)


def _seed_mean(aggregate: dict, split: str, feature: str, metric: str = "macro_f1") -> float:
    """Read an across-seed mechanism mean through the shared seed reader."""
    read = read_feature_metric(aggregate["across_seed"], split, feature, metric)
    if not read.available:
        raise ValueError(
            f"{split} {feature} {metric} has no across-seed mean: {read.message}"
        )
    return read.value


def _finite(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} is not a finite number: {value!r}")
    return float(value)


def _ci_errors(point: float, ci_low: float, ci_high: float) -> np.ndarray:
    if not ci_low <= point <= ci_high:
        raise ValueError(
            f"Confidence interval [{ci_low}, {ci_high}] does not contain {point}"
        )
    return np.asarray([[point - ci_low], [ci_high - point]])


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_figure(figure: Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        output_path = FIGURES_DIR / f"{stem}.{suffix}"
        save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = 300
        figure.savefig(output_path, **save_kwargs)
        print(f"wrote {output_path}")
    plt.close(figure)


def _rounded_box(
    axis: Axes,
    x_position: float,
    y_position: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = DARK,
    text_color: str = DARK,
    fontsize: float = 8.5,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x_position, y_position),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        x_position + width / 2,
        y_position + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.25,
    )
    return patch


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
    linewidth: float = 1.4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _schematic_axis(axis: Axes, title: str, panel_label: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, loc="left", pad=8)
    _panel_label(axis, panel_label)


def _load_mechanism_cohort_counts() -> tuple[Counter[str], Counter[str], int, int]:
    variants = _load_json(VALID_VARIANTS_JSON)
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"Expected a non-empty list in {VALID_VARIANTS_JSON}")

    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    genes: set[str] = set()
    for row_index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise TypeError(f"Variant row {row_index} is not an object")
        for field_name in ("source", LABEL_3CLASS_FIELD, "gene"):
            if field_name not in variant:
                raise KeyError(f"Variant row {row_index} lacks {field_name!r}")
        source_counts[str(variant["source"])] += 1
        class_counts[str(variant[LABEL_3CLASS_FIELD])] += 1
        genes.add(str(variant["gene"]))

    expected_sources = {SOURCE_GERASIMAVICIUS, SOURCE_CLINVAR_G2P}
    if set(source_counts) != expected_sources:
        raise ValueError(
            f"Unexpected mechanism sources: {sorted(source_counts)}; "
            f"expected {sorted(expected_sources)}"
        )
    if set(class_counts) != set(MECHANISM_CLASSES):
        raise ValueError(
            f"Unexpected mechanism classes: {sorted(class_counts)}; "
            f"expected {sorted(MECHANISM_CLASSES)}"
        )
    return source_counts, class_counts, len(genes), len(variants)


def figure1_study_design() -> None:
    source_counts, class_counts, gene_count, variant_count = (
        _load_mechanism_cohort_counts()
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    figure.subplots_adjust(wspace=0.18, hspace=0.27)

    cohort_axis = axes[0, 0]
    _schematic_axis(cohort_axis, "Mechanism cohort", "A")
    _rounded_box(
        cohort_axis,
        0.03,
        0.63,
        0.28,
        0.18,
        "Gerasimavicius et al.\n" f"{source_counts[SOURCE_GERASIMAVICIUS]:,} variants",
        facecolor="#EAF2F8",
        edgecolor=BLUE,
    )
    _rounded_box(
        cohort_axis,
        0.03,
        0.25,
        0.28,
        0.18,
        "G2P + ClinVar\n" f"{source_counts[SOURCE_CLINVAR_G2P]:,} variants",
        facecolor="#FDEDEC",
        edgecolor=RED,
    )
    _rounded_box(
        cohort_axis,
        0.40,
        0.43,
        0.25,
        0.20,
        "Gene-level labels\nLOF · GOF · DN",
        facecolor="#F8F9F9",
        edgecolor=GREY,
    )
    _arrow(cohort_axis, (0.31, 0.72), (0.40, 0.56))
    _arrow(cohort_axis, (0.31, 0.34), (0.40, 0.50))
    _rounded_box(
        cohort_axis,
        0.73,
        0.39,
        0.24,
        0.29,
        f"{variant_count:,} missense variants\n"
        f"{gene_count:,} genes\n\n"
        f"LOF {class_counts[LOF]:,}\n"
        f"GOF {class_counts[GOF]:,}\n"
        f"DN {class_counts[DN]:,}",
        facecolor="#F4F6F7",
        edgecolor=DARK,
    )
    _arrow(cohort_axis, (0.65, 0.53), (0.73, 0.53))
    cohort_axis.text(
        0.03,
        0.08,
        "Each missense variant inherits its gene's curated mechanism label.",
        fontsize=8,
        color=GREY,
        ha="left",
    )

    representation_axis = axes[0, 1]
    _schematic_axis(representation_axis, "Frozen ESM-2 representations", "B")
    _rounded_box(
        representation_axis,
        0.02,
        0.65,
        0.25,
        0.13,
        "Wildtype sequence\n…  [variant site]  …",
        facecolor="#EAF2F8",
        edgecolor=BLUE,
    )
    _rounded_box(
        representation_axis,
        0.02,
        0.30,
        0.25,
        0.13,
        "Mutant sequence\n…  [substitution]  …",
        facecolor="#FDEDEC",
        edgecolor=RED,
    )
    model_label = ESM2_MODEL.replace("esm2_", "ESM-2\n").replace("_UR50D", "")
    _rounded_box(
        representation_axis,
        0.36,
        0.46,
        0.25,
        0.20,
        f"{model_label}\nweights frozen",
        facecolor="#F4F6F7",
        edgecolor=DARK,
    )
    _arrow(representation_axis, (0.27, 0.71), (0.36, 0.59), color=BLUE)
    _arrow(representation_axis, (0.27, 0.36), (0.36, 0.52), color=RED)
    _rounded_box(
        representation_axis,
        0.70,
        0.67,
        0.27,
        0.12,
        "Wildtype mean embedding\n1,280 dimensions",
        facecolor="#EAF2F8",
        edgecolor=BLUE,
    )
    _rounded_box(
        representation_axis,
        0.70,
        0.29,
        0.27,
        0.12,
        "Mutant mean embedding\n1,280 dimensions",
        facecolor="#FDEDEC",
        edgecolor=RED,
    )
    _arrow(representation_axis, (0.61, 0.59), (0.70, 0.73), color=BLUE)
    _arrow(representation_axis, (0.61, 0.52), (0.70, 0.35), color=RED)
    _rounded_box(
        representation_axis,
        0.70,
        0.05,
        0.27,
        0.12,
        "Mutation delta\nmutant − wildtype",
        facecolor="#F4ECF7",
        edgecolor=PURPLE,
    )
    _arrow(representation_axis, (0.84, 0.29), (0.84, 0.17), color=PURPLE)
    representation_axis.text(
        0.02,
        0.08,
        f"Sequences longer than {MAX_SEQ_LEN:,} residues\nare windowed around the variant.",
        fontsize=8,
        color=GREY,
        ha="left",
        va="center",
    )

    split_axis = axes[1, 0]
    _schematic_axis(split_axis, "Held-out unit changes the transfer question", "C")
    split_axis.add_patch(
        Rectangle((0.02, 0.15), 0.46, 0.70, facecolor="#FAFAFA", edgecolor=LIGHT_GREY)
    )
    split_axis.add_patch(
        Rectangle((0.52, 0.15), 0.46, 0.70, facecolor="#FAFAFA", edgecolor=LIGHT_GREY)
    )
    split_axis.text(0.25, 0.79, "Gene holdout", ha="center", fontweight="bold")
    split_axis.text(0.75, 0.79, "Pfam-family holdout", ha="center", fontweight="bold")
    split_axis.text(0.08, 0.68, "Training", color=GREY, fontsize=8)
    split_axis.text(0.34, 0.68, "Test", color=GREY, fontsize=8)
    split_axis.text(0.58, 0.68, "Training", color=GREY, fontsize=8)
    split_axis.text(0.84, 0.68, "Test", color=GREY, fontsize=8)

    family_palette = (BLUE, RED, GREEN)
    gene_split_points = (
        (0.12, 0.55, 0),
        (0.18, 0.43, 0),
        (0.39, 0.50, 0),
        (0.11, 0.37, 1),
        (0.21, 0.32, 1),
        (0.38, 0.35, 2),
    )
    family_split_points = (
        (0.61, 0.55, 0),
        (0.70, 0.47, 0),
        (0.62, 0.37, 1),
        (0.72, 0.32, 1),
        (0.87, 0.52, 2),
        (0.91, 0.37, 2),
    )
    for x_position, y_position, family_index in gene_split_points + family_split_points:
        split_axis.scatter(
            x_position,
            y_position,
            s=150,
            color=family_palette[family_index],
            edgecolor=WHITE,
            linewidth=1.2,
            zorder=3,
        )
    split_axis.text(
        0.25,
        0.18,
        "A test gene is unseen, but its family\nmay be represented in training.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=GREY,
    )
    split_axis.text(
        0.75,
        0.18,
        "Every member of the test family\nis absent from training.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=GREY,
    )

    framework_axis = axes[1, 1]
    _schematic_axis(framework_axis, "Shared evaluation framework", "D")
    rows = (
        ("Disease mechanism", "WT · mutant · delta", "Gene · Pfam family"),
        ("Pathogenicity", "Delta", "Gene · Pfam family"),
        ("Enzyme type", "Wildtype", "Gene · Pfam family"),
        ("Folding stability", "Delta", "Random · domain · Pfam family"),
    )
    column_x = (0.03, 0.42, 0.69)
    column_widths = (0.36, 0.24, 0.28)
    header_y = 0.72
    row_height = 0.135
    headers = ("Biological target", "Representation", "Held-out unit")
    for x_position, width, header in zip(column_x, column_widths, headers):
        _rounded_box(
            framework_axis,
            x_position,
            header_y,
            width,
            0.12,
            header,
            facecolor=DARK,
            edgecolor=DARK,
            text_color=WHITE,
            fontsize=8,
        )
    for row_index, row in enumerate(rows):
        y_position = header_y - (row_index + 1) * row_height
        for column_index, (x_position, width) in enumerate(
            zip(column_x, column_widths)
        ):
            facecolor = "#F8F9F9" if row_index % 2 == 0 else WHITE
            _rounded_box(
                framework_axis,
                x_position,
                y_position,
                width,
                0.11,
                row[column_index],
                facecolor=facecolor,
                edgecolor=LIGHT_GREY,
                fontsize=7.5,
                linewidth=0.8,
            )
    framework_axis.text(
        0.03,
        0.08,
        "Frozen embeddings, task-specific probes, and uncertainty estimates matched to the held-out unit.",
        fontsize=8,
        color=GREY,
        ha="left",
    )

    _save_figure(figure, "figure1_study_design")


def _plot_per_class_auroc(
    axis: Axes,
    labels: tuple[str, ...],
    values: tuple[float, ...],
    *,
    panel_label: str,
    title: str,
) -> None:
    positions = np.arange(len(labels))
    axis.vlines(positions, 0.5, values, color=AUROC_COLOR, linewidth=2)
    axis.scatter(positions, values, color=AUROC_COLOR, s=55, zorder=3)
    axis.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    for position, value in zip(positions, values):
        axis.text(position, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("One-vs-rest AUROC")
    axis.set_title(title, loc="left")
    _panel_label(axis, panel_label)


def figure2_mechanism_delta() -> None:
    aggregate = _load_json(MECHANISM_AGGREGATE_JSON)
    geometry = _load_json(MAGNITUDE_DIRECTION_JSON)
    claim_2a = aggregate["claim_2a_ci_summary"]

    figure = plt.figure(figsize=(12, 8.2))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.0, 1.15), hspace=0.34, wspace=0.28
    )
    interval_axis = figure.add_subplot(grid[0, :])
    probe_axis = figure.add_subplot(grid[1, 0])
    auroc_axis = figure.add_subplot(grid[1, 1])

    seed_rows = claim_2a["per_seed"]
    seed_positions = np.arange(len(seed_rows))
    points = np.asarray([_finite(row["point"], "claim2a point") for row in seed_rows])
    lows = np.asarray([_finite(row["ci_low"], "claim2a ci_low") for row in seed_rows])
    highs = np.asarray(
        [_finite(row["ci_high"], "claim2a ci_high") for row in seed_rows]
    )
    interval_axis.errorbar(
        points,
        seed_positions,
        xerr=np.vstack((points - lows, highs - points)),
        fmt="o",
        color=PURPLE,
        ecolor=PURPLE,
        capsize=3,
    )
    reference = _finite(claim_2a["family_chance_floor"], "family reference")
    threshold = _finite(claim_2a["threshold"], "reference plus margin")
    interval_axis.axvline(reference, color=GREY, linestyle="--", linewidth=1.2)
    interval_axis.text(
        reference + 0.0004,
        0.55,
        "majority-class reference",
        va="center",
        ha="left",
        color=GREY,
        fontsize=8,
    )
    interval_axis.text(
        0.98,
        1.02,
        f"Preregistered threshold {threshold:.3f}\nAll upper bounds were below it",
        transform=interval_axis.transAxes,
        va="bottom",
        ha="right",
        color=RED,
        fontsize=8,
    )
    interval_axis.set_yticks(seed_positions)
    interval_axis.set_yticklabels([f"Seed {row['seed']}" for row in seed_rows])
    interval_axis.invert_yaxis()
    interval_axis.set_xlim(min(lows) - 0.005, max(highs) + 0.006)
    interval_axis.set_xlabel("Family-split macro-F1 (95% family-bootstrap CI)")
    interval_axis.set_title(
        "Preregistered delta probe matched the reference", loc="left"
    )
    _panel_label(interval_axis, "A")

    probe_labels = (
        "Preregistered\nPCA + logistic",
        "Exploratory\nlogistic",
        "Exploratory\nMLP",
    )
    gene_values = (
        _seed_mean(aggregate, "gene_split", "delta_mean"),
        geometry["mechanism"]["full"]["gene_split"]["logreg_macro_f1"]["mean"],
        geometry["mechanism"]["full"]["gene_split"]["mlp_macro_f1"]["mean"],
    )
    family_values = (
        _seed_mean(aggregate, "family_split", "delta_mean"),
        geometry["mechanism"]["full"]["family_split"]["logreg_macro_f1"]["mean"],
        geometry["mechanism"]["full"]["family_split"]["mlp_macro_f1"]["mean"],
    )
    positions = np.arange(len(probe_labels))
    for position, gene_value, family_value in zip(
        positions, gene_values, family_values
    ):
        probe_axis.plot(
            [position - 0.12, position + 0.12],
            [gene_value, family_value],
            color=LIGHT_GREY,
            linewidth=2,
            zorder=1,
        )
    probe_axis.scatter(
        positions - 0.12, gene_values, color=GENE_COLOR, s=55, label="Gene holdout"
    )
    probe_axis.scatter(
        positions + 0.12,
        family_values,
        color=FAMILY_COLOR,
        s=55,
        label="Family holdout",
    )
    probe_axis.axhline(reference, color=GREY, linestyle="--", linewidth=1)
    probe_axis.set_xticks(positions)
    probe_axis.set_xticklabels(probe_labels)
    probe_axis.set_ylim(0.26, 0.44)
    probe_axis.set_ylabel("Macro-F1 (five-seed mean)")
    probe_axis.set_title("Probe-dependent classification signal", loc="left")
    probe_axis.legend(frameon=False, loc="upper right")
    _panel_label(probe_axis, "B")

    auroc_values = tuple(
        _seed_mean(aggregate, "family_split", "delta_mean", f"auroc_{class_label}")
        for class_label in DISPLAY_CLASS_ORDER
    )
    _plot_per_class_auroc(
        auroc_axis,
        DISPLAY_CLASS_ORDER,
        auroc_values,
        panel_label="C",
        title="Weak family-split ranking signal",
    )

    _save_figure(figure, "figure2_mechanism_delta")


def figure3_family_information() -> None:
    clustering = _load_json(FAMILY_CLUSTERING_JSON)
    aggregate = _load_json(MECHANISM_AGGREGATE_JSON)
    leakage = _load_json(LEAKAGE_FRACTION_JSON)
    views = clustering["by_view"]
    view_keys = ("wt_mean", "mut_mean", "delta_mean")
    view_labels = ("Wildtype", "Mutant", "Delta")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7.2),
        gridspec_kw={"height_ratios": (1.0, 0.60)},
    )
    figure.subplots_adjust(wspace=0.30, hspace=0.44)

    accuracy_axis = axes[0, 0]
    positions = np.arange(len(view_keys))
    probe_blocks = [views[key]["family_probe"] for key in view_keys]
    for position, block, color in zip(
        positions, probe_blocks, (BLUE, CYAN, PURPLE)
    ):
        accuracy = block.get("accuracy")
        if accuracy is None:
            accuracy_axis.text(
                position, 0.03, "Unscorable", ha="center", va="bottom", fontsize=8
            )
            continue
        accuracy_axis.bar(
            position,
            accuracy,
            yerr=block.get("accuracy_std"),
            color=color,
            capsize=3,
        )
    baselines = [
        block.get("majority_baseline_acc")
        for block in probe_blocks
        if block.get("majority_baseline_acc") is not None
    ]
    if baselines:
        if not all(math.isclose(baseline, baselines[0]) for baseline in baselines):
            raise ValueError("Family-probe majority references differ across views")
        accuracy_axis.axhline(
            baselines[0], color=GREY, linestyle="--", linewidth=1
        )
        accuracy_axis.text(
            2.45,
            baselines[0],
            f"reference {baselines[0]:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            color=GREY,
        )
    accuracy_axis.set_xticks(positions)
    accuracy_axis.set_xticklabels(view_labels)
    accuracy_axis.set_ylim(0, 0.68)
    accuracy_axis.set_ylabel("Pfam-family accuracy")
    accuracy_axis.set_title(
        "Family can be decoded from absolute embeddings", loc="left"
    )
    _panel_label(accuracy_axis, "A")

    purity_axis = axes[0, 1]
    purity_points = []
    purity_lows = []
    purity_highs = []
    purity_nulls = []
    for view_key in view_keys:
        purity_cell = views[view_key]
        purity_points.append(purity_cell["knn5_purity"])
        purity_lows.append(purity_cell["knn5_purity_ci"]["ci_low"])
        purity_highs.append(purity_cell["knn5_purity_ci"]["ci_high"])
        purity_nulls.append(purity_cell["knn5_purity_null"])
    purity_points_array = np.asarray(purity_points)
    purity_axis.errorbar(
        positions,
        purity_points_array,
        yerr=np.vstack(
            (purity_points_array - purity_lows, purity_highs - purity_points_array)
        ),
        fmt="o",
        color=BLUE,
        capsize=3,
        label="Observed",
    )
    purity_axis.scatter(
        positions,
        purity_nulls,
        marker="o",
        facecolor=WHITE,
        edgecolor=GREY,
        label="Shuffled labels",
    )
    purity_axis.set_xticks(positions)
    purity_axis.set_xticklabels(view_labels)
    purity_axis.set_ylim(0, 0.34)
    purity_axis.set_ylabel(
        "Fraction of five nearest neighbours\nsharing the Pfam family"
    )
    purity_axis.set_title("Local geometry contains family structure", loc="left")
    purity_axis.legend(frameon=False)
    _panel_label(purity_axis, "B")

    gap_axis = axes[1, 0]
    gap_rows = aggregate["claim_2b_split_gap_summary"]["per_seed"]
    gap_positions = np.arange(len(gap_rows))
    gap_points = np.asarray([row["point_diff"] for row in gap_rows])
    gap_lows = np.asarray([row["ci_low"] for row in gap_rows])
    gap_highs = np.asarray([row["ci_high"] for row in gap_rows])
    gap_axis.errorbar(
        gap_points,
        gap_positions,
        xerr=np.vstack((gap_points - gap_lows, gap_highs - gap_points)),
        fmt="o",
        color=BLUE,
        capsize=3,
    )
    gap_axis.axvline(0, color=GREY, linewidth=1)
    gap_axis.set_yticks(gap_positions)
    gap_axis.set_yticklabels([f"Seed {row['seed']}" for row in gap_rows])
    gap_axis.invert_yaxis()
    gap_axis.set_xlabel("Wildtype gene-minus-family macro-F1 (95% CI)")
    gap_axis.set_title("Gene-held-out performance was higher in every seed", loc="left")
    _panel_label(gap_axis, "C")

    leakage_axis = axes[1, 1]
    leakage_cell = leakage["by_feature"]["wt_only_mean"]
    leakage_point = _finite(leakage_cell["leakage_fraction"], "leakage fraction")
    leakage_ci = leakage_cell["ci"]
    leakage_low = _finite(leakage_ci["ci_low"], "leakage ci low")
    leakage_high = _finite(leakage_ci["ci_high"], "leakage ci high")
    leakage_axis.errorbar(
        [leakage_point * 100],
        [0],
        xerr=_ci_errors(leakage_point * 100, leakage_low * 100, leakage_high * 100),
        fmt="o",
        color=PURPLE,
        capsize=4,
        markersize=7,
    )
    leakage_axis.text(
        leakage_point * 100,
        0.12,
        f"{leakage_point * 100:.1f}%\n[{leakage_low * 100:.1f}, {leakage_high * 100:.1f}]",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    leakage_axis.set_xlim(0, 65)
    leakage_axis.set_ylim(-0.35, 0.45)
    leakage_axis.set_yticks([])
    leakage_axis.set_xlabel(
        "Fraction of above-reference performance\nlost under family holdout (%)"
    )
    leakage_axis.set_title(
        "Leakage fraction with family-bootstrap interval", loc="left"
    )
    _panel_label(leakage_axis, "D")

    _save_figure(figure, "figure3_family_information")


def figure4_pathogenicity_conservation() -> None:
    pathogenicity = _load_json(PATHOGENICITY_CONTROL_JSON)
    geometry = _load_json(MAGNITUDE_DIRECTION_JSON)
    conservation = _load_json(CONSERVATION_AXIS_JSON)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    figure.subplots_adjust(wspace=0.34)

    transfer_axis = axes[0]
    delta_results = pathogenicity["by_feature"]["delta_mean"]
    probe_names = ("Logistic", "MLP")
    gene_values = (
        delta_results["logreg_gene"]["auroc_mean"],
        delta_results["mlp_gene"]["auroc_mean"],
    )
    family_values = (
        delta_results["logreg_family"]["auroc_mean"],
        delta_results["mlp_family"]["auroc_mean"],
    )
    positions = np.arange(len(probe_names))
    for position, gene_value, family_value in zip(
        positions, gene_values, family_values
    ):
        transfer_axis.plot(
            [position - 0.12, position + 0.12],
            [gene_value, family_value],
            color=LIGHT_GREY,
            linewidth=2,
        )
    transfer_axis.scatter(
        positions - 0.12, gene_values, color=GENE_COLOR, s=55, label="Gene holdout"
    )
    transfer_axis.scatter(
        positions + 0.12,
        family_values,
        color=FAMILY_COLOR,
        s=55,
        label="Family holdout",
    )
    transfer_axis.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    transfer_axis.set_xticks(positions)
    transfer_axis.set_xticklabels(probe_names)
    transfer_axis.set_ylim(0.5, 0.93)
    transfer_axis.set_ylabel("Pathogenicity AUROC")
    transfer_axis.set_title("Pathogenicity transferred across families", loc="left")
    transfer_axis.legend(frameon=False)
    _panel_label(transfer_axis, "A")

    geometry_axis = axes[1]
    component_keys = ("full", "dir", "mag")
    component_labels = ("Full delta", "Direction", "Magnitude")
    logreg_values = tuple(
        geometry["pathogenicity"][key]["family_split"]["logreg_auroc"]["mean"]
        for key in component_keys
    )
    mlp_values = tuple(
        geometry["pathogenicity"][key]["family_split"]["mlp_auroc"]["mean"]
        for key in component_keys
    )
    geometry_positions = np.arange(len(component_labels))
    width = 0.26
    geometry_axis.scatter(
        geometry_positions - width / 2,
        logreg_values,
        color=BLUE,
        s=50,
        label="Logistic",
    )
    geometry_axis.scatter(
        geometry_positions + width / 2,
        mlp_values,
        color=RED,
        s=50,
        marker="s",
        label="MLP",
    )
    geometry_axis.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    geometry_axis.set_xticks(geometry_positions)
    geometry_axis.set_xticklabels(component_labels, rotation=15, ha="right")
    geometry_axis.set_ylim(0.5, 0.93)
    geometry_axis.set_ylabel("Family-split AUROC")
    geometry_axis.set_title("Direction retained the pathogenicity signal", loc="left")
    geometry_axis.legend(frameon=False)
    _panel_label(geometry_axis, "B")

    conservation_axis = axes[2]
    comparison_keys = ("delta", "conservation", "conservation_plus_delta")
    comparison_labels = ("Delta", "Conservation", "Conservation\n+ delta")
    comparison_points = []
    comparison_lows = []
    comparison_highs = []
    for comparison_key in comparison_keys:
        cell = conservation["auroc_family_split_ci"][comparison_key]
        comparison_points.append(cell["point"])
        comparison_lows.append(cell["ci_low"])
        comparison_highs.append(cell["ci_high"])
    point_array = np.asarray(comparison_points)
    comparison_positions = np.arange(len(comparison_labels))
    conservation_axis.errorbar(
        comparison_positions,
        point_array,
        yerr=np.vstack((point_array - comparison_lows, comparison_highs - point_array)),
        fmt="o",
        color=PURPLE,
        capsize=3,
    )
    conservation_axis.set_xticks(comparison_positions)
    conservation_axis.set_xticklabels(comparison_labels)
    conservation_axis.set_ylim(0.81, 0.91)
    conservation_axis.set_ylabel("Family-split AUROC, seed 0 (95% CI)")
    conservation_axis.set_title(
        "Adding delta did not improve discrimination", loc="left"
    )
    paired_difference = conservation["claims"]["2E_delta_beyond_conservation"][
        "paired_diff"
    ]
    conservation_axis.text(
        0.98,
        0.04,
        "Combined − conservation\n"
        f"{paired_difference['point_diff']:+.3f} "
        f"[{paired_difference['ci_low']:+.3f}, {paired_difference['ci_high']:+.3f}]",
        transform=conservation_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=DARK,
    )
    _panel_label(conservation_axis, "C")

    _save_figure(figure, "figure4_pathogenicity_conservation")


def figure5_enzyme_classification() -> None:
    enzyme = _load_json(ENZYME_CLASSIFICATION_JSON)
    esm = enzyme["esm2_wt_embedding"]
    proteome = enzyme["proteome_features"]
    figure = plt.figure(figsize=(12, 7.2))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.52),
        wspace=0.30,
        hspace=0.45,
    )

    transfer_axis = figure.add_subplot(grid[0, 0])
    representation_labels = ("ESM-2 wildtype", "Proteome control")
    gene_values = (
        esm["logreg_gene_split"]["macro_f1_mean"],
        proteome["logreg_gene_split"]["macro_f1_mean"],
    )
    family_values = (
        esm["logreg_family_split"]["macro_f1_mean"],
        proteome["logreg_family_split"]["macro_f1_mean"],
    )
    positions = np.arange(len(representation_labels))
    for position, gene_value, family_value in zip(
        positions, gene_values, family_values
    ):
        transfer_axis.plot(
            [position - 0.12, position + 0.12],
            [gene_value, family_value],
            color=LIGHT_GREY,
            linewidth=2,
        )
    transfer_axis.scatter(
        positions - 0.12, gene_values, color=GENE_COLOR, s=55, label="Gene holdout"
    )
    transfer_axis.scatter(
        positions + 0.12,
        family_values,
        color=FAMILY_COLOR,
        s=55,
        label="Family holdout",
    )
    family_reference = _finite(
        esm["majority_reference"]["family_split"]["macro_f1_mean"],
        "enzyme family-split majority reference",
    )
    transfer_axis.axhline(
        family_reference, color=GREY, linestyle="--", linewidth=1
    )
    transfer_axis.set_xticks(positions)
    transfer_axis.set_xticklabels(representation_labels)
    transfer_axis.set_ylim(0.18, 0.88)
    transfer_axis.set_ylabel("Enzyme macro-F1 (five-seed mean)")
    transfer_axis.set_title(
        "Wildtype embedding supported enzyme classification", loc="left"
    )
    transfer_axis.legend(frameon=False)
    _panel_label(transfer_axis, "A")

    auroc_axis = figure.add_subplot(grid[0, 1])
    enzyme_auroc = tuple(
        esm["logreg_family_split"]["per_class_auroc_mean"][class_name]
        for class_name in ENZYME_CLASS_ORDER
    )
    display_labels = ("Kinase", "Oxidoreductase", "Non-enzyme", "Protease")
    _plot_per_class_auroc(
        auroc_axis,
        display_labels,
        enzyme_auroc,
        panel_label="B",
        title="High family-split ranking across enzyme classes",
    )
    auroc_axis.tick_params(axis="x", rotation=15)

    comparison_axis = figure.add_subplot(grid[1, :])
    task_difference = esm["paired_ci_logreg_minus_mechanism"]
    task_point = task_difference["point_diff"]
    task_low = task_difference["ci_low"]
    task_high = task_difference["ci_high"]
    comparison_axis.errorbar(
        [task_point],
        [1],
        xerr=_ci_errors(task_point, task_low, task_high),
        fmt="o",
        color=GREEN,
        capsize=4,
        markersize=7,
    )
    probe_difference = esm["paired_ci_mlp_minus_logreg"]
    probe_point = probe_difference["point_diff"]
    probe_low = probe_difference["ci_low"]
    probe_high = probe_difference["ci_high"]
    comparison_axis.add_patch(
        Rectangle(
            (-0.05, -0.34),
            0.10,
            0.68,
            facecolor=LIGHT_GREY,
            edgecolor="none",
            alpha=0.8,
            zorder=0,
        )
    )
    comparison_axis.errorbar(
        [probe_point],
        [0],
        xerr=_ci_errors(probe_point, probe_low, probe_high),
        fmt="o",
        color=PURPLE,
        capsize=4,
        markersize=7,
    )
    comparison_axis.axvline(0, color=GREY, linewidth=1)
    minimum_gap = enzyme["gate_evaluation"]["2G_minimum_f1_gap"]
    comparison_axis.plot(
        [minimum_gap, minimum_gap],
        [0.66, 1.34],
        color=RED,
        linestyle=":",
        linewidth=1.5,
    )
    comparison_axis.text(
        task_high + 0.018,
        1,
        f"{task_point:+.3f} [{task_low:+.3f}, {task_high:+.3f}]",
        ha="left",
        va="center",
        fontsize=8,
    )
    comparison_axis.text(
        probe_high + 0.018,
        0,
        f"{probe_point:+.3f} [{probe_low:+.3f}, {probe_high:+.3f}]",
        ha="left",
        va="center",
        fontsize=8,
    )
    comparison_axis.text(
        minimum_gap,
        1.34,
        f"minimum gap {minimum_gap:.2f}",
        color=RED,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    comparison_axis.text(
        0,
        -0.30,
        "equivalence range ±0.05",
        color=GREY,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    comparison_axis.set_xlim(-0.15, 0.60)
    comparison_axis.set_ylim(-0.45, 1.45)
    comparison_axis.set_yticks((1, 0))
    comparison_axis.set_yticklabels(("Enzyme − mechanism", "MLP − logistic"))
    comparison_axis.set_xlabel("Family-split macro-F1 difference (95% CI)")
    comparison_axis.set_title("Paired family-split comparisons", loc="left")
    _panel_label(comparison_axis, "C")

    _save_figure(figure, "figure5_enzyme_classification")


def figure6_folding_stability() -> None:
    stability = _load_json(STABILITY_SUMMARY_JSON)
    mlp = _load_json(STABILITY_MLP_SUMMARY_JSON)
    xgb = _load_json(STABILITY_XGB_SUMMARY_JSON)
    per_protein = _load_json(STABILITY_PER_PROTEIN_JSON)
    figure = plt.figure(figsize=(12, 6.6))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(0.48, 1.0),
        wspace=0.36,
        hspace=0.52,
    )

    transfer_axis = figure.add_subplot(grid[:, 0])
    split_names = ("Random", "Domain", "Pfam family")
    split_positions = np.arange(len(split_names))
    model_series = (
        (
            "Ridge (preregistered)",
            BLUE,
            "o",
            "-",
            1.0,
            (
                stability["delta_mean_random"]["across_seed"]["spearman_mean"],
                stability["delta_mean_domain"]["across_seed"]["spearman_mean"],
                stability["delta_mean_family"]["across_seed"]["spearman_mean"],
            ),
        ),
        (
            "MLP (exploratory)",
            RED,
            "s",
            "--",
            0.72,
            (
                mlp["mlp_random"]["across_seed"]["spearman_mean"],
                mlp["mlp_domain"]["across_seed"]["spearman_mean"],
                mlp["mlp_family"]["across_seed"]["spearman_mean"],
            ),
        ),
        (
            "XGBoost (exploratory)",
            GREEN,
            "^",
            "--",
            0.72,
            (
                xgb["xgb_random"]["across_seed"]["spearman_mean"],
                xgb["xgb_domain"]["across_seed"]["spearman_mean"],
                xgb["xgb_family"]["across_seed"]["spearman_mean"],
            ),
        ),
    )
    for model_name, color, marker, linestyle, alpha, values in model_series:
        transfer_axis.plot(
            split_positions,
            values,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=2.2 if linestyle == "-" else 1.5,
            alpha=alpha,
            label=model_name,
        )
    transfer_axis.set_xticks(split_positions)
    transfer_axis.set_xticklabels(split_names, rotation=15, ha="right")
    transfer_axis.set_ylim(0.48, 0.90)
    transfer_axis.set_ylabel("Spearman correlation (five-seed mean)")
    transfer_axis.set_title(
        "Stability signal decreased under stricter holdout", loc="left"
    )
    transfer_axis.legend(frameon=False)
    _panel_label(transfer_axis, "A")

    gap_axis = figure.add_subplot(grid[0, 1])
    gap = stability["3B_gap_ci"]
    gap_point = gap["point_diff"]
    gap_low = gap["ci_low"]
    gap_high = gap["ci_high"]
    gap_axis.errorbar(
        [gap_point],
        [0],
        xerr=_ci_errors(gap_point, gap_low, gap_high),
        fmt="o",
        color=RED,
        capsize=4,
        markersize=7,
    )
    family_dependence_boundary = stability["gates"]["3B"]["threshold"]
    gap_axis.axvline(
        family_dependence_boundary, color=RED, linestyle=":", linewidth=1.5
    )
    gap_axis.axvline(0, color=GREY, linewidth=1)
    gap_axis.text(
        gap_point,
        0.12,
        f"{gap_point:.3f}\n[{gap_low:.3f}, {gap_high:.3f}]",
        ha="center",
        va="bottom",
    )
    gap_axis.set_xlim(-0.01, 0.22)
    gap_axis.set_ylim(-0.25, 0.32)
    gap_axis.set_yticks([])
    gap_axis.set_xlabel("Random-minus-family Spearman correlation")
    gap_axis.set_title("Drop exceeded preregistered tolerance", loc="left")
    _panel_label(gap_axis, "B")

    domain_axis = figure.add_subplot(grid[1, 1])
    per_domain_map = per_protein["per_protein"]
    if not isinstance(per_domain_map, dict) or not per_domain_map:
        raise ValueError("Per-domain stability results are missing")
    domain_correlations = [
        _finite(cell["spearman"], f"{domain_name} Spearman")
        for domain_name, cell in per_domain_map.items()
    ]
    domain_axis.hist(
        domain_correlations, bins=np.linspace(0, 0.9, 19), color=CYAN, edgecolor=WHITE
    )
    domain_mean = stability["per_protein"]["spearman_mean"]
    domain_axis.axvline(
        domain_mean, color=BLUE, linewidth=1.5, label=f"Mean {domain_mean:.3f}"
    )
    domain_std = stability["per_protein"]["spearman_std"]
    std_ci = stability["per_protein"]["spearman_std_ci"]
    domain_axis.text(
        0.03,
        0.95,
        f"Across-domain SD {domain_std:.3f}\n95% CI [{std_ci['ci_low']:.3f}, {std_ci['ci_high']:.3f}]",
        transform=domain_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    domain_axis.set_xlabel("Per-domain Spearman correlation")
    domain_axis.set_ylabel("Domains")
    domain_axis.set_title("Performance varied across domains", loc="left")
    domain_axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.78))
    _panel_label(domain_axis, "C")

    _save_figure(figure, "figure6_folding_stability")


def figure_s1_single_source() -> None:
    aggregate = _load_json(MECHANISM_AGGREGATE_JSON)
    single_source = _load_json(SINGLE_SOURCE_AGGREGATE_JSON)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    figure.subplots_adjust(wspace=0.32)

    delta_axis = axes[0]
    split_labels = ("Gene holdout", "Family holdout")
    split_positions = np.arange(len(split_labels))
    delta_values = (
        _seed_mean(single_source, "gene_split", "delta_mean"),
        _seed_mean(single_source, "family_split", "delta_mean"),
    )
    subset_references = (
        single_source["subset_floor"]["gene_split"]["macro_f1_mean"],
        single_source["subset_floor"]["family_split"]["macro_f1_mean"],
    )
    delta_axis.scatter(
        split_positions - 0.08, delta_values, color=PURPLE, s=55, label="Delta"
    )
    delta_axis.scatter(
        split_positions + 0.08,
        subset_references,
        facecolor=WHITE,
        edgecolor=GREY,
        s=55,
        label="Subset reference",
    )
    delta_axis.set_xticks(split_positions)
    delta_axis.set_xticklabels(split_labels)
    delta_axis.set_ylim(0.26, 0.30)
    delta_axis.set_ylabel("Macro-F1 (five-seed mean)")
    delta_axis.set_title("Delta matched subset-specific references", loc="left")
    delta_axis.legend(frameon=False)
    _panel_label(delta_axis, "A")

    wildtype_axis = axes[1]
    cohort_labels = ("Merged cohort", "Gerasimavicius only")
    gene_values = (
        _seed_mean(aggregate, "gene_split", "wt_only_mean"),
        _seed_mean(single_source, "gene_split", "wt_only_mean"),
    )
    family_values = (
        _seed_mean(aggregate, "family_split", "wt_only_mean"),
        _seed_mean(single_source, "family_split", "wt_only_mean"),
    )
    cohort_positions = np.arange(len(cohort_labels))
    for position, gene_value, family_value in zip(
        cohort_positions, gene_values, family_values
    ):
        wildtype_axis.plot(
            [position - 0.12, position + 0.12],
            [gene_value, family_value],
            color=LIGHT_GREY,
            linewidth=2,
        )
    wildtype_axis.scatter(
        cohort_positions - 0.12,
        gene_values,
        color=GENE_COLOR,
        s=55,
        label="Gene holdout",
    )
    wildtype_axis.scatter(
        cohort_positions + 0.12,
        family_values,
        color=FAMILY_COLOR,
        s=55,
        label="Family holdout",
    )
    wildtype_axis.set_xticks(cohort_positions)
    wildtype_axis.set_xticklabels(cohort_labels)
    wildtype_axis.set_ylim(0.40, 0.65)
    wildtype_axis.set_ylabel("Wildtype macro-F1 (five-seed mean)")
    wildtype_axis.set_title("Wildtype split decrease replicated", loc="left")
    wildtype_axis.legend(frameon=False)
    _panel_label(wildtype_axis, "B")

    _save_figure(figure, "figureS1_single_source")


def figure_s2_stability_direction_ablation() -> None:
    projection = _load_json(STABILITY_PROJECTION_JSON)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    figure.subplots_adjust(wspace=0.35)

    score_axis = axes[0]
    labels = ("Baseline", "Stability direction\nremoved")
    points = (projection["baseline_f1_mean"], projection["projected_f1_mean"])
    errors = (projection["baseline_f1_std"], projection["projected_f1_std"])
    positions = np.arange(len(labels))
    score_axis.errorbar(
        positions, points, yerr=errors, fmt="o", color=PURPLE, capsize=3
    )
    score_axis.set_xticks(positions)
    score_axis.set_xticklabels(labels)
    score_axis.set_ylim(0.37, 0.42)
    score_axis.set_ylabel("Mechanism macro-F1 (five-seed mean ± SD)")
    score_axis.set_title("Mechanism score changed minimally", loc="left")
    _panel_label(score_axis, "A")

    difference_axis = axes[1]
    difference = projection["difference_ci"]
    difference_point = difference["point_diff"]
    difference_low = difference["ci_low"]
    difference_high = difference["ci_high"]
    difference_axis.errorbar(
        [difference_point],
        [0],
        xerr=_ci_errors(difference_point, difference_low, difference_high),
        fmt="o",
        color=PURPLE,
        capsize=4,
        markersize=7,
    )
    difference_axis.axvline(0, color=GREY, linewidth=1)
    difference_axis.axvline(0.01, color=RED, linestyle=":", linewidth=1.5)
    difference_axis.text(
        difference_point,
        0.12,
        f"{difference_point:+.4f}\n[{difference_low:+.4f}, {difference_high:+.4f}]",
        ha="center",
        va="bottom",
    )
    difference_axis.set_xlim(-0.006, 0.014)
    difference_axis.set_xticks((-0.005, 0.000, 0.005, 0.010))
    difference_axis.set_ylim(-0.35, 0.45)
    difference_axis.set_yticks([])
    difference_axis.set_xlabel("Projected minus baseline mechanism macro-F1")
    difference_axis.set_title("Paired family-bootstrap difference", loc="left")
    _panel_label(difference_axis, "B")

    _save_figure(figure, "figureS2_stability_direction_ablation")


FIGURE_FUNCTIONS = {
    "1": figure1_study_design,
    "2": figure2_mechanism_delta,
    "3": figure3_family_information,
    "4": figure4_pathogenicity_conservation,
    "5": figure5_enzyme_classification,
    "6": figure6_folding_stability,
    "S1": figure_s1_single_source,
    "S2": figure_s2_stability_direction_ablation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("all", *FIGURE_FUNCTIONS),
        default="all",
        help="Figure to generate (default: all)",
    )
    args = parser.parse_args()
    _configure_style()

    selected = (
        FIGURE_FUNCTIONS
        if args.figure == "all"
        else {args.figure: FIGURE_FUNCTIONS[args.figure]}
    )
    for figure_name, figure_function in selected.items():
        print(f"generating figure {figure_name}")
        figure_function()


if __name__ == "__main__":
    main()
