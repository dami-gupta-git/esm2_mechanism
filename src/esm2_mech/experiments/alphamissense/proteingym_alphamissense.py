"""Per-DMS AlphaMissense AUROC and Spearman analysis on ProteinGym human assays."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import (
    AM_CACHE_FILE,
    DATA_DIR as _DATA_DIR,
    PROTEINGYM_CACHE_DIR,
    RESULTS_DIR as _RESULTS_DIR,
)

DATA = PROTEINGYM_CACHE_DIR
RESULTS = _RESULTS_DIR / "proteingym_alphamissense"
AM_FILE = AM_CACHE_FILE

DMS_INDEX = DATA / "DMS_substitutions.csv"
DMS_DIR = DATA / "DMS_ProteinGym_substitutions"
MAP_CACHE = DATA / "mnemonic_to_acc.json"
SCORE_CACHE = DATA / "am_scores_proteingym.json"


def load_mnemonic_map(mnemonics: list[str]) -> dict[str, str | None]:
    """Map UniProt mnemonics to accessions, caching only definitive answers.

    A mnemonic that genuinely has no entry is cached as None so it is not queried
    again. A transient failure is left out of the cache entirely: writing it as a
    permanent "no accession" would drop that assay from every later run with no
    way to tell the two apart.
    """
    cache: dict[str, str | None] = {}
    if MAP_CACHE.exists():
        try:
            with open(MAP_CACHE) as _f:
                cache = json.load(_f)
        except json.JSONDecodeError:
            print(f"  {MAP_CACHE.name} is corrupt — refetching", file=sys.stderr)
            MAP_CACHE.unlink()
            cache = {}
    missing = [m for m in mnemonics if m not in cache]
    if not missing:
        return cache
    print(f"querying UniProt for {len(missing)} mnemonics...")
    transient_failures = []
    for i, m in enumerate(missing, 1):
        url = f"https://rest.uniprot.org/uniprotkb/{m}.tsv?fields=accession"
        try:
            req = Request(url, headers={"User-Agent": "claude-script/1.0"})
            with urlopen(req, timeout=30) as r:
                lines = r.read().decode().strip().splitlines()
            # A 2xx with no data row is a definitive "no such mnemonic".
            cache[m] = lines[1].split("\t")[0] if len(lines) >= 2 else None
        except HTTPError as e:
            if e.code == 404:
                cache[m] = None
            else:
                transient_failures.append(m)
                print(f"  WARN: {m} -> {e} (not cached)", file=sys.stderr)
        except URLError as e:
            transient_failures.append(m)
            print(f"  WARN: {m} -> {e} (not cached)", file=sys.stderr)
        if i % 20 == 0:
            print(f"  {i}/{len(missing)}", file=sys.stderr)
        time.sleep(0.05)
    if transient_failures:
        print(
            f"  {len(transient_failures)} mnemonic(s) failed transiently and were "
            f"not cached; rerun to retry them: {transient_failures[:5]}",
            file=sys.stderr,
        )
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP_CACHE, "w") as _f:
        json.dump(cache, _f, indent=2, sort_keys=True)
    return cache


def collect_target_keys(
    assays: list[dict], acc_map: dict[str, str]
) -> tuple[dict[tuple[str, str], list[tuple[str, str]]], dict]:
    """Build (accession, variant) lookup index and per-assay metadata."""
    index: dict[tuple[str, str], list[tuple[str, str]]] = {}
    meta: dict[str, dict] = {}
    for a in assays:
        dms_id = a["DMS_id"]
        mnemonic = a["UniProt_ID"]
        acc = acc_map.get(mnemonic, "")
        dms_path = DMS_DIR / a["DMS_filename"]
        if not acc or not dms_path.exists():
            meta[dms_id] = {
                "skipped": True,
                "reason": "no_acc" if not acc else "no_dms_file",
            }
            continue
        try:
            df = pd.read_csv(dms_path, usecols=["mutant", "DMS_score", "DMS_score_bin"])
        except Exception as e:
            meta[dms_id] = {"skipped": True, "reason": f"dms_read: {e}"}
            continue
        df = df[~df["mutant"].astype(str).str.contains(":")].reset_index(drop=True)
        if df.empty:
            meta[dms_id] = {"skipped": True, "reason": "no_single_mutants"}
            continue
        meta[dms_id] = {
            "skipped": False,
            "mnemonic": mnemonic,
            "accession": acc,
            "n_variants_in_dms": int(len(df)),
            "dms_df": df,  # held in memory; ProteinGym DMS files are small
        }
        for mut in df["mutant"].astype(str).tolist():
            index.setdefault((acc, mut), []).append((dms_id, mut))
    return index, meta


def stream_am(index: dict[tuple[str, str], list]) -> dict[tuple[str, str], float]:
    needed = len(index)
    print(f"streaming AM, looking for {needed:,} (accession, variant) pairs")
    out: dict[tuple[str, str], float] = {}
    unparseable = 0
    with gzip.open(AM_FILE, "rt") as f:
        for i, line in enumerate(f):
            if i % 10_000_000 == 0 and i:
                print(
                    f"  read {i:,} rows; matched {len(out):,}/{needed:,}",
                    file=sys.stderr,
                )
            if not line or line.startswith("#") or line.startswith("uniprot_id"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            key = (parts[0], parts[1])
            if key in index:
                try:
                    out[key] = float(parts[2])
                except ValueError:
                    unparseable += 1
    if unparseable:
        print(
            f"  {unparseable} matched row(s) had an unparseable score and were dropped",
            file=sys.stderr,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxon", default="Human")
    ap.add_argument("--min-variants", type=int, default=20)
    ap.add_argument(
        "--use-score-cache",
        action="store_true",
        help="skip AM streaming if score cache exists",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    with open(DMS_INDEX) as _f:
        assays = list(csv.DictReader(_f))
    human = [a for a in assays if a["taxon"] == args.taxon]
    print(f"{args.taxon} assays in index: {len(human)}")

    mnemonics = sorted({a["UniProt_ID"] for a in human})
    acc_map = load_mnemonic_map(mnemonics)
    print(
        f"mapped {sum(1 for m in mnemonics if acc_map.get(m))}/{len(mnemonics)} mnemonics"
    )

    index, meta = collect_target_keys(human, acc_map)
    print(
        f"built lookup index: {len(index):,} (accession, variant) pairs across "
        f"{sum(1 for m in meta.values() if not m['skipped'])} usable assays"
    )

    if args.use_score_cache and SCORE_CACHE.exists():
        with open(SCORE_CACHE) as _f:
            cached = json.load(_f)
        scores = {tuple(k.split("|", 1)): float(v) for k, v in cached.items()}
        print(f"loaded {len(scores):,} cached scores from {SCORE_CACHE}")
    else:
        scores = stream_am(index)
        with open(SCORE_CACHE, "w") as _f:
            json.dump({f"{a}|{v}": s for (a, v), s in scores.items()}, _f)
        print(f"cached {len(scores):,} scores to {SCORE_CACHE}")

    per_assay = {}
    for dms_id, m in meta.items():
        if m["skipped"]:
            per_assay[dms_id] = {"skipped": True, "reason": m["reason"]}
            continue
        acc = m["accession"]
        df = m["dms_df"]
        s = df["mutant"].apply(lambda mu: scores.get((acc, mu))).to_numpy()
        mask = pd.notnull(s)
        n = int(mask.sum())
        if n < args.min_variants:
            per_assay[dms_id] = {"skipped": True, "reason": "too_few", "n": n}
            continue
        # DMS_score_bin==1 is functional; we invert so damaging=positive, matching ClinVar convention.
        bin_dmg = (df["DMS_score_bin"].astype(int).to_numpy()[mask] == 0).astype(int)
        sc = df["DMS_score"].to_numpy()[mask]
        ss = s[mask].astype(float)
        try:
            auroc = (
                float(roc_auc_score(bin_dmg, ss))
                if len(np.unique(bin_dmg)) >= 2
                else None
            )
        except Exception:
            auroc = None
        # Negate Spearman so positive means AM agrees with low fitness.
        rho_raw, _ = spearmanr(sc, ss)
        rho = -rho_raw if rho_raw == rho_raw else None
        per_assay[dms_id] = {
            "skipped": False,
            "mnemonic": m["mnemonic"],
            "accession": acc,
            "n_variants": n,
            "n_damaging": int(bin_dmg.sum()),
            "n_functional": int((1 - bin_dmg).sum()),
            "auroc": auroc,
            "spearman_neg": float(rho) if rho is not None else None,
            "coverage": float(n / m["n_variants_in_dms"]),
        }

    ok = [v for v in per_assay.values() if not v["skipped"]]
    print(f"\nassays scored: {len(ok)} / {len(meta)}")
    skipped_reasons = {}
    for v in per_assay.values():
        if v.get("skipped"):
            skipped_reasons.setdefault(v.get("reason"), 0)
            skipped_reasons[v["reason"]] += 1
    if skipped_reasons:
        print(f"skipped: {skipped_reasons}")

    aurocs = [v["auroc"] for v in ok if v["auroc"] is not None]
    rhos = [v["spearman_neg"] for v in ok if v["spearman_neg"] is not None]
    covs = [v["coverage"] for v in ok]

    summary = {
        "taxon": args.taxon,
        "n_assays_indexed": len(human),
        "n_assays_scored": len(ok),
        "median_coverage": float(np.median(covs)) if covs else None,
        "auroc": _dist(aurocs),
        "spearman": _dist(rhos),
    }

    print("\n=== AUROC across assays ===")
    print(json.dumps(summary["auroc"], indent=2))
    print("\n=== Spearman across assays ===")
    print(json.dumps(summary["spearman"], indent=2))
    print(
        f"\nmedian per-assay coverage (AM/DMS variants): {summary['median_coverage']:.3f}"
    )

    ranked = sorted(
        (
            (k, v)
            for k, v in per_assay.items()
            if not v["skipped"] and v["auroc"] is not None
        ),
        key=lambda kv: kv[1]["auroc"],
    )
    print("\nWORST 5 (by AUROC):")
    for k, v in ranked[:5]:
        print(
            f"  {k:55s}  AUROC={v['auroc']:.3f}  -ρ={v['spearman_neg']:.3f}  n={v['n_variants']}"
        )
    print("\nBEST 5 (by AUROC):")
    for k, v in ranked[-5:]:
        print(
            f"  {k:55s}  AUROC={v['auroc']:.3f}  -ρ={v['spearman_neg']:.3f}  n={v['n_variants']}"
        )

    with open(RESULTS / "per_assay.json", "w") as f:
        json.dump(per_assay, f, indent=2, sort_keys=True, default=str)
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS}/per_assay.json and summary.json")
    return 0


def _dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "min": float(np.min(xs)),
        "q25": float(np.quantile(xs, 0.25)),
        "median": float(np.median(xs)),
        "q75": float(np.quantile(xs, 0.75)),
        "max": float(np.max(xs)),
        "frac_below_0_7": float(sum(x < 0.7 for x in xs) / len(xs)),
        "frac_below_0_6": float(sum(x < 0.6 for x in xs) / len(xs)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
