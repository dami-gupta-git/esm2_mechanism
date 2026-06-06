"""
Assign each Tsuboyama natural domain to a Pfam family via HMMER, and report
family membership as a sanity check.

Pipeline:
  1. Load the parsed Tsuboyama variants, collect one WT sequence per domain.
  2. Write a FASTA and run `hmmscan --cut_ga` against the (hmmpress-ed) Pfam-A db.
  3. For each domain, take the best (lowest E-value) Pfam hit as its family.
     Domains with no hit are ORPHANS — absent from the map, excluded from
     family-split only (they still count for random/domain-split and per-domain).
  4. Write {domain: PfamID} to MEGASCALE_DOMAIN_FAMILIES_JSON and print a
     membership breakdown (families, sizes, singletons, orphans).

Requires `hmmscan` on PATH and PFAM_A_HMM hmmpress-ed (.h3{f,i,m,p} alongside it).

Usage:
  python -m esm2_mech.experiments.stability.build_domain_families
"""

import functools
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    PFAM_A_HMM,
    MEGASCALE_DOMAIN_FAMILIES_JSON,
)

print = functools.partial(print, flush=True)

HMMSCAN = "hmmscan"


def _wt_sequences(variants):
    """One WT sequence per domain, from the parsed variant records."""
    wt = {}
    for variant in variants:
        domain = variant["protein"]
        if domain not in wt:
            wt[domain] = variant["wt_seq"]
    return wt


def run_hmmscan(wt_seqs, hmm_db=None, cpu=4):
    """Run hmmscan --cut_ga on the WT sequences; return {domain: best_pfam_acc}.

    best = lowest full-sequence E-value among a domain's hits (tblout column 5).
    Pfam accessions are returned without the version suffix (PF00018.24 -> PF00018).
    Domains with no hit are simply absent from the returned dict.
    """
    hmm_db = str(hmm_db or PFAM_A_HMM)
    if not shutil.which(HMMSCAN):
        raise RuntimeError(f"{HMMSCAN} not found on PATH")
    if not os.path.exists(hmm_db):
        raise FileNotFoundError(hmm_db)
    # hmmscan needs the pressed binary indexes next to the .hmm
    if not os.path.exists(hmm_db + ".h3m"):
        raise FileNotFoundError(
            f"{hmm_db}.h3m — Pfam-A database is not hmmpress-ed"
        )

    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "domains.fasta")
        with open(fasta, "w") as handle:
            for domain, seq in wt_seqs.items():
                handle.write(f">{domain}\n{seq}\n")

        tblout = os.path.join(tmp, "pfam.tbl")
        subprocess.run(
            [HMMSCAN, "--cut_ga", "--cpu", str(cpu),
             "--tblout", tblout, "-o", os.path.join(tmp, "hmmscan.log"),
             hmm_db, fasta],
            check=True,
        )

        # tblout columns (1-based): target_name(1) accession(2) query/domain(3)
        #                           query_acc(4) full_evalue(5) full_score(6) ...
        # Family id = the Pfam accession (col 2, e.g. PF00018.24 -> PF00018). HMMER
        # writes "-" in the accession column for any model without an ACC field; in
        # that case fall back to the target NAME (col 1, always present and unique
        # per model) so unrelated domains are NOT all merged into one bogus "-"
        # family — which would force them into the same family-split fold.
        hits = defaultdict(list)
        n_malformed = 0
        n_acc_fallback = 0
        with open(tblout) as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    n_malformed += 1
                    continue
                target_name, pfam_acc, domain = parts[0], parts[1], parts[2]
                evalue = float(parts[4])
                if pfam_acc == "-":
                    family_id = target_name
                    n_acc_fallback += 1
                else:
                    family_id = pfam_acc.split(".")[0]
                hits[domain].append((evalue, family_id))

        if n_malformed:
            print(
                f"WARNING: skipped {n_malformed} tblout line(s) with < 5 columns "
                f"(malformed/truncated)"
            )
        if n_acc_fallback:
            print(
                f"WARNING: {n_acc_fallback} hit(s) had no Pfam accession ('-'); "
                f"used the target name as the family id instead"
            )

    return {d: min(hit_list)[1] for d, hit_list in hits.items()}


def build_family_map(variants=None, out_path=None):
    """Build and cache the {domain: PfamID} map; print a membership sanity check."""
    if variants is None:
        variants = load_tsuboyama_variants()
    out_path = str(out_path or MEGASCALE_DOMAIN_FAMILIES_JSON)

    wt_seqs = _wt_sequences(variants)
    print(f"Domains with variants: {len(wt_seqs)}")

    family_map = run_hmmscan(wt_seqs)
    orphans = sorted(d for d in wt_seqs if d not in family_map)

    print_membership(wt_seqs, family_map, orphans)

    atomic_write_json(out_path, family_map, indent=2)
    print(f"\nWrote {len(family_map)} domain→Pfam assignments to {out_path}")
    if orphans:
        print(f"{len(orphans)} orphan domains (no Pfam) excluded from family-split")
    return family_map


def print_membership(wt_seqs, family_map, orphans):
    """Print the family-membership sanity check: per-family domain counts."""
    members = defaultdict(list)
    for domain, family in family_map.items():
        members[family].append(domain)

    sizes = Counter({fam: len(ds) for fam, ds in members.items()})
    singletons = [fam for fam, count in sizes.items() if count == 1]
    multi = [fam for fam, count in sizes.items() if count > 1]
    domains_in_multi = sum(count for count in sizes.values() if count > 1)

    print("\n── Family membership ──")
    print(f"  domains total            : {len(wt_seqs)}")
    print(f"  assigned a Pfam family   : {len(family_map)}")
    print(f"  orphans (no Pfam)        : {len(orphans)}")
    print(f"  distinct Pfam families   : {len(sizes)}")
    print(f"  singleton families       : {len(singletons)}")
    print(f"  multi-member families    : {len(multi)}")
    print(
        f"  domains in multi families: {domains_in_multi} "
        f"({100 * domains_in_multi / max(1, len(family_map)):.0f}% of assigned)"
    )

    print("\n  Per-family counts (multi-member, descending):")
    for fam, count in sizes.most_common():
        if count == 1:
            continue
        example = ", ".join(sorted(members[fam])[:6])
        more = "" if count <= 6 else f", +{count - 6} more"
        print(f"    {fam}  n={count:<3}  {example}{more}")

    if orphans:
        print("\n  Orphan domains (no Pfam family):")
        print("    " + " ".join(orphans))


if __name__ == "__main__":
    build_family_map()
