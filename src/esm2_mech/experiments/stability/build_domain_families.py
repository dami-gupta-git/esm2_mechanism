"""Assign Tsuboyama domains to Pfam families via HMMER and report membership."""

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
    """Return {domain: wt_seq} from parsed variant records."""
    wt = {}
    for variant in variants:
        domain = variant["protein"]
        if domain not in wt:
            wt[domain] = variant["wt_seq"]
    return wt


def run_hmmscan(wt_seqs, hmm_db=None, cpu=4):
    """Run hmmscan --cut_ga on WT sequences; return {domain: best_pfam_acc}."""
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

        # HMMER writes "-" in the accession column for models without an ACC field;
        # fall back to target NAME so unrelated domains aren't merged into one "-" family.
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
    """Build and cache {domain: PfamID}; print membership sanity check."""
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
    """Print per-family domain counts."""
    members = defaultdict(list)
    for domain, family in family_map.items():
        members[family].append(domain)

    sizes = Counter({fam: len(ds) for fam, ds in members.items()})
    singletons = [fam for fam, count in sizes.items() if count == 1]
    multi = [fam for fam, count in sizes.items() if count > 1]
    domains_in_multi = sum(count for count in sizes.values() if count > 1)

    print("")
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
