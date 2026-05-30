"""
Synthetic variant dataset for pipeline testing.

NOT for scientific use — only validates that the embedding pipeline runs
end-to-end when the real OSF / UniProt data is unavailable.
"""

import numpy as np


def make_synthetic_variants(seed: int = 42) -> list[dict]:
    """Return a minimal list of synthetic variant dicts with the same schema as merged_variants.json."""
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.RandomState(seed)
    genes_mechanisms = [
        ("TP53", "P04637", "GOF"),
        ("KRAS", "P01116", "GOF"),
        ("EGFR", "P00533", "GOF"),
        ("BRAF", "P15056", "GOF"),
        ("PIK3CA", "P42336", "GOF"),
        ("MYC", "P01106", "GOF"),
        ("CTNNB1", "P35222", "GOF"),
        ("IDH1", "O75874", "GOF"),
        ("TP53", "P04637", "DN"),
        ("SMAD2", "Q15796", "DN"),
        ("SMAD3", "P84022", "DN"),
        ("SMAD4", "Q13485", "DN"),
        ("RUNX1", "Q01196", "DN"),
        ("PAX5", "Q02548", "DN"),
        ("WT1", "P19544", "DN"),
        ("SOX9", "P48436", "DN"),
        ("BRCA1", "P38398", "HI"),
        ("BRCA2", "P51587", "HI"),
        ("RB1", "P06400", "HI"),
        ("PTEN", "P60484", "HI"),
        ("VHL", "P40337", "AR"),
        ("CFTR", "P13569", "AR"),
        ("HEXA", "P06865", "AR"),
        ("MUTYH", "Q9UIF7", "AR"),
    ]
    variants = []
    for gene, uniprot, mech in genes_mechanisms:
        label_3class = "LOF" if mech in ("HI", "AR") else mech
        for _ in range(15):
            pos = int(rng.randint(1, 300))
            wt = rng.choice(aa_list)
            mut = rng.choice([aa for aa in aa_list if aa != wt])
            variants.append(
                {
                    "gene": gene,
                    "uniprot_id": uniprot,
                    "aa_pos": pos,
                    "aa_wt": wt,
                    "aa_mut": mut,
                    "mechanism": mech,
                    "label_3class": label_3class,
                    "foldx_ddg": float(rng.randn()),
                    "clinvar_id": f"SYNTH_{gene}_{pos}{wt}{mut}",
                    "source": "synthetic",
                }
            )
    return variants
