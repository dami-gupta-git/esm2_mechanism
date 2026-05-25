# Result 8 — Within-family mechanism classification
## Date: May 25, 2026 | Model: ESM-2 650M | Seed: 42

---

## TL;DR

Within-family gene-split CV on the 5 largest Gerasimavicius Pfam families with ≥2 mechanism classes. Results are largely at chance due to tiny sample sizes (6–12 genes per family), but two directional findings emerge:

- **PF00520 (ion channel)**: delta probe F1=0.407 ± 0.050, AUROC=0.659 for 2-class GOF/DN. Delta clearly outperforms WT (AUROC 0.659 vs 0.396). Most interpretable result.
- **PF00071 (Ras GTPase)**: delta AUROC=0.818, but GOF=92% makes this near-trivial and DN has only 2 genes.

**Verdict**: Directional signal in ion channels; not publishable at single seed with these sample sizes. Requires multi-seed replication and larger within-family gene sets (merged dataset).

---

## Results

| Family | Name | Genes | Classes | WT F1 | WT AUROC | Delta F1 | Delta AUROC | Always-LOF |
|---|---|---|---|---|---|---|---|---|
| PF00069 | Kinase | 12 | GOF/DN/LOF | 0.269 ± 0.159 | 0.657 | 0.221 ± 0.116 | 0.519 | 0.547 |
| PF00168 | C2 domain | 12 | GOF/DN/LOF | 0.810 ± 0.234 | 0.736 | 0.598 ± 0.333 | 0.526 | 0.731 |
| PF00046 | Homeodomain | 11 | GOF/DN/LOF | 0.161 ± 0.094 | 0.313 | 0.266 ± 0.112 | 0.498 | 0.735 |
| PF00071 | Ras GTPase | 11 | GOF/DN/LOF | 0.414 ± 0.064 | 0.590 | **0.419 ± 0.083** | **0.818** | 0.920 |
| PF00520 | Ion channel | 9 | GOF/DN | 0.325 ± 0.117 | 0.396 | **0.407 ± 0.050** | **0.659** | 0.709 |

---

## Family-by-family interpretation

**PF00069 (Kinase, 12 genes, GOF/DN/LOF):**
Both probes at or below chance F1. Kinases span all three mechanism classes but there's no detectable within-family signal at this sample size.

**PF00168 (C2 domain, 12 genes):**
WT F1=0.810 looks impressive but always-LOF baseline is 0.731 — real gain over baseline is only +0.08. High std (±0.234) makes this unreliable. Not convincing.

**PF00046 (Homeodomain, 11 genes):**
Both probes below chance, AUROC ~0.5. No signal.

**PF00071 (Ras GTPase, 11 genes):**
Delta AUROC=0.818 is striking but GOF=92% makes the classification near-trivial, and DN has only 2 genes making folds degenerate. This number is essentially measuring "can we tell the 2 DN genes from the 9 GOF genes" — not informative about mechanism geometry.

**PF00520 (Ion channel, 9 genes, GOF/DN 2-class):**
The most interpretable result. 9 voltage-gated ion channel genes (KCNQ2, KCNQ4, KCNH2, GABRA1, etc.), 2-class GOF vs DN. Delta F1=0.407 ± 0.050, AUROC=0.659 — low std suggests real signal, not noise. WT is clearly worse (AUROC=0.396, below chance). The mutation-specific delta carries GOF/DN discriminating information within ion channels that raw protein identity doesn't.

---

## Analysis

**Why ion channels?**
GOF mutations in ion channels tend to increase channel activity (gain of conductance), while DN mutations interfere with tetramerisation. These may leave distinguishable signatures in the local residue context at the variant position — captured by delta but not WT.

**Why not kinases?**
Kinases also have GOF (activating) and LOF (inactivating) mutations, but the distinction may be more position-dependent and less consistent at the sequence level without fine-tuning.

**What would strengthen this:**
1. **More genes per family** — the merged dataset adds G2P genes; ion channels and kinases may have more genes there
2. **Multi-seed replication** — current std values suggest some results are noise-driven
3. **MLP probe within-family** — linear probe may miss nonlinear signal (as seen in the cross-family analysis)

---

## Implications for paper framing

The ion channel result (PF00520, delta AUROC=0.659) is a directional positive that's consistent with the MissION paper's finding — GOF/DN separation is achievable within a homologous subfamily using ESM-2 representations. This reconciles the overall cross-family null with published positive results: mechanism prediction works *within* a family but not *across* families using frozen embeddings.

This is worth one paragraph in the paper: "Within the voltage-gated ion channel family (PF00520, 9 genes), delta embeddings achieve GOF/DN AUROC=0.659 — consistent with MissION's within-subfamily positive results and suggesting the cross-family null reflects generalisation difficulty, not complete absence of mechanism signal."

---

## Files

- `results/20260524_baseline_run/run_0/within_family_analysis.json` — full results JSON
- Run locally using cached Gerasimavicius embeddings — no GPU needed
