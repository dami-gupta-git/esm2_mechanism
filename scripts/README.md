# Scripts

## Core pipeline

**`experiment.py`**
Full baseline pipeline. Downloads Gerasimavicius dataset, fetches UniProt sequences and Pfam families, extracts ESM-2 650M embeddings, fits stability subspace, runs linear probe with gene-split and family-split CV, baselines, and probe direction orthogonality. Writes `run_0/final_info.json`. Requires GPU for embedding extraction.

**`experiment_mlp.py`**
Nonlinear probes on cached delta embeddings. PyTorch MLP (1280→256→64→3) with dropout, class weighting, and gene-split early stopping. Also runs GBM, RF (PCA-50), and kNN probes. Use `--family_split` to run family-split CV alongside gene-split. CPU-only (reads cached .npy files). Produces `mlp_results_seed{N}.json`.

## Analysis scripts

**`family_clustering.py`**
Diagnostic: quantifies how strongly ESM-2 embeddings cluster by Pfam family. Computes silhouette, k-NN family purity (k=5,10) vs shuffled null, within/between cosine distance ratio, and a linear family-probe accuracy on gene-level embeddings. Run this to test whether apparent mechanism signal could be family-mediated homology leakage. Produces `family_clustering.json`.

**`family_split_baselines.py`**
Runs all baselines (WT-only, mutant-only, WT+mutant concat, delta mean, delta per-residue, one-hot AA, FoldX ΔΔG, AlphaMissense) under both gene-split and family-split CV. Prints Δ(gene−family) for each feature — positive values indicate homology leakage. Produces `family_split_baselines.json`.

**`pathogenicity_control.py`**
Positive control: predicts ClinVar pathogenic vs benign on the same gene set using the same pipeline. Three phases: (1) fetch ClinVar variants (CPU), (2) extract ESM-2 embeddings (GPU), (3) run linear + MLP probes under gene-split and family-split (CPU). Validates pipeline soundness — if pathogenicity AUROC ≥ 0.85 and survives family-split, the mechanism null result is interpretable. Produces `pathogenicity_control.json`.


## Data scripts

**`fetch_clinvar_variants.py`**
Fetches ClinVar pathogenic/likely-pathogenic missense variants for a gene list (reads `merged_gene_list.tsv`). Resume-safe, rate-limited to ≤3 NCBI requests/second. Outputs `clinvar_variants.tsv`. Used to get variants for G2P genes that lack Gerasimavicius coverage.

## Visualisation

**`plot.py`**
Matplotlib plots from `final_info.json`: AUROC bar chart per class, probe direction cosine matrix heatmap, variance-explained bar chart. Run as `python plot.py run_0`.
