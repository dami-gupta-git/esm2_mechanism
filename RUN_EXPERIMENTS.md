# Running AI Scientist on RunPod

## Overview

The workflow is:
1. Generate hypotheses locally (`--ideas-only`)
2. SSH into RunPod, pull the branch, run the baseline experiment to build `run_0/`
3. Run `launch_scientist.py` — it uses Aider + Claude to implement and run each idea
4. Pull results back locally

---

## 1. SSH Key Setup (one-time)

Generate a key on your Mac and add it to the RunPod pod:

```bash
# Generate key
ssh-keygen -t ed25519 -f ~/.ssh/id_runpod -C "runpod" -N ""

# Copy public key — paste this into the RunPod web terminal (see below)
cat ~/.ssh/id_runpod.pub
```

In the RunPod web terminal for your pod, run:
```bash
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
```

Then connect using the direct pod IP (find it in the RunPod console under "Connect"):
```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_runpod root@<POD_IP> -p <PORT>
```

Note: the `ssh.runpod.io` proxy requires a key registered in RunPod account settings. The direct IP method above always works once the key is in `authorized_keys`.

---

## 2. One-Time Pod Setup

Run once on a fresh pod:

```bash
# Install system deps
apt-get update -y && apt-get install -y tmux

# Install Python deps
pip3 install -r requirements.txt

# Install flash-attn (required for Evo2, compiles from source — takes ~15 min)
pip3 install flash-attn --no-build-isolation

# Patch Evo2 checkpoint loader (required for PyTorch >= 2.4)
sed -i 's/weights_only=True/weights_only=False/' \
  /usr/local/lib/python3.11/dist-packages/vortex/model/utils.py
```

---

## 3. Clone and Set Up the Repo

```bash
cd /workspace
git clone https://github.com/dami-gupta-git/dami-AI-Scientist.git
cd dami-AI-Scientist
git checkout <branch>   # e.g. evo2-supervised
```




---

## 5. Run the Baseline Experiment on RunPod

The baseline must be run first — it fetches gene sequences, extracts embeddings, and writes `run_0/final_info.json` which AI Scientist uses as the starting point.

```bash
# On RunPod, in a tmux session so it survives disconnection
tmux new-session -d -s baseline \
  'cd /workspace/dami-AI-Scientist && \
   git pull && \
   python3 templates/evo2_function/experiment.py --out_dir templates/evo2_function/run_0 \
   2>&1 | tee /tmp/baseline.log; echo DONE >> /tmp/baseline.log'

# Monitor
tail -f /tmp/baseline.log
```

Expected duration: ~1-2 hours (gene fetch ~45 min + Evo2 model download ~30 min + embedding extraction ~30 min).

When complete, `run_0/` will contain:
- `data/dataset.json` — gene sequences and labels (178 genes)
- `data/embeddings_evo2_7b.npy` — Evo2 embeddings (178 × 4096)
- `final_info.json` — baseline metrics (LR/SVM/MLP AUROCs)

Pull the cached data locally to avoid re-fetching next time:
```bash
# On your Mac
scp -i ~/.ssh/id_runpod -P <PORT> \
  root@<POD_IP>:/workspace/dami-AI-Scientist/templates/evo2_function/run_0/data/dataset.json \
  templates/evo2_function/run_0/data/dataset.json
```

---

## 6. Run AI Scientist

```bash
# On RunPod
tmux new-session -d -s aiscientist \
  'cd /workspace/dami-AI-Scientist && \
   ANTHROPIC_API_KEY=<key> OPENAI_API_KEY=<key> \
   python3 launch_scientist.py \
     --experiment evo2_function \
     --model claude-sonnet-4-5 \
     --skip-idea-generation \
     --skip-novelty-check \
     --no-writeup \
   2>&1 | tee /tmp/aiscientist.log; echo DONE >> /tmp/aiscientist.log'

# Monitor
tail -f /tmp/aiscientist.log
```

Key flags:
| Flag | Purpose |
|---|---|
| `--skip-idea-generation` | Use existing `ideas.json` instead of generating new ones |
| `--skip-novelty-check` | Skip Semantic Scholar API (no key needed) |
| `--no-writeup` | Skip LaTeX paper generation (faster) |
| `--num-ideas N` | Only run the first N novel ideas |
| `--model claude-sonnet-4-5` | Required — older claude-3-5-sonnet models return 404 |

To generate papers, omit `--no-writeup` and install LaTeX first:
```bash
apt-get install -y texlive-full   # ~2 GB, takes ~10 min
```

---

## 7. Pull Results Locally

Results are written to `results/<experiment>/<timestamp>_<idea_name>/`.

```bash
# On your Mac — pull all results for an experiment
scp -r -i ~/.ssh/id_runpod -P <PORT> \
  root@<POD_IP>:/workspace/dami-AI-Scientist/results/evo2_function/ \
  results/evo2_function/
```

Each result folder contains:
- `notes.txt` — idea description + per-run results
- `log.txt` — full Aider + experiment log
- `final_info.json` — metrics
- `*.pdf` — paper (if `--no-writeup` was not set)
- `run_*/` — per-seed experiment outputs and plots

---

## 8. Current Experiments

| Branch | Experiment | Status |
|---|---|---|
| `evo2-xgboost` | Evo2 XGBoost probe comparison | In progress |
| `esm2-mechanism` | ESM-2 delta-embeddings — GOF/DN/LOF mechanism geometry | Ready to run |
| `evo2-supervised` | Evo2 supervised probe comparison (LR vs SVM vs MLP) | Previously run |
| `esm2-depmap` | ESM-2 vs DepMap Mantel test | Previously run |

## Running esm2_mechanism

This is a standalone repo — not part of dami-AI-Scientist. Clone it directly on RunPod.

**Clone the repo on RunPod:**
```bash
cd /workspace
git clone https://github.com/dami-gupta-git/esm2_mechanism.git
cd esm2_mechanism
```

**Baseline run on RunPod (~2-2.5 hours on A100):**
```bash
tmux new-session -d -s baseline \
  'cd /workspace/esm2_mechanism && \
   git pull && \
   python3 scripts/experiment.py --out_dir results/run_0 \
   2>&1 | tee /tmp/baseline.log; echo DONE >> /tmp/baseline.log'

tail -f /tmp/baseline.log
```

Time breakdown:
- OSF dataset download + parse: ~5 min
- UniProt sequence fetch (~1200 genes): ~20 min
- Pfam family fetch (~1200 genes): ~20 min
- AlphaMissense scores (~8000 variants, rate-limited): ~30-45 min
- ESM-2 650M embedding extraction (~8000 variant pairs): ~30-45 min
- Probes + baselines + orthogonality: ~15 min

**Pull results back locally:**
```bash
# On your Mac
scp -r -i ~/.ssh/id_runpod -P <PORT> \
  root@<POD_IP>:/workspace/esm2_mechanism/results/run_0 \
  results/
```

**Pull cached data locally to avoid re-fetching next time:**
```bash
scp -r -i ~/.ssh/id_runpod -P <PORT> \
  root@<POD_IP>:/workspace/esm2_mechanism/data/raw/ \
  data/raw/
scp -r -i ~/.ssh/id_runpod -P <PORT> \
  root@<POD_IP>:/workspace/esm2_mechanism/data/embeddings/ \
  data/embeddings/
```

## Current RunPod Connection

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_runpod root@216.81.245.125 -p 10075
```
