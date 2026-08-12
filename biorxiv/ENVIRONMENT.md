# run_biorxiv environment

What the numbers were produced under. Reports cite this file in Provenance.

Two environments, because the GPU steps do not run locally. Each result file records which one
produced it via the runbook's step table; a report covering both must say so.

## CPU environment (local — all probe and bootstrap steps)

```
python           3.13.7   macOS-15.6-arm64
numpy            2.2.5
scipy            1.17.1
scikit-learn     1.8.0
pandas           2.3.3
torch            2.10.0
biopython        1.86
matplotlib       3.10.8
openpyxl         3.1.5
requests         2.32.5
tqdm             4.67.1
joblib           1.5.3
pytest           9.1.1
```

**scikit-learn 1.8.0 is safe for this code.** `multi_class=` was removed from `LogisticRegression`
in 1.8, and `CLAUDE.md` records that as a hazard. No module passes it — the only `multi_class`
strings in the tree are a `multi_class_flag` data column in `fetch_annotations.py`, unrelated to
sklearn. Multinomial is the 1.8 default, which is the behaviour the probes want.

**`fair-esm` and `xgboost` are not installed here.** Both are on the result path but only for GPU
steps, so this environment cannot run them:

| Missing package | Steps it blocks |
|---|---|
| `fair-esm` | Experiment 5 step 3, `conservation_axis --extract` (masked-LM forward pass) |
| `xgboost` | Experiment 7 step 4, `megascale_mlp --xgboost` |

Neither blocks the CPU work. Record the RunPod environment below before those steps run — a report
citing a GPU-produced number against the CPU version list would be wrong.

## GPU environment (RunPod)

Not yet recorded. Capture it on the pod before Experiment 5 step 3 and Experiment 7 step 4:

```bash
python -c "
import platform, importlib.metadata as md
print('python', platform.python_version(), '|', platform.platform())
for p in ['numpy','scipy','scikit-learn','pandas','torch','fair-esm','xgboost','biopython','joblib']:
    try: print(f'{p}=={md.version(p)}')
    except Exception: print(f'{p}: MISSING')
"
```

Also record the CUDA and driver versions, since the permutation refits and the megascale probe are
the steps whose cost and numerics depend on them.
