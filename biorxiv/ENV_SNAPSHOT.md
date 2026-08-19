# run_biorxiv environment snapshot

Captured for precondition 0.7, so later reports can cite the exact package versions the run's
numbers were computed under. Two machines: local (CPU probe/bootstrap steps) and pod (GPU
embedding/extraction steps). HEAD at capture time: `b502952`.

## Local

```
python 3.13.7 | macOS-15.6-arm64-arm-64bit-Mach-O
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
pandas==3.0.3
torch==2.12.0
fair-esm==2.0.0
xgboost==3.2.0
biopython==1.87
joblib==1.5.3
```

## Pod (root@216.243.220.222:10625)

```
python 3.12.3 | Linux-6.8.0-90-generic-x86_64-with-glibc2.39
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.9.0
pandas==2.2.2
torch==2.13.0
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
```
