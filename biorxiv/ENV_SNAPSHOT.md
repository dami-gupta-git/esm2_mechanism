# run_biorxiv environment snapshot

Captured for precondition 0.7, so later reports can cite the exact package versions the run's
numbers were computed under. Two machines: local (CPU probe/bootstrap steps) and pod (GPU
embedding/extraction steps). The original precondition snapshot was captured at HEAD `b502952`.

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

## Pod before environment rebuild (root@216.243.220.222:10625)

This environment was captured before the pod virtual environment was recreated during Experiment
7. It applies to the earlier pod steps and to processes that were already running when the virtual
environment was replaced.

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

## Pod after environment rebuild (root@216.243.220.222:10625)

The virtual environment was rebuilt before the successful final run of step 7.2. These versions
were verified directly on the pod on 2026-08-19. They also apply to steps 7.3 and 7.4 if those
steps finish without another environment rebuild.

```
python 3.12.3 | Linux-6.8.0-90-generic-x86_64-with-glibc2.39
numpy==2.2.6
scipy==1.18.0
scikit-learn==1.9.0
pandas==2.3.3
torch==2.13.0+cu130
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
```

## Final Experiment 8 pod

This environment produced the final step 8.1 result at clean commit `6937c85`. The source snapshot
is retained with the result as `results/run_biorxiv/enzyme_classification/environment_snapshot.txt`.

```
python 3.12.13 | Linux-6.8.0-90-generic-x86_64-with-glibc2.35
numpy==2.2.6
scipy==1.18.0
scikit-learn==1.9.0
pandas==2.3.3
torch==2.13.0
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
```

## Experiment 7 nonlinear-probe pod

This environment produced the final step 7.3 result at clean commit `6937c85`. The source snapshot is retained as
`results/run_biorxiv/megascale_stability/environment_snapshot_step_7_3.txt`.

```
python 3.12.3 | Linux-6.8.0-90-generic-x86_64-with-glibc2.39
numpy==2.4.6
scipy==1.15.3
scikit-learn==1.9.0
pandas==3.0.3
torch==2.13.0+cu130
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
cuml-cu12==26.8.0
libcuml-cu12==26.8.0
GPU: NVIDIA H100 80GB HBM3
NVIDIA driver: 580.126.09
```

## Experiment 7 XGBoost and baseline pod

This environment produced steps 7.4 and 7.5 at clean commit `6937c85`.
The source snapshot is retained as
`results/run_biorxiv/megascale_stability/environment_snapshot_steps_7_4_7_5.txt`.

```
python 3.12.13 | Linux-6.8.0-90-generic-x86_64-with-glibc2.35
numpy==2.2.6
scipy==1.18.0
scikit-learn==1.9.0
pandas==2.3.3
torch==2.13.0+cu130
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
GPU: NVIDIA H100 80GB HBM3
NVIDIA driver: 580.126.09
```

## Clean provenance rerun, pod 1

This environment produced clean reruns of steps 4.2 through 4.6, 5.5, and 6.7 at commit
`c9945b43dbc279af988ce888febd570fd1e2d5df`. The working tree was clean. The source snapshot is
retained as `results/run_biorxiv/environment_snapshot_clean_rerun_pod1.txt`.

```
python 3.12.3 | Linux-6.8.0-90-generic-x86_64-with-glibc2.39
numpy==2.2.6
scipy==1.18.0
scikit-learn==1.9.0
pandas==2.3.3
torch==2.13.0
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
CPU: Intel Xeon Platinum 8470, 208 logical CPUs
GPU: NVIDIA H100 80GB HBM3
NVIDIA driver: 580.126.09
```

## Clean provenance rerun, pod 2

This environment produced the clean reruns of steps 4.7 and 6.2 at commit
`c9945b43dbc279af988ce888febd570fd1e2d5df`. The working tree was clean. The source snapshot is
retained as `results/run_biorxiv/environment_snapshot_clean_rerun_pod2.txt`.

```
python 3.12.13 | Linux-6.8.0-90-generic-x86_64-with-glibc2.35
numpy==2.2.6
scipy==1.18.0
scikit-learn==1.9.0
pandas==2.3.3
torch==2.13.0
fair-esm==2.0.0
xgboost==3.4.1
biopython==1.88
joblib==1.5.3
CPU: Intel Xeon Platinum 8470, 208 logical CPUs
GPU: NVIDIA H100 80GB HBM3
NVIDIA driver: 580.126.09
```
