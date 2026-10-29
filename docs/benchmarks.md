# Benchmark protocol and recorded measurements

This page describes the benchmark evidence currently checked into the repository. It does not introduce new measurements. [`RESULTS.md`](RESULTS.md) contains the full tables, figures, run identifiers, and protocol caveats; regenerate experiments before treating a result as current for a modified environment or configuration.

## RUL model comparison

The committed baseline snapshot uses seed 42, a 125-cycle target cap, a 30-cycle history, an engine-level 80/20 fit-validation split, and at most 20 LSTM epochs. It ran with Python 3.12.10 and TensorFlow 2.21.0 on CPU. Metrics below aggregate eligible windows from the supplied test trajectories.

| Subset | Selected by validation MAE | Held-out test MAE | Held-out test RMSE |
|---|---|---:|---:|
| FD001 | LSTM | 10.31 cycles | 13.94 cycles |
| FD002 | Ridge | 15.39 cycles | 19.36 cycles |
| FD003 | Ridge | 11.25 cycles | 15.57 cycles |
| FD004 | Ridge | 13.08 cycles | 18.00 cycles |

These are one-seed window-level benchmark scores. FD003 demonstrates why the selection rule matters: Ridge is selected on validation even though the LSTM has a lower test MAE. The test set is not used to select the winner. These results are not terminal-engine scores, official per-engine PHM08 scores, or operational fleet accuracy.

## Optional C++ sensor-kernel timings

The recorded parity benchmark samples up to 10,000 standardized training rows per subset. It runs on a Windows/MSVC/OpenMP host and compares PCA fit-plus-projection and K-Means on shared PCA scores. The exact timing environment and run outputs are available in [`web/public/results/benchmark.json`](../web/public/results/benchmark.json).

| Subset | PCA sklearn (ms) | PCA C++ (ms) | K-Means speedup | Adjusted Rand index |
|---|---:|---:|---:|---:|
| FD001 | 8.73 | 13.62 | 1.21x | 0.994 |
| FD002 | 7.61 | 15.12 | 3.46x | 1.000 |
| FD003 | 5.25 | 16.43 | 1.74x | 0.996 |
| FD004 | 7.74 | 12.00 | 3.11x | 1.000 |

The PCA kernel is slower than scikit-learn in this snapshot despite numerical subspace parity. K-Means speedups are local microbenchmark observations, not end-to-end pipeline gains or RUL-inference speedups. Timings can change with CPU, compiler, thread settings, and build flags.

## Reliability-study measurements

The committed reliability snapshot is a separate experiment. It reports repeated-seed Ridge results and single-seed LSTM/MLP/CNN comparisons, plus terminal-engine, synthetic stress, short-history, and leave-one-subset-out metrics. Consult the run and caveats in [`RESULTS.md`](RESULTS.md) and protocol in [`RELIABILITY_AND_ROBUSTNESS.md`](RELIABILITY_AND_ROBUSTNESS.md); do not compare terminal-engine rows directly with the baseline's overlapping-window table.

## Regeneration

```powershell
python -m src.train_all --datasets FD001 FD002 FD003 FD004 --epochs 20
python scripts/export_web_data.py
```

For C++ parity timings, install/build the optional extension and run:

```powershell
python -m pip install ./cpp_engine
python -m src.run_cpp_benchmark --datasets FD001 FD002 FD003 FD004 --max-rows 10000
```

New experiment artifacts are written to ignored `outputs/` and `models/`. The web application changes only after the export bundle is reviewed and committed.

## Not measured

This repository does not currently provide a controlled comparison of end-to-end throughput, peak memory, model serving latency, power consumption, production failure rate, or operational savings. No values for those measures should be inferred from the model accuracy or C++ microbenchmark tables.
