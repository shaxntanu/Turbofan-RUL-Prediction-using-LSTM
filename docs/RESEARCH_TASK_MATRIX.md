# Research Task Matrix

This matrix maps the requirements from the provided research papers to the implementation in this project.

## Paper 1: ML-Based Predictive Turbofan Engine Health Model

| Original Paper Task | Original Target | Project Equivalent | Implementation File | Status | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Feature Screening (PCA/Pearson) | TGT | Sensor Correlation/Variance Analysis | `src/data_cleaner.py` | Partially Implemented | Basic variance check done; PCA missing. |
| Detrending & Segmentation | TGT | Trajectory Analysis | TBD | Missing | NASA data lacks overhaul history. |
| Comparative Modeling | TGT | Baseline vs LSTM | `src/pipeline.py` | Partially Implemented | Baseline regressor logic missing. |
| Many-to-One LSTM | TGT | Many-to-One LSTM | `src/lstm_model.py` | Implemented | Adapted to RUL target. |
| Robustness/Outlier Analysis | TGT | Per-engine RMSE/IQR | TBD | Missing | Needs full execution of pipeline. |
| Uncertainty Quantification | TGT | Prediction Intervals | TBD | Missing | Not yet implemented. |

## Paper 2: Generalisable Predictive Health Models

| Original Paper Task | Original Target | Project Equivalent | Implementation File | Status | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Domain-Specific Training | TGT | Subset-Specific Training | `src/train_all.py` | Implemented | Needs execution. |
| Cross-Engine Knowledge Transfer | TGT | Cross-Subset Generalisation | TBD | Missing | Not yet implemented. |
| Source $\rightarrow$ Target Adaptation | TGT | Transfer learning FD(x) $\rightarrow$ FD(y) | TBD | Missing | Not yet implemented. |
| Generalisation Evaluation | TGT | Cross-domain Metrics | TBD | Missing | Not yet implemented. |
| Transfer-Learning Correctness | TGT | Domain Leakage Prevention | `src/data_splitter.py` | Implemented | Verified for in-domain. |
