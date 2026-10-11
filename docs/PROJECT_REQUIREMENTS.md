# Project Requirements Specification

## 1. Project Title

Turbofan Remaining Useful Life (RUL) Prediction using LSTM

## 2. Problem Statement

Predict the remaining useful life (RUL) of aircraft turbofan engines from multivariate sensor time-series data using Long Short-Term Memory (LSTM) neural networks. The problem involves learning degradation patterns from historical sensor readings to forecast how many operational cycles remain before engine failure.

## 3. Background/Context

Turbofan engines are critical components in commercial aviation. Predictive maintenance through RUL estimation enables:

- Reduced unplanned downtime
- Optimized maintenance scheduling
- Improved safety and reliability
- Cost savings through condition-based maintenance

The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset provides a benchmark for prognostics research, offering simulated run-to-failure data under realistic flight conditions with varying operational settings and fault modes.

## 4. Motivation

- **Academic**: Demonstrate deep learning application to prognostics and health management (PHM)
- **Engineering**: Build a reproducible ML pipeline for time-series regression
- **Research**: Establish a baseline LSTM model on a standard benchmark dataset
- **Practical**: Create a framework that can be extended to real-world predictive maintenance scenarios

## 5. Project Objectives

### Primary Goal

Develop a reproducible LSTM-based model to predict turbofan engine RUL from multivariate sensor time-series data, achieving competitive performance on the NASA C-MAPSS benchmark dataset.

### Secondary Goals

- Establish a modular, well-documented ML pipeline
- Create comprehensive evaluation metrics and visualization
- Ensure reproducibility through deterministic execution and version control
- Provide an interactive interface for model exploration
- Document the complete experimental process

## 6. Scope

### In Scope

- Data ingestion and preprocessing of NASA C-MAPSS dataset
- Exploratory data analysis and sensor behavior characterization
- Feature engineering for time-series degradation modeling
- RUL target construction from run-to-failure trajectories
- Sequence generation for LSTM training
- LSTM model architecture design and implementation
- Training, validation, and testing procedures
- Performance evaluation using regression metrics
- Visualization of predictions and degradation patterns
- Model artifact management and versioning
- Reproducibility infrastructure (seeds, configuration hashing)
- Documentation and reporting

### Out of Scope

- Real-time deployment in production environments
- Hardware-in-the-loop validation
- Integration with actual aircraft systems
- Multi-fault mode diagnosis beyond RUL prediction
- Physics-informed modeling approaches
- Transfer learning to other engine types
- Distributed training infrastructure
- Real-time streaming inference

## 7. Intended Users/Stakeholders

- **Primary**: Academic researchers, ML engineering students
- **Secondary**: PHM researchers, predictive maintenance practitioners
- **Tertiary**: Faculty evaluators, future researchers extending this work

## 8. Functional Requirements

### 8.1 Data Ingestion (Required)

- Download and parse NASA C-MAPSS dataset files
- Support all four dataset subsets (FD001, FD002, FD003, FD004)
- Handle train/test split structure for each subset
- Parse 26-column format (unit ID, cycle, 3 settings, 21 sensors)
- Validate data integrity (missing values, data types, ranges)

### 8.2 Data Preprocessing (Required)

- Compute RUL labels for training data (max_cycle - current_cycle)
- Handle test data RUL computation using provided ground truth
- Identify and handle constant/near-constant sensors
- Perform sensor normalization/scaling
- Implement feature selection if justified by EDA
- Generate training/validation/test splits at engine level (not row level)

### 8.3 Sequence Generation (Required)

- Generate sliding window sequences from time-series data
- Support configurable window length
- Handle variable-length sequences (padding/truncation strategies)
- Preserve temporal ordering within sequences
- Ensure no data leakage between train/validation/test engines

### 8.4 LSTM Model (Required)

- Implement LSTM architecture for sequence-to-scalar regression
- Support configurable hyperparameters (layers, units, dropout, etc.)
- Implement training loop with validation monitoring
- Include early stopping based on validation loss
- Implement model checkpointing
- Support configurable batch size and learning rate

### 8.5 Baseline Model (Required)

- Implement a simple baseline for comparison (e.g., linear regression on aggregated features, or naive persistence model)
- Document baseline performance for comparison

### 8.6 Evaluation (Required)

- Compute regression metrics: MAE, RMSE
- Evaluate on held-out test set
- Perform engine-level error analysis
- Visualize predicted vs actual RUL
- Analyze prediction errors across degradation stages

### 8.7 Visualization (Required)

- Sensor time-series plots for sample engines
- Degradation trajectory visualization
- Prediction vs ground truth scatter plots
- Error distribution plots
- Training/validation loss curves

### 8.8 Reproducibility (Required)

- Fix random seeds for all stochastic components
- Implement configuration hashing for experiment tracking
- Log all hyperparameters and package versions
- Save model artifacts with metadata
- Document exact dataset version used

### 8.9 Documentation (Required)

- Comprehensive README with installation and usage instructions
- Inline code documentation
- Architecture documentation
- Experiment documentation with results

### 8.10 Interactive Application (Recommended)

- Streamlit-based interface for exploring results
- Ability to visualize predictions for specific engines
- Interactive parameter exploration (if feasible)

## 9. Non-Functional Requirements

### 9.1 Performance

- Training should complete within reasonable time on a standard laptop (e.g., < 2 hours for FD001, < 6 hours for larger subsets)
- Inference latency should be suitable for batch prediction (not real-time)
- Pipeline should handle varying dataset sizes efficiently

### 9.2 Reliability

- Pipeline should handle edge cases (empty sequences, missing data)
- Graceful error handling and logging
- Validation checks at each pipeline stage

### 9.3 Maintainability

- Modular code structure with clear separation of concerns
- Consistent coding style and naming conventions
- Comprehensive comments and docstrings

### 9.4 Reproducibility

- Deterministic execution given same configuration
- Version-controlled dependencies
- Clear documentation of experimental setup

### 9.5 Usability

- Clear command-line interface for running experiments
- Meaningful error messages
- Organized output structure

### 9.6 Scalability (Optional)

- Code should be structured to allow extension to larger datasets
- Avoid hard-coded assumptions about dataset size

## 10. Input Requirements

### 10.1 Dataset

- NASA C-MAPSS dataset files for all subsets (FD001, FD002, FD003, FD004)
- Files per subset: train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt
- Alternative: Hugging Face dataset version (SoyVitou/NASA-C-MAPSS-Turbofan-Engine)
- File format: Space-separated values, 26 columns

### 10.2 Configuration

- Hyperparameters (window size, LSTM architecture, training parameters)
- Random seed
- Dataset subset selection
- Output directory paths

### 10.3 Data Characteristics

- Unit ID: Engine identifier
- Cycle: Time step in operational cycles
- Operational settings: 3 columns (setting_1, setting_2, setting_3)
- Sensor measurements: 21 columns (sensor_1 to sensor_21)
- Training: ~20,000+ rows (FD001), ~50,000+ rows (FD002), ~20,000+ rows (FD003), ~60,000+ rows (FD004)
- Test: ~13,000+ rows (FD001), ~34,000+ rows (FD002), ~13,000+ rows (FD003), ~45,000+ rows (FD004)

## 11. Output Requirements

### 11.1 Model Artifacts

- Trained LSTM model weights (e.g., .h5, .pt, or .pkl format)
- Model architecture configuration
- Training history (loss curves)

### 11.2 Evaluation Results

- Test set metrics (MAE, RMSE) for all subsets (FD001, FD002, FD003, FD004)
- Per-engine predictions for each subset
- Error analysis reports per subset
- Comparison with baseline for each subset
- Cross-subset performance comparison

### 11.3 Visualizations

- Sensor degradation plots
- Prediction accuracy plots
- Training curves
- Error distribution plots

### 11.4 Documentation

- Analysis summary report
- Configuration metadata
- Package version information

## 12. ML Requirements

### 12.1 Model Type

- Sequence-to-scalar regression using LSTM
- Alternative architectures may be explored (GRU, bidirectional LSTM) as extensions

### 12.2 Training Requirements

- Train/validation split at engine level (not random row split)
- Validation monitoring for early stopping
- Checkpointing of best model
- Learning rate scheduling (optional)

### 12.3 Evaluation Requirements

- Metrics: MAE (primary), RMSE (secondary)
- Engine-level analysis (not just aggregate metrics)
- Error analysis across RUL ranges

### 12.4 Generalization

- Model should generalize to unseen engines from same distribution
- Test set must be completely held out during training

## 13. Data Requirements

### 13.1 Dataset Source

- NASA C-MAPSS Jet Engine Simulated Data
- Public domain, no usage restrictions
- Citation required for academic use

### 13.2 Dataset Structure

- 4 subsets: FD001, FD002, FD003, FD004
- Baseline: FD001 (simplest: 1 operating condition, 1 fault mode)
- 100 training engines, 100 test engines for FD001
- 26 columns per row

### 13.3 Data Quality

- Simulated data with sensor noise
- No missing values in original dataset
- Some sensors may be constant or near-constant
- Operational settings affect sensor readings

## 14. Visualization Requirements

### 14.1 EDA Visualizations

- Sensor time-series for sample engines
- Distribution of sensor values
- Correlation heatmaps
- RUL distribution

### 14.2 Model Performance Visualizations

- Predicted vs actual RUL scatter plots
- Residual plots
- Error distribution histograms
- Training/validation loss curves

### 14.3 Degradation Analysis

- Sensor degradation trajectories
- Prediction accuracy vs remaining cycles
- Per-engine prediction plots

## 15. Evaluation Requirements

### 15.1 Metrics

- **Primary**: Mean Absolute Error (MAE)
- **Secondary**: Root Mean Square Error (RMSE)
- **Optional**: R² score, MAPE (if appropriate)

### 15.2 Validation Strategy

- Engine-level train/validation split (e.g., 80/20 of engines)
- No data leakage between engines
- Final evaluation on completely held-out test set

### 15.3 Baseline Comparison

- Compare against simple baseline (e.g., linear regression or persistence model)
- Report relative improvement

### 15.4 Error Analysis

- Analyze errors across RUL ranges (early vs late degradation)
- Identify engines with poor predictions
- Investigate systematic biases

## 16. Reproducibility Requirements

### 16.1 Deterministic Execution

- Fixed random seed for NumPy, TensorFlow/PyTorch, Python
- Deterministic data splits
- Reproducible model initialization

### 16.2 Configuration Tracking

- Hash of all hyperparameters
- Logging of exact package versions
- Timestamped experiment outputs

### 16.3 Artifact Management

- Versioned model saves
- Metadata with each artifact
- Clear directory structure

### 16.4 Documentation

- Step-by-step reproduction instructions
- Exact dataset source and version
- Complete dependency specification

## 17. Deployment/Application Requirements

### 17.1 Local Execution

- Must run on standard laptop hardware
- No cloud infrastructure required
- CPU-based training acceptable (GPU optional)

### 17.2 Interactive Application (Recommended)

- Streamlit-based dashboard
- Local execution via `streamlit run`
- Visualization of model predictions
- Exploration of sensor data

### 17.3 Export Capabilities

- Export predictions to CSV
- Export model for inference
- Export figures in standard formats (PNG, PDF)

## 18. Constraints

### 18.1 Computational

- Training should be feasible on consumer hardware
- Memory usage should be reasonable for laptop (e.g., < 8GB RAM)
- No distributed computing infrastructure

### 18.2 Time

- Project scope appropriate for academic mini-project timeline
- Focus on quality over breadth of experiments

### 18.3 Dataset

- Limited to publicly available datasets
- Cannot fabricate or modify ground truth labels
- Must respect dataset licensing (public domain for C-MAPSS)

### 18.4 Scope

- Single dataset subset for baseline (FD001)
- Extensions to other subsets as future work
- Focus on LSTM, not exhaustive architecture search

## 19. Assumptions

### 19.1 Dataset

- NASA C-MAPSS dataset is representative of turbofan degradation patterns
- Simulated data provides valid learning signal for real-world application
- Sensor noise is realistic and not pathological

### 19.2 Model

- LSTM architecture is suitable for capturing temporal degradation patterns
- Sufficient training data exists to learn meaningful patterns
- Hyperparameters can be tuned within reasonable search space

### 19.3 Evaluation

- Test set provides valid estimate of generalization performance
- MAE and RMSE are appropriate metrics for RUL prediction
- Engine-level splitting prevents data leakage

### 19.4 Environment

- User has Python 3.12+ installed
- Sufficient disk space for dataset and artifacts (~500MB)
- Internet access for initial dataset download

## 20. Risks

### 20.1 Technical Risks

- **Overfitting**: Model may memorize training engines rather than learn generalizable patterns
- **Data leakage**: Improper splitting could contaminate validation/test sets
- **Hyperparameter sensitivity**: Performance may be sensitive to hyperparameter choices
- **Computational limitations**: Training time may exceed expectations

### 20.2 Data Risks

- **Sensor quality**: Some sensors may be uninformative or constant
- **Domain shift**: Simulated data may not reflect real-world complexity
- **Label noise**: RUL labels are deterministic but may not reflect true failure modes

### 20.3 Project Risks

- **Scope creep**: Adding too many features beyond baseline LSTM
- **Reproducibility issues**: Undocumented dependencies or random state issues
- **Documentation gaps**: Insufficient documentation for future reproduction

### 20.4 Mitigation Strategies

- Strict engine-level train/validation/test splits
- Early stopping and regularization
- Comprehensive logging and configuration tracking
- Clear scoping and incremental development

## 21. Success Criteria

### 21.1 Functional Success

- [ ] Pipeline runs end-to-end without errors
- [ ] Model trains successfully and converges
- [ ] Predictions generated for test set
- [ ] Evaluation metrics computed and reported
- [ ] Visualizations generated successfully

### 21.2 Quality Success

- [ ] LSTM model outperforms simple baseline
- [ ] MAE on test set is competitive with literature values for FD001
- [ ] Code is modular and well-documented
- [ ] Experiments are reproducible from documentation

### 21.3 Engineering Success

- [ ] Clear separation of concerns in codebase
- [ ] Configuration-driven design
- [ ] Comprehensive error handling
- [ ] Organized output structure

### 21.4 Documentation Success

- [ ] README provides clear installation and usage instructions
- [ ] Architecture is documented
- [ ] Experimental results are reported
- [ ] Code has inline documentation

## 22. Acceptance Criteria

### 22.1 Minimum Viable Product

- Working pipeline from data ingestion to prediction
- Trained LSTM model with saved weights
- Test set evaluation with MAE and RMSE
- Basic visualizations (predictions vs actual, loss curves)
- README with installation instructions

### 22.2 Enhanced Product

- Baseline model comparison
- Comprehensive EDA with visualizations
- Engine-level error analysis
- Streamlit interactive application
- Configuration hashing and reproducibility features
- Extended documentation (architecture, experiments)

### 22.3 Excellence Indicators

- Hyperparameter tuning with systematic search
- Comparison across multiple dataset subsets
- Ablation studies (window size, feature selection)
- Advanced visualizations (degradation trajectories)
- Publication-ready figures and reports

## 23. Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| Data ingestion | Implemented | Bundled C-MAPSS FD001-FD004 with schema and engine-cycle validation |
| Preprocessing | Implemented | Capped RUL, engine-level validation split, training-only scaling, 30-cycle windows |
| LSTM model | Implemented | Stacked 64/32-unit TensorFlow/Keras sequence model |
| Baseline model | Implemented | Window-summary Ridge regression |
| Evaluation | Implemented | MAE/RMSE, validation model selection, intervals, per-engine robustness |
| Research analyses | Implemented | Transfer, sensor/PCA screening, detrending and life-stage proxy |
| Visualization | Implemented | Matplotlib performance, uncertainty, sensor, and error figures |
| Reproducibility | Implemented | Seeded CLI runs with configuration and package metadata |
| Documentation | Implemented | Usage, architecture, protocol, dataset, and research scope |
| Streamlit app | Implemented | Exploration, saved results, and CSV/TXT prediction workflow |
| Web dashboard | Implemented | React/Vite results viewer with committed experiment data |
| Docker | Available | Root `Dockerfile`; GPU behavior depends on the host runtime |

---

**Document Version**: 1.1  
**Last Updated**: 2026-10-11  
**Status**: Core requested pipeline and applications implemented; performance criteria remain empirical and are reported per subset.
