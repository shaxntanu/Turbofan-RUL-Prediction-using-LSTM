# Technology Stack & Engineering Specification

> **Implemented stack:** The authoritative dependencies are `requirements.txt` and `web/package.json`. The project uses TensorFlow/Keras (not PyTorch), Python 3.12 for the supported training environment, scikit-learn, pandas/NumPy, Matplotlib/Seaborn, Streamlit, and React/Vite with Chart.js. The sections below retain rationale and design notes from the initial specification; proposal labels do not imply missing implementation.

TensorFlow training was exercised in a Python 3.12 environment on CPU. On native Windows, TensorFlow 2.11 and later does not provide built-in CUDA GPU execution; use CPU or a supported WSL2 setup for GPU workflows.

## 1. Programming Language

### Python 3.12+

**Selection**: Python 3.12 or newer

**Rationale**:
- Industry standard for ML and data science
- Extensive ecosystem for deep learning (TensorFlow, PyTorch)
- Compatibility with modern scientific computing packages
- Reference project uses Python 3.12+ (scipy 1.18.0 requirement)

**Alternatives Considered**:
- Python 3.11: Rejected due to scipy version compatibility
- Python 3.13+: Not tested, but likely compatible

**Status**: Required

---

## 2. Package/Dependency Management

### pip with requirements.txt

**Selection**: Standard pip with pinned requirements.txt

**Rationale**:
- Simple, widely understood
- Exact version pinning for reproducibility
- Consistent with reference project
- No additional tooling overhead

**Alternatives Considered**:
- Poetry: More sophisticated but adds complexity
- Conda: Good for environment management but heavier
- pipenv: Combines pip and virtualenv but less standard

**Status**: Required

**Implementation**:
```txt
# requirements.txt
pandas==3.0.0
numpy==2.3.5
scikit-learn==1.9.0
tensorflow==2.18.0  # or torch==2.5.0
matplotlib==3.10.8
seaborn==0.13.2
streamlit==1.62.0
joblib==1.5.3
```

---

## 3. Numerical Computing

### NumPy

**Version**: 2.3.5 (pinned)

**Used For**:
- Array operations and numerical computations
- Sequence generation and manipulation
- Mathematical operations for preprocessing

**Rationale**:
- Fundamental package for scientific computing
- Efficient array operations
- Standard dependency for all ML packages
- Consistent with reference project

**Alternatives Considered**:
- None - NumPy is de facto standard

**Status**: Required

---

## 4. Data Processing

### Pandas

**Version**: 3.0.0 (pinned)

**Used For**:
- Data loading and manipulation
- Dataframe operations for sensor data
- Group-by operations per engine
- RUL computation

**Rationale**:
- Industry standard for data manipulation
- Efficient handling of tabular data
- Rich API for data cleaning and transformation
- Consistent with reference project

**Alternatives Considered**:
- Polars: Faster but less mature ecosystem
- Pure NumPy: More complex for tabular operations

**Status**: Required

---

## 5. Visualization

### Matplotlib

**Version**: 3.10.8 (pinned)

**Used For**:
- Static figure generation (EDA, performance plots)
- Training loss curves
- Sensor time-series plots
- Prediction vs actual scatter plots

**Rationale**:
- Standard plotting library
- Full control over figure customization
- Publication-quality figures
- Consistent with reference project

**Alternatives Considered**:
- Plotly: Interactive but heavier
- Seaborn: Built on matplotlib, used for statistical plots

**Status**: Required

### Seaborn

**Version**: 0.13.2 (pinned)

**Used For**:
- Statistical visualizations
- Correlation heatmaps
- Distribution plots
- Enhanced styling

**Rationale**:
- Statistical plotting built on matplotlib
- Attractive default styles
- Simplifies complex statistical plots
- Consistent with reference project

**Alternatives Considered**:
- Pure matplotlib: More verbose for statistical plots

**Status**: Required

---

## 6. Machine Learning

### scikit-learn

**Version**: 1.9.0 (pinned)

**Used For**:
- Baseline model (LinearRegression)
- Data splitting (train_test_split)
- Scaling (MinMaxScaler, StandardScaler)
- Metrics (mean_absolute_error, mean_squared_error)

**Rationale**:
- Comprehensive ML library
- Standard preprocessing tools
- Consistent API
- Consistent with reference project

**Alternatives Considered**:
- Custom implementations: Unnecessary for standard operations

**Status**: Required

---

## 7. Deep Learning Framework

### TensorFlow / Keras (Primary Choice)

**Version**: 2.18.0 (proposed)

**Used For**:
- LSTM model implementation
- Model training and validation
- Callbacks (early stopping, checkpointing)
- Model serialization

**Rationale**:
- Mature, production-ready framework
- Keras API is intuitive for sequential models
- Good documentation and community support
- Built-in callbacks for training utilities
- Widely used in industry and academia

**Alternatives Considered**:
- **PyTorch**: Excellent framework, more flexible, but Keras API is simpler for this use case
- **JAX**: Emerging framework, less mature for this use case

**Decision**: TensorFlow/Keras recommended for simplicity and built-in training utilities. PyTorch is acceptable as an alternative if preferred.

**Status**: Required (either TensorFlow or PyTorch)

### PyTorch (Alternative)

**Version**: 2.5.0 (if chosen)

**Used For**:
- LSTM model implementation
- Training loop
- Model serialization

**Rationale**:
- More flexible and Pythonic
- Strong research community
- Dynamic computation graph

**Status**: Alternative to TensorFlow

---

## 8. Experiment/Configuration Management

### Python Configuration Objects

**Selection**: Dataclasses or simple dictionaries for configuration

**Used For**:
- Hyperparameter management
- Path configuration
- Experiment settings

**Rationale**:
- Simple, no additional dependencies
- Type-safe with dataclasses
- Easy to serialize to JSON
- Consistent with reference project approach

**Alternatives Considered**:
- Hydra: Powerful but adds complexity
- MLflow: Overkill for this project scope
- Weights & Biases: External service, not needed

**Status**: Required

---

## 9. Testing

### pytest

**Version**: 8.4.2 (pinned)

**Used For**:
- Unit tests for individual modules
- Integration tests for pipeline
- Regression tests

**Rationale**:
- Standard Python testing framework
- Simple fixtures and assertions
- Good test discovery
- Consistent with reference project

**Alternatives Considered**:
- unittest: Built-in but less feature-rich
- nose2: Less popular

**Status**: Required

---

## 10. Linting/Formatting

### Black (Optional)

**Version**: 24.x (if used)

**Used For**:
- Code formatting
- Consistent style

**Rationale**:
- Opinionated formatter, no configuration needed
- Industry standard
- Consistent code style

**Alternatives Considered**:
- autopep8: Less opinionated
- Yapf: Google's formatter

**Status**: Optional (recommended)

### flake8 (Optional)

**Version**: 7.x (if used)

**Used For**:
- Linting
- Style checking

**Rationale**:
- Combines pyflakes, pycodestyle, and McCabe
- Standard tool

**Status**: Optional (recommended)

---

## 11. Logging

### Python logging module

**Selection**: Standard library logging

**Used For**:
- Pipeline execution logging
- Debug information
- Progress tracking

**Rationale**:
- Built-in, no dependencies
- Flexible configuration
- Standard practice

**Alternatives Considered**:
- loguru: More features but adds dependency

**Status**: Required

---

## 12. Serialization/Model Artifacts

### Joblib

**Version**: 1.5.3 (pinned)

**Used For**:
- Saving scaler objects
- Saving preprocessing artifacts
- Efficient serialization of numpy arrays

**Rationale**:
- Efficient for numpy arrays
- Standard for scikit-learn objects
- Consistent with reference project

**Alternatives Considered**:
- pickle: Standard but less efficient for arrays
- dill: More features but overkill

**Status**: Required

### HDF5 / SavedModel (for LSTM)

**Used For**:
- Saving trained LSTM model
- Model weights and architecture

**Rationale**:
- TensorFlow: SavedModel format or .h5 (Keras)
- PyTorch: .pt or .pth format

**Status**: Required (framework-specific)

---

## 13. Documentation

### Markdown

**Used For**:
- README.md
- Documentation in docs/
- Inline code documentation

**Rationale**:
- Standard for GitHub documentation
- Readable on GitHub
- Easy to write

**Alternatives Considered**:
- reStructuredText: More complex
- Jupyter notebooks: Good for tutorials but not for project docs

**Status**: Required

### Docstrings

**Style**: Google or NumPy style

**Used For**:
- Function and class documentation
- Module-level documentation

**Rationale**:
- Standard Python practice
- IDE support
- Readable

**Status**: Required

---

## 14. Dashboard/Application Framework

### Streamlit

**Version**: 1.62.0 (pinned)

**Used For**:
- Interactive dashboard
- Model exploration
- Prediction interface

**Rationale**:
- Pure Python, no frontend knowledge required
- Fast development
- Good for ML demos
- Consistent with reference project

**Alternatives Considered**:
- Dash: More complex, requires React knowledge
- Gradio: Simpler but less customizable
- Flask/Django: Overkill for this use case

**Status**: Recommended

---

## 15. Deployment Platform

### Local Execution

**Selection**: Local execution on user's machine

**Rationale**:
- Academic mini-project scope
- No production deployment requirement
- Simpler infrastructure
- Reference project uses local + Vercel (static)

**Alternatives Considered**:
- Cloud deployment (AWS, GCP): Overkill
- Docker: Optional for reproducibility

**Status**: Required (local execution)

### Docker (Optional)

**Used For**:
- Reproducible environment
- Consistent dependency management

**Rationale**:
- Ensures consistent environment
- Good for reproducibility
- Reference project includes Dockerfile

**Status**: Optional (recommended for reproducibility)

---

## 16. Git/GitHub Workflow

### Git

**Used For**:
- Version control
- Collaboration
- Experiment tracking

**Rationale**:
- Industry standard
- GitHub integration
- Branching for experiments

**Status**: Required

### GitHub

**Used For**:
- Repository hosting
- Issue tracking
- Documentation hosting

**Rationale**:
- Free for public repositories
- Standard platform
- Good documentation support

**Status**: Required

---

## 17. CI/CD

### GitHub Actions (Optional)

**Used For**:
- Automated testing
- Linting checks
- Dependency validation

**Rationale**:
- Ensures code quality
- Catches issues early
- Standard practice

**Status**: Optional (recommended)

---

## 18. Directory Structure

### Project Layout

```
Turbofan-RUL-Prediction-using-LSTM/
├── data/                           # Dataset files (git-ignored if large)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── eda.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   ├── target_construction.py
│   ├── normalization.py
│   ├── sequence_generator.py
│   ├── data_splitter.py
│   ├── baseline_model.py
│   ├── lstm_model.py
│   ├── trainer.py
│   ├── validator.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   ├── report_generator.py
│   ├── pipeline.py
│   ├── config.py
│   └── paths.py
├── models/                         # Saved models (git-ignored)
│   ├── lstm_model.h5
│   ├── scaler.pkl
│   ├── training_history.json
│   └── experiment_metadata.json
├── outputs/                        # Generated outputs (git-ignored)
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── docs/                           # Documentation
│   ├── PROJECT_REQUIREMENTS.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── TECHNOLOGY_STACK.md
│   ├── DATASET_AND_DATA_PIPELINE.md
│   └── EXPERIMENTATION_AND_EVALUATION.md
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_sequence_generator.py
│   └── test_pipeline.py
├── streamlit_app.py                # Streamlit dashboard
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
└── LICENSE                         # MIT License
```

**Rationale**:
- Clear separation of concerns
- Consistent with reference project structure
- Standard Python project layout
- Easy to navigate

---

## 19. Naming Conventions

### Python Code

**Files**: `snake_case.py`
- Example: `data_loader.py`, `lstm_model.py`

**Classes**: `PascalCase`
- Example: `LSTMModel`, `DataLoader`

**Functions/Variables**: `snake_case`
- Example: `load_data()`, `window_size`

**Constants**: `UPPER_SNAKE_CASE`
- Example: `RANDOM_SEED`, `WINDOW_SIZE`

**Private members**: `_leading_underscore`
- Example: `_internal_function()`

### Directories

**Directories**: `snake_case`
- Example: `src/`, `models/`, `outputs/`

### Model Artifacts

**Model files**: Descriptive names with version/hash
- Example: `lstm_model_fd001_99c7a66.h5`

**Scaler files**: Descriptive names
- Example: `scaler_minmax_fd001.pkl`

### Configuration

**Config files**: `config.py` or YAML/JSON
- Example: `config.py` with dataclass

---

## 20. Configuration Conventions

### Configuration Object

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # Data
    dataset: str = "FD001"
    data_path: str = "data/"
    
    # Preprocessing
    window_size: int = 30
    normalize: bool = True
    scaler_type: str = "minmax"  # or "standard"
    
    # Model
    lstm_units: list = None
    dropout_rate: float = 0.2
    dense_units: int = 16
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    # Reproducibility
    random_seed: int = 42
    
    # Paths
    output_dir: str = "outputs/"
    model_dir: str = "models/"
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32]
```

### Configuration Hashing

```python
import hashlib
import json

def compute_config_hash(config: ExperimentConfig) -> str:
    """Compute unique hash for configuration."""
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
```

---

## 21. Environment Variables

### Optional Environment Variables

```bash
# Optional: Override default paths
DATA_DIR=/path/to/data
OUTPUT_DIR=/path/to/outputs
MODEL_DIR=/path/to/models

# Optional: GPU configuration (TensorFlow)
CUDA_VISIBLE_DEVICES=0

# Optional: Random seed override
RANDOM_SEED=42
```

**Rationale**: Environment variables allow flexibility without code changes. All defaults should work without environment variables.

**Status**: Optional

---

## 22. Dependency Organization

### requirements.txt Structure

```txt
# Core dependencies
pandas==3.0.0
numpy==2.3.5
scikit-learn==1.9.0

# Deep learning (choose one)
tensorflow==2.18.0
# OR
# torch==2.5.0

# Visualization
matplotlib==3.10.8
seaborn==0.13.2

# Application
streamlit==1.62.0

# Utilities
joblib==1.5.3

# Testing
pytest==8.4.2

# Optional: Development tools
# black==24.1.1
# flake8==7.0.0
```

### Optional requirements-dev.txt

```txt
# Development tools
black==24.1.1
flake8==7.0.0
mypy==1.8.0
```

---

## 23. Artifact Storage Strategy

### Model Artifacts

**Location**: `models/`

**Naming**: `{model_type}_{dataset}_{hash}.{ext}`
- Example: `lstm_fd001_99c7a66.h5`

**Versioning**: Include config hash in filename

**Metadata**: Save `experiment_metadata.json` with:
- Config hash
- All hyperparameters
- Package versions
- Timestamp
- Training metrics

### Data Artifacts

**Location**: `outputs/`

**Subdirectories**:
- `figures/`: All visualizations
- `metrics/`: JSON/CSV metrics
- `reports/`: Markdown reports

**Naming**: `{stage}_{description}.{ext}`
- Example: `training_loss_curves.png`, `test_metrics.json`

### Dataset Storage

**Location**: `data/`

**Policy**: 
- Raw dataset files can be git-ignored if large
- Processed datasets should be versioned or regenerated
- Document dataset source and version in README

---

## 24. Dataset Storage Strategy

### Raw Dataset

**Location**: `data/`

**Files**:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

**Git Policy**: 
- Option 1: Commit if small enough
- Option 2: Git-ignore and provide download instructions
- Option 3: Use Git LFS for large files

**Recommendation**: Git-ignore raw dataset files and provide download script in README.

### Processed Dataset

**Location**: `data/processed/` (optional)

**Policy**: 
- Regenerate from raw data via pipeline
- Do not commit processed data
- Document processing steps

---

## 25. Generated-File Policy

### Files to Commit

- Source code (`src/`)
- Documentation (`docs/`, `README.md`)
- Configuration files (`requirements.txt`, `.gitignore`)
- Test files (`tests/`)
- Streamlit app (`streamlit_app.py`)

### Files to Git-Ignore

- Dataset files (`data/*.txt`)
- Model artifacts (`models/*.h5`, `models/*.pkl`)
- Generated outputs (`outputs/`)
- Python cache (`__pycache__/`, `*.pyc`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)

### .gitignore Template

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Dataset
data/*.txt
data/processed/

# Models
models/*.h5
models/*.pkl
models/*.pt

# Outputs
outputs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
venv/
env/
.venv/
```

---

## 26. Git Tracking Policy

### Branching Strategy

**Main branch**: `main`
- Production-ready code
- Stable documentation

**Feature branches**: `feature/description`
- Experimental features
- New modules

**Experiment branches**: `experiment/description`
- Hyperparameter experiments
- Alternative approaches

### Commit Messages

**Format**: `type: description`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Test changes
- `exp`: Experiment results

**Examples**:
- `feat: add LSTM model implementation`
- `fix: correct RUL computation for test data`
- `docs: update architecture documentation`
- `exp: window size 30 vs 50 comparison`

---

## 27. Technology Summary Table

| Technology | Version | Purpose | Required/Optional | Rationale |
|------------|---------|---------|-------------------|-----------|
| Python | 3.12 | Language/runtime | Required for TensorFlow path | Supported project environment |
| pip | - | Package manager | Required | Simple, standard |
| NumPy | `requirements.txt` | Numerical computing | Required | Sequence and metric arrays |
| Pandas | `requirements.txt` | Data processing | Required | C-MAPSS trajectories and artifacts |
| TensorFlow/Keras | `>=2.18.0` | Deep learning | Required for LSTM runs | Sequence model and training callbacks |
| scikit-learn | `requirements.txt` | ML utilities | Required | Ridge baseline, scaling, and metrics |
| Matplotlib/Seaborn | `requirements.txt` | Visualization | Required | Experiment figures |
| Streamlit | `requirements.txt` | Dashboard | Included | Interactive exploration/prediction |
| Joblib | `requirements.txt` | Serialization | Required | Baseline and scaler artifacts |
| React/React DOM | `web/package.json` | Web UI | Included | Results dashboard |
| Vite | `web/package.json` | Frontend build | Included | Development and production bundles |
| Chart.js | `web/package.json` | Interactive charts | Included | Dashboard comparisons and metrics |
| Lucide React | `web/package.json` | Icons | Included | Dashboard controls and navigation |
| pytest | `requirements.txt` | Test framework | Development | Repository unit tests |
| Docker | Root `Dockerfile` | Containerization | Optional | Reproducible service packaging |
| Git | - | Version control | Required | Standard |
| GitHub | - | Repository hosting | Required | Platform |
| Docker | - | Containerization | Optional | Reproducibility |

---

## 28. Implementation Status

| Technology | Status | Notes |
|------------|--------|-------|
| Python 3.12 | Implemented | Used for the TensorFlow LSTM workflow |
| pip requirements | Implemented | See root dependency file |
| TensorFlow/Keras | Implemented | LSTM training and model serialization |
| scikit-learn | Implemented | Ridge model and preprocessing |
| Matplotlib/Seaborn | Implemented | Static experiment figures |
| Streamlit | Implemented | Interactive local app |
| React/Vite/Chart.js | Implemented | Research results dashboard |
| Git/GitHub | Implemented | Per-file commit/push workflow |
| Docker | Available | Root `Dockerfile`; host capabilities vary |

---

**Document Version**: 1.1  
**Last Updated**: 2026-10-11  
**Status**: Core stack implemented; dependency sources of truth are the root and web package manifests.
