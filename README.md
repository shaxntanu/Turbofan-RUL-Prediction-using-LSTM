# Turbofan RUL Prediction using LSTM

This project implements a deep learning pipeline to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset.

## 🚀 Project Overview
The system transforms multivariate sensor time-series data into RUL predictions using a stacked LSTM (Long Short-Term Memory) network. It includes a full pipeline from raw data ingestion to an interactive Streamlit dashboard.

## 🛠️ Tech Stack
- **Language**: Python 3.12+
- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **UI**: Streamlit
- **Testing**: Pytest

## 📂 Project Structure
- `src/`: Core source code (modular pipeline)
- `data/`: Raw and processed datasets
- `models/`: Saved model weights (`.h5`) and scalers (`.pkl`)
- `outputs/`: Figures and evaluation metrics
- `tests/`: Unit tests for the data pipeline
- `streamlit_app.py`: Interactive prediction dashboard

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM.git
   cd Turbofan-RUL-Prediction-using-LSTM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 How to Run

### 1. Train the Model
To run the end-to-end pipeline (ingestion $\rightarrow$ preprocessing $\rightarrow$ training $\rightarrow$ evaluation):
```bash
py src/pipeline.py
```
This will save the trained model to `models/final_lstm_model.h5` and generate performance plots in `outputs/figures/`.

### 2. Launch the Dashboard
To explore the model and make predictions interactively:
```bash
streamlit run streamlit_app.py
```

## 📈 Methodology
The pipeline follows these stages:
1. **Data Ingestion**: Parsing NASA C-MAPSS `.txt` files.
2. **Labeling**: Computing RUL per engine based on run-to-failure trajectories.
3. **Cleaning**: Removing constant/uninformative sensors.
4. **Normalization**: Scaling features using `MinMaxScaler` (fitted on training data only).
5. **Sequencing**: Generating sliding window sequences for LSTM input.
6. **Training**: Stacked LSTM with Early Stopping and Model Checkpointing.
7. **Evaluation**: Metrics calculated using MAE and RMSE.
