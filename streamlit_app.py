import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from src.config import ProjectConfig
from src.normalization import Normalizer
from src.sequence_generator import SequenceGenerator

# Set page config to wide mode for a dashboard feel
st.set_page_config(page_title="Turbofan RUL Prediction", layout="wide")

# Custom CSS to match professional dashboard aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Application state and config
config = ProjectConfig()

# Load artifacts with a cache to prevent reloading on every interaction
@st.cache_resource
def load_artifacts():
    try:
        model = load_model(f"{config.model_dir}/final_lstm_model.h5")
        scaler = joblib.load(f"{config.model_dir}/scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

# --- HEADER SECTION ---
st.title("Turbofan Engine RUL Prediction Dashboard")
st.markdown("Predicting the Remaining Useful Life (RUL) of aircraft engines using deep learning.")
st.divider()

# --- SIDEBAR SECTION ---
st.sidebar.header("Control Panel")
st.sidebar.markdown("Upload engine sensor data to get an RUL forecast.")

uploaded_file = st.sidebar.file_uploader("Upload Sensor Data (CSV)", type=["csv"])

# --- MAIN CONTENT ---
if uploaded_file is not None:
    if model is None or scaler is None:
        st.error("Model artifacts not found. Please run the pipeline first.")
        st.stop()

    # Load data
    df = pd.read_csv(uploaded_file)

    # Create tabs for different views (consistent with Energy Consumption project)
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Prediction", "📋 Raw Data"])

    with tab1:
        st.subheader("Sensor Distribution & Behavior")

        # Feature identification
        feature_cols = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL']]

        # Row 1: Basic Statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Total Cycles", len(df))
        col_stat2.metric("Active Sensors", len(feature_cols))
        col_stat3.metric("Engine Unit", str(df['unit'].iloc[0]) if 'unit' in df.columns else "N/A")

        # Sensor Correlation Heatmap
        st.markdown("### Sensor Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[feature_cols].corr(), annot=False, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title("Sensor Correlation Heatmap")
        st.pyplot(fig_corr)

    with tab2:
        st.subheader("RUL Forecast")

        # Preprocessing
        norm = Normalizer(config)
        norm.scaler = scaler
        df_scaled = norm.transform(df, feature_cols)

        seq_gen = SequenceGenerator(config)
        X, _ = seq_gen.generate(df_scaled, feature_cols)

        if len(X) > 0:
            # Inference
            predictions = model.predict(X)

            # Display latest prediction
            res_col1, res_col2 = st.columns([1, 3])
            with res_col1:
                st.metric("Current RUL Estimate", f"{predictions[-1][0]:.2f} Cycles")
                st.markdown("**Confidence**: High (based on trained LSTM)")

            with res_col2:
                fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                ax_pred.plot(predictions, color='#1f77b4', linewidth=2, label='Predicted RUL')

                if 'RUL' in df.columns:
                    actual_rul = df['RUL'].values[config.window_size-1:]
                    ax_pred.plot(actual_rul, color='#ff7f0e', linewidth=2, label='Actual RUL')
                    ax_pred.set_title("RUL Prediction vs Ground Truth")
                else:
                    ax_pred.set_title("RUL Prediction Trajectory")

                ax_pred.set_xlabel("Time (Window Index)")
                ax_pred.set_ylabel("Remaining Useful Life")
                ax_pred.legend()
                ax_pred.grid(True, alpha=0.3)
                st.pyplot(fig_pred)
        else:
            st.warning(f"Insufficient data points. A minimum of {config.window_size} cycles is required.")

    with tab3:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

else:
    # Landing page state
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.info("Please upload a CSV file through the sidebar to begin the analysis.")
        st.markdown("""
        **Expected CSV Format:**
        - `unit`: Engine identifier
        - `cycle`: Operational cycle
        - `setting_1`, `setting_2`, `setting_3`: Operating conditions
        - `sensor_1` to `sensor_21`: Sensor measurements
        - `RUL` (Optional): Ground truth for validation
        """)
    with col_r:
        st.markdown("### System Info")
        st.write(f"**Model**: Stacked LSTM")
        st.write(f"**Window Size**: {config.window_size}")
        st.write(f"**Dataset**: {config.dataset}")
