import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.config import ProjectConfig
from src.normalization import Normalizer
from src.sequence_generator import SequenceGenerator

st.set_page_config(page_title="Turbofan RUL Predictor", layout="wide")

st.title("✈️ Turbofan Engine RUL Prediction")
st.markdown("Predicting the Remaining Useful Life (RUL) of aircraft engines using LSTM networks.")

# Load Config and Artifacts
config = ProjectConfig()
try:
    model = load_model(f"{config.model_dir}/final_lstm_model.h5")
    scaler = joblib.load(f"{config.model_dir}/scaler.pkl")
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Sidebar for Data Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Preprocessing
    feature_cols = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL']]

    # Normalization
    norm = Normalizer(config)
    norm.scaler = scaler # Use the pre-loaded scaler
    df_scaled = norm.transform(df, feature_cols)

    # Sequence Generation
    seq_gen = SequenceGenerator(config)
    X, _ = seq_gen.generate(df_scaled, feature_cols)

    if len(X) > 0:
        # Prediction
        predictions = model.predict(X)

        # Visuals
        st.subheader("RUL Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Latest RUL Prediction", f"{predictions[-1][0]:.2f} Cycles")

        with col2:
            fig, ax = plt.subplots()
            ax.plot(predictions, color='red', label='Predicted RUL')
            if 'RUL' in df.columns:
                actual_rul = df['RUL'].values[config.window_size-1:]
                ax.plot(actual_rul, color='blue', label='Actual RUL')
                ax.set_title("RUL Prediction vs Actual")
                ax.legend()
            else:
                ax.set_title("RUL Prediction Trajectory")
            st.pyplot(fig)
    else:
        st.warning(f"Not enough data points. Need at least {config.window_size} cycles.")
else:
    st.info("Please upload a CSV file containing engine sensor data to begin.")
