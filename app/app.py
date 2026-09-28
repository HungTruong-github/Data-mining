import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Repeat Purchase Prediction", layout="wide")
st.title("Dashboard Du Doan Khach Hang Mua Lai (Repeat Purchase)")

# Load model and metadata
@st.cache_resource
def load_model_and_metadata():
    model_dir = Path('models/classification')
    pipeline = joblib.load(model_dir / 'best_classifier_pipeline.joblib')
    with open(model_dir / 'classification_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return pipeline, metadata

try:
    pipeline, metadata = load_model_and_metadata()
    st.success(f"Model loaded: {metadata.get('selected_model', 'Unknown')}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("---")
st.subheader("1. Upload Customer Data (CSV)")
uploaded_file = st.file_uploader("Upload customer features CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Input data:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        with st.spinner('Processing...'):
            feature_cols = metadata.get('feature_columns', [])
            missing_cols = [c for c in feature_cols if c not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                X_input = df[feature_cols].copy()

                # Use pipeline directly (no manual preprocessing)
                predictions = pipeline.predict(X_input)
                probabilities = pipeline.predict_proba(X_input)[:, 1]

                df['Predicted_Repeat'] = predictions
                df['Repeat_Probability'] = probabilities.round(4)

                st.write("---")
                st.subheader("2. Prediction Results")
                st.dataframe(df)

                repeat_count = (predictions == 1).sum()
                no_repeat_count = (predictions == 0).sum()
                st.metric("Predicted Repeat", f"{repeat_count} ({repeat_count/len(df)*100:.1f}%)")
                st.metric("Predicted No Repeat", f"{no_repeat_count} ({no_repeat_count/len(df)*100:.1f}%)")
