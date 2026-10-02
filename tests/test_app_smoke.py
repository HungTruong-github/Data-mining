"""
Smoke test for Streamlit App (app/app.py) using streamlit.testing.v1.AppTest
"""
import io
import json
import pytest
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest

# Path resolved from tests/ directory
APP_PATH = Path(__file__).resolve().parent.parent / "app" / "app.py"


def test_app_loads_successfully():
    """Verify that app/app.py runs without unhandled exceptions and loads model."""
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert len(at.exception) == 0, f"App threw unexpected exception: {at.exception}"
    assert len(at.title) > 0
    assert "Repeat Purchase" in at.title[0].value


def test_pipeline_prediction_on_sample_data():
    """Verify that model pipeline loaded by app can predict on sample data."""
    import joblib
    model_path = Path("models/classification/best_classifier_pipeline.joblib")
    meta_path = Path("models/classification/classification_metadata.json")
    
    assert model_path.exists(), "Model artifact missing"
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    feature_cols = meta.get("feature_columns", [])
    assert len(feature_cols) > 0
    
    features_csv = Path("data/processed/repeat_purchase_features.csv")
    if features_csv.exists():
        df = pd.read_csv(features_csv).head(5)
        X_sample = df[feature_cols].copy()
        preds = pipeline.predict(X_sample)
        probs = pipeline.predict_proba(X_sample)[:, 1]
        
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
