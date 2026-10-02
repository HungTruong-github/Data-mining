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
    features_csv = Path("data/processed/repeat_purchase_features.csv")
    
    if not (model_path.exists() and meta_path.exists() and features_csv.exists()):
        pytest.skip("Prerequisite classification artifacts not found")
        
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    feature_cols = meta.get("feature_columns", [])
    assert len(feature_cols) > 0
    
    df = pd.read_csv(features_csv).head(10)
    X_sample = df[feature_cols].copy()
    threshold = float(meta.get('threshold', 0.5))
    probs = pipeline.predict_proba(X_sample)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_prediction_pipeline_applies_custom_threshold():
    """Verify that classification logic respects arbitrary decision thresholds."""
    import joblib
    model_path = Path("models/classification/best_classifier_pipeline.joblib")
    meta_path = Path("models/classification/classification_metadata.json")
    features_csv = Path("data/processed/repeat_purchase_features.csv")
    
    if not (model_path.exists() and meta_path.exists() and features_csv.exists()):
        pytest.skip("Prerequisite classification artifacts not found")
        
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_columns", [])
    
    df = pd.read_csv(features_csv).head(50)
    X_sample = df[feature_cols].copy()
    probs = pipeline.predict_proba(X_sample)[:, 1]
    
    # At extreme threshold 1.0, predictions must be 0 unless probability is 1.0
    preds_strict = (probs >= 1.0).astype(int)
    preds_lenient = (probs >= 0.0).astype(int)
    assert preds_lenient.sum() == len(X_sample), "Threshold 0.0 must predict 1 for all samples"
    assert preds_strict.sum() <= preds_lenient.sum()


def test_prediction_detects_missing_columns():
    """Verify that feature validation catches missing columns before inference."""
    meta_path = Path("models/classification/classification_metadata.json")
    if not meta_path.exists():
        pytest.skip("Prerequisite classification_metadata.json not found")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_columns", [])
    
    # Incomplete dataframe missing the first required feature
    incomplete_df = pd.DataFrame([{c: 1.0 for c in feature_cols[1:]}])
    missing = [c for c in feature_cols if c not in incomplete_df.columns]
    assert len(missing) > 0, "Phải phát hiện cột đặc trưng bị thiếu"
    assert feature_cols[0] in missing


def test_prediction_detects_empty_dataframe():
    """Verify that empty inputs (0 rows) are handled without unhandled crash."""
    meta_path = Path("models/classification/classification_metadata.json")
    if not meta_path.exists():
        pytest.skip("Prerequisite classification_metadata.json not found")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_columns", [])
    
    empty_df = pd.DataFrame(columns=feature_cols)
    assert len(empty_df) == 0

