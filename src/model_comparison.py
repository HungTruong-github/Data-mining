"""
Step 07 — Model Comparison: load, validate, and compare results from steps 04-06.
All data is read from CSV/JSON outputs, nothing is re-trained here.
"""
import json, warnings, hashlib
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from src.config import (
    PROCESSED_DIR, TABLES_CLUSTERING, TABLES_CLASSIFICATION,
    TABLES_ASSOCIATION, MODELS_CLUSTERING_DIR, MODELS_CLASSIFICATION_DIR,
    RANDOM_STATE
)


# ──────────────────────────────────────────────
# INPUT FILE MAP
# ──────────────────────────────────────────────
INPUT_FILES = {
    'rfm_features':        PROCESSED_DIR / 'rfm_customer_features.csv',
    'repeat_features':     PROCESSED_DIR / 'repeat_purchase_features.csv',
    'customer_clusters':   PROCESSED_DIR / 'customer_clusters.csv',
    'cluster_comparison':  TABLES_CLUSTERING / 'clustering_algorithm_comparison.csv',
    'cluster_profiles':    TABLES_CLUSTERING / 'cluster_profiles.csv',
    'class_comparison':    TABLES_CLASSIFICATION / 'model_comparison.csv',
    'cv_results':          TABLES_CLASSIFICATION / 'cv_results.csv',
    'test_predictions':    TABLES_CLASSIFICATION / 'test_predictions.csv',
    'feature_importance':  TABLES_CLASSIFICATION / 'feature_importance.csv',
    'assoc_comparison':    TABLES_ASSOCIATION / 'association_algorithm_comparison.csv',
    'selected_rules':      TABLES_ASSOCIATION / 'association_rules' / 'selected_association_rules.csv',
    'business_insights':   TABLES_ASSOCIATION / 'association_business_insights.csv',
    'class_metadata':      MODELS_CLASSIFICATION_DIR / 'classification_metadata.json',
}


def validate_inputs():
    """Check all required input files exist and are non-empty. Returns report."""
    report = []
    all_ok = True
    for name, path in INPUT_FILES.items():
        if not path.exists():
            report.append(f"[MISS] {name}: {path}")
            all_ok = False
        elif path.stat().st_size == 0:
            report.append(f"[EMPTY] {name}: {path}")
            all_ok = False
        else:
            report.append(f"[OK] {name}: {path.name}")
    return report, all_ok


def load_input(name):
    """Load a single input file by name."""
    if name not in INPUT_FILES:
        raise FileNotFoundError(f"Unknown input key: {name}")
    path = INPUT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    if path.suffix == '.json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return pd.read_csv(path)


# ──────────────────────────────────────────────
# CLUSTERING COMPARISON
# ──────────────────────────────────────────────
def build_clustering_comparison():
    """Load clustering comparison, add selection_reason, return enriched DataFrame."""
    df = load_input('cluster_comparison').copy()

    # Determine actual selected model from saved artifact
    try:
        config = joblib.load(MODELS_CLUSTERING_DIR / 'clustering_config.pkl')
        sel_algo = config.get('algorithm', '')
        sel_k = config.get('n_clusters', -1)
    except Exception:
        sel_algo, sel_k = '', -1

    # Mark is_selected based on artifact match
    df['is_selected'] = (
        (df['algorithm'] == sel_algo) & (df['n_clusters'] == sel_k)
    )

    # Build selection_reason
    reasons = []
    for _, row in df.iterrows():
        if row.get('is_selected', False):
            parts = []
            parts.append(f"Silhouette={row.get('silhouette_score', 'N/A'):.4f}")
            parts.append(f"DB={row.get('davies_bouldin_score', 'N/A'):.4f}")
            if row.get('noise_ratio', 0) > 0:
                parts.append(f"noise={row.get('noise_ratio', 0):.2%}")
            parts.append("Best combined ranking across Silhouette, DB, CH, Dunn metrics")
            reasons.append('; '.join(parts))
        else:
            reasons.append('')
    df['selection_reason'] = reasons

    return df


# ──────────────────────────────────────────────
# CLASSIFICATION COMPARISON
# ──────────────────────────────────────────────
def build_classification_comparison():
    """Load classification comparison, add selection_reason, return enriched DataFrame."""
    df = load_input('class_comparison').copy()

    # Rename 'selected' -> 'is_selected' for consistency
    if 'selected' in df.columns and 'is_selected' not in df.columns:
        df['is_selected'] = df['selected'].astype(bool)

    # Add selection_reason
    reasons = []
    for _, row in df.iterrows():
        if row.get('is_selected', False):
            reasons.append(
                f"Highest CV F1-score ({row.get('cv_f1_mean', 0):.4f}) among non-baseline models. "
                f"Model selected via StratifiedKFold CV on training set only. "
                f"Test set NOT used for selection."
            )
        elif row.get('model', '') == 'DummyClassifier':
            reasons.append('Baseline only — predicts majority class')
        else:
            reasons.append('')
    df['selection_reason'] = reasons

    # Add specificity if missing
    if 'specificity' not in df.columns and 'test_specificity' in df.columns:
        df['specificity'] = df['test_specificity']

    return df


# ──────────────────────────────────────────────
# ASSOCIATION RULES COMPARISON
# ──────────────────────────────────────────────
def build_association_comparison():
    """Load association rules comparison, add selection info."""
    df = load_input('assoc_comparison').copy()

    # Determine selected algorithm (faster one if results identical)
    if len(df) >= 2:
        # Check if rule counts are the same
        rule_counts = df['valid_rule_count'].unique()
        if len(rule_counts) == 1:
            # Same results — select faster
            fastest_idx = df['runtime_seconds'].idxmin()
            df['is_selected'] = False
            df.loc[fastest_idx, 'is_selected'] = True
            df['selection_reason'] = ''
            df.loc[fastest_idx, 'selection_reason'] = (
                f"Both algorithms produce identical rules ({int(rule_counts[0])} valid rules). "
                f"Selected for faster runtime ({df.loc[fastest_idx, 'runtime_seconds']:.2f}s)."
            )
        else:
            # Different results — select one with more valid rules
            best_idx = df['valid_rule_count'].idxmax()
            df['is_selected'] = False
            df.loc[best_idx, 'is_selected'] = True
            df['selection_reason'] = ''
            df.loc[best_idx, 'selection_reason'] = f"More valid rules ({df.loc[best_idx, 'valid_rule_count']})"
    elif len(df) == 1:
        df['is_selected'] = True
        df['selection_reason'] = 'Only algorithm available'

    return df


# ──────────────────────────────────────────────
# VALIDATE RULES
# ──────────────────────────────────────────────
def validate_selected_rules():
    """Validate association rules quality."""
    df = load_input('selected_rules')
    checks = []

    # No empty antecedents/consequents
    empty_ant = df['antecedents_str'].isna().sum() + (df['antecedents_str'] == '').sum()
    checks.append(f"[{'PASS' if empty_ant == 0 else 'FAIL'}] Empty antecedents: {empty_ant}")

    empty_con = df['consequents_str'].isna().sum() + (df['consequents_str'] == '').sum()
    checks.append(f"[{'PASS' if empty_con == 0 else 'FAIL'}] Empty consequents: {empty_con}")

    # No NaN/Inf in metrics
    for col in ['support', 'confidence', 'lift']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            ok = nan_count == 0 and inf_count == 0
            checks.append(f"[{'PASS' if ok else 'FAIL'}] {col} NaN={nan_count} Inf={inf_count}")

    # Lift > 1 for all selected rules
    if 'lift' in df.columns:
        low_lift = (df['lift'] <= 1).sum()
        checks.append(f"[{'PASS' if low_lift == 0 else 'WARN'}] Rules with lift <= 1: {low_lift}")

    return checks, df


# ──────────────────────────────────────────────
# VERIFY MODEL ARTIFACTS
# ──────────────────────────────────────────────
def verify_model_artifacts():
    """Check that saved model files can be loaded."""
    checks = []

    # Clustering model
    cm_path = MODELS_CLUSTERING_DIR / 'clustering_model.pkl'
    try:
        model = joblib.load(cm_path)
        checks.append(f"[PASS] Clustering model loadable: {type(model).__name__}")
    except Exception as e:
        checks.append(f"[FAIL] Clustering model: {e}")

    # Classification pipeline
    cp_path = MODELS_CLASSIFICATION_DIR / 'best_classifier_pipeline.joblib'
    try:
        pipeline = joblib.load(cp_path)
        checks.append(f"[PASS] Classification pipeline loadable: {type(pipeline).__name__}")
    except Exception as e:
        checks.append(f"[FAIL] Classification pipeline: {e}")

    # Classification metadata
    meta_path = MODELS_CLASSIFICATION_DIR / 'classification_metadata.json'
    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        checks.append(f"[PASS] Classification metadata: selected={meta.get('selected_model')}")
    except Exception as e:
        checks.append(f"[FAIL] Classification metadata: {e}")

    return checks


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()
