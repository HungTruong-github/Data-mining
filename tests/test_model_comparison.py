"""
Tests for Step 07: Model Comparison and Insights.
Uses synthetic data for unit tests — NOT for reporting real results.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Fixtures ──
@pytest.fixture
def sample_cluster_comparison():
    return pd.DataFrame({
        'algorithm': ['K-Means', 'K-Means', 'GMM'],
        'n_clusters': [2, 3, 2],
        'silhouette_score': [0.45, 0.38, 0.42],
        'davies_bouldin_score': [0.9, 1.1, 0.95],
        'calinski_harabasz_score': [500, 400, 480],
        'dunn_approximation': [0.3, 0.25, 0.28],
        'noise_ratio': [0, 0, 0],
        'min_cluster_size': [100, 50, 90],
        'max_cluster_pct': [0.6, 0.5, 0.55],
        'runtime_seconds': [0.5, 0.6, 0.8],
        'is_selected': [True, False, False],
    })


@pytest.fixture
def sample_class_comparison():
    return pd.DataFrame({
        'model': ['DummyClassifier', 'LogisticRegression', 'RandomForest'],
        'cv_f1_mean': [0.72, 0.68, 0.70],
        'cv_f1_std': [0.01, 0.02, 0.02],
        'cv_roc_auc_mean': [0.5, 0.73, 0.71],
        'test_accuracy': [0.57, 0.70, 0.68],
        'test_balanced_accuracy': [0.5, 0.71, 0.67],
        'test_precision': [0.57, 0.79, 0.73],
        'test_recall': [1.0, 0.63, 0.69],
        'test_f1': [0.73, 0.70, 0.71],
        'test_roc_auc': [0.5, 0.78, 0.74],
        'test_average_precision': [0.57, 0.84, 0.81],
        'test_specificity': [0, 0.78, 0.66],
        'tn': [0, 227, 191], 'fp': [290, 63, 99],
        'fn': [0, 141, 119], 'tp': [384, 243, 265],
        'runtime_seconds': [0.03, 0.03, 0.19],
        'selected': [False, False, True],
        'random_state': [42, 42, 42],
    })


def test_clustering_comparison_schema(sample_cluster_comparison):
    required = ['algorithm', 'n_clusters', 'silhouette_score', 'is_selected']
    for col in required:
        assert col in sample_cluster_comparison.columns


def test_classification_comparison_schema(sample_class_comparison):
    required = ['model', 'cv_f1_mean', 'test_f1', 'test_roc_auc', 'selected']
    for col in required:
        assert col in sample_class_comparison.columns


def test_no_nan_in_key_metrics(sample_class_comparison):
    for col in ['test_f1', 'test_roc_auc']:
        assert sample_class_comparison[col].isna().sum() == 0


def test_selected_not_dummy(sample_class_comparison):
    selected = sample_class_comparison[sample_class_comparison['selected'] == True]
    assert all(selected['model'] != 'DummyClassifier')


def test_segment_insights_have_required_cols():
    from src.insights import build_customer_segment_insights
    try:
        df = build_customer_segment_insights()
        required = ['cluster_id', 'customer_count', 'recency_mean', 'frequency_mean', 'monetary_mean']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Input files not available")


def test_action_plan_not_empty():
    from src.insights import build_customer_segment_insights, build_action_plan
    try:
        seg = build_customer_segment_insights()
        actions = build_action_plan(seg)
        assert len(actions) > 0
        assert 'strategy' in actions.columns
        assert 'recommended_action' in actions.columns
    except FileNotFoundError:
        pytest.skip("Input files not available")


def test_feature_insights_schema():
    from src.insights import build_feature_insights
    try:
        df = build_feature_insights()
        assert 'feature' in df.columns
        assert 'importance' in df.columns
        assert 'rank' in df.columns
        assert 'interpretation' in df.columns
        assert 'limitation' in df.columns
    except FileNotFoundError:
        pytest.skip("Input files not available")


def test_product_insights_no_empty_rules():
    from src.insights import build_product_association_insights
    try:
        df = build_product_association_insights()
        assert df['antecedents'].notna().all()
        assert df['consequents'].notna().all()
        assert (df['lift'] > 0).all()
    except FileNotFoundError:
        pytest.skip("Input files not available")


def test_missing_input_fails_clearly():
    from src.model_comparison import load_input
    with pytest.raises(FileNotFoundError):
        load_input('nonexistent_key')


def test_summary_json_schema():
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports',
                                '07_model_comparison_and_insights_summary.json')
    if not os.path.exists(summary_path):
        pytest.skip("Summary JSON not yet generated")
    with open(summary_path) as f:
        data = json.load(f)
    assert 'random_state' in data
    assert 'selected_clustering_model' in data
    assert 'selected_classification_model' in data
    assert 'validation_status' in data
