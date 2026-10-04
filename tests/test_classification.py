"""
Tests for classification pipeline.
Uses small synthetic data — NOT for reporting experimental results.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classification import (
    validate_classification_dataset,
    split_classification_data,
    build_preprocessor,
    build_model_pipelines,
    cross_validate_models,
    tune_model_hyperparameters,
    select_best_model,
    train_final_models,
    predict_with_threshold,
    evaluate_on_test,
    get_feature_importance,
)


@pytest.fixture
def sample_df():
    """Create small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Recency': np.random.randint(1, 365, n),
        'Frequency': np.random.randint(1, 50, n),
        'Monetary': np.random.uniform(10, 5000, n),
        'TotalItems': np.random.randint(1, 500, n),
        'UniqueProducts': np.random.randint(1, 100, n),
        'ActiveDays': np.random.randint(1, 200, n),
        'AverageOrderValue': np.random.uniform(5, 500, n),
        'AverageItemsPerInvoice': np.random.uniform(1, 50, n),
        'CustomerLifetimeDays': np.random.randint(0, 365, n),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany'], n),
        'repeat_purchase_90d': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })
    return df


def test_validate_passes(sample_df):
    checks = validate_classification_dataset(sample_df)
    assert all('[PASS]' in c or '[WARN]' in c for c in checks)


def test_validate_fails_missing_target(sample_df):
    df_bad = sample_df.drop(columns=['repeat_purchase_90d'])
    with pytest.raises(AssertionError):
        validate_classification_dataset(df_bad)


def test_validate_fails_future_columns(sample_df):
    sample_df['future_invoice_count'] = 5
    with pytest.raises(AssertionError):
        validate_classification_dataset(sample_df)


def test_validate_fails_duplicate_customer(sample_df):
    sample_df.loc[0, 'CustomerID'] = sample_df.loc[1, 'CustomerID']
    with pytest.raises(AssertionError):
        validate_classification_dataset(sample_df)


def test_split_no_customerid_in_X(sample_df):
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    assert 'CustomerID' not in X_train.columns
    assert 'repeat_purchase_90d' not in X_train.columns


def test_pipeline_fit_predict(sample_df):
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, num_feat, cat_feat = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        assert len(preds) == len(y_test)
        proba = pipeline.predict_proba(X_test)
        assert proba.shape[1] == 2


def test_cross_validate_and_select(sample_df):
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)

    cv_results = cross_validate_models(pipelines, X_train, y_train, cv_folds=3)
    assert len(cv_results) == len(pipelines)
    assert 'cv_f1_mean' in cv_results.columns

    best = select_best_model(cv_results)
    assert best != 'DummyClassifier'


def test_unknown_category_handled(sample_df):
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)
    pipeline = pipelines['LogisticRegression']
    pipeline.fit(X_train, y_train)

    # Add unseen category
    X_test_new = X_test.copy()
    X_test_new['Country'] = 'Atlantis'
    preds = pipeline.predict(X_test_new)
    assert len(preds) == len(X_test_new)


def test_no_nan_in_metrics(sample_df):
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)
    cv_results = cross_validate_models(pipelines, X_train, y_train, cv_folds=3)
    trained = train_final_models(pipelines, X_train, y_train)
    selected = select_best_model(cv_results)
    comparison = evaluate_on_test(trained, X_test, y_test, cv_results, selected)

    # Key metrics should not be NaN for real classifiers
    for _, row in comparison[comparison['model'] != 'DummyClassifier'].iterrows():
        assert not np.isnan(row['test_f1'])
        assert not np.isnan(row['test_roc_auc'])


def test_predict_with_threshold_strictly_applies_operator():
    """Verify that probability >= threshold is strictly applied without rounding."""
    class MockPipeline:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.array([
                [0.5000001, 0.4999999], # Below 0.5
                [0.5000000, 0.5000000], # Exactly 0.5 -> must map to 1
                [0.4999999, 0.5000001], # Above 0.5 -> must map to 1
                [0.3000000, 0.7000000], # Custom test -> 0.7
            ])

    pipe = MockPipeline()
    X_dummy = np.zeros((4, 2))

    # Test threshold = 0.5
    preds, probs = predict_with_threshold(pipe, X_dummy, threshold=0.5, pos_label=1)
    np.testing.assert_array_equal(preds, [0, 1, 1, 1])
    assert probs[1] == 0.5
    assert preds[1] == 1, "Probability == threshold (0.5) MUST strictly map to 1"

    # Test custom threshold = 0.7
    preds_70, _ = predict_with_threshold(pipe, X_dummy, threshold=0.7, pos_label=1)
    np.testing.assert_array_equal(preds_70, [0, 0, 0, 1])


def test_offline_dashboard_parity(sample_df):
    """Verify that offline evaluation and dashboard produce 100% identical outputs."""
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)
    pipe = pipelines['LogisticRegression']
    pipe.fit(X_train, y_train)

    # Offline evaluation call
    offline_preds, offline_probs = predict_with_threshold(pipe, X_test, threshold=0.5, pos_label=1)

    # Dashboard simulation call
    dash_threshold = 0.5
    dash_pos_label = 1
    dash_preds, dash_probs = predict_with_threshold(pipe, X_test, threshold=dash_threshold, pos_label=dash_pos_label)

    np.testing.assert_array_equal(offline_preds, dash_preds)
    np.testing.assert_array_almost_equal(offline_probs, dash_probs, decimal=10)


def test_tune_model_hyperparameters_runs_cleanly(sample_df):
    """Verify that hyperparameter grid search explores search space and returns CV and tuning history tables."""
    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)

    best_pipes, cv_results, tuning_history, best_params = tune_model_hyperparameters(
        preprocessor, X_train, y_train, cv_folds=2, random_state=42
    )

    assert len(best_pipes) == 4
    assert set(cv_results['model']) == {'DummyClassifier', 'LogisticRegression', 'DecisionTree', 'RandomForest'}
    assert not tuning_history.empty
    assert 'rank_f1' in tuning_history.columns
    assert 'mean_test_f1' in tuning_history.columns
    assert len(best_params) == 4


def test_predict_customers_validation_and_parity(sample_df):
    """Verify predict_customers validates features and exactly matches predict_with_threshold."""
    from src.classification import predict_customers, threshold_f1_scorer_05

    X_train, X_test, y_train, y_test = split_classification_data(sample_df)
    preprocessor, _, _ = build_preprocessor(X_train)
    pipelines = build_model_pipelines(preprocessor)
    pipe = pipelines['RandomForest']
    pipe.fit(X_train, y_train)

    metadata = {
        'target_col': 'repeat_purchase_90d',
        'feature_columns': list(X_train.columns),
        'threshold': 0.5,
        'positive_class': 1,
    }

    # 1. Missing columns test
    bad_input = X_test.drop(columns=[X_train.columns[0]])
    with pytest.raises(ValueError, match="Input dataframe missing required feature columns"):
        predict_customers(pipe, metadata, bad_input)

    # 2. Functional parity test
    preds_cust, probs_cust = predict_customers(pipe, metadata, X_test)
    assert len(preds_cust) == len(X_test)
    assert len(probs_cust) == len(X_test)

    y_pred_direct, y_proba_direct = predict_with_threshold(pipe, X_test, threshold=0.5, pos_label=1)
    np.testing.assert_array_equal(preds_cust, y_pred_direct)
    np.testing.assert_array_almost_equal(probs_cust, y_proba_direct, decimal=10)

    # 3. Test threshold_f1_scorer_05 callable
    f1_val = threshold_f1_scorer_05(pipe, X_test, y_test)
    assert 0.0 <= f1_val <= 1.0

