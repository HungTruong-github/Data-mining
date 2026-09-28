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
    select_best_model,
    train_final_models,
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
