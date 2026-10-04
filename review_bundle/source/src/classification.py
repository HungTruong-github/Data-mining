"""
Module classification: Repeat Purchase Classification Pipeline.
Dự đoán khách hàng có mua lại trong 90 ngày (repeat_purchase_90d).
Tuân thủ nghiêm ngặt: không data leakage, không chọn model bằng test set.
"""
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

from src.config import RANDOM_STATE, TEST_SIZE

# ====================================================================
# COLUMNS TO EXCLUDE
# ====================================================================
FORBIDDEN_COLS = [
    'CustomerID', 'repeat_purchase_90d',
    'future_invoice_count', 'future_revenue',
    'cutoff_date', 'FirstPurchaseDate', 'LastPurchaseDate',
]

DEFAULT_NUMERIC_FEATURES = [
    'Recency', 'Frequency', 'Monetary', 'TotalItems', 'UniqueProducts',
    'ActiveDays', 'AverageOrderValue', 'AverageItemsPerInvoice',
    'CustomerLifetimeDays',
]

DEFAULT_CATEGORICAL_FEATURES = ['Country']


# ====================================================================
# 1. VALIDATE
# ====================================================================
def validate_classification_dataset(df, target_col='repeat_purchase_90d'):
    """
    Validate classification dataset. Raises AssertionError on failure.
    Returns list of check messages.
    """
    checks = []

    # Target exists
    assert target_col in df.columns, f"Target '{target_col}' not found in columns: {list(df.columns)}"
    checks.append(f"[PASS] Target '{target_col}' exists")

    # Target values only 0 and 1
    unique_vals = set(df[target_col].dropna().unique())
    assert unique_vals.issubset({0, 1}), f"Target has unexpected values: {unique_vals}"
    checks.append(f"[PASS] Target only contains 0/1")

    # No missing target
    missing_target = df[target_col].isnull().sum()
    assert missing_target == 0, f"Target has {missing_target} missing values"
    checks.append(f"[PASS] No missing target values")

    # CustomerID exists and no duplicates
    assert 'CustomerID' in df.columns, "CustomerID column missing"
    dup_count = df['CustomerID'].duplicated().sum()
    assert dup_count == 0, f"CustomerID has {dup_count} duplicates"
    checks.append(f"[PASS] No duplicate CustomerID ({len(df)} unique)")

    # No forbidden columns leak into features
    for col in ['future_invoice_count', 'future_revenue']:
        assert col not in df.columns, f"Forbidden column '{col}' found - DATA LEAKAGE!"
    checks.append("[PASS] No future columns (leakage check)")

    # Both classes present
    class_counts = df[target_col].value_counts()
    assert len(class_counts) == 2, f"Target must have 2 classes, found {len(class_counts)}"
    checks.append(f"[PASS] Both classes present: {dict(class_counts)}")

    # No NaN/Inf in numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['CustomerID', target_col]]
    for col in numeric_cols:
        nan_count = df[col].isnull().sum()
        inf_count = np.isinf(df[col]).sum() if df[col].dtype != 'object' else 0
        if nan_count > 0:
            checks.append(f"[WARN] {col} has {nan_count} NaN values (will be imputed)")
        if inf_count > 0:
            checks.append(f"[WARN] {col} has {inf_count} Inf values")

    return checks


# ====================================================================
# 2. SPLIT
# ====================================================================
def split_classification_data(df, target_col='repeat_purchase_90d',
                              test_size=None, random_state=None):
    """
    Split data into X_train, X_test, y_train, y_test.
    Returns feature matrices WITHOUT CustomerID or target.
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    y = df[target_col].astype(int)

    # Determine feature columns
    drop_cols = [c for c in FORBIDDEN_COLS if c in df.columns]
    # Also drop UniqueInvoices if it equals Frequency
    if 'UniqueInvoices' in df.columns and 'Frequency' in df.columns:
        if (df['UniqueInvoices'] == df['Frequency']).all():
            drop_cols.append('UniqueInvoices')

    X = df.drop(columns=drop_cols, errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ====================================================================
# 3. PREPROCESSOR
# ====================================================================
def build_preprocessor(X_train):
    """
    Build ColumnTransformer for numeric (impute+scale) and categorical (impute+onehot).
    """
    all_cols = list(X_train.columns)
    numeric_features = [c for c in DEFAULT_NUMERIC_FEATURES if c in all_cols]
    categorical_features = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in all_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )

    return preprocessor, numeric_features, categorical_features


# ====================================================================
# 4. MODEL PIPELINES
# ====================================================================
def build_model_pipelines(preprocessor, random_state=None):
    """
    Build dict of {name: Pipeline} for each model.
    """
    if random_state is None:
        random_state = RANDOM_STATE

    models = {
        'DummyClassifier': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DummyClassifier(strategy='most_frequent', random_state=random_state)),
        ]),
        'LogisticRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000, random_state=random_state,
                class_weight='balanced', solver='lbfgs'
            )),
        ]),
        'DecisionTree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                random_state=random_state, class_weight='balanced', max_depth=10
            )),
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, random_state=random_state,
                class_weight='balanced', n_jobs=-1
            )),
        ]),
    }

    return models


# ====================================================================
# 5. PREDICTION WITH UNIFIED THRESHOLD FUNCTION & CUSTOM SCORER
# ====================================================================
def predict_with_threshold(pipeline, X, threshold=0.5, pos_label=1):
    """
    Unified function to compute positive class probabilities and predict binary labels.
    
    Decision Rule:
        y_pred = 1 if prob >= threshold else 0
        
    Key guarantees:
    - probability == threshold is strictly mapped to class 1 (>= operator).
    - No rounding is performed on probability prior to threshold comparison.
    - Positive class column index is determined dynamically from classes_ (pipeline or classifier).
    - Guaranteed 100% parity between offline evaluation, CSV exports, and Streamlit dashboard.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline or classifier model
        Fitted model pipeline.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    threshold : float, default=0.5
        Decision threshold for positive class.
    pos_label : int or str, default=1
        Label of the positive class.
        
    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels (0 or 1).
    y_proba : np.ndarray of shape (n_samples,)
        Unrounded predicted probabilities for the positive class.
    """
    if hasattr(pipeline, 'classes_'):
        classes = list(pipeline.classes_)
    elif hasattr(pipeline, 'named_steps') and hasattr(pipeline.named_steps.get('classifier'), 'classes_'):
        classes = list(pipeline.named_steps['classifier'].classes_)
    else:
        classes = [0, 1]

    if hasattr(pipeline, 'predict_proba'):
        try:
            pos_idx = classes.index(pos_label)
        except ValueError:
            pos_idx = 1 if len(classes) > 1 else 0
        probs = pipeline.predict_proba(X)[:, pos_idx]
        preds = (probs >= threshold).astype(int)
    else:
        preds = pipeline.predict(X).astype(int)
        probs = preds.astype(float)

    return preds, probs


def threshold_f1_scorer_05(estimator, X, y):
    """
    Scikit-learn compatible scorer enforcing probability >= 0.5 decision threshold convention.
    Uses predict_with_threshold to guarantee complete mathematical parity between
    cross-validation grid search, test set evaluation, and interactive dashboard predictions.
    Pickle-safe at module level for multi-process cross-validation.
    """
    y_pred, _ = predict_with_threshold(estimator, X, threshold=0.5, pos_label=1)
    return f1_score(y, y_pred, pos_label=1, zero_division=0)


def predict_customers(pipeline, metadata, input_df):
    """
    Unified production prediction function for repeat purchase inference.
    Shared by Streamlit web application (app/app.py), runner scripts, notebooks, and test suite.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    metadata : dict
        Classification metadata dictionary containing 'feature_columns', 'threshold', 'positive_class'.
    input_df : pd.DataFrame
        Input DataFrame containing customer features.
        
    Returns
    -------
    y_pred : np.ndarray
        Binary predictions (0 or 1).
    y_proba : np.ndarray
        Unrounded positive class probabilities.
    """
    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary.")

    feature_cols = metadata.get('feature_columns', [])
    if not feature_cols:
        raise ValueError("Metadata must contain 'feature_columns' list.")

    missing = [c for c in feature_cols if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required feature columns: {missing}")

    if len(input_df) == 0:
        raise ValueError("Input dataframe is empty (0 rows).")

    X = input_df[feature_cols].copy()
    threshold = float(metadata.get('threshold', 0.5))
    pos_label = metadata.get('positive_class', 1)

    return predict_with_threshold(pipeline, X, threshold=threshold, pos_label=pos_label)


# ====================================================================
# 6. CROSS-VALIDATION AND HYPERPARAMETER TUNING
# ====================================================================
def cross_validate_models(model_pipelines, X_train, y_train,
                          cv_folds=5, random_state=None):
    """
    Cross-validate all models on training data ONLY using unified threshold scorer.
    Returns DataFrame with CV results.
    """
    if random_state is None:
        random_state = RANDOM_STATE

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'f1': threshold_f1_scorer_05,
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }

    cv_results = []
    for name, pipeline in model_pipelines.items():
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring=scoring, return_train_score=False, n_jobs=-1
            )
        elapsed = time.time() - start_time

        cv_results.append({
            'model': name,
            'cv_f1_mean': float(np.mean(scores['test_f1'])),
            'cv_f1_std': float(np.std(scores['test_f1'])),
            'cv_roc_auc_mean': float(np.mean(scores['test_roc_auc'])),
            'cv_roc_auc_std': float(np.std(scores['test_roc_auc'])),
            'cv_average_precision_mean': float(np.mean(scores['test_average_precision'])),
            'cv_average_precision_std': float(np.std(scores['test_average_precision'])),
            'cv_runtime_seconds': round(elapsed, 2),
        })

    return pd.DataFrame(cv_results)


def tune_model_hyperparameters(preprocessor, X_train, y_train, cv_folds=5, random_state=None):
    """
    Perform bounded hyperparameter tuning with GridSearchCV strictly on training set.
    Evaluates LR, DT, RF across defined parameter grids using StratifiedKFold and threshold_f1_scorer_05.
    Also evaluates DummyClassifier baseline under identical folds.
    
    Returns:
    --------
    best_pipelines : dict of {model_name: best_Pipeline}
    cv_results_df : pd.DataFrame comparing best model configs across CV F1, ROC-AUC, AP
    tuning_history_df : pd.DataFrame detailing all tested configurations and fold scores
    best_params : dict of {model_name: dict of best parameters}
    """
    from sklearn.model_selection import GridSearchCV
    if random_state is None:
        random_state = RANDOM_STATE

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    raw_pipelines = {
        'DummyClassifier': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DummyClassifier(strategy='most_frequent', random_state=random_state))
            ]),
            {}
        ),
        'LogisticRegression': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=random_state, solver='lbfgs'))
            ]),
            {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__class_weight': ['balanced', None],
            }
        ),
        'DecisionTree': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=random_state))
            ]),
            {
                'classifier__max_depth': [3, 5, 10],
                'classifier__min_samples_leaf': [1, 5, 20],
                'classifier__class_weight': ['balanced', None],
            }
        ),
        'RandomForest': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=random_state, n_jobs=-1))
            ]),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10],
                'classifier__min_samples_leaf': [1, 5],
                'classifier__class_weight': ['balanced', None],
            }
        )
    }

    best_pipelines = {}
    cv_summary_rows = []
    tuning_history_rows = []
    best_params = {}

    scoring_dict = {
        'f1': threshold_f1_scorer_05,
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }

    for name, (base_pipe, param_grid) in raw_pipelines.items():
        t0 = time.time()
        if param_grid:
            gs = GridSearchCV(
                base_pipe, param_grid, cv=cv, scoring=threshold_f1_scorer_05,
                return_train_score=False, n_jobs=-1, refit=True
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            best_p = gs.best_params_

            cv_res = gs.cv_results_
            for i in range(len(cv_res['params'])):
                row = {
                    'model': name,
                    'params': str(cv_res['params'][i]),
                    'mean_test_f1': float(cv_res['mean_test_score'][i]),
                    'std_test_f1': float(cv_res['std_test_score'][i]),
                    'rank_f1': int(cv_res['rank_test_score'][i]),
                    'mean_fit_time_seconds': float(cv_res['mean_fit_time'][i]),
                }
                for f in range(cv_folds):
                    row[f'split{f}_test_f1'] = float(cv_res[f'split{f}_test_score'][i])
                tuning_history_rows.append(row)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_pipe.fit(X_train, y_train)
            best_pipe = base_pipe
            best_p = {'strategy': 'most_frequent'}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                best_pipe, X_train, y_train, cv=cv, scoring=scoring_dict, return_train_score=False, n_jobs=-1
            )
        elapsed = time.time() - t0

        best_pipelines[name] = best_pipe
        best_params[name] = best_p

        cv_summary_rows.append({
            'model': name,
            'cv_f1_mean': float(np.mean(scores['test_f1'])),
            'cv_f1_std': float(np.std(scores['test_f1'])),
            'cv_roc_auc_mean': float(np.mean(scores['test_roc_auc'])),
            'cv_roc_auc_std': float(np.std(scores['test_roc_auc'])),
            'cv_average_precision_mean': float(np.mean(scores['test_average_precision'])),
            'cv_average_precision_std': float(np.std(scores['test_average_precision'])),
            'cv_runtime_seconds': round(elapsed, 2),
            'best_params': json.dumps(best_p),
        })

    cv_results_df = pd.DataFrame(cv_summary_rows)
    tuning_history_df = pd.DataFrame(tuning_history_rows) if tuning_history_rows else pd.DataFrame()
    return best_pipelines, cv_results_df, tuning_history_df, best_params


# ====================================================================
# 7. SELECT BEST MODEL (by CV, NOT test set)
# ====================================================================
def select_best_model(cv_results_df, metric='cv_f1_mean', secondary_metric='cv_average_precision_mean'):
    """
    Select best model based on cross-validation metric.
    NOT based on test set performance.
    """
    # Exclude DummyClassifier from selection
    candidates = cv_results_df[cv_results_df['model'] != 'DummyClassifier'].copy()
    if candidates.empty:
        raise ValueError("No valid candidate models found!")

    # Sort by primary metric, then secondary
    candidates = candidates.sort_values(
        [metric, secondary_metric], ascending=[False, False]
    )

    best_model_name = candidates.iloc[0]['model']
    return best_model_name


# ====================================================================
# 8. TRAIN FINAL MODELS (on full training set)
# ====================================================================
def train_final_models(model_pipelines, X_train, y_train):
    """
    Train all model pipelines on full training data.
    Returns dict of trained pipelines.
    """
    trained = {}
    for name, pipeline in model_pipelines.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        trained[name] = pipeline
    return trained


# ====================================================================
# 9. EVALUATE ON TEST SET
# ====================================================================
def evaluate_on_test(trained_pipelines, X_test, y_test, cv_results_df, selected_model,
                     threshold=0.5, pos_label=1):
    """
    Evaluate all trained models on test set (final evaluation only).
    Uses unified predict_with_threshold logic so offline evaluation and dashboard are 100% consistent.
    Returns comprehensive metrics DataFrame.
    """
    results = []
    for name, pipeline in trained_pipelines.items():
        start_time = time.time()
        y_pred, y_proba = predict_with_threshold(pipeline, X_test, threshold=threshold, pos_label=pos_label)

        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
        except Exception:
            roc_auc = np.nan
            avg_precision = np.nan

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        elapsed = time.time() - start_time

        # Merge with CV results
        cv_row = cv_results_df[cv_results_df['model'] == name]
        cv_f1_mean = cv_row['cv_f1_mean'].values[0] if not cv_row.empty else np.nan
        cv_f1_std = cv_row['cv_f1_std'].values[0] if not cv_row.empty else np.nan
        cv_roc_mean = cv_row['cv_roc_auc_mean'].values[0] if not cv_row.empty else np.nan
        cv_ap_mean = cv_row['cv_average_precision_mean'].values[0] if not cv_row.empty else np.nan

        results.append({
            'model': name,
            'cv_f1_mean': round(cv_f1_mean, 4),
            'cv_f1_std': round(cv_f1_std, 4),
            'cv_roc_auc_mean': round(cv_roc_mean, 4),
            'cv_average_precision_mean': round(cv_ap_mean, 4),
            'test_accuracy': round(accuracy_score(y_test, y_pred), 4),
            'test_balanced_accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
            'test_precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'test_recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'test_f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'test_roc_auc': round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
            'test_average_precision': round(avg_precision, 4) if not np.isnan(avg_precision) else np.nan,
            'test_specificity': round(specificity, 4),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'runtime_seconds': round(elapsed, 4),
            'selected': name == selected_model,
            'random_state': RANDOM_STATE,
        })

    return pd.DataFrame(results)


# ====================================================================
# 10. FEATURE IMPORTANCE
# ====================================================================
def get_feature_importance(trained_pipeline, feature_names):
    """
    Extract feature importance from the trained pipeline's classifier step.
    """
    classifier = trained_pipeline.named_steps['classifier']
    preprocessor = trained_pipeline.named_steps['preprocessor']

    # Get transformed feature names
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = feature_names

    # Clean up names
    clean_names = []
    for n in transformed_names:
        if n.startswith('num__'):
            clean_names.append(n.replace('num__', ''))
        elif n.startswith('cat__'):
            clean_names.append(n.replace('cat__', ''))
        else:
            clean_names.append(n)

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame({'feature': clean_names, 'importance': [np.nan]*len(clean_names)})

    fi_df = pd.DataFrame({
        'feature': clean_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return fi_df


# ====================================================================
# 11. SAVE ARTIFACTS
# ====================================================================
def save_classification_artifacts(trained_pipelines, selected_model, metadata,
                                  models_dir, tables_dir, feature_importance_df=None,
                                  tuning_history_df=None):
    """
    Save all model pipelines, metadata JSON, feature importance, and tuning history.
    """
    models_dir = Path(models_dir)
    tables_dir = Path(tables_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save each pipeline
    pipeline_names = {
        'LogisticRegression': 'logistic_regression_pipeline.joblib',
        'DecisionTree': 'decision_tree_pipeline.joblib',
        'RandomForest': 'random_forest_pipeline.joblib',
        'DummyClassifier': 'dummy_classifier_pipeline.joblib',
    }

    for name, pipeline in trained_pipelines.items():
        filename = pipeline_names.get(name, f'{name.lower()}_pipeline.joblib')
        joblib.dump(pipeline, models_dir / filename)

    # Save best model separately
    if selected_model in trained_pipelines:
        joblib.dump(trained_pipelines[selected_model],
                    models_dir / 'best_classifier_pipeline.joblib')

    # Save metadata
    with open(models_dir / 'classification_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)

    # Save feature importance
    if feature_importance_df is not None:
        feature_importance_df.to_csv(tables_dir / 'feature_importance.csv', index=False)

    # Save tuning history
    if tuning_history_df is not None and not tuning_history_df.empty:
        tuning_history_df.to_csv(tables_dir / 'cv_tuning_history.csv', index=False)

    return True
