"""
Runner: 05_repeat_purchase_classification.
Chay tu project root: python notebooks/run_05_repeat_purchase_classification.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

from src.config import (
    PROCESSED_DIR, FIGURES_CLASSIFICATION, TABLES_CLASSIFICATION,
    MODELS_CLASSIFICATION_DIR, RANDOM_STATE, TEST_SIZE, REPEAT_PURCHASE_WINDOW_DAYS
)
from src.classification import (
    validate_classification_dataset, split_classification_data,
    build_preprocessor, build_model_pipelines, cross_validate_models,
    select_best_model, train_final_models, evaluate_on_test,
    get_feature_importance, save_classification_artifacts
)
from src.evaluation import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_model_comparison, plot_feature_importance, save_classification_report_csv
)

TARGET_COL = 'repeat_purchase_90d'

def main():
    total_start = time.time()
    print("=" * 60)
    print("  05. REPEAT PURCHASE CLASSIFICATION")
    print("=" * 60)

    # --- STEP 1: Load Data ---
    print("\n--- STEP 1: Load and Validate Data ---")
    data_path = PROCESSED_DIR / 'repeat_purchase_features.csv'
    assert data_path.exists(), f"Data file not found: {data_path}"
    df = pd.read_csv(data_path)
    print(f"  Loaded: {len(df):,} customers x {df.shape[1]} columns")

    # Validate
    checks = validate_classification_dataset(df, TARGET_COL)
    for c in checks:
        print(f"  {c}")

    # Class distribution
    class_counts = df[TARGET_COL].value_counts().sort_index()
    print(f"\n  Class Distribution:")
    print(f"    Label 0 (No repeat): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    Label 1 (Repeat)   : {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    # Save data summary
    summary = {
        'total_customers': len(df),
        'n_features': df.shape[1] - 2,  # minus CustomerID and target
        'class_0': int(class_counts.get(0, 0)),
        'class_1': int(class_counts.get(1, 0)),
        'class_ratio': round(class_counts.get(1, 0) / len(df), 4),
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
    }
    pd.DataFrame([summary]).to_csv(TABLES_CLASSIFICATION / 'classification_data_summary.csv', index=False)

    # --- STEP 2: Split Data ---
    print("\n--- STEP 2: Train/Test Split ---")
    X_train, X_test, y_train, y_test = split_classification_data(df, TARGET_COL)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test class dist:  {dict(y_test.value_counts().sort_index())}")

    # --- STEP 3: Build Preprocessing Pipeline ---
    print("\n--- STEP 3: Build Preprocessing Pipeline ---")
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    # --- STEP 4: Build Model Pipelines ---
    print("\n--- STEP 4: Build Model Pipelines ---")
    model_pipelines = build_model_pipelines(preprocessor)
    for name in model_pipelines:
        print(f"  - {name}")

    # --- STEP 5: Cross-Validation (Model Selection) ---
    print("\n--- STEP 5: Cross-Validation on TRAINING SET ONLY ---")
    cv_results = cross_validate_models(model_pipelines, X_train, y_train)
    print("\n  CV Results:")
    for _, row in cv_results.iterrows():
        print(f"    {row['model']:25s} CV-F1={row['cv_f1_mean']:.4f} +/- {row['cv_f1_std']:.4f}  "
              f"CV-AUC={row['cv_roc_auc_mean']:.4f}  CV-AP={row['cv_average_precision_mean']:.4f}")
    cv_results.to_csv(TABLES_CLASSIFICATION / 'cv_results.csv', index=False)

    # --- STEP 6: Select Best Model (by CV, NOT test) ---
    print("\n--- STEP 6: Select Best Model ---")
    selected_model = select_best_model(cv_results)
    print(f"  Selected: {selected_model} (based on CV F1-score)")
    print(f"  NOTE: Test set was NOT used for model selection.")

    # --- STEP 7: Train All Models on Full Training Set ---
    print("\n--- STEP 7: Train Final Models ---")
    trained_pipelines = train_final_models(model_pipelines, X_train, y_train)
    print(f"  Trained {len(trained_pipelines)} models on full training set.")

    # --- STEP 8: Evaluate on Test Set (FINAL, ONE-TIME) ---
    print("\n--- STEP 8: Final Evaluation on Test Set ---")
    comparison_df = evaluate_on_test(trained_pipelines, X_test, y_test, cv_results, selected_model)
    comparison_df.to_csv(TABLES_CLASSIFICATION / 'model_comparison.csv', index=False)

    print("\n  Test Set Results:")
    for _, row in comparison_df.iterrows():
        marker = " <<< SELECTED" if row['selected'] else ""
        print(f"    {row['model']:25s} F1={row['test_f1']:.4f}  AUC={row['test_roc_auc']:.4f}  "
              f"Prec={row['test_precision']:.4f}  Rec={row['test_recall']:.4f}{marker}")

    # --- STEP 9: Save Predictions ---
    print("\n--- STEP 9: Save Test Predictions ---")
    best_pipeline = trained_pipelines[selected_model]
    y_pred_best = best_pipeline.predict(X_test)
    try:
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    except:
        y_proba_best = np.full(len(y_pred_best), np.nan)

    pred_df = X_test.copy()
    if 'CustomerID' in df.columns:
        pred_df.insert(0, 'CustomerID', df.loc[X_test.index, 'CustomerID'].values)
    pred_df[TARGET_COL + '_actual'] = y_test.values
    pred_df[TARGET_COL + '_predicted'] = y_pred_best
    pred_df[TARGET_COL + '_probability'] = y_proba_best
    pred_df.to_csv(TABLES_CLASSIFICATION / 'test_predictions.csv', index=False)

    # Classification report
    report_df = save_classification_report_csv(
        y_test, y_pred_best,
        TABLES_CLASSIFICATION / 'classification_report.csv'
    )

    # --- STEP 10: Feature Importance ---
    print("\n--- STEP 10: Feature Importance ---")
    fi_df = get_feature_importance(best_pipeline, numeric_features + categorical_features)
    print("  Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    # --- STEP 11: Visualizations ---
    print("\n--- STEP 11: Visualizations ---")
    plot_confusion_matrix(y_test, y_pred_best, selected_model,
                          save_path=FIGURES_CLASSIFICATION / 'confusion_matrix_best_model.png')
    print("  [OK] confusion_matrix_best_model.png")
    plot_roc_curves(trained_pipelines, X_test, y_test,
                    save_path=FIGURES_CLASSIFICATION / 'roc_curves.png')
    print("  [OK] roc_curves.png")
    plot_precision_recall_curves(trained_pipelines, X_test, y_test,
                                save_path=FIGURES_CLASSIFICATION / 'precision_recall_curves.png')
    print("  [OK] precision_recall_curves.png")
    plot_model_comparison(comparison_df,
                          save_path=FIGURES_CLASSIFICATION / 'model_metric_comparison.png')
    print("  [OK] model_metric_comparison.png")
    plot_feature_importance(fi_df, selected_model,
                            save_path=FIGURES_CLASSIFICATION / 'feature_importance_best_model.png')
    print("  [OK] feature_importance_best_model.png")

    # --- STEP 12: Save Models ---
    print("\n--- STEP 12: Save Models and Metadata ---")
    metadata = {
        'target_col': TARGET_COL,
        'feature_columns': list(X_train.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'selected_model': selected_model,
        'selection_metric': 'cv_f1_mean',
        'threshold': 0.5,
        'training_row_count': len(X_train),
        'test_row_count': len(X_test),
        'class_distribution': {
            'train_0': int((y_train == 0).sum()),
            'train_1': int((y_train == 1).sum()),
            'test_0': int((y_test == 0).sum()),
            'test_1': int((y_test == 1).sum()),
        },
        'window_days': REPEAT_PURCHASE_WINDOW_DAYS,
        'class_weight': 'balanced',
    }

    save_classification_artifacts(
        trained_pipelines, selected_model, metadata,
        MODELS_CLASSIFICATION_DIR, TABLES_CLASSIFICATION, fi_df
    )
    print(f"  [OK] Saved all models to {MODELS_CLASSIFICATION_DIR}")

    # --- STEP 13: Final Validation ---
    print("\n--- STEP 13: Final Validation ---")
    required_files = [
        TABLES_CLASSIFICATION / 'model_comparison.csv',
        TABLES_CLASSIFICATION / 'classification_report.csv',
        TABLES_CLASSIFICATION / 'test_predictions.csv',
        TABLES_CLASSIFICATION / 'classification_data_summary.csv',
        TABLES_CLASSIFICATION / 'cv_results.csv',
        TABLES_CLASSIFICATION / 'feature_importance.csv',
        FIGURES_CLASSIFICATION / 'confusion_matrix_best_model.png',
        FIGURES_CLASSIFICATION / 'roc_curves.png',
        FIGURES_CLASSIFICATION / 'precision_recall_curves.png',
        FIGURES_CLASSIFICATION / 'model_metric_comparison.png',
        FIGURES_CLASSIFICATION / 'feature_importance_best_model.png',
        MODELS_CLASSIFICATION_DIR / 'best_classifier_pipeline.joblib',
        MODELS_CLASSIFICATION_DIR / 'classification_metadata.json',
    ]
    all_ok = True
    for f in required_files:
        if f.exists() and f.stat().st_size > 0:
            print(f"  [OK] {f.name}")
        else:
            print(f"  [FAIL] {f.name}")
            all_ok = False

    assert all_ok, "Some output files are missing or empty!"

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICATION COMPLETE in {elapsed:.2f}s")
    print(f"  Selected Model: {selected_model}")
    print(f"  Customers: {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
