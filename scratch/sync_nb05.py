import json
from pathlib import Path

def create_cell(cell_type, source, outputs=None, execution_count=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]] if source else []
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

cells = []

# Cell 0: Header
cells.append(create_cell("markdown", """# 05. Repeat Purchase Classification

**Mục tiêu bài toán:**
- Dự đoán khả năng khách hàng quay lại mua hàng trong vòng 90 ngày (`repeat_purchase_90d`).
- So sánh các thuật toán: `DummyClassifier` (baseline), `LogisticRegression`, `DecisionTree`, và `RandomForest`.
- **Tối ưu siêu tham số & Chọn mô hình hoàn toàn dựa trên 5-Fold Stratified Cross-Validation trên Training Set.**
- Nghiêm cấm rò rỉ dữ liệu (Data Leakage): Test set chỉ được dùng cho đánh giá duy nhất một lần cuối cùng.
- Đảm bảo tính nhất quán tuyệt đối giữa Cross-Validation và Inference bằng cơ chế phân loại xác suất với ngưỡng cố định (`probability >= 0.5`)."""))

# Cell 1: Imports
cells.append(create_cell("code", """import sys, os
sys.path.insert(0, os.path.abspath('..'))
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
%matplotlib inline

from src.config import (
    PROJECT_ROOT, PROCESSED_DIR, FIGURES_CLASSIFICATION, TABLES_CLASSIFICATION,
    MODELS_CLASSIFICATION_DIR, RANDOM_STATE, TEST_SIZE, REPEAT_PURCHASE_WINDOW_DAYS
)
from src.classification import (
    validate_classification_dataset, split_classification_data,
    build_preprocessor, tune_model_hyperparameters, select_best_model,
    predict_with_threshold, evaluate_on_test,
    get_feature_importance, save_classification_artifacts
)
from src.evaluation import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_model_comparison, plot_feature_importance, save_classification_report_csv
)

TARGET_COL = 'repeat_purchase_90d'
print(f"Classification module loaded successfully. TARGET_COL = '{TARGET_COL}'")"""))

# Cell 2: Data Understanding & Leakage Prevention
cells.append(create_cell("markdown", """## 1. Tải và Kiểm tra Dữ liệu (Leakage Check)
Dữ liệu đầu vào là `repeat_purchase_features.csv` được trích xuất từ 90 ngày quan sát lịch sử đầu tiên.  
Các trường thông tin thuộc khoảng thời gian tương lai (`future_*`) đã được loại bỏ hoàn toàn khỏi tập đặc trưng dự báo để ngăn chặn rò rỉ thông tin trước khi chia tách."""))

# Cell 3: Load Data Code
cells.append(create_cell("code", """df = pd.read_csv(PROCESSED_DIR / 'repeat_purchase_features.csv')
print(f"Tổng số khách hàng: {len(df):,} | Số cột: {df.shape[1]}")

checks = validate_classification_dataset(df, TARGET_COL)
for c in checks:
    print(c)

class_counts = df[TARGET_COL].value_counts().sort_index()
print(f"\\nPhân bố nhãn mục tiêu:")
print(f"  Nhãn 0 (Không mua lại): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  Nhãn 1 (Mua lại 90d)  : {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")"""))

# Cell 4: Train/Test Split
cells.append(create_cell("markdown", """## 2. Phân tách Dữ liệu (Train/Test Split) & Preprocessing Pipeline
- Phân chia `80% Train / 20% Test` có phân tầng theo biến mục tiêu (`stratify=y`) để duy trì tỷ lệ lớp thiểu số/đa số.
- Thiết lập `ColumnTransformer`:
  + **Numeric features**: `SimpleImputer(strategy='median')` kết hợp `StandardScaler()`.
  + **Categorical features**: `SimpleImputer(strategy='most_frequent')` kết hợp `OneHotEncoder(handle_unknown='ignore')`.
- Pipeline tiền xử lý được đóng gói cùng estimator để chống rò rỉ dữ liệu khi huấn luyện lẫn suy luận thực tế."""))

# Cell 5: Split Code
cells.append(create_cell("code", """X_train, X_test, y_train, y_test = split_classification_data(df, TARGET_COL)
print(f"Kích thước tập Train: {len(X_train):,} dòng | Test: {len(X_test):,} dòng")

preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
print(f"Đặc trưng số ({len(numeric_features)}): {numeric_features}")
print(f"Đặc trưng phân loại ({len(categorical_features)}): {categorical_features}")"""))

# Cell 6: Hyperparameter Tuning Theory & Rationale
cells.append(create_cell("markdown", r"""## 3. Tinh chỉnh Siêu tham số & Đánh giá Cross-Validation (Chỉ trên Train Set)

### Căn cứ phương pháp & Không gian tìm kiếm (Search Spaces):
1. **Phân biệt Fixed vs Tuned Parameters:**
   - **Tham số cố định (Fixed):** `random_state=42` (đảm bảo tính tái lập), `max_iter=1000` cho Logistic Regression (đảm bảo hội tụ).
   - **Tham số tinh chỉnh (Tuned):**
     - *LogisticRegression:* Độ mạnh điều chuẩn `C: [0.1, 1.0, 10.0]` và trọng số lớp `class_weight: ['balanced', None]` (6 cấu hình).
     - *DecisionTree:* Độ sâu tối đa `max_depth: [3, 5, 10]`, số mẫu tối thiểu ở nút lá `min_samples_leaf: [1, 5, 20]`, `class_weight: ['balanced', None]` (18 cấu hình).
     - *RandomForest:* Số cây `n_estimators: [100, 200]`, độ sâu `max_depth: [5, 10]`, `min_samples_leaf: [1, 5]`, `class_weight: ['balanced', None]` (16 cấu hình).
   - Tổng cộng **40 cấu hình ứng viên** được kiểm thử qua **5-Fold Stratified CV** (200 lượt fit trên tập train).

2. **Tính đồng nhất của Ngưỡng quyết định (Threshold Parity):**
   - Thay vì sử dụng hàm `predict()` mặc định của scikit-learn (có thể dẫn đến sai lệch biên quyết định giữa các fold), quy trình sử dụng bộ đo chuyên biệt `threshold_f1_scorer_05` kết hợp `predict_with_threshold(..., threshold=0.5, pos_label=1)`. Quy tắc phân loại nghiêm ngặt: $P(y=1) \ge 0.5 \Rightarrow \hat{y}=1$, bảo toàn tính nhất quán tuyệt đối giữa Cross-Validation, kiểm thử và suy luận ứng dụng.

3. **Nguyên tắc chọn mô hình & Minh bạch học thuật (Academic Transparency):**
   - **Vai trò của DummyClassifier:** Mô hình dự báo lớp đa số (most_frequent) đạt điểm F1 thô biểu kiến cao (~0.72) do tập dữ liệu có ~57% nhãn positive. Tuy nhiên, `DummyClassifier` hoàn toàn vô giá trị trong thực tiễn vì ROC-AUC đúng bằng 0.5000 (không có khả năng phân biệt khách hàng). Vì vậy, DummyClassifier chỉ đóng vai trò mốc chuẩn kiểm soát tối thiểu (sanity baseline).
   - **Quy tắc chọn:** Mô hình được chọn là mô hình học máy có điểm `cv_f1_mean` cao nhất trong số các mô hình có khả năng học (excluding Dummy baseline).
   - **Minh bạch:** Toàn bộ quá trình chọn cấu hình và mô hình tối ưu được chốt độc lập trên tập train qua CV. Tập kiểm tra (test set) tuyệt đối không được tham gia vào việc tối ưu hay lựa chọn cấu hình."""))

# Cell 7: Tuning Code
cells.append(create_cell("code", """trained_pipelines, cv_results, tuning_history, best_params = tune_model_hyperparameters(
    preprocessor, X_train, y_train, cv_folds=5, random_state=RANDOM_STATE
)

print("Kết quả tinh chỉnh tốt nhất theo từng thuật toán (5-Fold CV trên tập Train):")
display(cv_results[['model', 'cv_f1_mean', 'cv_f1_std', 'cv_roc_auc_mean', 'cv_average_precision_mean']])

selected_model = select_best_model(cv_results)
print(f"\\n>>> Mô hình chiến thắng được chọn: {selected_model}")
print(f">>> Siêu tham số tối ưu: {best_params.get(selected_model, {})}")"""))

# Cell 8: Test Set Evaluation Markdown
cells.append(create_cell("markdown", """## 4. Huấn luyện Mô hình Cuối & Đánh giá Thực nghiệm trên Test Set

Các mô hình với cấu hình tối ưu đã được huấn luyện trên toàn bộ tập train (`refit=True`).  
Dưới đây là kết quả kiểm chứng **duy nhất một lần trên tập Test (20% dữ liệu)** để đối chiếu khả năng tổng quát hóa (Generalization)."""))

# Cell 9: Evaluate Test Code
cells.append(create_cell("code", """comparison_df = evaluate_on_test(
    trained_pipelines, X_test, y_test, cv_results, selected_model, threshold=0.5, pos_label=1
)

display(comparison_df[['model', 'cv_f1_mean', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall', 'test_accuracy', 'selected']])"""))

# Cell 10: Inference & Predictions
cells.append(create_cell("markdown", """## 5. Dự báo trên Tập Test bằng Ngưỡng Nghiêm ngặt (Strict Threshold Inference)
Sử dụng hàm suy luận `predict_with_threshold(pipeline, X, threshold=0.5, pos_label=1)` để xuất nhãn phân loại và xác suất dự báo."""))

# Cell 11: Predictions Code
cells.append(create_cell("code", """best_pipeline = trained_pipelines[selected_model]
y_pred_best, y_proba_best = predict_with_threshold(best_pipeline, X_test, threshold=0.5, pos_label=1)

pred_df = X_test.copy()
if 'CustomerID' in df.columns:
    pred_df.insert(0, 'CustomerID', df.loc[X_test.index, 'CustomerID'].values)
pred_df[TARGET_COL + '_actual'] = y_test.values
pred_df[TARGET_COL + '_predicted'] = y_pred_best
pred_df[TARGET_COL + '_probability'] = y_proba_best

display(pred_df.head())"""))

# Cell 12: Visualizations - Confusion Matrix
cells.append(create_cell("code", """plot_confusion_matrix(y_test, y_pred_best, selected_model)"""))

# Cell 13: Visualizations - ROC Curves
cells.append(create_cell("code", """plot_roc_curves(trained_pipelines, X_test, y_test)"""))

# Cell 14: Visualizations - PR Curves
cells.append(create_cell("code", """plot_precision_recall_curves(trained_pipelines, X_test, y_test)"""))

# Cell 15: Visualizations - Model Comparison
cells.append(create_cell("code", """plot_model_comparison(comparison_df)"""))

# Cell 16: Feature Importance
cells.append(create_cell("markdown", """## 6. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance)
Trích xuất độ quan trọng của đặc trưng (Gini Impurity Decrease) từ mô hình `RandomForestClassifier` tốt nhất."""))

# Cell 17: Feature Importance Code
cells.append(create_cell("code", """fi_df = get_feature_importance(best_pipeline, numeric_features + categorical_features)
display(fi_df.head(15))

plot_feature_importance(fi_df, selected_model)"""))

# Cell 18: Save Artifacts Markdown
cells.append(create_cell("markdown", """## 7. Lưu trữ Mô hình và Metadata Đầy đủ (Persistence & Audit Trail)"""))

# Cell 19: Save Artifacts Code
cells.append(create_cell("code", """try:
    commit_res = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    git_commit = commit_res.stdout.strip()
except Exception:
    git_commit = 'unknown'

metadata = {
    'target_col': TARGET_COL,
    'feature_columns': list(X_train.columns),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'cv_folds': 5,
    'cv_scoring': 'f1 (threshold >= 0.5 via predict_with_threshold)',
    'search_spaces': {
        'LogisticRegression': {'classifier__C': [0.1, 1.0, 10.0], 'classifier__class_weight': ['balanced', None]},
        'DecisionTree': {'classifier__max_depth': [3, 5, 10], 'classifier__min_samples_leaf': [1, 5, 20], 'classifier__class_weight': ['balanced', None]},
        'RandomForest': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [5, 10], 'classifier__min_samples_leaf': [1, 5], 'classifier__class_weight': ['balanced', None]},
    },
    'selected_model': selected_model,
    'selection_metric': 'cv_f1_mean',
    'selection_rationale': 'Selected based on 5-fold CV F1 score among learned models (excluding Dummy baseline). Note that DummyClassifier achieves higher F1 due to class imbalance but has zero discriminative power (ROC-AUC=0.5000).',
    'threshold': 0.5,
    'threshold_convention': 'Standard default convention (0.5); probability >= threshold strictly mapped to positive class (1)',
    'positive_class': 1,
    'decision_rule': 'probability >= threshold',
    'best_hyperparameters': best_params,
    'actual_model_parameters': best_pipeline.named_steps['classifier'].get_params(),
    'training_row_count': len(X_train),
    'test_row_count': len(X_test),
    'class_distribution': {
        'train_0': int((y_train == 0).sum()),
        'train_1': int((y_train == 1).sum()),
        'test_0': int((y_test == 0).sum()),
        'test_1': int((y_test == 1).sum()),
    },
    'window_days': REPEAT_PURCHASE_WINDOW_DAYS,
    'run_timestamp': datetime.now().isoformat(),
    'git_commit': git_commit,
}

# Lưu toàn bộ artifact bảng, mô hình và metadata
save_classification_artifacts(
    trained_pipelines, selected_model, metadata,
    MODELS_CLASSIFICATION_DIR, TABLES_CLASSIFICATION, fi_df,
    tuning_history_df=tuning_history
)

# Lưu bảng so sánh, báo cáo phân loại, dự đoán test và tóm tắt dữ liệu
comparison_df.to_csv(TABLES_CLASSIFICATION / 'model_comparison.csv', index=False)
cv_results.to_csv(TABLES_CLASSIFICATION / 'cv_results.csv', index=False)
tuning_history.to_csv(TABLES_CLASSIFICATION / 'cv_tuning_history.csv', index=False)
pred_df.to_csv(TABLES_CLASSIFICATION / 'test_predictions.csv', index=False)

summary_df = pd.DataFrame([{
    'total_customers': len(df),
    'n_features': df.shape[1] - 2,
    'class_0': int(class_counts.get(0, 0)),
    'class_1': int(class_counts.get(1, 0)),
    'class_ratio': round(class_counts.get(1, 0) / len(df), 4),
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
}])
summary_df.to_csv(TABLES_CLASSIFICATION / 'classification_data_summary.csv', index=False)

save_classification_report_csv(
    y_test, y_pred_best,
    TABLES_CLASSIFICATION / 'classification_report.csv'
)

print("[PASS] Đã đồng bộ và lưu đầy đủ toàn bộ artifact mô hình, bảng kết quả và metadata.")"""))

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/05_repeat_purchase_classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Created synchronized notebooks/05_repeat_purchase_classification.ipynb successfully!")
