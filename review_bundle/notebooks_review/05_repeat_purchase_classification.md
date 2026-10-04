# Review Notebook: 05_repeat_purchase_classification.ipynb
*Source Path: `D:/Project/Data-mininng/notebooks/05_repeat_purchase_classification.ipynb`*

---

# 05. Repeat Purchase Classification

**Mục tiêu bài toán:**
- Dự đoán khả năng khách hàng quay lại mua hàng trong vòng 90 ngày (`repeat_purchase_90d`).
- So sánh các thuật toán: `DummyClassifier` (baseline), `LogisticRegression`, `DecisionTree`, và `RandomForest`.
- **Tối ưu siêu tham số & Chọn mô hình hoàn toàn dựa trên 5-Fold Stratified Cross-Validation trên Training Set.**
- Nghiêm cấm rò rỉ dữ liệu (Data Leakage): Test set chỉ được dùng cho đánh giá duy nhất một lần cuối cùng.
- Đảm bảo tính nhất quán tuyệt đối giữa Cross-Validation và Inference bằng cơ chế phân loại xác suất với ngưỡng cố định (`probability >= 0.5`).

```python
# [Cell 1 - Execution Count: 1]
import sys, os
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
print(f"Classification module loaded successfully. TARGET_COL = '{TARGET_COL}'")
```

**Output (stdout):**
```text
Classification module loaded successfully. TARGET_COL = 'repeat_purchase_90d'
```

## 1. Tải và Kiểm tra Dữ liệu (Leakage Check)
Dữ liệu đầu vào là `repeat_purchase_features.csv` được trích xuất từ 90 ngày quan sát lịch sử đầu tiên.  
Các trường thông tin thuộc khoảng thời gian tương lai (`future_*`) đã được loại bỏ hoàn toàn khỏi tập đặc trưng dự báo để ngăn chặn rò rỉ thông tin trước khi chia tách.

```python
# [Cell 3 - Execution Count: 2]
df = pd.read_csv(PROCESSED_DIR / 'repeat_purchase_features.csv')
print(f"Tổng số khách hàng: {len(df):,} | Số cột: {df.shape[1]}")

checks = validate_classification_dataset(df, TARGET_COL)
for c in checks:
    print(c)

class_counts = df[TARGET_COL].value_counts().sort_index()
print(f"\nPhân bố nhãn mục tiêu:")
print(f"  Nhãn 0 (Không mua lại): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  Nhãn 1 (Mua lại 90d)  : {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")
```

**Output (stdout):**
```text
Tổng số khách hàng: 3,368 | Số cột: 13
[PASS] Target 'repeat_purchase_90d' exists
[PASS] Target only contains 0/1
[PASS] No missing target values
[PASS] No duplicate CustomerID (3368 unique)
[PASS] No future columns (leakage check)
[PASS] Both classes present: {1: np.int64(1919), 0: np.int64(1449)}

Phân bố nhãn mục tiêu:
  Nhãn 0 (Không mua lại): 1,449 (43.0%)
  Nhãn 1 (Mua lại 90d)  : 1,919 (57.0%)
```

## 2. Phân tách Dữ liệu (Train/Test Split) & Preprocessing Pipeline
- Phân chia `80% Train / 20% Test` có phân tầng theo biến mục tiêu (`stratify=y`) để duy trì tỷ lệ lớp thiểu số/đa số.
- Thiết lập `ColumnTransformer`:
  + **Numeric features**: `SimpleImputer(strategy='median')` kết hợp `StandardScaler()`.
  + **Categorical features**: `SimpleImputer(strategy='most_frequent')` kết hợp `OneHotEncoder(handle_unknown='ignore')`.
- Pipeline tiền xử lý được đóng gói cùng estimator để chống rò rỉ dữ liệu khi huấn luyện lẫn suy luận thực tế.

```python
# [Cell 5 - Execution Count: 3]
X_train, X_test, y_train, y_test = split_classification_data(df, TARGET_COL)
print(f"Kích thước tập Train: {len(X_train):,} dòng | Test: {len(X_test):,} dòng")

preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
print(f"Đặc trưng số ({len(numeric_features)}): {numeric_features}")
print(f"Đặc trưng phân loại ({len(categorical_features)}): {categorical_features}")
```

**Output (stdout):**
```text
Kích thước tập Train: 2,694 dòng | Test: 674 dòng
```

**Output (stdout):**
```text
Đặc trưng số (9): ['Recency', 'Frequency', 'Monetary', 'TotalItems', 'UniqueProducts', 'ActiveDays', 'AverageOrderValue', 'AverageItemsPerInvoice', 'CustomerLifetimeDays']
Đặc trưng phân loại (1): ['Country']
```

## 3. Tinh chỉnh Siêu tham số & Đánh giá Cross-Validation (Chỉ trên Train Set)

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
   - **Minh bạch:** Toàn bộ quá trình chọn cấu hình và mô hình tối ưu được chốt độc lập trên tập train qua CV. Tập kiểm tra (test set) tuyệt đối không được tham gia vào việc tối ưu hay lựa chọn cấu hình.

```python
# [Cell 7 - Execution Count: 4]
trained_pipelines, cv_results, tuning_history, best_params = tune_model_hyperparameters(
    preprocessor, X_train, y_train, cv_folds=5, random_state=RANDOM_STATE
)

print("Kết quả tinh chỉnh tốt nhất theo từng thuật toán (5-Fold CV trên tập Train):")
display(cv_results[['model', 'cv_f1_mean', 'cv_f1_std', 'cv_roc_auc_mean', 'cv_average_precision_mean']])

selected_model = select_best_model(cv_results)
print(f"\n>>> Mô hình chiến thắng được chọn: {selected_model}")
print(f">>> Siêu tham số tối ưu: {best_params.get(selected_model, {})}")
```

**Output (stdout):**
```text
Kết quả tinh chỉnh tốt nhất theo từng thuật toán (5-Fold CV trên tập Train):
```

**Result Display:**
```text
model  cv_f1_mean  cv_f1_std  cv_roc_auc_mean  \
0     DummyClassifier    0.725940   0.000344         0.500000   
1  LogisticRegression    0.706858   0.020601         0.732363   
2        DecisionTree    0.697778   0.022159         0.718116   
3        RandomForest    0.710218   0.017473         0.737225   

   cv_average_precision_mean  
0                   0.569785  
1                   0.801045  
2                   0.762044  
3                   0.809899
```

**Output (stdout):**
```text
>>> Mô hình chiến thắng được chọn: RandomForest
>>> Siêu tham số tối ưu: {'classifier__class_weight': None, 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 100}
```

## 4. Huấn luyện Mô hình Cuối & Đánh giá Thực nghiệm trên Test Set

Các mô hình với cấu hình tối ưu đã được huấn luyện trên toàn bộ tập train (`refit=True`).  
Dưới đây là kết quả kiểm chứng **duy nhất một lần trên tập Test (20% dữ liệu)** để đối chiếu khả năng tổng quát hóa (Generalization).

```python
# [Cell 9 - Execution Count: 5]
comparison_df = evaluate_on_test(
    trained_pipelines, X_test, y_test, cv_results, selected_model, threshold=0.5, pos_label=1
)

display(comparison_df[['model', 'cv_f1_mean', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall', 'test_accuracy', 'selected']])
```

**Result Display:**
```text
model  cv_f1_mean  test_f1  test_roc_auc  test_precision  \
0     DummyClassifier      0.7259   0.7259        0.5000          0.5697   
1  LogisticRegression      0.7069   0.7471        0.7761          0.7520   
2        DecisionTree      0.6978   0.6898        0.7424          0.7735   
3        RandomForest      0.7102   0.7375        0.7756          0.7434   

   test_recall  test_accuracy  selected  
0       1.0000         0.5697     False  
1       0.7422         0.7136     False  
2       0.6224         0.6810     False  
3       0.7318         0.7033      True
```

## 5. Dự báo trên Tập Test bằng Ngưỡng Nghiêm ngặt (Strict Threshold Inference)
Sử dụng hàm suy luận `predict_with_threshold(pipeline, X, threshold=0.5, pos_label=1)` để xuất nhãn phân loại và xác suất dự báo.

```python
# [Cell 11 - Execution Count: 6]
best_pipeline = trained_pipelines[selected_model]
y_pred_best, y_proba_best = predict_with_threshold(best_pipeline, X_test, threshold=0.5, pos_label=1)

pred_df = X_test.copy()
if 'CustomerID' in df.columns:
    pred_df.insert(0, 'CustomerID', df.loc[X_test.index, 'CustomerID'].values)
pred_df[TARGET_COL + '_actual'] = y_test.values
pred_df[TARGET_COL + '_predicted'] = y_pred_best
pred_df[TARGET_COL + '_probability'] = y_proba_best

display(pred_df.head())
```

**Result Display:**
```text
CustomerID  Recency  Frequency  Monetary  TotalItems  UniqueProducts  \
2719       17117      198          1    116.20          68               9   
512        13198       31          5   3009.18        2369             182   
2593       16889      105          3   1578.67         925             112   
1214       14436        9          1     89.06         129               9   
2783       17239      223          1    382.50         150              15   

      ActiveDays         Country  AverageOrderValue  AverageItemsPerInvoice  \
2719           1  United Kingdom         116.200000               68.000000   
512            5  United Kingdom         601.836000              473.800000   
2593           3  United Kingdom         526.223333              308.333333   
1214           1  United Kingdom          89.060000              129.000000   
2783           1  United Kingdom         382.500000              150.000000   

      CustomerLifetimeDays  repeat_purchase_90d_actual  \
2719                     0                           0   
512                    241                           1   
2593                   142                           0   
1214                     0                           0   
2783                     0                           0   

      repeat_purchase_90d_predicted  repeat_purchase_90d_probability  
2719                              0                         0.362649  
512                               1                         0.867196  
2593                              1                         0.759515  
1214                              0                         0.383963  
2783                              0                         0.387927
```

```python
# [Cell 12 - Execution Count: 7]
plot_confusion_matrix(y_test, y_pred_best, selected_model)
```

**Result Display:**
```text
<Figure size 700x600 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 13 - Execution Count: 8]
plot_roc_curves(trained_pipelines, X_test, y_test)
```

**Result Display:**
```text
<Figure size 1000x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 14 - Execution Count: 9]
plot_precision_recall_curves(trained_pipelines, X_test, y_test)
```

**Result Display:**
```text
<Figure size 1000x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 15 - Execution Count: 10]
plot_model_comparison(comparison_df)
```

**Result Display:**
```text
<Figure size 1400x700 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 6. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance)
Trích xuất độ quan trọng của đặc trưng (Gini Impurity Decrease) từ mô hình `RandomForestClassifier` tốt nhất.

```python
# [Cell 17 - Execution Count: 11]
fi_df = get_feature_importance(best_pipeline, numeric_features + categorical_features)
display(fi_df.head(15))

plot_feature_importance(fi_df, selected_model)
```

**Result Display:**
```text
feature  importance
0                Frequency    0.189034
1               ActiveDays    0.177873
2     CustomerLifetimeDays    0.129228
3           UniqueProducts    0.122418
4                 Monetary    0.110082
5               TotalItems    0.095898
6                  Recency    0.078317
7        AverageOrderValue    0.053512
8   AverageItemsPerInvoice    0.037088
9   Country_United Kingdom    0.002602
10         Country_Germany    0.001803
11          Country_France    0.000975
12     Country_Switzerland    0.000368
13         Country_Belgium    0.000343
14           Country_Spain    0.000193
```

**Result Display:**
```text
<Figure size 1200x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 7. Lưu trữ Mô hình và Metadata Đầy đủ (Persistence & Audit Trail)

```python
# [Cell 19 - Execution Count: 12]
try:
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

print("[PASS] Đã đồng bộ và lưu đầy đủ toàn bộ artifact mô hình, bảng kết quả và metadata.")
```

**Output (stdout):**
```text
[PASS] Đã đồng bộ và lưu đầy đủ toàn bộ artifact mô hình, bảng kết quả và metadata.
```

