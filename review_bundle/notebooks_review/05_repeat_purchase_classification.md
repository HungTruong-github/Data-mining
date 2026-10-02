# Review Notebook: 05_repeat_purchase_classification.ipynb
*Source Path: `d:/Project/Data-mininng/notebooks/05_repeat_purchase_classification.ipynb`*

---

# 05. Repeat Purchase Classification

**Mục tiêu:**
- Dự đoán khách hàng có mua lại trong 90 ngày (`repeat_purchase_90d`).
- So sánh ít nhất 4 thuật toán: DummyClassifier (baseline), Logistic Regression, Decision Tree, Random Forest.
- **Chọn model bằng Cross-Validation trên tập huấn luyện** (KHÔNG dùng test set để chọn model).
- Test set chỉ được dùng cho đánh giá cuối cùng duy nhất một lần.
- Pipeline preprocessing + model có thể tái sử dụng cho Dashboard Streamlit.

```python
# [Cell 1 - Execution Count: 1]
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

%matplotlib inline
```

## 1. Tải và Kiểm tra Dữ liệu
Dữ liệu đầu vào là `repeat_purchase_features.csv` đã được tạo ở bước Feature Engineering (03).  
Kiểm tra chống data leakage: không có `future_invoice_count`, `future_revenue`, không duplicate.

```python
# [Cell 3 - Execution Count: 2]
df = pd.read_csv(PROCESSED_DIR / 'repeat_purchase_features.csv')
print(f"Customers: {len(df):,} | Columns: {df.shape[1]}")

checks = validate_classification_dataset(df, TARGET_COL)
for c in checks:
    print(c)

# Class distribution
class_counts = df[TARGET_COL].value_counts().sort_index()
print(f"\nClass Balance:")
print(f"  0 (No Repeat): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  1 (Repeat):    {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")
```

**Output (stdout):**
```text
Customers: 3,368 | Columns: 13
[PASS] Target 'repeat_purchase_90d' exists
[PASS] Target only contains 0/1
[PASS] No missing target values
[PASS] No duplicate CustomerID (3368 unique)
[PASS] No future columns (leakage check)
[PASS] Both classes present: {1: np.int64(1919), 0: np.int64(1449)}

Class Balance:
  0 (No Repeat): 1,449 (43.0%)
  1 (Repeat):    1,919 (57.0%)
```

## 2. Train/Test Split & Preprocessing Pipeline
- Chia dữ liệu theo `stratify=y` để giữ tỷ lệ class.
- `ColumnTransformer` xử lý numeric (Impute + Scale) và categorical (Impute + OneHotEncode).
- **Pipeline preprocessing được gắn liền với model** để tái sử dụng cho Dashboard.

```python
# [Cell 5 - Execution Count: 3]
X_train, X_test, y_train, y_test = split_classification_data(df, TARGET_COL)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

model_pipelines = build_model_pipelines(preprocessor)
print(f"\nModels to compare: {list(model_pipelines.keys())}")
```

**Output (stdout):**
```text
Train: 2,694 | Test: 674
Numeric features: ['Recency', 'Frequency', 'Monetary', 'TotalItems', 'UniqueProducts', 'ActiveDays', 'AverageOrderValue', 'AverageItemsPerInvoice', 'CustomerLifetimeDays']
Categorical features: ['Country']

Models to compare: ['DummyClassifier', 'LogisticRegression', 'DecisionTree', 'RandomForest']
```

## 3. Cross-Validation (Chọn Model)
> **QUAN TRỌNG**: Model được chọn hoàn toàn dựa trên kết quả Cross-Validation trên tập huấn luyện.  
> Test set CHƯA được sử dụng ở bước này.

```python
# [Cell 7 - Execution Count: 4]
cv_results = cross_validate_models(model_pipelines, X_train, y_train)
display(cv_results.sort_values('cv_f1_mean', ascending=False))

selected_model = select_best_model(cv_results)
print(f"\n>>> Model được chọn (bằng CV): {selected_model}")
print(f">>> Test set CHƯA được dùng để chọn model.")
```

**Result Display:**
```text
model  cv_f1_mean  cv_f1_std  cv_roc_auc_mean  cv_roc_auc_std  \
0     DummyClassifier    0.725940   0.000344         0.500000        0.000000   
3        RandomForest    0.672584   0.019140         0.699079        0.013734   
1  LogisticRegression    0.664984   0.014968         0.732265        0.018326   
2        DecisionTree    0.650245   0.016162         0.637491        0.019611   

   cv_average_precision_mean  cv_average_precision_std  cv_runtime_seconds  
0                   0.569785                  0.000423                3.28  
3                   0.784951                  0.013322                2.52  
1                   0.801027                  0.013208                5.80  
2                   0.681687                  0.021032                4.26
```

**Output (stdout):**
```text
>>> Model được chọn (bằng CV): RandomForest
>>> Test set CHƯA được dùng để chọn model.
```

## 4. Huấn luyện & Đánh giá cuối cùng trên Test Set
Sau khi chọn model bằng CV, huấn luyện tất cả model trên toàn bộ tập training.  
Đánh giá trên test set **chỉ một lần duy nhất** (không tuning thêm).

```python
# [Cell 9 - Execution Count: 5]
trained_pipelines = train_final_models(model_pipelines, X_train, y_train)
comparison_df = evaluate_on_test(trained_pipelines, X_test, y_test, cv_results, selected_model)
display(comparison_df[['model', 'cv_f1_mean', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall', 'selected']])
```

**Result Display:**
```text
model  cv_f1_mean  test_f1  test_roc_auc  test_precision  \
0     DummyClassifier      0.7259   0.7259        0.5000          0.5697   
1  LogisticRegression      0.6650   0.7043        0.7763          0.7941   
2        DecisionTree      0.6502   0.6848        0.6828          0.7159   
3        RandomForest      0.6726   0.7086        0.7438          0.7280   

   test_recall  selected  
0       1.0000     False  
1       0.6328     False  
2       0.6562     False  
3       0.6901      True
```

## 5. Trực quan hóa

```python
# [Cell 11 - Execution Count: 6]
best_pipeline = trained_pipelines[selected_model]
y_pred_best = best_pipeline.predict(X_test)

plot_confusion_matrix(y_test, y_pred_best, selected_model)
```

**Result Display:**
```text
<Figure size 700x600 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 12 - Execution Count: 7]
plot_roc_curves(trained_pipelines, X_test, y_test)
```

**Result Display:**
```text
<Figure size 1000x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 13 - Execution Count: 8]
plot_precision_recall_curves(trained_pipelines, X_test, y_test)
```

**Result Display:**
```text
<Figure size 1000x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 14 - Execution Count: 9]
plot_model_comparison(comparison_df)
```

**Result Display:**
```text
<Figure size 1400x700 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 6. Feature Importance

```python
# [Cell 16 - Execution Count: 10]
fi_df = get_feature_importance(best_pipeline, numeric_features + categorical_features)
display(fi_df.head(15))
plot_feature_importance(fi_df, selected_model)
```

**Result Display:**
```text
feature  importance
0                  Recency    0.139253
1                 Monetary    0.136043
2           UniqueProducts    0.129772
3               TotalItems    0.127866
4        AverageOrderValue    0.123478
5   AverageItemsPerInvoice    0.117268
6     CustomerLifetimeDays    0.081577
7                Frequency    0.057643
8               ActiveDays    0.056645
9   Country_United Kingdom    0.007406
10         Country_Germany    0.004079
11          Country_France    0.003214
12           Country_Spain    0.001761
13         Country_Belgium    0.001720
14     Country_Switzerland    0.001583
```

**Result Display:**
```text
<Figure size 1200x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 7. Lưu kết quả

```python
# [Cell 18 - Execution Count: 11]
# Save tables
comparison_df.to_csv(TABLES_CLASSIFICATION / 'model_comparison.csv', index=False)
cv_results.to_csv(TABLES_CLASSIFICATION / 'cv_results.csv', index=False)
fi_df.to_csv(TABLES_CLASSIFICATION / 'feature_importance.csv', index=False)

save_classification_report_csv(y_test, y_pred_best, TABLES_CLASSIFICATION / 'classification_report.csv')

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

summary = {
    'total_customers': len(df), 'class_0': int((df[TARGET_COL]==0).sum()),
    'class_1': int((df[TARGET_COL]==1).sum()), 'random_state': RANDOM_STATE, 'test_size': TEST_SIZE,
}
pd.DataFrame([summary]).to_csv(TABLES_CLASSIFICATION / 'classification_data_summary.csv', index=False)

# Save models
import json
metadata = {
    'target_col': TARGET_COL, 'feature_columns': list(X_train.columns),
    'numeric_features': numeric_features, 'categorical_features': categorical_features,
    'random_state': RANDOM_STATE, 'test_size': TEST_SIZE, 'selected_model': selected_model,
    'selection_metric': 'cv_f1_mean', 'threshold': 0.5,
    'training_row_count': len(X_train), 'test_row_count': len(X_test),
    'window_days': REPEAT_PURCHASE_WINDOW_DAYS, 'class_weight': 'balanced',
}
save_classification_artifacts(trained_pipelines, selected_model, metadata,
                              MODELS_CLASSIFICATION_DIR, TABLES_CLASSIFICATION, fi_df)
print("All outputs saved successfully.")
```

**Output (stdout):**
```text
All outputs saved successfully.
```

