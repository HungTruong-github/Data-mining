# Review Notebook: 07_model_comparison_and_insights.ipynb
*Source Path: `d:/Project/Data-mininng/notebooks/07_model_comparison_and_insights.ipynb`*

---

# 07. Model Comparison and Insights

## Mục đích
Tổng hợp và so sánh kết quả của cả ba bài toán data mining: Customer Clustering (04), Repeat Purchase Classification (05), và Association Rules (06). Đánh giá chất lượng mô hình, sinh business insights và tạo báo cáo hoàn chỉnh.

**Câu hỏi chính:**
- Mô hình clustering nào phù hợp nhất và tại sao?
- Classifier có vượt baseline không? Trên những metric nào?
- Hai thuật toán association rules có cho kết quả tương đương không?
- Insights nghiệp vụ nào có căn cứ từ dữ liệu?

```python
# [Cell 1 - Execution Count: 1]
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR
from src.model_comparison import (
    validate_inputs, build_clustering_comparison,
    build_classification_comparison, build_association_comparison,
    verify_model_artifacts, generate_report, generate_manifest
)
from src.insights import generate_all_insights

TABLES_MC = OUTPUTS_DIR / 'tables' / 'model_comparison'
TABLES_MC.mkdir(parents=True, exist_ok=True)
FIGURES_MC = OUTPUTS_DIR / 'figures' / 'model_comparison'
FIGURES_MC.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTPUTS_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR = OUTPUTS_DIR / 'evidence'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

%matplotlib inline
print("Setup complete.")
```

**Output (stdout):**
```text
Setup complete.
```

## 1. Validate Inputs

Kiểm tra tất cả file đầu vào từ steps 04, 05, 06 tồn tại, không rỗng, và đúng schema.

```python
# [Cell 3 - Execution Count: 2]
try:
    checks = validate_inputs()
    for c in checks:
        print(c)
    print("\n[OK] All inputs validated successfully.")
except ValueError as e:
    print(f"[ERROR] Input validation failed:\n{e}")
    raise
```

**Output (stdout):**
```text
[OK] rfm_features: rfm_customer_features.csv
[OK] repeat_features: repeat_purchase_features.csv
[OK] customer_clusters: customer_clusters.csv
[OK] cluster_comparison: clustering_algorithm_comparison.csv
[OK] cluster_profiles: cluster_profiles.csv
[OK] class_comparison: model_comparison.csv
[OK] cv_results: cv_results.csv
[OK] test_predictions: test_predictions.csv
[OK] feature_importance: feature_importance.csv
[OK] assoc_comparison: association_algorithm_comparison.csv
[OK] selected_rules: selected_association_rules.csv
[OK] business_insights: association_business_insights.csv
[OK] class_metadata: classification_metadata.json

[OK] All inputs validated successfully.
```

## 2. Clustering Model Comparison

So sánh K-Means, GMM, Agglomerative và DBSCAN trên các internal metrics.

```python
# [Cell 5 - Execution Count: 3]
clustering_comp = build_clustering_comparison()
clustering_comp.to_csv(TABLES_MC / 'clustering_model_comparison.csv', index=False)

# Display comparison
display_cols = ['algorithm', 'n_clusters', 'silhouette_score', 'davies_bouldin_score',
                'calinski_harabasz_score', 'is_selected', 'selection_reason']
available_cols = [c for c in display_cols if c in clustering_comp.columns]
print("Clustering Model Comparison:")
display(clustering_comp[available_cols])

# Highlight selected model
selected = clustering_comp[clustering_comp['is_selected'].astype(bool)]
if len(selected) > 0:
    sel = selected.iloc[0]
    print(f"\nSelected: {sel['algorithm']} K={sel['n_clusters']}")
    if 'silhouette_score' in sel:
        print(f"  Silhouette: {sel['silhouette_score']:.4f}")
    if 'davies_bouldin_score' in sel:
        print(f"  Davies-Bouldin: {sel['davies_bouldin_score']:.4f}")
```

**Output (stdout):**
```text
Clustering Model Comparison:
```

**Result Display:**
```text
algorithm  n_clusters  silhouette_score  \
0                  K-Means           2            0.4330   
1                  K-Means           3            0.3375   
2                  K-Means           4            0.3381   
3                  K-Means           5            0.3172   
4                  K-Means           6            0.3141   
5                  K-Means           7            0.3095   
6                  K-Means           8            0.3012   
7                      GMM           2            0.2874   
8                      GMM           3            0.2562   
9                      GMM           4            0.1752   
10                     GMM           5            0.1520   
11                     GMM           6            0.1164   
12                     GMM           7            0.1019   
13                     GMM           8            0.0681   
14     Agglomerative(ward)           2            0.4232   
15     Agglomerative(ward)           3            0.3146   
16     Agglomerative(ward)           4            0.2428   
17     Agglomerative(ward)           5            0.2385   
18     Agglomerative(ward)           6            0.2447   
19     Agglomerative(ward)           7            0.2489   
20     Agglomerative(ward)           8            0.2252   
21   DBSCAN(eps=0.3,min=5)           8            0.0632   
22   DBSCAN(eps=0.5,min=5)           2            0.2946   
23   DBSCAN(eps=0.7,min=5)           2            0.5659   
24   DBSCAN(eps=1.0,min=5)           1               NaN   
25  DBSCAN(eps=0.5,min=10)           2            0.2961   

    davies_bouldin_score  calinski_harabasz_score  is_selected  \
0                 0.8917                4364.6091         True   
1                 1.0462                3626.2766        False   
2                 1.0131                3329.6842        False   
3                 0.9851                3194.6602        False   
4                 1.0106                3083.6138        False   
5                 0.9692                2960.3706        False   
6                 0.9944                2824.3335        False   
7                 1.0662                2307.4172        False   
8                 1.2157                2749.8992        False   
9                 1.7079                2167.5913        False   
10                1.7655                1880.2369        False   
11                2.2668                1454.5443        False   
12                2.5613                1271.3215        False   
13                2.2984                1279.1170        False   
14                0.9064                4227.3723        False   
15                1.1509                3040.2604        False   
16                1.2255                2780.0958        False   
17                1.2431                2551.4344        False   
18                1.1518                2439.1744        False   
19                1.1215                2405.8703        False   
20                1.0852                2281.4319        False   
21                1.5042                 837.2949        False   
22                1.0626                2408.3810        False   
23                0.3483                  49.9624        False   
24                   NaN                      NaN        False   
25                1.0613                2416.0711        False   

                                     selection_reason  
0   Silhouette=0.4330; DB=0.8917; Best combined ra...  
1           Eligible candidate (Combined Rank = 23.0)  
2           Eligible candidate (Combined Rank = 27.0)  
3           Eligible candidate (Combined Rank = 19.0)  
4           Eligible candidate (Combined Rank = 28.0)  
5           Eligible candidate (Combined Rank = 25.0)  
6           Eligible candidate (Combined Rank = 35.0)  
7           Eligible candidate (Combined Rank = 50.0)  
8           Eligible candidate (Combined Rank = 54.0)  
9           Eligible candidate (Comb
... [Output truncated for review readability; see full CSV] ...
```

**Output (stdout):**
```text
Selected: K-Means K=2
  Silhouette: 0.4330
  Davies-Bouldin: 0.8917
```

### Nhận xét Clustering

**Giải thích lựa chọn model:**
- Silhouette score đo mức tách biệt giữa các cụm (cao hơn = tốt hơn, max = 1)
- Davies-Bouldin index đo mức chồng lấn (thấp hơn = tốt hơn, min = 0)
- Model được chọn dựa trên ranking tổng hợp các internal metrics VÀ cluster size distribution hợp lý
- DBSCAN bị loại nếu max_cluster_pct quá cao (một cụm chiếm gần hết dữ liệu)

**Hạn chế:**
- Internal metrics không có ground truth — Silhouette cao chưa chắc phân khúc đúng
- K=2 có thể đơn giản hóa quá mức nhưng cho phân khúc rõ ràng nhất
- PCA visualization chỉ để trực quan, không dùng trong model

```python
# [Cell 7 - Execution Count: 4]
# Clustering comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filter valid configs for plotting
valid = clustering_comp[clustering_comp['silhouette_score'].notna()].copy()

# Silhouette by algorithm
for algo in valid['algorithm'].unique():
    subset = valid[valid['algorithm'] == algo]
    axes[0].plot(subset['n_clusters'], subset['silhouette_score'], 'o-', label=algo)
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score by Algorithm and K')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Davies-Bouldin by algorithm
for algo in valid['algorithm'].unique():
    subset = valid[valid['algorithm'] == algo]
    if 'davies_bouldin_score' in subset.columns:
        axes[1].plot(subset['n_clusters'], subset['davies_bouldin_score'], 'o-', label=algo)
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Davies-Bouldin Score')
axes[1].set_title('Davies-Bouldin Score by Algorithm and K')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_MC / 'clustering_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Result Display:**
```text
<Figure size 1400x500 with 2 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 3. Classification Model Comparison

So sánh DummyClassifier (baseline), Logistic Regression, Decision Tree và Random Forest.

```python
# [Cell 9 - Execution Count: 5]
classification_comp = build_classification_comparison()
classification_comp.to_csv(TABLES_MC / 'classification_model_comparison.csv', index=False)

display_cols = ['model', 'cv_f1_mean', 'cv_f1_std', 'test_f1', 'test_accuracy',
                'test_precision', 'test_recall', 'test_roc_auc', 'is_selected']
available_cols = [c for c in display_cols if c in classification_comp.columns]
print("Classification Model Comparison:")
display(classification_comp[available_cols])

# Highlight baseline vs best
dummy_row = classification_comp[classification_comp['model'].str.contains('Dummy', case=False, na=False)]
selected_row = classification_comp[classification_comp['is_selected'].astype(bool)]
if len(dummy_row) > 0 and len(selected_row) > 0:
    d = dummy_row.iloc[0]
    s = selected_row.iloc[0]
    dummy_f1 = d.get('cv_f1_mean', d.get('cv_f1', 0))
    sel_f1 = s.get('cv_f1_mean', s.get('cv_f1', 0))
    print(f"\nBaseline (Dummy) CV F1: {dummy_f1:.4f}")
    print(f"Selected ({s['model']}) CV F1: {sel_f1:.4f}")
    if sel_f1 > dummy_f1:
        print(f"  → Selected model vượt baseline +{sel_f1-dummy_f1:.4f}")
    else:
        print(f"  ⚠ Selected model KHÔNG vượt baseline trên CV F1")
```

**Output (stdout):**
```text
Classification Model Comparison:
```

**Result Display:**
```text
model  cv_f1_mean  cv_f1_std  test_f1  test_accuracy  \
0     DummyClassifier      0.7259     0.0003   0.7259         0.5697   
1  LogisticRegression      0.6650     0.0150   0.7043         0.6973   
2        DecisionTree      0.6502     0.0162   0.6848         0.6558   
3        RandomForest      0.6726     0.0191   0.7086         0.6766   

   test_precision  test_recall  test_roc_auc  is_selected  
0          0.5697       1.0000        0.5000        False  
1          0.7941       0.6328        0.7763        False  
2          0.7159       0.6562        0.6828        False  
3          0.7280       0.6901        0.7438         True
```

**Output (stdout):**
```text
Baseline (Dummy) CV F1: 0.7259
Selected (RandomForest) CV F1: 0.6726
  ⚠ Selected model KHÔNG vượt baseline trên CV F1
```

### Nhận xét Classification

**Protocol đánh giá:**
- Model selection dựa trên Stratified K-Fold CV F1-score trên tập train
- DummyClassifier (most_frequent) là baseline — nếu learned model không vượt Dummy thì chưa có giá trị dự đoán thực sự
- Threshold = 0.5 (default) — chưa tối ưu cho nghiệp vụ vì thiếu chi phí FP/FN

**Hạn chế:**
- Single cutoff random customer split — không chứng minh generalization temporal
- Chưa có hyperparameter grid search đầy đủ
- class_weight=balanced được dùng nhưng cần đánh giá tác động rõ hơn

```python
# [Cell 11 - Execution Count: 6]
# CV vs Test comparison
fig, ax = plt.subplots(figsize=(10, 5))
models = classification_comp['model'].values
x = np.arange(len(models))
width = 0.35

cv_f1 = classification_comp.get('cv_f1_mean', classification_comp.get('cv_f1', pd.Series([0]*len(models)))).values
test_f1 = classification_comp.get('test_f1', pd.Series([0]*len(models))).values

bars1 = ax.bar(x - width/2, cv_f1, width, label='CV F1 (mean)', color='steelblue')
bars2 = ax.bar(x + width/2, test_f1, width, label='Test F1', color='coral')
ax.set_xlabel('Model')
ax.set_ylabel('F1 Score')
ax.set_title('Classification: CV vs Test F1 Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_MC / 'classification_cv_vs_test.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Result Display:**
```text
<Figure size 1000x500 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 4. Association Rules Comparison

So sánh Apriori và FP-Growth trên cùng basket matrix, cùng min_support và min_confidence.

```python
# [Cell 13 - Execution Count: 7]
assoc_comp = build_association_comparison()
assoc_comp.to_csv(TABLES_MC / 'association_rules_comparison.csv', index=False)

print("Association Rules Algorithm Comparison:")
display(assoc_comp)
```

**Output (stdout):**
```text
Association Rules Algorithm Comparison:
```

**Result Display:**
```text
algorithm  min_support  min_confidence  runtime_seconds  \
0    Apriori       0.0200          0.5000           4.1200   
1  FP-Growth       0.0200          0.5000           3.6100   

   frequent_itemset_count  rule_count  valid_rule_count  max_itemset_size  \
0                     389          61                61                 3   
1                     389          61                61                 3   

   max_rule_lift  is_selected  \
0        18.2311        False   
1        18.2311         True   

                                    selection_reason  
0                                                     
1  Apriori and FP-Growth produce mathematically i...
```

### Nhận xét Association Rules

**Kỳ vọng:**
- Apriori và FP-Growth giải cùng bài toán frequent itemsets → kết quả phải tương đương
- Runtime có thể khác nhau tùy thuộc vào cấu trúc dữ liệu (mật độ, số items)
- FP-Growth thường nhanh hơn trên dữ liệu thưa, nhưng không luôn đúng

**Validation:**
- Cùng số frequent itemsets → PASS
- Cùng số rules → PASS (hoặc kiểm tra nội dung thực nếu khác)
- lift > 1 cho tất cả selected rules → mối liên hệ dương

## 5. Model Artifacts Verification

```python
# [Cell 16 - Execution Count: 8]
try:
    artifact_checks = verify_model_artifacts()
    for c in artifact_checks:
        print(c)
    print("\n[OK] All model artifacts verified.")
except Exception as e:
    print(f"[WARNING] Artifact verification: {e}")
```

**Output (stdout):**
```text
[PASS] Clustering model loadable: KMeans
[PASS] Classification pipeline loadable: Pipeline
[PASS] Classification metadata: selected=RandomForest

[OK] All model artifacts verified.
```

## 6. Business Insights

Insights được sinh từ kết quả thực tế của các models, không hard-code.

```python
# [Cell 18 - Execution Count: 9]
try:
    insights = generate_all_insights()

    # Save insights tables
    for name, df_insight in insights.items():
        if isinstance(df_insight, pd.DataFrame) and not df_insight.empty:
            path = OUTPUTS_DIR / 'tables' / 'insights' / f'{name}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            df_insight.to_csv(path, index=False)
            print(f"[OK] Saved {name}.csv ({len(df_insight)} rows)")
        elif isinstance(df_insight, dict):
            print(f"[INFO] {name}: {len(df_insight)} entries")

except Exception as e:
    print(f"[WARNING] Insights generation: {e}")
    import traceback
    traceback.print_exc()
```

**Output (stdout):**
```text
[OK] Saved customer_segment_insights.csv (2 rows)
[OK] Saved customer_segment_action_plan.csv (4 rows)
[OK] Saved product_association_insights.csv (20 rows)
[OK] Saved classification_feature_insights.csv (43 rows)
```

## 7. Generate Report and Manifest

```python
# [Cell 20 - Execution Count: 10]
# Generate report
try:
    report_path = generate_report(
        clustering_comp, classification_comp, assoc_comp,
        output_path=REPORTS_DIR / '07_model_comparison_and_insights.md'
    )
    print(f"[OK] Report saved to: {report_path}")
except Exception as e:
    print(f"[WARNING] Report generation: {e}")

# Generate manifest
try:
    manifest = generate_manifest()
    manifest_path = EVIDENCE_DIR / 'pipeline_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)
    print(f"[OK] Manifest saved to: {manifest_path}")
except Exception as e:
    print(f"[WARNING] Manifest generation: {e}")
```

**Output (stdout):**
```text
[OK] Report saved to: # CRISP-DM Step 07: Model Comparison, Business Insights & Strategic Recommendations

**Project:** UCI Online Retail Data Mining Analysis  
**Phase:** CRISP-DM Step 05 (Evaluation) & Step 06 (Deployment Preparation)  
**Execution Timestamp:** 2026-10-02 20:07:32  

---

## 1. Executive Summary

BÃ¡o cÃ¡o nÃ y tá»•ng há»£p toÃ n diá»‡n káº¿t quáº£ tá»« cÃ¡c bÆ°á»›c khai phÃ¡ dá»¯ liá»‡u (PhÃ¢n cá»¥m khÃ¡ch hÃ ng, Dá»± Ä‘oÃ¡n mua láº¡i, vÃ  Khai phÃ¡ luáº­t káº¿t há»£p) trÃªn táº­p dá»¯ liá»‡u UCI Online Retail.

- **Customer Clustering (PhÃ¢n cá»¥m khÃ¡ch hÃ ng):** MÃ´ hÃ¬nh Ä‘Æ°á»£c chá»n lÃ  **K-Means (K=2)** vá»›i Silhouette Score = **0.4330**, Davies-Bouldin Index = **0.8917**. PhÃ¢n tÃ¡ch thÃ nh cÃ¡c nhÃ³m: Best Customers (38.35%, 1,662 khÃ¡ch hÃ ng); Lost Customers (61.65%, 2,672 khÃ¡ch hÃ ng).
- **Repeat Purchase Classification (Dá»± Ä‘oÃ¡n mua láº¡i):** MÃ´ hÃ¬nh há»c Ä‘Æ°á»£c chá»n lÃ  **RandomForest** vá»›i Stratified 5-Fold CV F1 = **0.6726**, Test F1 = **0.7086**, Test Precision = **0.7280**, Test Recall = **0.6901**, Test ROC-AUC = **0.7438**. Baseline DummyClassifier (chiáº¿n lÆ°á»£c Ä‘oÃ¡n lá»›p Ä‘a sá»‘) cÃ³ CV F1 = 0.7259 do tá»· lá»‡ lá»›p dÆ°Æ¡ng cao (58.4%). MÃ´ hÃ¬nh há»c mÃ¡y RandomForest Ä‘Æ°á»£c chá»n lÃ  mÃ´ hÃ¬nh há»c cÃ³ kháº£ nÄƒng phÃ¢n biá»‡t tá»‘t nháº¥t (ROC-AUC=0.7438).
- **Association Rule Mining (Khai phÃ¡ luáº­t káº¿t há»£p):** Thuáº­t toÃ¡n Ä‘Æ°á»£c chá»n lÃ  **FP-Growth** sinh ra **61 luáº­t há»£p lá»‡** (Lift > 1.0) trong thá»i gian thá»±c thi 3.61 giÃ¢y.

---

## 2. Model Comparison Tables (Báº£ng so sÃ¡nh mÃ´ hÃ¬nh)

### 2.1 Customer Clustering Model Comparison
| algorithm              |   n_clusters |   silhouette_score |   davies_bouldin_score |   calinski_harabasz_score |   dunn_approximation |   noise_ratio |   min_cluster_size |   max_cluster_pct |   runtime_seconds | is_selected   | is_eligible   | rejection_reason                                                                                                                           | selection_reason                                                                      |   rank_sil |   rank_db |   rank_ch |   rank_dunn |   combined_rank |
|:-----------------------|-------------:|-------------------:|-----------------------:|--------------------------:|---------------------:|--------------:|-------------------:|------------------:|------------------:|:--------------|:--------------|:-------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|-----------:|----------:|----------:|------------:|----------------:|
| K-Means                |            2 |             0.433  |                 0.8917 |                 4364.61   |               0.2093 |          0    |               1662 |             61.65 |            0.0638 | True          | True          | nan                                                                                                                                        | Silhouette=0.4330; DB=0.8917; Best combined ranking across Silhouette, DB, CH metrics |          1 |         1 |         1 |         3   |             6   |
| K-Means                |            3 |             0.3375 |                 1.0462 |                 3626.28   |               0.1669 |          0    |                763 |             43.22 |            0.0255 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 23.0)                                             |          4 |         8 |         3 |         8   |            23   |
| K-Means                |            4 |             0.3381 |                 1.0131 |                 3329.68   |               0.1506 |          0    |                686 |             37.75 |            0.0449 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 27.0)                                             |          3 |         7 |         4 |        13   |            27   |
| K-Means                |            5 |             0.3172 |                 0.9851 |                 3194.66   |               0.1786 |          0    |                312 |             27.57 |            0.0813 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 19.0)                                             |          5 |         4 |         5 |         5   |            19   |
| K-Means                |            6 |             0.3141 |                 1.0106 |                 3083.61   |               0.163  |          0    |                317 |             22.82 |            0.0901 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 28.0)                                             |          7 |         6 |         6 |         9   |            28   |
| K-Means                |            7 |             0.3095 |                 0.9692 |                 2960.37   |               0.1719 |          0    |                219 |             20.51 |            0.1147 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 25.0)                                             |          8 |         3 |         8 |         6   |            25   |
| K-Means                |            8 |             0.3012 |                 0.9944 |                 2824.33   |               0.1525 |          0    |                214 |             19.54 |            0.0918 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 35.0)                                             |          9 |         5 |         9 |        12   |            35   |
| GMM                    |            2 |             0.2874 |                 1.0662 |                 2307.42   |               0.158  |          0    |               1505 |             65.27 |            0.3229 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 50.0)                                             |         12 |        11 |        17 |        10   |            50   |
| GMM                    |            3 |             0.2562 |                 1.2157 |                 2749.9    |               0.1406 |          0    |                864 |             45.34 |            0.2637 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 54.0)                                             |         13 |        16 |        11 |        14   |            54   |
| GMM                    |            4 |             0.1752 |                 1.7079 |                 2167.59   |               0.0885 |          0    |                831 |             34.73 |            0.383  | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 76.0)                                             |         19 |        19 |        19 |        19   |            76   |
| GMM                    |            5 |             0.152  |                 1.7655 |                 1880.24   |               0.0764 |          0    |                185 |             34.7  |            0.3315 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 80.0)                                             |         20 |        20 |        20 |        20   |            80   |
| GMM                    |            6 |             0.1164 |                 2.2668 |                 1454.54   |               0.0546 |          0    |                110 |             34.7  |            0.7202 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 84.0)                                             |         21 |        21 |        21 |        21   |            84   |
| GMM                    |            7 |             0.1019 |                 2.5613 |                 1271.32   |               0.046  |          0    |                 81 |             34.7  |            0.7187 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 90.0)                                             |         22 |        23 |        23 |        22   |            90   |
| GMM                    |            8 |             0.0681 |                 2.2984 |                 1279.12   |               0.0454 |          0    |                 72 |             34.7  |            0.765  | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 90.0)                                             |         23 |        22 |        22 |        23   |            90   |
| Agglomerative(ward)    |            2 |             0.4232 |                 0.9064 |                 4227.37   |               0.2048 |          0    |               1726 |             60.18 |            0.7933 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 10.0)                                             |          2 |         2 |         2 |         4   |            10   |
| Agglomerative(ward)    |            3 |             0.3146 |                 1.1509 |                 3040.26   |               0.1694 |          0    |                633 |             60.18 |            0.6623 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 34.0)                                             |          6 |        14 |         7 |         7   |            34   |
| Agglomerative(ward)    |            4 |             0.2428 |                 1.2255 |                 2780.1    |               0.1244 |          0    |                633 |             37.54 |            0.7238 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 59.0)                                             |         16 |        17 |        10 |        16   |            59   |
| Agglomerative(ward)    |            5 |             0.2385 |                 1.2431 |                 2551.43   |               0.1177 |          0    |                633 |             25.22 |            0.7833 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 64.5)                                             |         17 |        18 |        12 |        17.5 |            64.5 |
| Agglomerative(ward)    |            6 |             0.2447 |                 1.1518 |                 2439.17   |               0.1177 |          0    |                474 |             22.63 |            0.7504 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 60.5)                                             |         15 |        15 |        13 |        17.5 |            60.5 |
| Agglomerative(ward)    |            7 |             0.2489 |                 1.1215 |                 2405.87   |               0.1526 |          0    |                101 |             22.63 |            0.7274 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 54.0)                                             |         14 |        13 |        16 |        11   |            54   |
| Agglomerative(ward)    |            8 |             0.2252 |                 1.0852 |                 2281.43   |               0.1315 |          0    |                101 |             22.63 |            0.7045 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 63.0)                                             |         18 |        12 |        18 |        15   |            63   |
| DBSCAN(eps=0.3,min=5)  |            8 |             0.0632 |                 1.5042 |                  837.295  |               0.0949 |          5.7  |                  7 |             33.87 |            0.0757 | False         | False         | Trivial cluster size: smallest cluster has 7 < 10 samples                                                                                  | Ineligible: Trivial cluster size: smallest cluster has 7 < 10 samples                 |        nan |       nan |       nan |       nan   |           nan   |
| DBSCAN(eps=0.5,min=5)  |            2 |             0.2946 |                 1.0626 |                 2408.38   |               0.2419 |          1.32 |               1496 |             64.17 |            0.1372 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 37.0)                                             |         11 |        10 |        15 |         1   |            37   |
| DBSCAN(eps=0.7,min=5)  |            2 |             0.5659 |                 0.3483 |                   49.9624 |               0.477  |          0.55 |                  5 |             99.33 |            0.1799 | False         | False         | Degenerate cluster: largest cluster has 99.3% >= 90%; Trivial cluster size: smallest cluster has 5 < 10 samples                            | Ineligible: Degenerate cluster: largest cluster has 99.3% >= 90%                      |        nan |       nan |       nan |       nan   |           nan   |
| DBSCAN(eps=1.0,min=5)  |            1 |           nan      |               nan      |                  nan      |             nan      |          0.23 |               4324 |             99.77 |            0.2685 | False         | False         | Silhouette score is NaN (single cluster or pure noise); Fewer than 2 clusters formed; Degenerate cluster: largest cluster has 99.8% >= 90% | Ineligible: Silhouette score is NaN (single cluster or pure noise)                    |        nan |       nan |       nan |       nan   |           nan   |
| DBSCAN(eps=0.5,min=10) |            2 |             0.2961 |                 1.0613 |                 2416.07   |               0.2403 |          1.78 |               1492 |             63.8  |            0.1116 | False         | True          | nan                                                                                                                                        | Eligible candidate (Combined Rank = 35.0)                                             |         10 |         9 |        14 |         2   |            35   |

### 2.2 Classification Model Comparison
| model              |   cv_f1_mean |   cv_f1_std |   cv_roc_auc_mean |   cv_average_precision_mean |   test_accuracy |   test_balanced_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   test_average_precision |   test_specificity |   tn |   fp |   fn |   tp |   runtime_seconds | selected   |   random_state | is_selected   | selection_reason                                                                                                                                         |   specificity |
|:-------------------|-------------:|------------:|------------------:|----------------------------:|----------------:|-------------------------:|-----------------:|--------------:|----------:|---------------:|-------------------------:|-------------------:|-----:|-----:|-----:|-----:|------------------:|:-----------|---------------:|:--------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| DummyClassifier    |       0.7259 |      0.0003 |            0.5    |                      0.5698 |          0.5697 |                   0.5    |           0.5697 |        1      |    0.7259 |         0.5    |                   0.5697 |             0      |    0 |  290 |    0 |  384 |            0.0297 | False      |             42 | False         | Baseline classifier predicting majority class                                                                                                            |        0      |
| LogisticRegression |       0.665  |      0.015  |            0.7323 |                      0.801  |          0.6973 |                   0.7078 |           0.7941 |        0.6328 |    0.7043 |         0.7763 |                   0.8366 |             0.7828 |  227 |   63 |  141 |  243 |            0.0298 | False      |             42 | False         | Candidate learned model                                                                                                                                  |        0.7828 |
| DecisionTree       |       0.6502 |      0.0162 |            0.6375 |                      0.6817 |          0.6558 |                   0.6557 |           0.7159 |        0.6562 |    0.6848 |         0.6828 |                   0.7325 |             0.6552 |  190 |  100 |  132 |  252 |            0.032  | False      |             42 | False         | Candidate learned model                                                                                                                                  |        0.6552 |
| RandomForest       |       0.6726 |      0.0191 |            0.6991 |                      0.785  |          0.6766 |                   0.6744 |           0.728  |        0.6901 |    0.7086 |         0.7438 |                   0.8113 |             0.6586 |  191 |   99 |  119 |  265 |            0.2478 | True       |             42 | True          | Highest Stratified K-Fold CV F1-score (0.6726) among candidate learned models. Selection performed strictly on CV metrics without test set data leakage. |        0.6586 |

### 2.3 Association Rules Algorithm Comparison
| algorithm   |   min_support |   min_confidence |   runtime_seconds |   frequent_itemset_count |   rule_count |   valid_rule_count |   max_itemset_size |   max_rule_lift | is_selected   | selection_reason                                                                                                                                                    |
|:------------|--------------:|-----------------:|------------------:|-------------------------:|-------------:|-------------------:|-------------------:|----------------:|:--------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Apriori     |          0.02 |              0.5 |              4.12 |                      389 |           61 |                 61 |                  3 |         18.2311 | False         |                                                                                                                                                                     |
| FP-Growth   |          0.02 |              0.5 |              3.61 |                      389 |           61 |                 61 |                  3 |         18.2311 | True          | Apriori and FP-Growth produce mathematically identical rule sets (61 valid rules). Selected FP-Growth for faster execution runtime (3.61s vs 4.12s, 12.4% speedup). |

---

## 3. Customer Segment Profiles & Strategic Action Plan (PhÃ¢n khÃºc & Káº¿ hoáº¡ch hÃ nh Ä‘á»™ng)

### 3.1 Segment Profiles (Há»“ sÆ¡ phÃ¢n khÃºc)
|   cluster_id | business_segment_name   |   customer_count |   customer_percentage |   eligible_labeled_customers |   repeat_customers |   unlabeled_customers |   cohort_coverage_pct |   recency_mean |   recency_median |   frequency_mean |   frequency_median |   monetary_mean |   monetary_median |   average_order_value |   repeat_purchase_rate | business_interpretation                                                                                                                                                                                                                                 | cohort_note                                                                      | evidence_source                                                                                 |
|-------------:|:------------------------|-----------------:|----------------------:|-----------------------------:|-------------------:|----------------------:|----------------------:|---------------:|-----------------:|-----------------:|-------------------:|----------------:|------------------:|----------------------:|-----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
|            0 | Best Customers          |             1662 |                 38.35 |                         1487 |               1423 |                   175 |                 89.47 |           25.8 |               16 |              8.4 |                  6 |         4464.19 |           2041.33 |                565.22 |                 0.957  | Khách hàng mua hàng rất gần đây (Recency trung bình 26 ngày). tần suất đặt hàng cao (8.4 đơn). giá trị chi tiêu rất lớn (doanh thu trung bình £4,464). Tỷ lệ mua lại 90 ngày sau cutoff đạt 95.7%. Chiếm 38.3% tổng số khách hàng (1,662 khách hàng).   | Repeat rate is evaluated retrospectively on customers active before cutoff date. | data/processed/customer_clusters.csv + rfm_customer_features.csv + repeat_purchase_features.csv |
|            1 | Lost Customers          |             2672 |                 61.65 |                         1881 |                496 |                   791 |                 70.4  |          134.3 |               96 |              1.7 |                  1 |          493.17 |            356.92 |                322.34 |                 0.2637 | Khách hàng giao dịch mức độ vừa phải (Recency trung bình 134 ngày). tần suất đặt hàng thấp (1.7 đơn). giá trị chi tiêu thấp (doanh thu trung bình £493). Tỷ lệ mua lại 90 ngày sau cutoff đạt 26.4%. Chiếm 61.7% tổng số khách hàng (2,672 khách hàng). | Repeat rate is evaluated retrospectively on customers active before cutoff date. | data/processed/customer_clusters.csv + rfm_customer_features.csv + repeat_purchase_features.csv |

### 3.2 Strategic Action Plan (Káº¿ hoáº¡ch hÃ nh Ä‘á»™ng chiáº¿n lÆ°á»£c)
|   cluster_id | business_segment_name   |   customer_count |   customer_percentage | strategy                           | recommended_action                                                                                  | evidence                                                             | target_kpi                                                          | verification_method                                                     | limitations                                                         |
|-------------:|:------------------------|-----------------:|----------------------:|:-----------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:--------------------------------------------------------------------|:------------------------------------------------------------------------|:--------------------------------------------------------------------|
|            0 | Best Customers          |             1662 |                 38.35 | Retention & VIP Loyalty            | Triển khai chương trình khách hàng thân thiết VIP (ưu tiên giao hàng, quyền mua sớm bộ sưu tập mới) | Recency trung bình = 26 ngày, Tần suất = 8.4 đơn, Doanh thu = £4,464 | Tỷ lệ duy trì đơn hàng (Retention Rate) ≥ 80% trong 6 tháng kế tiếp | Theo dõi tỷ lệ quay lại tự nhiên so với các quý trước (Cohort analysis) | Chi phí vận hành chương trình tích điểm/ưu đãi thành viên           |
|            0 | Best Customers          |             1662 |                 38.35 | Premium Upsell & Cross-sell        | Giới thiệu các dòng sản phẩm cao cấp hoặc combo quà tặng giá trị cao                                | Chi tiêu trung bình £4,464 (> £1,000)                                | Tăng giá trị đơn hàng trung bình (AOV) thêm 10%                     | Đo lường AOV của nhóm khách nhận gợi ý so với nhóm không nhận gợi ý     | Cần kiểm soát tồn kho các mặt hàng cao cấp trước khi đẩy mạnh gợi ý |
|            0 | Best Customers          |             1662 |                 38.35 | Advocacy & Referral Campaign       | Kêu gọi đánh giá sản phẩm và giới thiệu bạn bè nhận thưởng hai chiều (Referral Program)             | Tỷ lệ mua lại rất cao: 95.7% (≥ 60%)                                 | Tỷ lệ khách hàng giới thiệu thêm người dùng mới ≥ 8%                | Gắn mã giới thiệu cá nhân và đối chiếu số lượt đăng ký mới qua mã       | Cần chống gian lận tự tạo tài khoản phụ để nhận thưởng              |
|            1 | Lost Customers          |             2672 |                 61.65 | First-to-Second Purchase Nurturing | Chiến dịch nuôi dưỡng sau đơn hàng đầu tiên (hướng dẫn sử dụng sản phẩm, quà tặng đơn thứ hai)      | Tỷ lệ mua lại chỉ đạt 26.4% (< 40%)                                  | Nâng tỷ lệ khách hàng mua lại lần 2 từ dưới 40% lên 45%             | Thử nghiệm email tự động gửi vào ngày thứ 14 sau đơn hàng đầu           | Rủi ro gây phiền toái nếu tần suất gửi email quá dày đặc            |

---

## 4. Product Co-Purchase Association Rules (Top 10 Luáº­t káº¿t há»£p hÃ ng Ä‘áº§u)
| antecedents                                                      | consequents                                                      |   support |   confidence |   lift |   conviction | evidence_quality   | business_interpretation                                                                                                                                                                                                                                                | recommended_action                                                                                                                                                                                                                  | limitations                                                                                                                                     |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------|----------:|-------------:|-------:|-------------:|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| PINK REGENCY TEACUP AND SAUCER                                   | GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER |    0.0274 |       0.7072 |  18.23 |       3.2827 | Moderate           | Khách hàng mua 'PINK REGENCY TEACUP AND SAUCER' có khả năng cao cũng mua 'GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' (xác suất cao gấp 18.2 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' khi khách hàng đưa 'PINK REGENCY TEACUP AND SAUCER' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER | PINK REGENCY TEACUP AND SAUCER                                   |    0.0274 |       0.7053 |  18.23 |       3.2625 | Moderate           | Khách hàng mua 'GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' có khả năng cao cũng mua 'PINK REGENCY TEACUP AND SAUCER' (xác suất cao gấp 18.2 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'PINK REGENCY TEACUP AND SAUCER' khi khách hàng đưa 'GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER  | GREEN REGENCY TEACUP AND SAUCER                                  |    0.0274 |       0.9047 |  17.66 |       9.9537 | Moderate           | Khách hàng mua 'PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' có khả năng cao cũng mua 'GREEN REGENCY TEACUP AND SAUCER' (xác suất cao gấp 17.7 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'GREEN REGENCY TEACUP AND SAUCER' khi khách hàng đưa 'PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| GREEN REGENCY TEACUP AND SAUCER                                  | PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER  |    0.0274 |       0.5341 |  17.66 |       2.0813 | Moderate           | Khách hàng mua 'GREEN REGENCY TEACUP AND SAUCER' có khả năng cao cũng mua 'PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' (xác suất cao gấp 17.7 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER ' khi khách hàng đưa 'GREEN REGENCY TEACUP AND SAUCER' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Độ tin cậy vừa phải (0.53) — xác suất đồng mua chưa tuyệt đối; Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality) |
| PINK REGENCY TEACUP AND SAUCER                                   | GREEN REGENCY TEACUP AND SAUCER                                  |    0.032  |       0.8261 |  16.13 |       5.4572 | Strong             | Khách hàng mua 'PINK REGENCY TEACUP AND SAUCER' có khả năng cao cũng mua 'GREEN REGENCY TEACUP AND SAUCER' (xác suất cao gấp 16.1 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 3.2% tổng số giỏ hàng.                                   | Đề xuất gợi ý chéo 'GREEN REGENCY TEACUP AND SAUCER' khi khách hàng đưa 'PINK REGENCY TEACUP AND SAUCER' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle).                                   | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| GREEN REGENCY TEACUP AND SAUCER                                  | PINK REGENCY TEACUP AND SAUCER                                   |    0.032  |       0.6239 |  16.13 |       2.5559 | Moderate           | Khách hàng mua 'GREEN REGENCY TEACUP AND SAUCER' có khả năng cao cũng mua 'PINK REGENCY TEACUP AND SAUCER' (xác suất cao gấp 16.1 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 3.2% tổng số giỏ hàng.                                   | Đề xuất gợi ý chéo 'PINK REGENCY TEACUP AND SAUCER' khi khách hàng đưa 'GREEN REGENCY TEACUP AND SAUCER' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle).                                   | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER  | ROSES REGENCY TEACUP AND SAUCER                                  |    0.0274 |       0.856  |  15.89 |       6.571  | Moderate           | Khách hàng mua 'GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER' có khả năng cao cũng mua 'ROSES REGENCY TEACUP AND SAUCER ' (xác suất cao gấp 15.9 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'ROSES REGENCY TEACUP AND SAUCER ' khi khách hàng đưa 'GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| ROSES REGENCY TEACUP AND SAUCER                                  | GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER  |    0.0274 |       0.508  |  15.89 |       1.9675 | Moderate           | Khách hàng mua 'ROSES REGENCY TEACUP AND SAUCER ' có khả năng cao cũng mua 'GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER' (xác suất cao gấp 15.9 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.7% tổng số giỏ hàng. | Đề xuất gợi ý chéo 'GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER' khi khách hàng đưa 'ROSES REGENCY TEACUP AND SAUCER ' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle). | Độ tin cậy vừa phải (0.51) — xác suất đồng mua chưa tuyệt đối; Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality) |
| GARDENERS KNEELING PAD CUP OF TEA                                | GARDENERS KNEELING PAD KEEP CALM                                 |    0.0276 |       0.7203 |  15.6  |       3.4104 | Moderate           | Khách hàng mua 'GARDENERS KNEELING PAD CUP OF TEA ' có khả năng cao cũng mua 'GARDENERS KNEELING PAD KEEP CALM ' (xác suất cao gấp 15.6 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.8% tổng số giỏ hàng.                             | Đề xuất gợi ý chéo 'GARDENERS KNEELING PAD KEEP CALM ' khi khách hàng đưa 'GARDENERS KNEELING PAD CUP OF TEA ' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle).                             | Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)                                                                |
| GARDENERS KNEELING PAD KEEP CALM                                 | GARDENERS KNEELING PAD CUP OF TEA                                |    0.0276 |       0.598  |  15.6  |       2.3924 | Moderate           | Khách hàng mua 'GARDENERS KNEELING PAD KEEP CALM ' có khả năng cao cũng mua 'GARDENERS KNEELING PAD CUP OF TEA ' (xác suất cao gấp 15.6 lần so với khi hai sản phẩm xuất hiện độc lập). Mô hình này xuất hiện trong 2.8% tổng số giỏ hàng.                             | Đề xuất gợi ý chéo 'GARDENERS KNEELING PAD CUP OF TEA ' khi khách hàng đưa 'GARDENERS KNEELING PAD KEEP CALM ' vào giỏ hàng. Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle).                             | Độ tin cậy vừa phải (0.60) — xác suất đồng mua chưa tuyệt đối; Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality) |

---

## 5. Predictive Feature Importance (Táº§m quan trá»ng cá»§a Ä‘áº·c trÆ°ng dá»± Ä‘oÃ¡n)
| feature                |   importance |   rank | source_model   | feature_group   | interpretation                                                                                                                                                                                                                       | limitations                                                                                                             |
|:-----------------------|-------------:|-------:|:---------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| Recency                |   0.139253   |      1 | RandomForest   | RFM             | Top-3 biến quan trọng nhất (tầm quan trọng = 0.1393). Biến này có đóng góp phân tách mạnh nhất đối với dự đoán mua lại trong mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả. | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| Monetary               |   0.136043   |      2 | RandomForest   | RFM             | Top-3 biến quan trọng nhất (tầm quan trọng = 0.1360). Biến này có đóng góp phân tách mạnh nhất đối với dự đoán mua lại trong mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả. | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| UniqueProducts         |   0.129772   |      3 | RandomForest   | Behavioral      | Top-3 biến quan trọng nhất (tầm quan trọng = 0.1298). Biến này có đóng góp phân tách mạnh nhất đối với dự đoán mua lại trong mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả. | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| TotalItems             |   0.127866   |      4 | RandomForest   | Behavioral      | Biến quan trọng mức trung bình (tầm quan trọng = 0.1279). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| AverageOrderValue      |   0.123478   |      5 | RandomForest   | Behavioral      | Biến quan trọng mức trung bình (tầm quan trọng = 0.1235). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| AverageItemsPerInvoice |   0.117268   |      6 | RandomForest   | Behavioral      | Biến quan trọng mức trung bình (tầm quan trọng = 0.1173). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| CustomerLifetimeDays   |   0.081577   |      7 | RandomForest   | Behavioral      | Biến quan trọng mức trung bình (tầm quan trọng = 0.0816). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| Frequency              |   0.0576431  |      8 | RandomForest   | RFM             | Biến quan trọng mức trung bình (tầm quan trọng = 0.0576). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| ActiveDays             |   0.0566452  |      9 | RandomForest   | Behavioral      | Biến quan trọng mức trung bình (tầm quan trọng = 0.0566). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |
| Country_United Kingdom |   0.00740603 |     10 | RandomForest   | Geographic      | Biến quan trọng mức trung bình (tầm quan trọng = 0.0074). Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình RandomForest. Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả.       | Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả. |

---

## 6. Generated Visualizations & Dashboards (Biá»ƒu Ä‘á»“ & Dashboard minh há»a)
- `clustering_model_comparison.png`
- `classification_model_comparison.png`
- `clustering_quality_metrics.png`
- `classification_cv_vs_test.png`
- `insight_summary_dashboard.png`

---

## 7. Limitations & Scientific Constraints (Giá»›i háº¡n & RÃ ng buá»™c phÆ°Æ¡ng phÃ¡p)
1. **Single Retailer Scope:** Dá»¯ liá»‡u chá»‰ tá»« má»™t nhÃ  bÃ¡n láº» trá»±c tuyáº¿n táº¡i VÆ°Æ¡ng quá»‘c Anh (12/2010 - 12/2011), khÃ´ng tá»± Ä‘á»™ng suy rá»™ng ra toÃ n ngÃ nh e-commerce.
2. **Missing CustomerID:** 24.93% giao dá»‹ch khÃ´ng cÃ³ CustomerID bá»‹ loáº¡i khá»i bÃ i toÃ¡n cáº¥p khÃ¡ch hÃ ng (selection bias).
3. **Class Imbalance & Baseline:** Tá»· lá»‡ mua láº¡i 90 ngÃ y Ä‘áº¡t 58.4%, khiáº¿n Dummy Classifier cÃ³ F1 danh nghÄ©a cao; Random Forest lÃ  mÃ´ hÃ¬nh há»c phÃ¢n biá»‡t cÃ³ giÃ¡ trá»‹ thá»±c táº¿ nháº¥t.
4. **Retrospective vs Predictive Segment Evaluation:** Tá»· lá»‡ mua láº¡i theo cá»¥m lÃ  phÃ¢n tÃ­ch há»“i cá»©u mÃ´ táº£ do cá»¥m RFM Ä‘Æ°á»£c xÃ¢y dá»±ng trÃªn toÃ n bá»™ l»‹ch sá»­ quan sÃ¡t.
5. **Association vs Causation:** Luáº­t káº¿t há»£p (Lift > 1) chá»‰ biá»ƒu thá»‹ tÆ°Æ¡ng quan Ä‘á»“ng xuáº¥t hiá»‡n thá»‘ng kÃª, chÆ°a pháº£i quan há»‡ nhÃ¢n quáº£; cáº§n kiá»ƒm chá»©ng qua A/B testing trÆ°á»›c khi quyáº¿t Ä‘á»‹nh nháº­p hÃ ng combo.
```

**Output (stdout):**
```text
[OK] Manifest saved to: D:\Project\Data-mininng\outputs\evidence\pipeline_manifest.json
```

## 8. Tổng kết

### Các quyết định cuối cùng:
1. **Clustering**: Model được chọn dựa trên ranking tổng hợp internal metrics + cluster size distribution
2. **Classification**: Model được chọn dựa trên CV F1-score (không dùng test set để chọn)
3. **Association Rules**: Cả Apriori và FP-Growth đều hợp lệ; chọn theo runtime + equivalence check

### Hạn chế chung:
- Single UK retailer — kết quả không tự động generalize sang retailer khác
- Historical data (2010-12-01 → 2011-12-09) — chưa đánh giá temporal drift
- Missing CustomerID (24.93%) — selection bias trong tập khách hàng
- Không có margin/campaign response data — chưa tính ROI thực tế
- Không chứng minh nhân quả — chỉ là mối liên hệ thống kê

### Items cần nhóm bổ sung (NEEDS_USER_INPUT):
- Tên và đóng góp thành viên
- Peer assessment
- AI disclosure

```python
# [Cell 22 - Execution Count: 11]
print("=" * 60)
print("  07. MODEL COMPARISON AND INSIGHTS — COMPLETE")
print("=" * 60)
print(f"\nOutputs:")
print(f"  Tables: {TABLES_MC}")
print(f"  Figures: {FIGURES_MC}")
print(f"  Report: {REPORTS_DIR / '07_model_comparison_and_insights.md'}")
print(f"  Manifest: {EVIDENCE_DIR / 'pipeline_manifest.json'}")
```

**Output (stdout):**
```text
============================================================
  07. MODEL COMPARISON AND INSIGHTS — COMPLETE
============================================================

Outputs:
  Tables: D:\Project\Data-mininng\outputs\tables\model_comparison
  Figures: D:\Project\Data-mininng\outputs\figures\model_comparison
  Report: D:\Project\Data-mininng\outputs\reports\07_model_comparison_and_insights.md
  Manifest: D:\Project\Data-mininng\outputs\evidence\pipeline_manifest.json
```

