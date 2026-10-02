# Review Notebook: 04_customer_clustering.ipynb
*Source Path: `D:/Project/Data-mininng/notebooks/04_customer_clustering.ipynb`*

---

# 04. Customer Clustering (Phân cụm Khách hàng)

**Mục tiêu:**
- Sử dụng dữ liệu RFM để phân nhóm khách hàng bằng nhiều thuật toán.
- So sánh các thuật toán: K-Means, GMM, Agglomerative Clustering, DBSCAN.
- Đánh giá bằng các metric: Silhouette Score, Davies-Bouldin, Calinski-Harabasz, và Dunn Index (sử dụng Dunn Approximation).
- Lưu lại mô hình và gán tên nhóm (Business Segment) cho khách hàng.

```python
# [Cell 1 - Execution Count: 1]
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import PROCESSED_DIR, TABLES_CLUSTERING, MODELS_CLUSTERING_DIR, FIGURES_CLUSTERING
from src.clustering import (
    prepare_clustering_features, run_kmeans, run_gmm, run_agglomerative, run_dbscan,
    build_comparison_table, select_best_model, create_cluster_profiles, assign_business_names,
    save_clustering_results, pca_2d
)
from src.visualization import (
    plot_elbow_curve, plot_silhouette_comparison, plot_metrics_comparison,
    plot_cluster_distribution, plot_cluster_rfm_boxplots, plot_pca_2d, plot_rfm_heatmap
)

# Render biểu đồ trực tiếp trong cell output
%matplotlib inline
```

## 1. Chuẩn bị Dữ liệu
- Load features RFM.
- Chuyển đổi theo pipeline: `RFM -> log1p -> StandardScaler`.
- Kiểm tra missing, duplicate, và giá trị âm.

```python
# [Cell 3 - Execution Count: 2]
df = pd.read_csv(PROCESSED_DIR / 'rfm_clustering_features.csv')
print(f"Số lượng khách hàng: {len(df):,}")

# Validate data
assert df['CustomerID'].isnull().sum() == 0, 'Missing CustomerID'
assert df['CustomerID'].duplicated().sum() == 0, 'Duplicate CustomerID'
assert (df['Recency'] >= 0).all() and (df['Frequency'] >= 1).all() and (df['Monetary'] > 0).all(), 'Invalid RFM values'

rfm_cols = ['Recency', 'Frequency', 'Monetary']
X_raw, X_log, X_scaled, scaler, feature_names, customer_ids = prepare_clustering_features(df, rfm_cols)

print("\nThống kê mô tả dữ liệu gốc (RFM):")
display(df[rfm_cols].describe())
```

**Output (stdout):**
```text
Số lượng khách hàng: 4,334

Thống kê mô tả dữ liệu gốc (RFM):
```

**Result Display:**
```text
Recency    Frequency       Monetary
count  4334.000000  4334.000000    4334.000000
mean     92.703046     4.245962    2015.973153
std     100.177047     7.634989    8903.673825
min       1.000000     1.000000       3.750000
25%      18.000000     1.000000     304.240000
50%      51.000000     2.000000     662.565000
75%     143.000000     5.000000    1631.622500
max     374.000000   206.000000  279138.020000
```

## 2. So sánh Thuật toán
Thực thi vòng lặp qua K-Means, GMM, Agglomerative (với K từ 2 đến 8) và DBSCAN (nhiều cấu hình eps/min_samples).

```python
# [Cell 5 - Execution Count: 3]
k_range = range(2, 9)
all_results = []
kmeans_results = []

for k in k_range:
    labels, model, metrics = run_kmeans(X_scaled, k)
    kmeans_results.append(('K-Means', k, labels, model, metrics))
    all_results.append(('K-Means', k, labels, model, metrics))

for k in k_range:
    labels, model, metrics = run_gmm(X_scaled, k)
    all_results.append(('GMM', k, labels, model, metrics))

for k in k_range:
    labels, model, metrics = run_agglomerative(X_scaled, k, linkage='ward')
    all_results.append(('Agglomerative(ward)', k, labels, model, metrics))

dbscan_configs = [(0.3, 5), (0.5, 5), (0.7, 5), (1.0, 5), (0.5, 10)]
for eps, min_s in dbscan_configs:
    labels, model, metrics = run_dbscan(X_scaled, eps=eps, min_samples=min_s)
    all_results.append(('DBSCAN', metrics['n_clusters'], labels, model, metrics))

comparison_df = build_comparison_table(all_results)
display(comparison_df.sort_values('silhouette_score', ascending=False).head(10))
```

**Result Display:**
```text
algorithm  n_clusters  silhouette_score  davies_bouldin_score  \
23  DBSCAN(eps=0.7,min=5)           2            0.5659                0.3483   
0                 K-Means           2            0.4330                0.8917   
14    Agglomerative(ward)           2            0.4232                0.9064   
2                 K-Means           4            0.3381                1.0131   
1                 K-Means           3            0.3375                1.0462   
3                 K-Means           5            0.3172                0.9851   
15    Agglomerative(ward)           3            0.3146                1.1509   
4                 K-Means           6            0.3141                1.0106   
5                 K-Means           7            0.3095                0.9692   
6                 K-Means           8            0.3012                0.9944   

    calinski_harabasz_score  dunn_approximation  noise_ratio  \
23                  49.9624              0.4770         0.55   
0                 4364.6091              0.2093         0.00   
14                4227.3723              0.2048         0.00   
2                 3329.6842              0.1506         0.00   
1                 3626.2766              0.1669         0.00   
3                 3194.6602              0.1786         0.00   
15                3040.2604              0.1694         0.00   
4                 3083.6138              0.1630         0.00   
5                 2960.3706              0.1719         0.00   
6                 2824.3335              0.1525         0.00   

    min_cluster_size  max_cluster_pct  runtime_seconds  is_selected  
23                 5            99.33           0.2589        False  
0               1662            61.65           0.2385        False  
14              1726            60.18           0.7059        False  
2                686            37.75           0.0859        False  
1                763            43.22           0.0695        False  
3                312            27.57           0.1046        False  
15               633            60.18           0.8381        False  
4                317            22.82           0.0902        False  
5                219            20.51           0.1491        False  
6                214            19.54           0.0995        False
```

## 3. Trực quan hóa So sánh Metric
Sinh biểu đồ live để đánh giá xu hướng Silhouette Score, Davies-Bouldin, Calinski-Harabasz và Dunn Approximation.

```python
# [Cell 7 - Execution Count: 4]
plot_elbow_curve(kmeans_results)
```

**Result Display:**
```text
<Figure size 1000x600 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 8 - Execution Count: 5]
plot_silhouette_comparison(comparison_df, k_range)
```

**Result Display:**
```text
<Figure size 1000x600 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 9 - Execution Count: 6]
plot_metrics_comparison(comparison_df, k_range)
```

**Result Display:**
```text
<Figure size 1600x1200 with 4 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 4. Lựa chọn Mô hình & Phân khúc (Profiles)
- Chọn mô hình có điểm xếp hạng tổng hợp (combined rank) tốt nhất.
- Lọc bỏ các mô hình có ít hơn 2 cụm, cụm quá bé (<10) hoặc 1 cụm quá lớn (>90%).

```python
# [Cell 11 - Execution Count: 7]
best_labels, best_model, best_metrics = select_best_model(all_results, comparison_df)
best_algo = best_metrics['algorithm']
best_k = best_metrics['n_clusters']

print(f"Thuật toán được chọn tối ưu: {best_algo} (K={best_k})")
print(f"Silhouette Score: {best_metrics['silhouette_score']:.4f}")
print(f"Davies-Bouldin Index: {best_metrics['davies_bouldin_score']:.4f}")
print(f"Calinski-Harabasz: {best_metrics['calinski_harabasz_score']:.1f}")
print(f"Dunn Approximation: {best_metrics['dunn_approximation']:.4f}")

print("\n--- Bảng xếp hạng và đánh giá điều kiện hợp lệ toàn bộ 26 cấu hình ---")
display(comparison_df[['algorithm', 'n_clusters', 'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score', 'noise_ratio', 'max_cluster_pct', 'is_eligible', 'rejection_reason', 'is_selected', 'combined_rank']].sort_values('combined_rank'))

profiles = create_cluster_profiles(df, best_labels, rfm_cols)
profiles = assign_business_names(profiles)
name_map = dict(zip(profiles['Cluster'].astype(int), profiles['BusinessName']))

print("\n--- Hồ sơ phân khúc khách hàng ---")
display(profiles)

```

**Output (stdout):**
```text
Thuật toán được chọn tối ưu: K-Means (K=2)
Silhouette Score: 0.4330
Davies-Bouldin Index: 0.8917
Calinski-Harabasz: 4364.6
Dunn Approximation: 0.2093

--- Bảng xếp hạng và đánh giá điều kiện hợp lệ toàn bộ 26 cấu hình ---
```

**Result Display:**
```text
algorithm  n_clusters  silhouette_score  \
0                  K-Means           2            0.4330   
14     Agglomerative(ward)           2            0.4232   
3                  K-Means           5            0.3172   
1                  K-Means           3            0.3375   
5                  K-Means           7            0.3095   
2                  K-Means           4            0.3381   
4                  K-Means           6            0.3141   
15     Agglomerative(ward)           3            0.3146   
6                  K-Means           8            0.3012   
25  DBSCAN(eps=0.5,min=10)           2            0.2961   
22   DBSCAN(eps=0.5,min=5)           2            0.2946   
7                      GMM           2            0.2874   
8                      GMM           3            0.2562   
19     Agglomerative(ward)           7            0.2489   
16     Agglomerative(ward)           4            0.2428   
18     Agglomerative(ward)           6            0.2447   
20     Agglomerative(ward)           8            0.2252   
17     Agglomerative(ward)           5            0.2385   
9                      GMM           4            0.1752   
10                     GMM           5            0.1520   
11                     GMM           6            0.1164   
13                     GMM           8            0.0681   
12                     GMM           7            0.1019   
21   DBSCAN(eps=0.3,min=5)           8            0.0632   
23   DBSCAN(eps=0.7,min=5)           2            0.5659   
24   DBSCAN(eps=1.0,min=5)           1               NaN   

    davies_bouldin_score  calinski_harabasz_score  noise_ratio  \
0                 0.8917                4364.6091         0.00   
14                0.9064                4227.3723         0.00   
3                 0.9851                3194.6602         0.00   
1                 1.0462                3626.2766         0.00   
5                 0.9692                2960.3706         0.00   
2                 1.0131                3329.6842         0.00   
4                 1.0106                3083.6138         0.00   
15                1.1509                3040.2604         0.00   
6                 0.9944                2824.3335         0.00   
25                1.0613                2416.0711         1.78   
22                1.0626                2408.3810         1.32   
7                 1.0662                2307.4172         0.00   
8                 1.2157                2749.8992         0.00   
19                1.1215                2405.8703         0.00   
16                1.2255                2780.0958         0.00   
18                1.1518                2439.1744         0.00   
20                1.0852                2281.4319         0.00   
17                1.2431                2551.4344         0.00   
9                 1.7079                2167.5913         0.00   
10                1.7655                1880.2369         0.00   
11                2.2668                1454.5443         0.00   
13                2.2984                1279.1170         0.00   
12                2.5613                1271.3215         0.00   
21                1.5042                 837.2949         5.70   
23                0.3483                  49.9624         0.55   
24                   NaN                      NaN         0.23   

    max_cluster_pct  is_eligible  \
0             61.65         True   
14            60.18         True   
3             27.57         True   
1             43.22         True   
5             20.51         True   
2             37.75         True   
4             22.82         True   
15            60.18         True   
6             19.54         True   
25            63.80         True   
22            64.17         True   
7             65.27         True   
8             45.34         True   
19            22.63         True   
16            37.54         True   
18            22.63 
... [Output truncated for review readability; see full CSV] ...
```

**Output (stdout):**
```text
--- Hồ sơ phân khúc khách hàng ---
```

**Result Display:**
```text
Cluster  CustomerCount  CustomerPercentage  MeanRecency  MedianRecency  \
0        0           1662               38.35        25.82           16.0   
1        1           2672               61.65       134.30           96.0   

   MeanFrequency  MedianFrequency  MeanMonetary  MedianMonetary  \
0           8.40              6.0       4464.19         2041.33   
1           1.66              1.0        493.17          356.92   

   TotalMonetary    BusinessName  
0     7419490.14  Best Customers  
1     1317737.50  Lost Customers
```

## 5. Trực quan hóa Khách hàng & Lưu kết quả

```python
# [Cell 13 - Execution Count: 8]
plot_cluster_distribution(best_labels, len(df), name_map, best_algo, best_k)
```

**Result Display:**
```text
<Figure size 1000x600 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 14 - Execution Count: 9]
plot_cluster_rfm_boxplots(df, best_labels, name_map, rfm_cols, best_algo, best_k)
```

**Result Display:**
```text
<Figure size 2000x700 with 3 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 15 - Execution Count: 10]
pca_df, pca_model = pca_2d(X_scaled, best_labels)
plot_pca_2d(pca_df, pca_model, name_map, best_algo, best_k)
```

**Result Display:**
```text
<Figure size 1200x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 16 - Execution Count: 11]
plot_rfm_heatmap(profiles, best_k)
```

**Result Display:**
```text
<Figure size 1000x600 with 2 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 17 - Execution Count: 12]
# Save outputs matching runner exact process
comparison_df.to_csv(TABLES_CLUSTERING / 'clustering_algorithm_comparison.csv', index=False)
profiles.to_csv(TABLES_CLUSTERING / 'cluster_profiles.csv', index=False)

df_result = df[['CustomerID'] + rfm_cols].copy()
df_result['Cluster'] = best_labels
df_result['Algorithm'] = best_algo
df_result['FeatureVersion'] = 'log1p_scaled'
df_result['BusinessSegment'] = df_result['Cluster'].map(name_map)
df_result.to_csv(PROCESSED_DIR / 'customer_clusters.csv', index=False)

config = {
    'algorithm': best_algo, 'n_clusters': best_k, 'feature_names': feature_names,
    'feature_version': 'log1p_scaled', 'metrics': best_metrics, 'n_customers': len(df)
}
save_clustering_results(best_model, scaler, best_labels, config, MODELS_CLUSTERING_DIR)
print("Saved all results successfully.")
```

**Output (stdout):**
```text
[OK] Model saved to D:\Project\Data-mininng\models\clustering
Saved all results successfully.
```

