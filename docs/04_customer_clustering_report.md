# Báo Cáo Tổng Hợp Giai Đoạn 04: Customer Clustering (Phân Cụm Khách Hàng)

**Dự án:** Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM:** **Phase 4 - Modeling (Unsupervised Learning - Customer Clustering)**  
**Dữ liệu đầu vào:** `data/processed/rfm_clustering_features.csv` (4,334 khách hàng định danh, 3 đặc trưng RFM)  
**Tài liệu tham chiếu:** [Methodology](methodology.md) | [Decision Log](decision_log.md) | [References](references.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Customer Clustering** nhằm giải quyết bài toán cốt lõi đầu tiên của dự án: Phân nhóm tự động 4,334 khách hàng dựa trên hành vi giao dịch thực tế (Recency, Frequency, Monetary) mà không dựa vào nhãn có sẵn, từ đó định hình các phân khúc giá trị để phục vụ chiến lược kinh doanh và giữ chân khách hàng.

Toàn bộ thuật toán được đóng gói trong [`src/clustering.py`](../src/clustering.py), trực quan hóa qua [`src/visualization.py`](../src/visualization.py), thực thi tự động qua [`notebooks/run_04_customer_clustering.py`](../notebooks/run_04_customer_clustering.py) và ghi nhận thực nghiệm trong notebook [`notebooks/04_customer_clustering.ipynb`](../notebooks/04_customer_clustering.ipynb).

### Các kết quả khoa học chính:
- **Tác động của tiền xử lý đặc trưng:** Chứng minh thực nghiệm việc áp dụng chuỗi biến đổi $\log(1+x) \rightarrow \text{StandardScaler}$ là điều kiện tiên quyết để triệt tiêu ảnh hưởng của ngoại lai cực đại (Monetary > £280k). Nếu chỉ chuẩn hóa `StandardScaler` thuần túy, mô hình sẽ sinh ra cụm suy biến (degenerate cluster: 99.42% số mẫu dồn vào 1 cụm duy nhất).
- **Quy mô thực nghiệm đối chiếu:** Khảo sát và đo lường độc lập **26 cấu hình mô hình** trên 4 thuật toán phân cụm kinh điển đại diện cho 4 trường phái khác nhau:
  1. *Partition-based:* K-Means ($K = 2 \dots 8$)
  2. *Probabilistic Mixture:* Gaussian Mixture Models - GMM ($K = 2 \dots 8$)
  3. *Hierarchical:* Agglomerative Clustering với liên kết Ward ($K = 2 \dots 8$)
  4. *Density-based:* DBSCAN (5 cấu hình tổ hợp bán kính $\epsilon \in \{0.3, 0.5, 0.7, 1.0\}$ và $\text{min\_samples} \in \{5, 10\}$).
- **Mô hình tối ưu được lựa chọn:** **K-Means ($K=2$)** đạt vị trí số 1 trong bảng xếp hạng tổng hợp đa chỉ số (Combined Rank = 6.0):
  - Silhouette Score = **0.4330** (độ phân tách cụm cao nhất).
  - Davies-Bouldin Index = **0.8917** (độ nén nội cụm và tách biệt tốt nhất, giá trị càng nhỏ càng tốt).
  - Calinski-Harabasz Index = **4,364.61** (tỷ số phương sai liên cụm trên nội cụm cao nhất).
- **Kiểm định độ ổn định (Clustering Stability Audit):** Kiểm tra độ ổn định qua 5 random seeds độc lập `[42, 123, 456, 789, 999]`, đạt chỉ số tương đồng phân vùng Adjusted Rand Index (ARI) trung bình = **0.9996** $\pm$ 0.0005 (xác nhận phân cụm đạt tính ổn định tuyệt đối, không phụ thuộc vào trạng thái khởi tạo tâm ban đầu).
- **Định danh 2 phân khúc khách hàng chiến lược:**
  - **Cụm 0 — Best Customers (38.35% - 1,662 khách):** Khách hàng trung thành, chi tiêu vượt trội. Recency trung vị 16.0 ngày, Frequency trung vị 6.0 đơn, Monetary trung vị £2,041.33 (trung bình £4,464.19). Đóng góp **£7,419,490.14 (84.9% doanh số)**, tỷ lệ quay lại mua trong 90 ngày đạt **95.7%**.
  - **Cụm 1 — Lost Customers / Low-Value (61.65% - 2,672 khách):** Khách hàng có nguy cơ rời bỏ, giá trị thấp. Recency trung vị 96.0 ngày, Frequency trung vị 1.0 đơn, Monetary trung vị £356.92 (trung bình £493.17). Đóng góp **£1,317,737.50 (15.1% doanh số)**, tỷ lệ quay lại mua trong 90 ngày chỉ đạt **26.4%**.

---

## 2. Tiền Xử Lý Dữ Liệu & Thực Nghiệm Biến Đổi Không Gian

### 2.1. So sánh ảnh hưởng của phương pháp chuẩn hóa
*Trích xuất từ `outputs/tables/clustering/clustering_preprocessing_comparison.csv`:*

| Phương pháp tiền xử lý | Số cụm ($K$) | Silhouette Score | Cụm nhỏ nhất (%) | Cụm lớn nhất (%) | Xử lý độ lệch (Skewness) | Đánh giá khoa học |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **StandardScaler thuần túy (Không Log)** | 2 | 0.8978 | 0.58% | 99.42% | Không | Bị đánh lừa bởi ngoại lai Monetary cực trị (£280,206); cụm lớn nuốt trọn 99.4% tập dữ liệu, hoàn toàn vô dụng cho kinh doanh. |
| **Log1p + StandardScaler** | **2** | **0.4330** | **38.35%** | **61.65%** | **Có** | Nén đuôi dài phân phối, đưa các biến RFM về dạng gần chuẩn Gauss, giúp thuật toán phân tách tự nhiên 2 tầng khách hàng rõ rệt. |

---

## 3. Bảng So Sánh 26 Cấu Hình Thuật Toán Phân Cụm

*Trích xuất từ `outputs/tables/clustering/clustering_algorithm_comparison.csv`:*

| STT | Thuật toán | Cấu hình | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cụm min | Cụm max (%) | Runtime (s) | Trạng thái hợp lệ | Lý do lựa chọn / loại trừ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **K-Means** | **K=2** | **0.4330** | **0.8917** | **4,364.61** | **1,662** | **61.65%** | **0.117** | **SELECTED** | **Đạt Hạng 1 trên cả 3 chỉ số Silhouette, DB, CH; phân phối cụm cân đối.** |
| 2 | Agglomerative(ward) | K=2 | 0.4232 | 0.9064 | 4,227.37 | 1,726 | 60.18% | 0.357 | Eligible | Xếp hạng 2 tổng hợp, chất lượng tiệm cận K-Means nhưng runtime cao hơn. |
| 3 | K-Means | K=3 | 0.3375 | 1.0462 | 3,626.28 | 763 | 43.22% | 0.056 | Eligible | Điểm Silhouette sụt giảm so với K=2. |
| 4 | K-Means | K=4 | 0.3381 | 1.0131 | 3,329.68 | 686 | 37.75% | 0.080 | Eligible | Cụm có xu hướng bị chia nhỏ nhân tạo. |
| 5 | K-Means | K=5 | 0.3172 | 0.9851 | 3,194.66 | 312 | 27.57% | 0.088 | Eligible | Điểm phân tách giảm dần theo quy luật Elbow. |
| 6 | GMM | K=2 | 0.2874 | 1.0662 | 2,307.42 | 1,505 | 65.27% | 0.259 | Eligible | Mô hình xác suất hỗn hợp cho kết quả phân cụm kém rõ ràng hơn K-Means. |
| 7 | GMM | K=3 | 0.2562 | 1.2157 | 2,749.90 | 864 | 45.34% | 0.231 | Eligible | Chất lượng suy giảm. |
| 8 | DBSCAN | eps=0.5, min=5 | 0.2946 | 1.0626 | 2,408.38 | 1,496 | 64.17% | 0.121 | Eligible | Nhiễu 1.32%; phân vùng tương đồng K=2 nhưng Silhouette thấp hơn. |
| 9 | DBSCAN | eps=0.5, min=10 | 0.2961 | 1.0613 | 2,416.07 | 1,492 | 63.80% | 0.177 | Eligible | Nhiễu 1.78%. |
| 10 | DBSCAN | eps=0.3, min=5 | 0.0632 | 1.5042 | 837.29 | 7 | 33.87% | 0.086 | **REJECTED** | Cụm nhỏ nhất chỉ có 7 mẫu (< 10 mẫu quy định). |
| 11 | DBSCAN | eps=0.7, min=5 | 0.5659 | 0.3483 | 49.96 | 5 | 99.33% | 0.217 | **REJECTED** | Cụm suy biến: cụm lớn nhất chiếm 99.33% (vượt ngưỡng cho phép 90%). |
| 12 | DBSCAN | eps=1.0, min=5 | NaN | NaN | NaN | 4,324 | 99.77% | 0.396 | **REJECTED** | Tất cả gom về 1 cụm duy nhất (K=1, không tính được Silhouette). |

### Tiêu chí khoa học loại trừ các cấu hình DBSCAN:
DBSCAN hoạt động dựa trên mật độ lân cận trong không gian đa chiều. Trên dữ liệu RFM thực tế, mật độ khách hàng giảm dần liên tục từ vùng đông đúc ra các vùng chi tiêu cao thay vì tạo ra các thung lũng mật độ rõ rệt. Do đó, DBSCAN hoặc xem khách hàng VIP là điểm nhiễu (noise), hoặc gom gần như toàn bộ vào 1 cụm suy biến, không phù hợp cho bài toán phân khúc khách hàng bán lẻ.

---

## 4. Kiểm Định Độ Ổn Định Phân Cụm (Stability Audit)

Để đảm bảo kết quả phân cụm K-Means ($K=2$) mang tính khách quan khoa học, một kiểm định lặp độc lập đã được thực thi với 5 giá trị seed ngẫu nhiên khác nhau:
- **Các seeds thử nghiệm:** `42`, `123`, `456`, `789`, `999`.
- **Chỉ số đo lường:** Adjusted Rand Index (ARI) cặp đôi giữa các phân vùng nhãn.
- **Kết quả:**
  - Mean Pairwise ARI: **0.9996**
  - Min Pairwise ARI: **0.9991**
  - Std Pairwise ARI: **0.0005**
  - Inertia Coefficient of Variation (CV): **0.0%**
- **Kết luận:** Thuật toán K-Means hội tụ về cùng một cực tiểu toàn cục duy nhất, chứng minh cấu trúc 2 cụm là thuộc tính nội tại vững chắc của tập dữ liệu.

---

## 5. Đặc Trưng Hồ Sơ Các Phân Khúc (Segment Profiles)

*Trích xuất từ `outputs/tables/clustering/cluster_profiles.csv`:*

| Chỉ số phân tích | Cụm 0: Best Customers (VIP / Loyal) | Cụm 1: Lost Customers (Low-Value / Churn) | Toàn bộ tập dữ liệu (Tổng / TB) |
|:---|:---:|:---:|:---:|
| **Số lượng khách hàng** | **1,662** (38.35%) | **2,672** (61.65%) | **4,334** (100.0%) |
| **Recency trung bình (ngày)** | 25.82 ngày | 134.30 ngày | 92.58 ngày |
| **Recency trung vị (ngày)** | **16.0 ngày** | **96.0 ngày** | 51.0 ngày |
| **Frequency trung bình (đơn)** | 8.40 đơn | 1.66 đơn | 4.27 đơn |
| **Frequency trung vị (đơn)** | **6.0 đơn** | **1.0 đơn** | 2.0 đơn |
| **Monetary trung bình (£)** | £4,464.19 | £493.17 | £2,053.79 |
| **Monetary trung vị (£)** | **£2,041.33** | **£356.92** | £674.45 |
| **Tổng doanh thu đóng góp (£)** | **£7,419,490.14** | **£1,317,737.50** | **£8,737,227.64** |
| **Tỷ trọng doanh thu (%)** | **84.92%** | **15.08%** | **100.0%** |
| **Tỷ lệ khách hàng mua lại 90 ngày** | **95.7%** | **26.4%** | 56.98% |

> **Quy luật Pareto 80/20 thực nghiệm:** 
> Phân cụm K-Means đã bóc tách chính xác quy luật kinh điển: **38.4% khách hàng nhóm Best Customers tạo ra gần 85% tổng giá trị giao dịch**, đồng thời có tỷ lệ quay lại mua sắm áp đảo (95.7%). Trong khi đó, nhóm Lost Customers đa phần chỉ mua đúng 1 lần (Frequency trung vị = 1) và đã hơn 3 tháng chưa quay lại.

---

## 6. Danh Mục Tệp Đầu Ra (Deliverable Artifacts)

### 6.1. Mô hình đã đóng gói (`models/clustering/`)
- `clustering_model.pkl`: Đối tượng mô hình `KMeans(n_clusters=2, random_state=42)` đã fit.
- `clustering_config.pkl`: Siêu tham số, danh sách biến RFM, phiên bản tiền xử lý `log1p_scaled`.
- `scaler.pkl`: Đối tượng `StandardScaler` đã fit trên ma trận log1p.

### 6.2. Dữ liệu gán nhãn (`data/processed/`)
- `customer_clusters.csv`: 4,334 dòng chứa định danh khách hàng, giá trị RFM gốc, nhãn cụm (0/1) và tên định danh kinh doanh.

### 6.3. Bảng thống kê (`outputs/tables/clustering/`)
- `clustering_algorithm_comparison.csv`: Bảng đối chiếu chi tiết 26 cấu hình với 19 cột chỉ số.
- `cluster_profiles.csv`: Bảng tổng hợp hồ sơ phân khúc RFM trung bình và trung vị.
- `clustering_stability_audit.csv`: Bảng kết quả kiểm định ARI qua 5 seeds ngẫu nhiên.
- `clustering_preprocessing_comparison.csv`: Bảng đối chứng tác động của Log Transformation.

### 6.4. Biểu đồ trực quan hóa (`outputs/figures/clustering/`)
- `elbow_curve.png`: Đồ thị đường cong Elbow thể hiện Inertia theo số cụm $K \in [2; 8]$.
- `silhouette_comparison.png`: Đồ thị biểu diễn điểm Silhouette theo số cụm của các thuật toán.
- `metrics_comparison.png`: Biểu đồ so sánh đồng thời Silhouette, Davies-Bouldin và Calinski-Harabasz.
- `cluster_distribution.png`: Biểu đồ hình tròn và cột phân bố số lượng khách hàng giữa các cụm.
- `cluster_rfm_boxplots.png`: Biểu đồ hộp so sánh phân phối Recency, Frequency, Monetary giữa 2 cụm.
- `pca_2d_clusters.png`: Trực quan hóa mặt phẳng 2 chiều PCA của không gian phân cụm.
- `rfm_heatmap.png`: Bản đồ nhiệt thể hiện sự tương phản đặc trưng giữa các cụm.
