# Decision Log — Online Retail Data Mining

Mỗi quyết định phân tích quan trọng được ghi lại với: ID, vấn đề, phương án, lý do, evidence, mã nguồn thực thi.

## Preprocessing Decisions

| ID | Quyết định | Phương án chọn | Phương án loại | Lý do | Evidence |
|---|---|---|---|---|---|
| D01 | Xử lý missing CustomerID | Loại khỏi aggregate cấp khách hàng, giữ cho phân tích invoice/product | Impute bằng mode, tạo ID giả | 24.93% missing — impute sẽ tạo khách hàng ảo; giữ cho EDA sản phẩm vì vẫn là giao dịch thật | NB02 cleaning_summary.csv |
| D02 | Xử lý missing Description | Map theo StockCode (mode), còn lại → "Unknown Product" | Loại toàn bộ dòng thiếu | Chỉ 0.27% missing; mapping phục hồi được phần lớn | NB02 |
| D03 | Exact duplicate | Loại bỏ duplicate trên 8 cột gốc | Giữ toàn bộ | 5,268 dòng (0.97%) — giả định là ghi nhận lặp; nguyên nhân không biết chắc | NB02 |
| D04 | Cancellation (prefix C) | Loại khỏi tập mua hàng hợp lệ | Đối soát với hóa đơn gốc | Đối soát phức tạp; loại toàn bộ dòng hủy là phương pháp đơn giản, Monetary không phải net value | NB02 |
| D05 | StockCode phi sản phẩm | Loại POST, DOT, M, m, C2, D, S, BANK CHARGES, AMAZONFEE, CRUK, B | Giữ tất cả | Đây là phí/dịch vụ/điều chỉnh, không phải sản phẩm vật lý | src/config.py |
| D06 | Outlier Quantity/UnitPrice | IQR fences, loại bỏ extreme outliers | Giữ + log transform, Winsorize | Phương án đơn giản; đánh giá tác động downstream cần thực nghiệm | NB02 |

## Feature Engineering Decisions

| ID | Quyết định | Phương án chọn | Lý do | Evidence |
|---|---|---|---|---|
| D07 | Temporal split | Feature period trước cutoff, label period 90 ngày sau cutoff | Ngăn data leakage — features không nhìn thấy tương lai | NB03, test_feature_engineering.py |
| D08 | Frequency = unique InvoiceNo | Count distinct invoices, không phải dòng giao dịch | Một hóa đơn có nhiều dòng; Frequency đo số lần mua, không phải số item | NB03 |
| D09 | reference_date = last_observation + 1 day | Quy ước thời gian | Recency tối thiểu = 1 ngày; log1p(0) vẫn xác định nhưng quy ước này chuẩn hơn | NB03 |
| D10 | Basket key = StockCode | Dùng StockCode làm item identifier | Description có nhiều biến thể cho cùng StockCode; dùng Description sẽ tách sản phẩm | NB03 |

## Clustering Decisions

| ID | Quyết định | Phương án chọn | Lý do | Evidence |
|---|---|---|---|---|
| D11 | Preprocessing cho clustering | log1p(RFM) → StandardScaler | RFM phân phối lệch phải; log giảm skew, StandardScaler chuẩn hóa khoảng cách | NB04, clustering comparison |
| D12 | Thuật toán so sánh | K-Means, GMM, Agglomerative(Ward), DBSCAN | 4 thuật toán đại diện: partition, mixture model, hierarchical, density-based | NB04 |
| D13 | Chọn K | Dựa trên Silhouette, Davies-Bouldin, Calinski-Harabasz + cluster size distribution | Không có nhãn chuẩn → dùng internal metrics + policy nghiệp vụ (min cluster size) | NB04, clustering_algorithm_comparison.csv |

## Classification Decisions

| ID | Quyết định | Phương án chọn | Lý do | Evidence |
|---|---|---|---|---|
| D14 | Baseline | DummyClassifier (most_frequent) | Đo giá trị thực sự của learned model; dummy không bị che giấu | NB05, model_comparison.csv |
| D15 | Model candidates | Logistic Regression, Decision Tree, Random Forest + Dummy | Từ đơn giản → phức tạp; LR interpretable, DT/RF non-linear | NB05 |
| D16 | Model selection | Chọn theo CV F1-score trên train, KHÔNG dùng test set | Tránh data leakage trong selection; test chỉ dùng đánh giá cuối | NB05, cv_results.csv |
| D17 | Threshold | 0.5 (default) | Chưa có chi phí marketing/biên lợi nhuận để tối ưu threshold nghiệp vụ | classification_metadata.json |
| D18 | class_weight | balanced | Bù đắp imbalance trong training; được đánh giá qua CV | classification_metadata.json |

## Association Rules Decisions

| ID | Quyết định | Phương án chọn | Lý do | Evidence |
|---|---|---|---|---|
| D19 | Thuật toán | Apriori và FP-Growth cùng tham số, so sánh runtime và kết quả | Cùng bài toán frequent itemsets; kiểm chứng equivalence | NB06 |
| D20 | min_support | 0.02 | Tương đương ≈ 267 hóa đơn (trên 13,369); sensitivity cần thử 0.01/0.03 | NB06 |
| D21 | min_confidence | 0.5 | Ngưỡng phổ biến; chỉ giữ rules có confidence ≥ 50% | NB06 |
| D22 | lift > 1 filter | Chỉ giữ rules có lift > 1 | lift = 1 nghĩa là antecedent/consequent độc lập → không có mối liên hệ | NB06 |