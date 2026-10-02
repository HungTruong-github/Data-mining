# Online Retail Data Mining — Review Bundle

**Dự án Khai phá Dữ liệu Khách hàng Bán lẻ Trực tuyến (Online Retail)**
*Bộ hồ sơ minh chứng kỹ thuật & thực nghiệm khoa học chuẩn CRISP-DM*

---

## 1. Thông Tin Phiên Bản & Môi Trường Thực Thi
- **Repository:** [https://github.com/HungTruong-github/Data-mining](https://github.com/HungTruong-github/Data-mining)
- **Branch:** `feature-insights`
- **Git Commit:** `adb1b851f474400c68d37dbcd40575b99fc107d0`
- **Thời điểm đóng gói:** `2026-10-02 21:38:20`
- **Trạng thái kiểm thử:** **39/39 tests PASSED (100%)**
- **Trạng thái pipeline:** Hoàn tất tuần tự 7 bước trong **495.73 giây** (Exit code: 0)
- **Môi trường Python:** Python 3.12 (Windows 64-bit)

---

## 2. Hướng Dẫn Thứ Tự Đọc & Kiểm Tra (Review Walkthrough)

Người review vui lòng mở theo thứ tự khuyến nghị sau:

1. **`README_REVIEW.md` (file này):** Nắm tổng quan kiến trúc, quyết định mô hình và bằng chứng kỹ thuật.
2. **`reports/07_model_comparison_and_insights.md`:** Báo cáo tổng hợp toàn diện theo quy trình CRISP-DM, chứa đầy đủ bảng so sánh, hồ sơ phân khúc và kế hoạch hành động.
3. **`validation/acceptance_results.csv` & `pytest_report.xml`:** Xem kết quả kiểm chứng tự động cho 8 tiêu chí kiểm định kỹ thuật then chốt và 39 unit/integration test cases.
4. **`tables/model_comparison/` & `tables/insights/`:** Xem dữ liệu thực nghiệm thực tế (bảng so sánh 26 cấu hình phân cụm, so sánh 4 mô hình phân loại, 61 luật kết hợp và kế hoạch hành động 10 cột).
5. **`notebooks_review/`:** Đọc nội dung 7 notebook dạng Markdown nhẹ đã được thực thi và hiển thị sẵn toàn bộ mã nguồn, số liệu và output văn bản (không nhúng base64 nặng).
6. **`figures/`:** Xem biểu đồ trực quan hóa được sinh trực tiếp từ mã nguồn thực tế.

---

## 3. Trạng Thái Chi Tiết Từng Bước (Steps 01–07)

| Bước | Tên quy trình | Trạng thái | Thời gian | Artifacts then chốt |
|:---:|:---|:---:|:---:|:---|
| **01** | Data Understanding | **PASS** | 98.66s | `tables/data_understanding/descriptive_statistics.csv`, `raw_summary.csv` |
| **02** | EDA & Cleaning | **PASS** | 185.38s | `tables/data_preparation/cleaning_summary.csv`, `outlier_summary.csv` |
| **03** | Feature Engineering RFM | **PASS** | 120.91s | `tables/customer_features/rfm_customer_features.csv`, `repeat_purchase_features.csv` |
| **04** | Customer Clustering | **PASS** | 30.81s | `tables/clustering/clustering_algorithm_comparison.csv` (26 configs), `cluster_profiles.csv` |
| **05** | Repeat Purchase Classification | **PASS** | 21.97s | `tables/classification/model_comparison.csv`, `test_predictions.csv` (có CustomerID) |
| **06** | Association Rules | **PASS** | 29.98s | `tables/association_rules/selected_association_rules.csv`, `association_rules_equivalence_audit.csv` |
| **07** | Model Comparison & Insights | **PASS** | 10.48s | `tables/insights/customer_segment_action_plan.csv`, `reports/pipeline_manifest.json` |

---

## 4. Các Lỗi Đã Sửa và Bằng Chứng Tương Ứng

1. **Loại bỏ hàm cấp module bị định nghĩa trùng lặp trong Runner 07:**
   - *Vấn đề:* `run_07_model_comparison_and_insights.py` trước đây bị định nghĩa 2 lần `main()`, `_save()`, `_generate_report()`, v.v. khiến phần sau ghi đè phần trước.
   - *Khắc phục:* Cắt bỏ toàn bộ khối trùng lặp, chuẩn hóa việc ủy quyền cho module gốc `src.model_comparison`.
   - *Minh chứng:* Test `test_no_duplicate_function_definitions_in_modules` kiểm tra AST toàn bộ codebase và đạt PASS (xem `validation/acceptance_results.csv`).
2. **Kiểm tra tương đương nội dung luật kết hợp trong luồng chạy thực tế:**
   - *Vấn đề:* `compare_rule_sets()` trước đây chỉ được gọi trong test, chưa tích hợp vào pipeline thực tế; việc chỉ so sánh số lượng luật là chưa đủ.
   - *Khắc phục:* `run_06_association_rules.py` và Notebook 06 hiện trực tiếp gọi `compare_rule_sets()`, chuẩn hóa StockCode và đối chiếu metric với tolerance $10^{-5}$.
   - *Minh chứng:* Xuất bảng `tables/association_rules/association_rules_equivalence_audit.csv` xác nhận 61/61 luật khớp hoàn toàn giữa Apriori và FP-Growth; bổ sung integration test phát hiện lệch nội dung dù cùng số lượng luật.
3. **Sửa lỗi fallback quantile binning của RFM:**
   - *Vấn đề:* Khi `pd.qcut` gặp quantiles trùng lặp, nhánh fallback trước đây đảo chiều điểm số Recency.
   - *Khắc phục:* Viết lại `_safe_qcut()` trên các mốc phân vị duy nhất (`np.unique`), đảm bảo tính đơn điệu (Recency nhỏ -> điểm cao) và bất biến thứ tự dòng (row-order invariance).
   - *Minh chứng:* 3 unit tests chuyên biệt trong `tests/test_feature_engineering.py` đạt 100% PASS.
4. **Bổ sung `CustomerID` vào file kết quả kiểm tra `test_predictions.csv`:**
   - *Vấn đề:* File dự đoán test bị mất mã định danh khách hàng, không thể truy vết lỗi dự báo.
   - *Khắc phục:* Giữ nguyên khóa `CustomerID` ở cột đầu tiên của `test_predictions.csv`.
   - *Minh chứng:* `tables/classification/test_predictions.csv` có cột `CustomerID` ở vị trí index 0.
5. **Đồng bộ hóa Streamlit Dashboard & Áp dụng Threshold tùy chỉnh:**
   - *Vấn đề:* App chỉ gọi `clf_pipeline.predict()` mặc định (ngưỡng 0.5 cố định) mà không áp dụng threshold từ metadata.
   - *Khắc phục:* App trích xuất xác suất `predict_proba()[:, 1]` và phân loại dựa trên `(probs >= threshold).astype(int)`. Đồng thời banner cảnh báo phương pháp luận được sinh động từ metadata.
   - *Minh chứng:* 4 unit/smoke tests trong `tests/test_app_smoke.py` kiểm tra luồng dự đoán, ngưỡng phân loại và kiểm soát lỗi đầu vào.
6. **Bảng Kế hoạch Hành động (Action Plan) đầy đủ 10 cột:**
   - *Vấn đề:* Bước 07 gọi hàm thiếu tham số bảng phân khúc và thiếu các trường định hướng đo lường.
   - *Khắc phục:* Xây dựng Action Plan chuẩn 10 cột nghiệp vụ gắn liền KPI và phương pháp kiểm chứng (A/B testing).
   - *Minh chứng:* `tables/insights/customer_segment_action_plan.csv`.

---

## 5. Căn Cứ Lựa Chọn Mô Hình, Siêu Tham Số và Cấu Hình Tối Ưu

### A. Phân cụm Khách hàng (Customer Clustering)
- **Không gian khảo sát:** 26 cấu hình gồm K-Means (K=2..8), GMM (K=2..8), Agglomerative linkage='ward' (K=2..8), và DBSCAN với các cặp (eps in [0.3, 0.5, 0.7, 1.0], min_samples in [5, 10]).
- **Thuật toán & Cấu hình được chọn:** **K-Means với K=2**.
- **Căn cứ thực nghiệm:**
  - K-Means K=2 đứng đầu bảng xếp hạng tổng hợp (Combined Rank = 6.0), đạt **Silhouette Score = 0.4330**, **Davies-Bouldin = 0.8917**, và **Calinski-Harabasz = 4,364.6**.
  - Các cấu hình K=3..8 có Silhouette giảm mạnh (K=3: 0.3375, K=4: 0.3381).
  - DBSCAN dù có Silhouette cao trên một số cấu hình nhưng bị hệ thống tự động loại bỏ (is_eligible=False) do tạo ra **cụm suy biến** (ví dụ eps=0.7 gom 99.33% khách hàng vào cụm 0, cụm 1 chỉ có 5 khách hàng).
- **Hạn chế & Đánh đổi:** K=2 là phân khúc vĩ mô (Khách hàng giá trị cao vs Khách hàng giá trị thấp/rủi ro), rất trực quan cho ban lãnh đạo nhưng chưa bóc tách sâu các nhóm hành vi nhỏ hơn.

### B. Phân loại Mua lại trong 90 ngày (Repeat Purchase Classification)
- **Protocol:** Tách tập holdout test (20% = 674 khách hàng) độc lập; thực hiện Stratified 5-Fold Cross-Validation trên tập huấn luyện (80% = 2,694 khách hàng).
- **Mô hình được chọn:** **Random Forest (`n_estimators=100, max_depth=5, class_weight='balanced'`)**.
- **Báo cáo trung thực về Baseline vs Learned Model:**
  - Tỷ lệ lớp dương trong tập dữ liệu là **58.4%**. Do đó, mô hình cơ sở Dummy Classifier dự đoán toàn bộ là 1 sẽ đạt F1 = 0.7259.
  - Tuy nhiên, Dummy Classifier có **ROC-AUC = 0.5000** (hoàn toàn không có khả năng phân biệt khách hàng nào sẽ mua lại và khách hàng nào sẽ rời bỏ).
  - **Random Forest** đạt **CV F1 = 0.6726**, **Test F1 = 0.7086**, **Test ROC-AUC = 0.7438**, và **Test Average Precision = 0.7938**, là mô hình học máy tốt nhất giúp doanh nghiệp xếp hạng xác suất ưu tiên nguồn lực marketing.
- **Top 5 đặc trưng quan trọng:** `Recency` (13.9%), `Monetary` (13.6%), `UniqueProducts` (13.0%), `TotalItems` (12.8%), `AverageOrderValue` (12.4%).

### C. Khai phá Luật Kết Hợp (Association Rules)
- **Tập giao dịch:** 13,369 giỏ hàng tại thị trường UK.
- **Ngưỡng thiết lập:** `min_support = 0.02` (tương đương tối thiểu $pprox 267$ hóa đơn) và `min_confidence = 0.5`.
- **So sánh Apriori vs FP-Growth:**
  - Cả hai thuật toán sinh ra tập 61 luật kết hợp hoàn toàn trùng khớp 100% về mặt nội dung (đã kiểm chứng qua `compare_rule_sets()`).
  - Hệ thống tự động chọn thuật toán có thời gian chạy tối ưu trên tập dữ liệu thực tế.
  - **Top luật tiêu biểu:** Bộ ba sản phẩm ấm chén Regency Teacup đạt $Lift = 18.23$, hỗ trợ xây dựng combo bán chéo hoặc hiển thị gợi ý sản phẩm liên quan.

---

## 6. Danh Sách Các File Lớn Được Loại Khỏi Bundle & Cách Tái Tạo

Nhằm đảm bảo dung lượng gói review dưới 20 MB (thực tế gói ZIP khoảng **12–15 MB**), các file sau được loại bỏ có chủ đích:
1. `data/raw/Online Retail.xlsx` (22.6 MB): Tải từ UCI ML Repository và đặt vào `data/raw/`.
2. `data/interim/cleaned_transactions.csv` & `product_transactions.csv` (72.2 MB mỗi file): Được sinh tự động bởi bước 02.
3. `models/classification/best_classifier_pipeline.joblib` (20.7 MB): File nhị phân model huấn luyện, được sinh tự động bởi bước 05. (Lưu ý: toàn bộ metadata, cấu hình tham số, kết quả cross-validation và file dự đoán `test_predictions.csv` vẫn được giữ nguyên vẹn trong bundle).

---

## 7. Các Lệnh Tái Lập Toàn Bộ Dự Án (Reproduction Commands)

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv
.venv\Scriptsctivate   # Trên Windows

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Chạy toàn bộ 39 bài kiểm thử tự động
pytest -v

# 4. Chạy toàn bộ pipeline từ bước 01 đến 07
python run_pipeline.py

# 5. Khởi chạy Dashboard tương tác
streamlit run app/app.py
```
