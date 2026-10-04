# Online Retail Data Mining — Review Bundle

**Dự án Khai phá Dữ liệu Khách hàng Bán lẻ Trực tuyến (Online Retail)**
*Bộ hồ sơ minh chứng kỹ thuật & thực nghiệm khoa học chuẩn CRISP-DM*

---

## 1. Thông Tin Phiên Bản & Môi Trường Thực Thi
- **Repository:** [https://github.com/HungTruong-github/Data-mining](https://github.com/HungTruong-github/Data-mining)
- **Branch:** `clean`
- **Git Commit:** `b899caae5d9a9a468f8905c3661003c28a2bba0f`
- **Thời điểm đóng gói:** `2026-10-04 23:46:15`
- **Trạng thái kiểm thử:** **47/47 tests PASSED (100% success rate, 0 failed)**
- **Trạng thái pipeline:** Hoàn tất tuần tự 7 bước (Exit code: 0, `pipeline_manifest.json`: PASS)
- **Môi trường Python:** Python 3.12 (Windows 64-bit)

---

## 2. Hướng Dẫn Thứ Tự Đọc & Kiểm Tra (Review Walkthrough)

1. **`README_REVIEW.md` (file này):** Nắm tổng quan kiến trúc, quyết định mô hình và bằng chứng kỹ thuật.
2. **`reports/07_model_comparison_and_insights.md`:** Báo cáo tổng hợp toàn diện theo quy trình CRISP-DM, chứa đầy đủ bảng so sánh, hồ sơ phân khúc và kế hoạch hành động.
3. **`validation/acceptance_results.csv` & `pytest_report.xml`:** Xem kết quả kiểm chứng tự động cho 12 tiêu chí kiểm định kỹ thuật then chốt và 47 unit/integration test cases.
4. **`tables/model_comparison/` & `tables/insights/`:** Xem dữ liệu thực nghiệm thực tế (bảng so sánh 26 cấu hình phân cụm, so sánh 4 mô hình phân loại, tuning history, 61 luật kết hợp và kế hoạch hành động 10 cột).
5. **`notebooks_review/`:** Đọc nội dung 7 notebook dạng Markdown nhẹ đã được thực thi và hiển thị sẵn toàn bộ mã nguồn, số liệu và output văn bản (không nhúng base64 nặng).
6. **`figures/`:** Xem biểu đồ trực quan hóa được sinh trực tiếp từ mã nguồn thực tế.

---

## 3. Trạng Thái Chi Tiết Từng Bước (Steps 01–07)

| Bước | Tên quy trình | Trạng thái | Artifacts then chốt |
|:---:|:---|:---:|:---|
| **01** | Data Understanding | **PASS** | `tables/data_understanding/descriptive_statistics.csv`, `raw_summary.csv` |
| **02** | EDA & Cleaning | **PASS** | `tables/data_preparation/cleaning_summary.csv`, `outlier_summary.csv` |
| **03** | Feature Engineering RFM | **PASS** | `tables/customer_features/rfm_customer_features.csv`, `repeat_purchase_features.csv` |
| **04** | Customer Clustering | **PASS** | `tables/clustering/clustering_algorithm_comparison.csv` (26 configs), `clustering_stability_audit.csv`, `clustering_preprocessing_comparison.csv` |
| **05** | Repeat Purchase Classification | **PASS** | `tables/classification/model_comparison.csv`, `cv_tuning_history.csv`, `test_predictions.csv` |
| **06** | Association Rules | **PASS** | `tables/association_rules/selected_association_rules.csv`, `association_rules_equivalence_audit.csv`, `rule_sensitivity_analysis.csv` |
| **07** | Model Comparison & Insights | **PASS** | `tables/insights/customer_segment_action_plan.csv`, `reports/pipeline_manifest.json` |

---

## 4. Các Lỗi Đã Sửa và Bằng Chứng Tương Ứng

1. **Thống nhất logic đánh giá phân lớp và Dashboard (`predict_with_threshold`):**
   - *Vấn đề:* Đánh giá offline dùng `predict()` còn dashboard dùng `prob >= threshold`. Tại biên xác suất 0.5, hai cách trả nhãn khác nhau.
   - *Khắc phục:* Xây dựng hàm dùng chung `predict_with_threshold()` xác định cột lớp dương động từ `classes_`, áp dụng toán tử không làm tròn `prob >= threshold`. Áp dụng đồng nhất cho offline metrics, confusion matrix, classification report, test predictions và app dashboard.
   - *Minh chứng:* Test `test_offline_dashboard_parity` kiểm chứng độ khớp nhãn 100% trên toàn bộ tập test (674 khách hàng) và tại điểm biên.

2. **Thực nghiệm siêu tham số có giới hạn trên tập train (`cv_tuning_history.csv`):**
   - *Vấn đề:* Trước đây các siêu tham số được đặt cố định, báo cáo gọi Random Forest là tốt nhất mà chưa so sánh đầy đủ với Logistic Regression (có ROC-AUC cao hơn) và DummyClassifier (có F1 cao do mất cân bằng).
   - *Khắc phục:* Triển khai `tune_model_hyperparameters()` với 5-fold CV GridSearch trên tập train cho Logistic Regression, Decision Tree và Random Forest. Ghi nhận `cv_tuning_history.csv` và giải thích tường minh lý do chọn RF theo CV F1 trong nhóm learned models.
   - *Minh chứng:* File `tables/classification/cv_tuning_history.csv` lưu toàn bộ 40 lượt fit, mean/std, runtime và cấu hình được chọn.

3. **Kiểm tra tương đương nội dung luật kết hợp (`association_rules_equivalence_audit.csv`):**
   - *Vấn đề:* Pipeline manifest ghi FAIL do thiếu file audit kiểm tra tương đương giữa Apriori và FP-Growth.
   - *Khắc phục:* Cả Runner 06 và Notebook 06 đều chạy `compare_rule_sets()` theo canonical StockCode, đối chiếu metric với tolerance $10^{-5}$, benchmark 3 lần lặp (median/IQR) và xuất `association_rules_equivalence_audit.csv`. Nếu phát hiện lệch, pipeline dừng ngay bằng exception.
   - *Minh chứng:* Bảng `tables/association_rules/association_rules_equivalence_audit.csv` xác nhận 61/61 luật trùng khớp hoàn toàn (max metric diff = 0.0).

4. **Kiểm định độ ổn định phân cụm và tác động preprocessing:**
   - *Vấn đề:* Thiếu kiểm chứng độ ổn định của K-Means qua nhiều seed và so sánh chuẩn hóa.
   - *Khắc phục:* Bổ sung `evaluate_clustering_stability()` tính ARI qua các seed (mean ARI = 0.9996) và `compare_preprocessing_impact()` so sánh StandardScaler vs Log1p + StandardScaler.
   - *Minh chứng:* Xuất `tables/clustering/clustering_stability_audit.csv` và `tables/clustering/clustering_preprocessing_comparison.csv`.

5. **Đồng bộ tiền tệ GBP (£), loại bỏ văn bản gán cứng và sửa mã hóa tiếng Việt:**
   - *Vấn đề:* Biểu đồ dùng ký hiệu $, dashboard tóm tắt gán cứng $4,464, báo cáo bị lỗi mã hóa.
   - *Khắc phục:* Thay toàn bộ sang GBP (£), trích xuất động các chỉ số phân khúc và đặc trưng vào biểu đồ `insight_summary_dashboard.png`, sửa mã hóa UTF-8 chuẩn cho file báo cáo.

---

## 5. Danh Sách Các File Lớn Được Loại Khỏi Bundle & Cách Tái Tạo

Nhằm đảm bảo dung lượng gói review dưới 20 MB (thực tế gói ZIP khoảng **12–15 MB**), các file sau được loại bỏ có chủ đích:
1. `data/raw/Online Retail.xlsx` (22.6 MB): Tải từ UCI ML Repository và đặt vào `data/raw/`.
2. `data/interim/cleaned_transactions.csv` & `product_transactions.csv` (72.2 MB mỗi file): Được sinh tự động bởi bước 02.
3. `models/classification/best_classifier_pipeline.joblib` (20.7 MB): File nhị phân model huấn luyện, được sinh tự động bởi bước 05. (Lưu ý: toàn bộ metadata, cấu hình tham số, kết quả cross-validation và file dự đoán `test_predictions.csv` vẫn được giữ nguyên vẹn trong bundle).

---

## 6. Các Lệnh Tái Lập Toàn Bộ Dự Án (Reproduction Commands)

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate   # Trên Windows

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Chạy toàn bộ 47 bài kiểm thử tự động
pytest -v

# 4. Chạy toàn bộ pipeline từ bước 01 đến 07
python run_pipeline.py

# 5. Khởi chạy Dashboard tương tác
streamlit run app/app.py
```
