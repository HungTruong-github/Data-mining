# Báo Cáo Tái Cấu Trúc Mã Nguồn & Bàn Giao Kỹ Thuật (Source Cleanup & Handover)

**Dự án:** Khai phá Dữ liệu Khách hàng Bán lẻ Trực tuyến (Online Retail Customer Analytics)  
**Nhánh Git:** `clean`  
**HEAD Commit mốc:** `841d5d20f736110e954450372fb42416e407a14e`  
**Chuẩn đánh giá:** Rubric Track B — Đồ án Cao học môn Khai phá Dữ liệu (CRISP-DM Framework)  
**Ngày thực hiện:** 05/10/2026  

---

## 1. Mục Tiêu & Phạm Vi Công Việc

Quá trình rà soát và tái cấu trúc mã nguồn được thực hiện trực tiếp trên workspace nhằm mục tiêu:
1. **Nâng cao tính mô-đun hóa và khả năng bảo trì:** Tách các tệp có quy mô quá lớn, phân tách rõ ràng giữa tầng logic xử lý khoa học dữ liệu và tầng trực quan hóa.
2. **Loại bỏ trùng lặp và thu gọn repository:** Xóa bỏ thư mục bản sao `review_bundle/` (166 file trùng lặp bị theo dõi trong Git), hợp nhất toàn bộ dữ liệu bảng biểu, minh chứng môi trường vào thư mục chính thức `outputs/`.
3. **Chuẩn hóa script tái lập:** Chuyển các công cụ đóng gói và thực thi từ thư mục tạm `scratch/` sang thư mục sản phẩm chuẩn `scripts/`, dọn dẹp các tệp nháp phát sinh trong quá trình dev.
4. **Bảo toàn 100% tính đúng đắn khoa học:** Đảm bảo toàn bộ 47 kiểm thử tự động (`pytest`) và 12 tiêu chí nghiệm thu kỹ thuật (`ACC-01` đến `ACC-12`) đạt trạng thái **PASS**.

---

## 2. Chi Tiết Các Thay Đổi Kỹ Thuật

### 2.1. Tách Mô-đun `src/model_comparison.py`
- **Hiện trạng trước sửa:** Tệp `src/model_comparison.py` dài gần 1,000 dòng (~999 dòng), tích hợp lẫn lộn giữa logic so sánh đa mô hình, trích xuất metrics và mã vẽ 10 biểu đồ Matplotlib/Seaborn.
- **Giải pháp tái cấu trúc:**
  - Tách toàn bộ logic vẽ biểu đồ và hàm điều phối `generate_all_step_07_figures` sang mô-đun mới [`src/model_comparison_figures.py`](file:///d:/Data-mininng/src/model_comparison_figures.py) (236 dòng).
  - Tái xuất khẩu (re-export) hàm `generate_all_step_07_figures` ngay trong [`src/model_comparison.py`](file:///d:/Data-mininng/src/model_comparison.py) (giảm xuống còn 786 dòng), đảm bảo tương thích ngược 100% cho mọi runner và test hiện hành.
  - Phân tách hoàn toàn trách nhiệm: `src/model_comparison.py` tập trung vào nghiệp vụ tổng hợp chỉ số, validation và xuất báo cáo; `src/model_comparison_figures.py` chuyên trách đồ họa trực quan.

### 2.2. Dọn Dẹp Mã Nguồn & Tinh Giản Imports
Rà soát AST và loại bỏ toàn bộ các import dư thừa, cải thiện tốc độ nạp mô-đun:
- [`app/app.py`](file:///d:/Data-mininng/app/app.py): Loại bỏ các import không sử dụng `io`, `numpy as np`, và `predict_with_threshold` (sử dụng thống nhất qua `predict_customers`). Sửa lỗi cấu trúc docstring ban đầu.
- [`run_pipeline.py`](file:///d:/Data-mininng/run_pipeline.py): Loại bỏ import `os` không dùng.
- [`notebooks/run_01_data_understanding.py`](file:///d:/Data-mininng/notebooks/run_01_data_understanding.py): Loại bỏ import dư thừa `get_raw_data_path`, `FIGURES_DIR`, `TABLES_DIR`.
- [`notebooks/run_02_eda_and_cleaning.py`](file:///d:/Data-mininng/notebooks/run_02_eda_and_cleaning.py): Loại bỏ import `get_raw_data_path`.
- [`notebooks/run_03_feature_engineering.py`](file:///d:/Data-mininng/notebooks/run_03_feature_engineering.py): Loại bỏ import `get_raw_data_path`.
- [`notebooks/run_04_customer_clustering.py`](file:///d:/Data-mininng/notebooks/run_04_customer_clustering.py): Chuyển import `compare_preprocessing_impact`, `evaluate_clustering_stability` từ trong thân hàm lên phần đầu tệp theo chuẩn PEP 8.

### 2.3. Hợp Nhất Minh Chứng & Thu Gọn Thư Mục Gốc
- **Hợp nhất minh chứng vào `outputs/`:**
  - `outputs/evidence/acceptance_results.csv` & `acceptance_results.json`: Ghi nhận trạng thái 12 tiêu chí nghiệm thu.
  - `outputs/evidence/environment/`: Lưu trữ `python_version.txt`, `pip_freeze.txt`, `hardware_specs.json`, `raw_data_provenance.md`.
  - `outputs/tables/customer_features/`: Chứa `customer_clusters.csv`, `repeat_purchase_features.csv`, `rfm_customer_features.csv`.
- **Chuyển dịch các script vận hành vào `scripts/`:**
  - [`scripts/build_review_bundle.py`](file:///d:/Data-mininng/scripts/build_review_bundle.py): Script chuẩn hóa đóng gói `review_bundle.zip` (đã chỉnh sửa tránh tạo file rác ở thư mục gốc).
  - [`scripts/execute_notebooks.py`](file:///d:/Data-mininng/scripts/execute_notebooks.py): Script thực thi toàn bộ 7 Jupyter Notebooks không cần mở IDE.
- **Xóa bỏ các tệp rác và tệp tạm đã được Git theo dõi:**
  - Thư mục `review_bundle/` (166 file trùng lặp).
  - Thư mục `scratch/` (các file script debug dùng 1 lần).
  - Các tệp đơn lẻ: `tree.txt`, `Prompt_Antigravity_Hoan_thien_Data_Mining_01-07.md`, `README_REVIEW.md` ở thư mục gốc.
- **Cập nhật `.gitignore`:** Bổ sung `review_bundle/` và `review_bundle.zip` để khi thực thi script đóng gói sẽ không gây bẩn cây Git.

---

## 3. Tổng Hợp Trạng Thái Kiểm Thử & Nghiệm Thu Khoa Học

### 3.1. Kết Quả Kiểm Thử Đơn Vị & Tích Hợp (Pytest Suite)
- **Tổng số ca kiểm thử:** 47 test cases.
- **Trạng thái:** **47 PASSED, 0 FAILED, 0 SKIPPED** (Tỷ lệ thành công: 100%).
- **Chi tiết phân bổ:**
  - `tests/test_preprocessing.py`: 7 tests passed (Làm sạch Invoice, phân loại StockCode, kiểm soát giá trị ngoại lai).
  - `tests/test_feature_engineering.py`: 8 tests passed (Tạo biến RFM, kiểm soát thứ tự phân vị monotonic, zero data leakage).
  - `tests/test_classification.py`: 8 tests passed (Tuning siêu tham số, tính nhất quán ngưỡng xác suất, test dataset validation).
  - `tests/test_model_comparison.py`: 19 tests passed (Không trùng lặp hàm AST, kiểm định tính tương đương Apriori/FP-Growth, Action Plan 10 cột, manifest SHA-256).
  - `tests/test_app_smoke.py`: 5 tests passed (Khởi tạo AppTest Streamlit không crash, kiểm tra ngưỡng suy luận và schema).

### 3.2. Bảng Kết Quả 12 Tiêu Chí Nghiệm Thu Kỹ Thuật (Acceptance Results)

| Mã tiêu chí | Yêu cầu kỹ thuật | Phương pháp kiểm chứng | Kết quả quan sát | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **ACC-01** | Action plan 10 cột hoàn chỉnh, sinh động | `test_model_comparison.py` | 2 chiến lược kinh doanh sinh động, đầy đủ 10 cột | **PASS** |
| **ACC-02** | Zero duplicate module-level functions | Quét AST toàn bộ file `.py` | 0 hàm trùng lặp trên toàn bộ kho mã nguồn | **PASS** |
| **ACC-03** | Đối chiếu nội dung tương đương Apriori & FP-Growth | `association_rules_equivalence_audit.csv` | 61 luật trùng khớp tuyệt đối (sai khác metric = 0.0) | **PASS** |
| **ACC-04** | Pipeline phát hiện sai lệch nội dung luật | `test_pipeline_detects_rule_content_mismatch` | Bắt lỗi mismatch khi gán nhãn giả lập | **PASS** |
| **ACC-05** | RFM quantile fallback đơn điệu & bất biến thứ tự dòng | `test_rfm_fallback_and_monotonicity` | Đơn điệu bảo toàn; bất biến thứ tự sắp xếp | **PASS** |
| **ACC-06** | `CustomerID` bảo toàn ở cột đầu của test predictions | Kiểm tra `test_predictions.csv` | `CustomerID` là cột 0, 0 giá trị null (674 khách hàng) | **PASS** |
| **ACC-07** | Dashboard và Offline Pipeline chia sẻ chung quy tắc dự đoán | `predict_with_threshold` qua test parity | Parity xác suất-ra-nhãn đạt 100% trên tập test | **PASS** |
| **ACC-08** | Pipeline manifest xác thực SHA-256 | `pipeline_manifest.json` | 100% tệp kết quả có mã băm SHA-256 hợp lệ | **PASS** |
| **ACC-09** | Grid search siêu tham số trên tập train folds | `cv_tuning_history.csv` | Ghi nhận 40 cấu hình ứng viên (200 fits) | **PASS** |
| **ACC-10** | Kiểm định ổn định phân cụm qua nhiều seed | `clustering_stability_audit.csv` | Mean pairwise ARI = 0.9996 (vượt ngưỡng 0.80) | **PASS** |
| **ACC-11** | Benchmark thuật toán luật kết hợp đa lần chạy | `rule_sensitivity_analysis.csv` | Đánh giá 3 lần lặp (median/IQR), quét lưới hỗ trợ | **PASS** |
| **ACC-12** | Tính toán lại metrics từ `test_predictions.csv` | `test_recalculated_metrics` | Test F1 tính lại (0.7375) khớp chính xác bảng so sánh | **PASS** |

---

## 4. Hướng Dẫn Vận Hành & Tái Lập (Reproduction Guide)

Hội đồng chấm thi và giảng viên có thể tái lập toàn bộ quy trình bằng các lệnh tiêu chuẩn:

```bash
# 1. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 2. Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# 3. Chạy toàn bộ kiểm thử tự động
python -m pytest -q

# 4. Thực thi toàn bộ pipeline 7 bước
python run_pipeline.py

# 5. Khởi chạy Dashboard tương tác
streamlit run app/app.py

# 6. (Tùy chọn) Đóng gói hồ sơ nộp bài review_bundle.zip
python scripts/build_review_bundle.py
```

---

## 5. Các Hạng Mục Chờ Nhóm Học Viên Hoàn Thiện Trước Khi Nộp

1. **Thông tin định danh nhóm học viên:**
   - Cập nhật thông tin thực tế vào bảng mục 1 của [`docs/team_tasks.md`](file:///d:/Data-mininng/docs/team_tasks.md): Họ và tên, MSHV, Email sinh viên, phân chia tỷ lệ đóng góp (%).
2. **Biên bản đánh giá chéo (Peer Assessment):**
   - Hoàn thiện bảng điểm đánh giá chéo nội bộ trong [`docs/team_tasks.md`](file:///d:/Data-mininng/docs/team_tasks.md) theo 5 tiêu chí đã quy định.
3. **Slide báo cáo thuyết minh:**
   - Sử dụng các biểu đồ chất lượng cao trong `outputs/figures/model_comparison/` (đặc biệt là `insight_summary_dashboard.png`) và các bảng số liệu trong `outputs/tables/` để chuẩn bị slide báo cáo cuối kỳ.
