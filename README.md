# Khai Phá Dữ Liệu Khách Hàng Bán Lẻ Trực Tuyến (Online Retail Analytics)

Dự án nghiên cứu và ứng dụng khai phá dữ liệu khách hàng dựa trên tập dữ liệu giao dịch thương mại điện tử **Online Retail Dataset** (UCI Machine Learning Repository), được xây dựng chuẩn hóa theo quy trình khoa học **CRISP-DM** phục vụ nghiệm thu môn học Data Mining (Track B - Cao học).

---

## 1. Mục Tiêu & Phạm Vi Phân Tích

Dự án giải quyết 3 bài toán khai phá dữ liệu cốt lõi trong bán lẻ:
1. **Phân tích RFM & Phân cụm khách hàng (Customer Clustering):** Đo lường Recency, Frequency, Monetary; thử nghiệm so sánh 26 cấu hình trên 4 thuật toán (K-Means, GMM, Agglomerative, DBSCAN) nhằm nhận diện phân khúc khách hàng giá trị cao và phân khúc rủi ro rời bỏ.
2. **Dự đoán hành vi mua lại trong 90 ngày (Repeat Purchase Classification):** Xây dựng pipeline học máy phân loại khách hàng có quay lại mua sắm sau khoảng thời gian quan sát hay không; so sánh Logistic Regression, Decision Tree, Random Forest với baseline DummyClassifier; tinh chỉnh siêu tham số trên tập huấn luyện qua 5-fold Stratified Cross-Validation.
3. **Khai phá luật kết hợp giỏ hàng (Market Basket Association Rules):** Khai phá các quy luật mua hàng đồng thời qua 2 thuật toán Apriori và FP-Growth; đối chiếu tính tương đương của tập luật và phân tích độ nhạy theo các ngưỡng min_support và min_confidence.
4. **Ứng dụng Dashboard Tương tác (Streamlit App):** Giao diện trực quan 4 tab hỗ trợ quản trị, tra cứu phân khúc, gợi ý sản phẩm bán chéo và suy luận xác suất mua lại.

---

## 2. Cấu Trúc Thư Mục Repository

```text
Data-mining/
├── app/
│   └── app.py                            # Streamlit Dashboard tương tác 4 tab
├── data/
│   └── raw/
│       └── README.md                     # Hướng dẫn tải và cấu hình dataset gốc
├── docs/                                 # Hồ sơ tài liệu phân tích, phương pháp & quản trị
│   ├── 01_data_understanding_report.md   # Báo cáo thấu hiểu dữ liệu ban đầu
│   ├── 02_data_preparation_report.md     # Báo cáo quy trình tiền xử lý & làm sạch
│   ├── JABES-2025-6-V164.pdf             # Bài báo nghiên cứu tham chiếu
│   ├── Rubric_FinalProject_DataMining_CaoHoc.pdf # Rubric đánh giá đồ án Track B
│   ├── business_understanding.md         # Phân tích bài toán kinh doanh
│   ├── data_dictionary.md                # Từ điển dữ liệu chi tiết
│   ├── decision_log.md                   # Nhật ký các quyết định kỹ thuật
│   ├── feature_decision_dictionary.md    # Từ điển và cơ sở lý luận tạo đặc trưng
│   ├── methodology.md                    # Phương pháp luận chi tiết theo CRISP-DM
│   ├── references.md                     # Danh mục tài liệu tham khảo
│   ├── rubric_evidence_matrix.md         # Ma trận minh chứng đối chiếu theo rubric
│   ├── source_cleanup_and_handover.md    # Biên bản tái cấu trúc & bàn giao kỹ thuật
│   └── team_tasks.md                     # Phân công nhiệm vụ & đánh giá chéo nội bộ
├── notebooks/                            # Jupyter Notebooks nghiên cứu & Runner scripts
│   ├── 01_data_understanding.ipynb       # Khảo sát dữ liệu & phân tích phân phối
│   ├── 02_eda_and_cleaning.ipynb         # EDA chuyên sâu, xử lý hủy hàng & ngoại lai
│   ├── 03_feature_engineering_rfm.ipynb  # Tạo đặc trưng RFM & nhãn mua lại 90 ngày
│   ├── 04_customer_clustering.ipynb      # Thử nghiệm & so sánh 26 cấu hình phân cụm
│   ├── 05_repeat_purchase_classification.ipynb # Huấn luyện, tuning & đánh giá phân loại
│   ├── 06_association_rules.ipynb        # Khai phá luật kết hợp Apriori & FP-Growth
│   ├── 07_model_comparison_and_insights.ipynb # Tổng hợp đa mô hình & chiến lược hành động
│   ├── run_01_data_understanding.py      # Runner bước 01
│   ├── run_02_eda_and_cleaning.py        # Runner bước 02
│   ├── run_03_feature_engineering.py     # Runner bước 03
│   ├── run_04_customer_clustering.py     # Runner bước 04
│   ├── run_05_repeat_purchase_classification.py # Runner bước 05
│   ├── run_06_association_rules.py       # Runner bước 06
│   └── run_07_model_comparison_and_insights.py # Runner bước 07
├── src/                                  # Các mô-đun mã nguồn dùng chung
│   ├── __init__.py
│   ├── association_rules.py              # Xử lý ma trận giỏ hàng, Apriori, FP-Growth
│   ├── classification.py                 # Pipeline phân loại, cross-validation, tuning
│   ├── clustering.py                     # Tiền xử lý phân cụm, đo lường Silhouette, ARI
│   ├── config.py                         # Cấu hình đường dẫn, hằng số phân tích
│   ├── data_loader.py                    # Nạp và kiểm tra tính toàn vẹn dữ liệu gốc
│   ├── evaluation.py                     # Hàm đánh giá phân loại, ma trận nhầm lẫn
│   ├── feature_engineering.py            # Tính RFM scores, phân vị quantile, basket
│   ├── insights.py                       # Sinh insight nghiệp vụ & Action Plan 10 cột
│   ├── model_comparison.py               # So sánh mô hình & xuất báo cáo CRISP-DM
│   ├── model_comparison_figures.py       # Trực quan hóa tổng hợp đa mô hình
│   ├── preprocessing.py                  # Pipeline làm sạch hóa đơn, lọc mã phi sản phẩm
│   ├── rfm_analysis.py                   # Tiện ích phân tích RFM bổ trợ
│   └── visualization.py                  # Thư viện đồ họa trực quan từng bước
├── .gitignore                            # Bỏ qua outputs, models, data trung gian, cache
├── LICENSE                               # Giấy phép mã nguồn mở
├── README.md                             # Tài liệu hướng dẫn cài đặt & vận hành dự án
├── requirements.txt                      # Danh mục thư viện phụ thuộc môi trường
└── run_pipeline.py                       # Script thực thi tuần tự pipeline đầu–cuối (01→07)
```

---

## 3. Cài Đặt Môi Trường

### 3.1. Yêu Cầu Hệ Thống
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux.
- **Phiên bản Python:** Python 3.10 trở lên (khuyến nghị Python 3.12).

### 3.2. Khởi Tạo Môi Trường Ảo & Cài Đặt Dependencies

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows (Command Prompt hoặc PowerShell):
.venv\Scripts\activate

# Trên macOS / Linux:
source .venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 4. Tải Dữ Liệu Gốc

Dự án sử dụng tập dữ liệu **Online Retail Dataset** từ UCI Machine Learning Repository. Do kích thước tệp lớn (~23 MB), dữ liệu không được đính kèm trực tiếp trong Git:

1. Tải tệp dữ liệu từ liên kết chính thức:
   🔗 [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
2. Đặt tệp `Online Retail.xlsx` (hoặc `Online Retail.csv`) vào thư mục:
   ```text
   data/raw/Online Retail.xlsx
   ```
   *(Chi tiết cấu trúc các cột và lưu ý dữ liệu có tại `data/raw/README.md`)*.

---

## 5. Hướng Dẫn Chạy Toàn Bộ Dự Án (Pipeline Execution)

### 5.1. Chạy Tự Động Đầu–Cuối (Khuyến Nghị)

Để tái lập toàn bộ quy trình khoa học từ làm sạch, trích xuất đặc trưng, huấn luyện các mô hình đến tổng hợp báo cáo:

```bash
python run_pipeline.py
```

`run_pipeline.py` sẽ thực thi tuần tự 7 bước qua các runner trong `notebooks/`, kiểm định tính sẵn sàng của đầu ra mỗi bước trước khi chuyển tiếp. Thời gian thực thi ước tính: ~5–7 phút.

### 5.2. Chạy Riêng Lẻ Từng Bước

Giảng viên và hội đồng có thể thực thi độc lập bất kỳ bước nào:

```bash
# Bước 01: Thấu hiểu dữ liệu ban đầu
python notebooks/run_01_data_understanding.py

# Bước 02: Tiền xử lý, làm sạch giao dịch & loại bỏ ngoại lai
python notebooks/run_02_eda_and_cleaning.py

# Bước 03: Tạo đặc trưng RFM & nhãn mua lại (90 ngày cutoff)
python notebooks/run_03_feature_engineering.py

# Bước 04: Thử nghiệm phân cụm & kiểm định độ ổn định K-Means
python notebooks/run_04_customer_clustering.py

# Bước 05: Huấn luyện phân loại, tuning siêu tham số qua Stratified CV
python notebooks/run_05_repeat_purchase_classification.py

# Bước 06: Khai phá luật kết hợp giỏ hàng Apriori & FP-Growth
python notebooks/run_06_association_rules.py

# Bước 07: So sánh đa mô hình, xuất biểu đồ tổng hợp & Kế hoạch hành động
python notebooks/run_07_model_comparison_and_insights.py
```

---

## 6. Khởi Chạy Dashboard Tương Tác (Streamlit Web App)

Sau khi pipeline hoàn tất và sinh ra mô hình cùng bảng kết quả, khởi chạy Dashboard tương tác bằng lệnh:

```bash
python -m streamlit run app/app.py
```

Giao diện web mở tại `http://localhost:8501` cung cấp 4 tab chức năng:
1. **Dự đoán Khách hàng Mua lại (Classification):** Nhập đặc trưng khách hàng hoặc tải tệp hàng loạt để dự đoán xác suất và nhãn mua lại (áp dụng cùng quy tắc ngưỡng đồng nhất với pipeline).
2. **Phân khúc Khách hàng & Kế hoạch Hành động (Clustering):** Khảo sát hồ sơ RFM từng cụm khách hàng và kế hoạch hành động 10 cột.
3. **Gợi ý Bán chéo Sản phẩm (Association Rules):** Tra cứu luật kết hợp sản phẩm mua kèm theo mức Support, Confidence, Lift.
4. **Báo cáo Kỹ thuật & Quản trị Mô hình:** Giám sát siêu dữ liệu, phân phối lớp tập huấn luyện và cảnh báo quản trị mô hình.

---

## 7. Vị Trí Các Kết Quả (Outputs) Sau Khi Chạy

Toàn bộ outputs được pipeline tự động khởi tạo và lưu trữ tại các thư mục tương ứng:

| Thư mục output | Nội dung được tạo ra | Tệp tiêu biểu |
|:---|:---|:---|
| `data/interim/` | Dữ liệu giao dịch đã làm sạch | `cleaned_transactions.csv`, `product_transactions.csv` |
| `data/processed/` | Bảng đặc trưng phục vụ mô hình | `rfm_customer_features.csv`, `repeat_purchase_features.csv`, `customer_clusters.csv` |
| `models/clustering/` | Mô hình phân cụm đã serialize | `clustering_model.pkl`, `clustering_config.pkl` |
| `models/classification/` | Pipeline phân loại tốt nhất | `best_classifier_pipeline.joblib`, `classification_metadata.json` |
| `outputs/tables/` | Toàn bộ bảng CSV kết quả chi tiết | `clustering_algorithm_comparison.csv`, `model_comparison.csv`, `customer_segment_action_plan.csv` |
| `outputs/figures/` | Toàn bộ biểu đồ trực quan hóa PNG | `insight_summary_dashboard.png`, `cluster_rfm_boxplots.png`, `roc_curves.png` |
| `outputs/reports/` | Báo cáo khoa học tổng hợp | `07_model_comparison_and_insights.md`, `07_model_comparison_and_insights_summary.json` |
| `outputs/evidence/` | Nhật ký thực thi & mã băm SHA-256 | `pipeline_execution.log`, `pipeline_manifest.json` |

---

## 8. Tóm Tắt Kết Quả Thực Nghiệm Khoa Học Chính

### 8.1. Phân Cụm Khách Hàng (Customer Clustering)
- **Mô hình được chọn:** **K-Means ($K=2$)** áp dụng trên bộ đặc trưng RFM qua biến đổi $\log(1+x)$ và `StandardScaler`.
- **Chỉ số đánh giá:** Silhouette Score = **0.4330**, Davies-Bouldin Index = **0.8917**.
- **Độ ổn định:** Mean Adjusted Rand Index (ARI) = **0.9996** qua nhiều giá trị random seeds độc lập.
- **Hồ sơ phân khúc:**
  - *Cụm 0 (Best Customers - 38.4%):* Giá trị chi tiêu cao (Recency trung vị 18 ngày, Frequency 8 đơn, Monetary £2,203).
  - *Cụm 1 (Low-Value Customers - 61.6%):* Giá trị chi tiêu thấp, ít quay lại (Recency trung vị 103 ngày, Frequency 2 đơn, Monetary £358).

### 8.2. Phân Loại Dự Đoán Khách Hàng Mua Lại (Repeat Purchase Classification)
- **Mô hình được chọn:** **Random Forest Classifier** (`n_estimators: 100, max_depth: 5, min_samples_leaf: 5`).
- **Hiệu năng kiểm chứng:**
  - Cross-Validation F1-Score (5-fold train): **0.7102** $\pm$ 0.0169.
  - Test F1-Score: **0.7375**, Test ROC-AUC: **0.7438**.
  - So sánh với Baseline: Dummy Classifier đạt F1 0.7259 nhưng ROC-AUC = 0.5000 (không có năng lực phân loại thực tế).
- **Ngưỡng quyết định thống nhất:** Xác suất $P(\text{repeat}) \ge 0.50$ được đồng bộ tuyệt đối giữa offline evaluation và Dashboard.

### 8.3. Khai Phá Luật Kết Hợp (Market Basket Association Rules)
- **Tính tương đương khoa học:** Khai phá ra **61 luật kết hợp hợp lệ** (Lift > 1, Support $\ge 0.02$, Confidence $\ge 0.50$). Cả hai thuật toán Apriori và FP-Growth cho kết quả trùng khớp hoàn toàn (sai khác metric = 0.0).
- **Quy luật tiêu biểu:** Các bộ sản phẩm đồ sứ *Regency Teacup and Saucer* có Lift cao nhất đạt **18.23** (Confidence 75.3%), là căn cứ cho chiến lược combo quà tặng và cross-sell giỏ hàng.

---

## 9. Đơn Vị Tiền Tệ & Bản Quyền

- **Đơn vị tiền tệ:** Toàn bộ số liệu tiền tệ trong mã nguồn, biểu đồ và báo cáo sử dụng **Bảng Anh (£ - GBP)** theo đúng dữ liệu gốc từ UCI.
- **Bản quyền:** Mã nguồn được phân phối theo giấy phép đính kèm trong tệp [LICENSE](file:///d:/Data-mininng/LICENSE).