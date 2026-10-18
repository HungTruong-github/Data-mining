# Khai Phá Dữ Liệu Khách Hàng Bán Lẻ Trực Tuyến (Online Retail Analytics)

Dự án nghiên cứu và ứng dụng khai phá dữ liệu khách hàng dựa trên tập dữ liệu giao dịch thương mại điện tử **Online Retail Dataset** (UCI Machine Learning Repository), được xây dựng chuẩn hóa theo quy trình khoa học **CRISP-DM** phục vụ nghiệm thu môn học Data Mining (Track B - Cao học).

---

## 1. Mục Tiêu & Phạm Vi Phân Tích

Dự án giải quyết 3 bài toán khai phá dữ liệu cốt lõi trong bán lẻ cùng 1 giải pháp ứng dụng tương tác:
1. **Phân tích RFM & Phân cụm khách hàng (Customer Clustering):** Đo lường Recency, Frequency, Monetary; thử nghiệm so sánh 26 cấu hình trên 4 thuật toán (K-Means, GMM, Agglomerative Hierarchical, DBSCAN) nhằm nhận diện phân khúc khách hàng giá trị cao và phân khúc rủi ro rời bỏ.
2. **Dự đoán hành vi mua lại trong 90 ngày (Repeat Purchase Classification):** Xây dựng pipeline học máy phân loại khách hàng có quay lại mua sắm sau khoảng thời gian quan sát hay không; so sánh Logistic Regression, Decision Tree, Random Forest với baseline DummyClassifier; tinh chỉnh siêu tham số trên tập huấn luyện qua 5-fold Stratified Cross-Validation.
3. **Khai phá luật kết hợp giỏ hàng (Market Basket Association Rules):** Khai phá các quy luật mua hàng đồng thời qua 2 thuật toán Apriori và FP-Growth; đối chiếu kiểm toán tính tương đương (Equivalence Audit) của tập luật và phân tích độ nhạy theo các ngưỡng min_support và min_confidence.
4. **Ứng dụng Dashboard Tương tác (Streamlit App):** Giao diện web trực quan 4 tab hỗ trợ quản trị mô hình, tra cứu phân khúc, gợi ý sản phẩm bán chéo (cross-sell) và suy luận xác suất mua lại.

---

## 2. Cấu Trúc Thư Mục Repository

```text
Data-mining/
├── app/
│   └── app.py                            # Streamlit Dashboard tương tác 4 tab
├── data/
│   ├── raw/
│   │   ├── Online Retail.xlsx            # Dataset gốc dạng Excel (tải từ UCI/Kaggle)
│   │   ├── Online Retail.csv             # Dataset gốc dạng CSV (ưu tiên nạp tự động)
│   │   └── README.md                     # Hướng dẫn tải và cấu hình dataset gốc
│   ├── interim/                          # Dữ liệu trung gian sau khi làm sạch giao dịch
│   │   ├── cleaned_transactions.csv      # Giao dịch mua hàng hợp lệ
│   │   ├── product_transactions.csv      # Dữ liệu sản phẩm phục vụ luật kết hợp
│   │   └── customer_transactions.csv     # Giao dịch có CustomerID định danh
│   └── processed/                        # Dữ liệu đặc trưng phục vụ huấn luyện mô hình
│       ├── rfm_customer_features.csv     # Đặc trưng RFM cấp khách hàng
│       ├── repeat_purchase_features.csv  # Bộ đặc trưng và nhãn mua lại 90 ngày
│       ├── customer_clusters.csv         # Dữ liệu khách hàng đã gán nhãn cụm
│       └── association_basket_matrix.csv # Ma trận nhị phân giỏ hàng (Basket Matrix)
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
├── models/                               # Mô hình học máy & siêu dữ liệu serialize
│   ├── classification/
│   │   ├── best_classifier_pipeline.joblib # Pipeline RandomForest tốt nhất
│   │   ├── classification_metadata.json   # Metadata, ngưỡng và siêu tham số
│   │   ├── decision_tree_pipeline.joblib  # Pipeline Decision Tree đã fit
│   │   ├── dummy_classifier_pipeline.joblib # Pipeline Dummy Classifier baseline
│   │   ├── logistic_regression_pipeline.joblib # Pipeline Logistic Regression đã fit
│   │   └── random_forest_pipeline.joblib  # Pipeline Random Forest đã fit
│   └── clustering/
│       ├── clustering_model.pkl          # Mô hình K-Means (K=2) đã fit
│       ├── clustering_config.pkl         # Cấu hình tiền xử lý và đặc trưng
│       └── scaler.pkl                    # StandardScaler đã fit trên dữ liệu log1p
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
├── outputs/                              # Toàn bộ outputs được sinh tự động
│   ├── evidence/                         # Nhật ký thực thi và manifest mã băm SHA-256
│   ├── figures/                          # Biểu đồ trực quan PNG (phân bổ theo stage)
│   ├── reports/                          # Báo cáo tổng hợp CRISP-DM (.md, .json)
│   ├── rules/                            # Tệp CSV các luật kết hợp đã khai phá
│   └── tables/                           # Toàn bộ bảng CSV kết quả chi tiết từng bước
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
├── .gitignore                            # Quy tắc loại trừ tệp tạm, môi trường ảo, cache
├── LICENSE                               # Giấy phép mã nguồn mở MIT
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
# 1. Khởi tạo môi trường ảo Python
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

Dự án sử dụng tập dữ liệu **Online Retail Dataset** từ UCI Machine Learning Repository (gồm 541,909 giao dịch từ 01/12/2010 đến 09/12/2011).

1. Tải tệp dữ liệu từ liên kết chính thức:
   - 🔗 [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
   - 🔗 [Kaggle E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
2. Đặt tệp `Online Retail.xlsx` hoặc `Online Retail.csv` vào thư mục:
   ```text
   data/raw/Online Retail.xlsx
   # hoặc
   data/raw/Online Retail.csv
   ```
   > **Lưu ý:** Bộ nạp dữ liệu (`src/data_loader.py`) hỗ trợ cả hai định dạng `.csv` và `.xlsx`, trong đó tự động ưu tiên `.csv` trước để tối ưu hóa tốc độ đọc dữ liệu gấp nhiều lần so với định dạng Excel nén. Chi tiết cấu trúc các cột và lưu ý dữ liệu có tại [`data/raw/README.md`](data/raw/README.md).

---

## 5. Hướng Dẫn Chạy Toàn Bộ Dự Án (Pipeline Execution)

### 5.1. Chạy Tự Động Toàn Bộ Pipeline Đầu–Cuối (Khuyến Nghị)

Để tái lập toàn bộ quy trình khoa học từ làm sạch, trích xuất đặc trưng, huấn luyện các mô hình đến tổng hợp báo cáo và kiểm tra tính toàn vẹn:

```bash
python run_pipeline.py
```

`run_pipeline.py` sẽ thực thi tuần tự 7 bước qua các runner trong `notebooks/`, đồng thời tự động kiểm định sự tồn tại và kích thước của các file outputs bắt buộc sau mỗi bước trước khi chuyển tiếp. Thời gian thực thi ước tính: ~5–7 phút. Nhật ký thực thi được tự động lưu tại `outputs/evidence/pipeline_execution.log`.

### 5.2. Chạy Riêng Lẻ Từng Bước

Người dùng và hội đồng đánh giá có thể thực thi độc lập bất kỳ bước nào qua các runner script:

```bash
# Bước 01: Thấu hiểu dữ liệu ban đầu, phân tích phân phối & missing values
python notebooks/run_01_data_understanding.py

# Bước 02: Tiền xử lý, lọc đơn hủy 'C', nợ xấu 'A', giá trị phi lý & StockCode phi sản phẩm
python notebooks/run_02_eda_and_cleaning.py

# Bước 03: Tạo đặc trưng RFM, chia khung thời gian quan sát & gán nhãn mua lại 90 ngày
python notebooks/run_03_feature_engineering.py

# Bước 04: Thử nghiệm 26 cấu hình phân cụm (K-Means, GMM, Agglomerative, DBSCAN) & kiểm định ARI
python notebooks/run_04_customer_clustering.py

# Bước 05: Huấn luyện phân loại, tuning siêu tham số qua 5-fold Stratified CV & đánh giá Test
python notebooks/run_05_repeat_purchase_classification.py

# Bước 06: Khai phá luật kết hợp giỏ hàng Apriori & FP-Growth, kiểm toán tính tương đương
python notebooks/run_06_association_rules.py

# Bước 07: So sánh đa mô hình, Action Plan 10 cột, biểu đồ tổng hợp & Báo cáo CRISP-DM
python notebooks/run_07_model_comparison_and_insights.py
```

### 5.3. Khám Phá Tương Tác Qua Jupyter Notebooks

Các Jupyter Notebooks tương ứng trong thư mục `notebooks/` (`01_data_understanding.ipynb` đến `07_model_comparison_and_insights.ipynb`) đã được biên tập đầy đủ với giải thích phương pháp luận chi tiết và biểu đồ trực quan, có thể mở bằng lệnh:

```bash
jupyter notebook notebooks/
```

---

## 6. Khởi Chạy Dashboard Tương Tác (Streamlit Web App)

Sau khi pipeline hoàn tất và sinh ra mô hình cùng bảng kết quả, khởi chạy Dashboard tương tác bằng lệnh:

```bash
streamlit run app/app.py
# hoặc
python -m streamlit run app/app.py
```

Giao diện web mở tại `http://localhost:8501` cung cấp 4 tab chức năng chuyên sâu:
1. **🎯 Dự đoán Khách hàng Mua lại (Classification):** Nhập dữ liệu khách hàng qua file CSV hoặc nhấn nút nạp 10 khách hàng mẫu; hệ thống áp dụng Scikit-Learn Pipeline (`RandomForest`) với quy tắc ngưỡng quyết định thống nhất ($P \ge 0.50$) để trả về xác suất và nhãn dự đoán, hỗ trợ tải file CSV kết quả.
2. **👥 Phân khúc Khách hàng & Kế hoạch Hành động (Clustering):** Khảo sát hồ sơ đặc trưng RFM từng cụm khách hàng (`Best Customers` vs `Lost Customers`) và kế hoạch hành động chiến lược 10 cột (chiến lược, hành động, bằng chứng dữ liệu, mục tiêu KPI, phương pháp kiểm chứng, giới hạn).
3. **🛒 Gợi ý Bán chéo Sản phẩm (Association Rules):** Cho phép chọn sản phẩm trong giỏ hàng từ danh sách thả xuống để tra cứu ngay các sản phẩm mua kèm có giá trị Lift cao nhất; hiển thị bảng Top 10 luật kết hợp mạnh nhất toàn hệ thống.
4. **📑 Báo cáo Kỹ thuật & Quản trị Mô hình (Model Governance & Lineage):** Giám sát thông tin phiên bản Run ID, Git commit, kiểm tra mã băm SHA-256 các output artifacts theo từng bước pipeline để chứng minh tính toàn vẹn khoa học.

---

## 7. Vị Trí Các Kết Quả (Outputs) Sau Khi Chạy

Toàn bộ outputs được pipeline tự động khởi tạo và lưu trữ đúng cấu trúc thư mục chuẩn hóa:

| Thư mục output | Nội dung được tạo ra | Các tệp tiêu biểu |
|:---|:---|:---|
| `data/interim/` | Dữ liệu giao dịch đã làm sạch | `cleaned_transactions.csv`, `product_transactions.csv`, `customer_transactions.csv` |
| `data/processed/` | Bảng đặc trưng phục vụ mô hình | `rfm_customer_features.csv`, `repeat_purchase_features.csv`, `customer_clusters.csv`, `association_basket_matrix.csv` |
| `models/clustering/` | Mô hình phân cụm đã serialize | `clustering_model.pkl`, `clustering_config.pkl`, `scaler.pkl` |
| `models/classification/` | Pipeline phân loại và metadata | `best_classifier_pipeline.joblib`, `classification_metadata.json`, `random_forest_pipeline.joblib`, `logistic_regression_pipeline.joblib`, `decision_tree_pipeline.joblib`, `dummy_classifier_pipeline.joblib` |
| `outputs/tables/data_understanding/` | Bảng thống kê thấu hiểu dữ liệu | `descriptive_statistics.csv`, `raw_summary.csv` |
| `outputs/tables/data_preparation/` | Bảng thống kê tiền xử lý & làm sạch | `cleaning_summary.csv`, `outlier_summary.csv`, `stockcode_classification.csv` |
| `outputs/tables/feature_engineering/` | Bảng tóm tắt đặc trưng RFM | `rfm_segment_summary.csv`, `repeat_purchase_class_balance.csv` |
| `outputs/tables/clustering/` | Bảng so sánh thuật toán & hồ sơ cụm | `clustering_algorithm_comparison.csv`, `cluster_profiles.csv`, `clustering_stability_audit.csv` |
| `outputs/tables/classification/` | Bảng kết quả phân loại & tuning CV | `model_comparison.csv`, `cv_results.csv`, `test_predictions.csv`, `feature_importance.csv` |
| `outputs/tables/association_rules/` | Bảng luật kết hợp & đối chiếu thuật toán | `association_algorithm_comparison.csv`, `selected_association_rules.csv`, `association_business_insights.csv`, `association_rules_equivalence_audit.csv` |
| `outputs/tables/insights/` | Bảng kế hoạch hành động & insight tổng hợp | `customer_segment_action_plan.csv`, `customer_segment_insights.csv`, `product_association_insights.csv`, `classification_feature_insights.csv` |
| `outputs/tables/model_comparison/` | Bảng so sánh đa mô hình tổng hợp | `clustering_model_comparison.csv`, `classification_model_comparison.csv`, `association_rules_comparison.csv` |
| `outputs/figures/` | Toàn bộ biểu đồ trực quan hóa PNG | `insight_summary_dashboard.png`, `cluster_rfm_boxplots.png`, `roc_curves.png`, `precision_recall_curves.png`, `confusion_matrix_best_model.png` |
| `outputs/reports/` | Báo cáo khoa học tổng hợp CRISP-DM | `07_model_comparison_and_insights.md`, `07_model_comparison_and_insights_summary.json` |
| `outputs/evidence/` | Nhật ký thực thi & mã băm SHA-256 | `pipeline_execution.log`, `pipeline_manifest.json` |

---

## 8. Tóm Tắt Kết Quả Thực Nghiệm Khoa Học Chính

### 8.1. Phân Cụm Khách Hàng (Customer Clustering)
- **Mô hình được chọn:** **K-Means ($K=2$)** áp dụng trên bộ 3 đặc trưng RFM qua biến đổi $\log(1+x)$ và chuẩn hóa `StandardScaler`.
- **Chỉ số đánh giá nội bộ:** Silhouette Score = **0.4330**, Davies-Bouldin Index = **0.8917**, Calinski-Harabasz Index = **4,364.61**.
- **Độ ổn định (Stability Audit):** Mean Pairwise Adjusted Rand Index (ARI) = **0.9996** $\pm$ 0.0005 qua 5 random seeds độc lập `[42, 123, 456, 789, 999]`.
- **Thử nghiệm đối chiếu:** Khảo sát 26 cấu hình trên 4 thuật toán (K-Means, GMM, Agglomerative, DBSCAN). Các cấu hình DBSCAN bị loại bỏ do hiện tượng cụm suy biến (cluster chiếm >90% hoặc số lượng mẫu <10).
- **Hồ sơ phân khúc (Tổng số 4,334 khách hàng định danh):**
  - *Cụm 0 (Best Customers — 1,662 khách, 38.35%):* Khách hàng thân thiết giá trị cao. Recency trung vị **16.0 ngày** (trung bình 25.82 ngày), Frequency trung vị **6.0 đơn** (trung bình 8.40 đơn), Monetary trung vị **£2,041.33** (trung bình £4,464.19). Đóng góp **£7,419,490.14** doanh số (chiếm **84.9%** tổng doanh thu). Tỷ lệ khách hàng mua lại trong 90 ngày đạt **95.7%**.
  - *Cụm 1 (Lost Customers / Low-Value — 2,672 khách, 61.65%):* Khách hàng chi tiêu thấp hoặc có nguy cơ rời bỏ. Recency trung vị **96.0 ngày** (trung bình 134.30 ngày), Frequency trung vị **1.0 đơn** (trung bình 1.66 đơn), Monetary trung vị **£356.92** (trung bình £493.17). Đóng góp **£1,317,737.50** doanh số (chiếm **15.1%** tổng doanh thu). Tỷ lệ khách hàng mua lại trong 90 ngày chỉ đạt **26.4%**.

### 8.2. Phân Loại Dự Đoán Khách Hàng Mua Lại (Repeat Purchase Classification)
- **Bài toán phân loại:** Dự đoán nhãn `repeat_purchase_90d` (có phát sinh đơn hàng trong 90 ngày tiếp theo kể từ ngày mốc 10/09/2011).
- **Mô hình được chọn:** **Random Forest Classifier** (`n_estimators: 100, max_depth: 5, min_samples_leaf: 5, criterion: gini`).
- **Phân chia dữ liệu:** Train set (2,694 mẫu: 1,159 negative, 1,535 positive — 57.0% positive); Test set holdout (674 mẫu: 290 negative, 384 positive — 56.97% positive).
- **Hiệu năng kiểm chứng:**
  - 5-fold Stratified CV trên Train: CV F1 = **0.7102** $\pm$ 0.0175, CV ROC-AUC = **0.7372**, CV Average Precision (PR-AUC) = **0.8099**.
  - Đánh giá trên tập Test độc lập:
    - Test F1-Score: **0.7375**
    - Test ROC-AUC: **0.7756**
    - Test Average Precision (PR-AUC): **0.8396**
    - Test Accuracy: **70.33%**
    - Test Precision: **74.34%**, Test Recall: **73.18%**, Test Specificity: **66.55%**
- **Đối chiếu với Baseline Dummy Classifier:** Dummy Classifier (chiến lược `most_frequent`) đạt Test F1 = 0.7259 chỉ nhờ vào tỷ lệ lớp dương chiếm 57.0%, nhưng Test ROC-AUC = 0.5000 và Specificity = 0.0 (hoàn toàn không nhận diện được khách hàng không quay lại). Random Forest vượt trội với ROC-AUC = 0.7756 và Specificity = 66.55%, chứng minh năng lực phân biệt thực tế rõ rệt.
- **Ngưỡng quyết định thống nhất:** Xác suất $P(\text{repeat}) \ge 0.50$ được đồng bộ tuyệt đối giữa pipeline đánh giá offline và Streamlit Dashboard.

### 8.3. Khai Phá Luật Kết Hợp (Market Basket Association Rules)
- **Tính tương đương khoa học:** Khai phá ra **61 luật kết hợp hợp lệ** từ 389 tập mục phổ biến với ngưỡng min_support $\ge 0.02$ (2%), min_confidence $\ge 0.50$ (50%), và Lift > 1.0. Cả hai thuật toán Apriori và FP-Growth cho kết quả trùng khớp hoàn toàn (sai khác các metric support, confidence, lift đều bằng 0.0). Thời gian thực thi: Apriori đạt 1.78s, FP-Growth đạt 2.11s.
- **Quy luật tiêu biểu:**
  - Dòng sản phẩm đồ sứ *Regency Teacup and Saucer* có Lift cao nhất đạt **18.23** (`PINK REGENCY TEACUP AND SAUCER` $\rightarrow$ `GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER`, Confidence 70.72%, Support 2.74%).
  - Luật có Confidence cao nhất đạt **90.47%** (`PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER` $\rightarrow$ `GREEN REGENCY TEACUP AND SAUCER`, Lift 17.66).
  - Đây là căn cứ khoa học vững chắc để thiết kế combo quà tặng và thuật toán gợi ý cross-sell trên giỏ hàng trực tuyến.

---

## 9. Đơn Vị Tiền Tệ, Tài Liệu Tham Khảo & Bản Quyền

- **Đơn vị tiền tệ:** Toàn bộ số liệu tiền tệ trong mã nguồn, biểu đồ và báo cáo sử dụng **Bảng Anh (£ - GBP)** theo đúng dữ liệu gốc từ UCI.
- **Tài liệu phương pháp luận:** Tham khảo chi tiết tại [`docs/business_understanding.md`](docs/business_understanding.md), [`docs/methodology.md`](docs/methodology.md), [`docs/data_dictionary.md`](docs/data_dictionary.md), và [`docs/rubric_evidence_matrix.md`](docs/rubric_evidence_matrix.md).
- **Bản quyền:** Mã nguồn được phân phối theo giấy phép đính kèm trong tệp [LICENSE](LICENSE).