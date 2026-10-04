# Data Mining & Customer Analytics: Online Retail

Dự án Khai phá Dữ liệu Khách hàng dựa trên tập dữ liệu Giao dịch Bán lẻ Trực tuyến (**Online Retail Dataset** — UCI ML Repository), áp dụng quy trình chuẩn CRISP-DM.

**Phạm vi phân tích:**
- Phân tích RFM và phân cụm khách hàng (K-Means, GMM, Agglomerative, DBSCAN)
- Phân loại dự đoán hành vi mua lại (Logistic Regression, Decision Tree, Random Forest)
- Khai phá luật kết hợp (Apriori vs FP-Growth)
- Dashboard tương tác (Streamlit)

**Nguồn dữ liệu:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — 541,909 giao dịch từ UK-based online retail, 2010-12-01 → 2011-12-09.

---

## 1. Cấu Trúc Dự Án

```text
Data-mining/
├── data/
│   ├── raw/                              # Dữ liệu gốc
│   │   └── Online Retail.xlsx
│   ├── interim/                          # Dữ liệu sau làm sạch
│   │   ├── cleaned_transactions.csv
│   │   ├── customer_transactions.csv
│   │   └── product_transactions.csv
│   └── processed/                        # Features cho modeling
│       ├── rfm_customer_features.csv
│       ├── rfm_clustering_features.csv
│       ├── repeat_purchase_features.csv
│       ├── customer_clusters.csv
│       ├── association_basket_long.csv
│       └── association_basket_matrix.csv
│
├── notebooks/                            # Jupyter Notebooks + Runner scripts
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda_and_cleaning.ipynb
│   ├── 03_feature_engineering_rfm.ipynb
│   ├── 04_customer_clustering.ipynb
│   ├── 05_repeat_purchase_classification.ipynb
│   ├── 06_association_rules.ipynb
│   ├── 07_model_comparison_and_insights.ipynb
│   └── run_*.py                          # Runner scripts (cùng logic với notebooks)
│
├── src/                                  # Source modules
│   ├── config.py                         # Cấu hình đường dẫn, tham số
│   ├── data_loader.py                    # Đọc dữ liệu gốc
│   ├── preprocessing.py                  # Làm sạch và tiền xử lý
│   ├── feature_engineering.py            # Tạo features RFM, behavioral, basket
│   ├── clustering.py                     # Thuật toán phân cụm
│   ├── classification.py                 # Huấn luyện classifier
│   ├── association_rules.py              # Apriori & FP-Growth
│   ├── evaluation.py                     # Đánh giá mô hình
│   ├── model_comparison.py               # So sánh và báo cáo
│   ├── insights.py                       # Sinh business insights
│   └── visualization.py                  # Biểu đồ
│
├── models/                               # Model artifacts
│   ├── clustering/
│   │   ├── clustering_model.pkl
│   │   ├── clustering_config.pkl
│   │   └── scaler.pkl
│   └── classification/
│       ├── best_classifier_pipeline.joblib
│       ├── classification_metadata.json
│       └── *_pipeline.joblib
│
├── outputs/                              # Kết quả
│   ├── figures/                          # Biểu đồ PNG
│   ├── tables/                           # Bảng CSV
│   ├── reports/                          # Báo cáo Markdown
│   └── evidence/                         # Manifest, logs
│
├── app/                                  # Streamlit Dashboard
│   └── app.py
│
├── docs/                                 # Tài liệu
│   ├── business_understanding.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── feature_decision_dictionary.md
│   ├── decision_log.md
│   ├── references.md
│   └── rubric_evidence_matrix.md
│
├── tests/                                # Automated tests (pytest suite, 100% pass)
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_classification.py
│   ├── test_model_comparison.py
│   └── test_app_smoke.py
│
├── run_pipeline.py                       # Pipeline đầu–cuối (01→07)
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Cài Đặt

### Yêu cầu
- Python 3.10+
- Raw data file: `data/raw/Online Retail.xlsx`

### Cài đặt dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Tải dữ liệu

Tải **Online Retail Dataset** từ UCI và đặt vào `data/raw/`:
- URL: https://archive.ics.uci.edu/dataset/352/online+retail
- File: `Online Retail.xlsx` (≈23 MB)

## 3. Chạy Pipeline

### Chạy toàn bộ pipeline (01→07):

```bash
python run_pipeline.py
```

Pipeline chạy tuần tự 7 bước, mỗi bước kiểm tra output trước khi tiếp tục.
Thời gian ước tính: ~7-8 phút.

### Chạy từng bước riêng:

```bash
python notebooks/run_01_data_understanding.py
python notebooks/run_02_eda_and_cleaning.py
python notebooks/run_03_feature_engineering.py
python notebooks/run_04_customer_clustering.py
python notebooks/run_05_repeat_purchase_classification.py
python notebooks/run_06_association_rules.py
python notebooks/run_07_model_comparison_and_insights.py
```

### Chạy tests:

```bash
python -m pytest -q
```

### Chạy Dashboard:

```bash
python -m streamlit run app/app.py
```

## 4. Kết Quả Chính

### 4.1 Customer Clustering
- Thuật toán: K-Means (K=2) trên RFM features (log1p + StandardScaler)
- Silhouette Score: 0.4330, Davies-Bouldin: 0.8917
- 2 phân khúc: Best Customers (38.4%) và Low-Value Customers (61.6%)

### 4.2 Repeat Purchase Classification
- Model: Random Forest (`n_estimators: 100, max_depth: 5, min_samples_leaf: 5, class_weight: None`)
- CV F1: 0.7102, Test F1: 0.7375, Test AUC: 0.7438
- Baseline (DummyClassifier): CV F1 0.7259 (không phân biệt khách hàng, ROC-AUC = 0.5000)
- Target: repeat_purchase_90d (mua lại trong 90 ngày)
- Ngưỡng suy luận thống nhất: `probability >= 0.5` đồng bộ giữa CV, test evaluation và Dashboard

### 4.3 Association Rules
- Thuật toán được chọn: Được chọn động theo runtime benchmark (Apriori hoặc FP-Growth; cả hai giải cùng bài toán frequent itemsets)
- 61 valid rules (lift > 1) trùng khớp tuyệt đối giữa Apriori và FP-Growth (max diff = 0.0)
- min_support=0.02 (tương đương 396 giỏ hàng trên 19,792 giao dịch), min_confidence=0.5
- Top rule lift: ~18.2 (PINK/GREEN/ROSES REGENCY TEACUP sets)

## 5. Tài Liệu Tham Khảo

Xem thư mục `docs/` cho:
- `business_understanding.md` — Bài toán nghiệp vụ
- `methodology.md` — Phương pháp CRISP-DM
- `data_dictionary.md` — Mô tả dữ liệu
- `feature_decision_dictionary.md` — Từ điển features
- `decision_log.md` — Nhật ký quyết định
- `references.md` — Tài liệu tham khảo

## 6. Đơn Vị Tiền Tệ

Tất cả giá trị tiền tệ trong project sử dụng **GBP (£)** — đồng Bảng Anh, theo dữ liệu gốc.

## 7. License

Xem file `LICENSE` để biết chi tiết.