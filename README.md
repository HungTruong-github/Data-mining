# Data Mining & Customer Analytics: Online Retail

Dự án Khai phá Dữ liệu Khách hàng dựa trên tập dữ liệu Giao dịch Bán lẻ Trực tuyến (**Online Retail Dataset**), áp dụng quy trình chuẩn CRISP-DM từ tiền xử lý, phân tích RFM, phân cụm khách hàng (K-Means), phân loại dự đoán hành vi mua lại (Classification), khai phá tập phổ biến & luật kết hợp (Market Basket Analysis), đến xây dựng Dashboard tương tác (Streamlit).

---

## 1. Cấu Trúc Dự Án (Project Structure)

```text
Data-mining/
│
├── data/
│   ├── raw/                              # Dữ liệu gốc (Online Retail.xlsx)
│   ├── interim/                          # Dữ liệu sau bước làm sạch sơ bộ
│   │   ├── cleaned_transactions.csv
│   │   └── valid_invoices.csv
│   └── processed/                        # Dữ liệu phục vụ huấn luyện mô hình
│       ├── customer_features.csv
│       ├── rfm_features.csv
│       ├── repeat_purchase_dataset.csv
│       ├── basket_transactions.csv
│       └── product_features.csv
│
├── notebooks/                            # Jupyter Notebooks nghiên cứu & phân tích
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda_and_cleaning.ipynb
│   ├── 03_feature_engineering_rfm.ipynb
│   ├── 04_customer_clustering.ipynb
│   ├── 05_repeat_purchase_classification.ipynb
│   ├── 06_association_rules.ipynb
│   └── 07_model_comparison_and_insights.ipynb
│
├── src/                                  # Mã nguồn Python chuẩn hóa
│   ├── __init__.py
│   ├── config.py                         # Cấu hình đường dẫn, tham số
│   ├── data_loader.py                    # Đọc và ghi dữ liệu
│   ├── preprocessing.py                  # Làm sạch và tiền xử lý dữ liệu
│   ├── feature_engineering.py            # Tạo đặc trưng khách hàng & giao dịch
│   ├── rfm_analysis.py                   # Tính RFM và phân nhóm khách hàng
│   ├── clustering.py                     # Thuật toán phân cụm K-Means
│   ├── classification.py                 # Huấn luyện mô hình phân loại mua lại
│   ├── association_rules.py              # Khai phá luật kết hợp Apriori/FP-Growth
│   ├── evaluation.py                     # Đo lường và đánh giá mô hình
│   └── visualization.py                  # Hàm vẽ biểu đồ chuyên nghiệp
│
├── models/                               # Lưu trữ trọng số mô hình đã huấn luyện
│   ├── clustering/
│   │   ├── kmeans_model.pkl
│   │   └── scaler.pkl
│   └── classification/
│       ├── logistic_regression.pkl
│       ├── decision_tree.pkl
│       ├── random_forest.pkl
│       └── best_model.pkl
│
├── outputs/                              # Kết quả đầu ra của pipeline
│   ├── figures/                          # Biểu đồ phân tích
│   ├── tables/                           # Báo cáo dạng bảng CSV
│   └── rules/                            # Danh sách các luật kết hợp
│
├── app/                                  # Ứng dụng Web Dashboard
│   └── app.py                            # Streamlit Interactive Dashboard
│
├── docs/                                 # Tài liệu kỹ thuật & nghiệp vụ
│   ├── business_understanding.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   └── team_tasks.md
│
├── report/                               # Báo cáo tổng kết dự án
│   ├── final_report.docx
│   ├── final_report.pdf
│   └── report_figures/
│
├── tests/                                # Unit tests
│   ├── test_preprocessing.py
│   └── test_feature_engineering.py
│
├── run_pipeline.py                       # Script chạy tự động toàn bộ quy trình
├── requirements.txt                      # Danh sách thư viện phụ thuộc
├── README.md                             # Hướng dẫn dự án
├── .gitignore                            # Cấu hình bỏ qua tệp nhạy cảm/rác
└── LICENSE                               # Giấy phép bản quyền
```

---

## 2. Cài Đặt Môi Trường (Installation)

1. **Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv .venv
   # Trên Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Trên Linux/macOS:
   source .venv/bin/activate
   ```

2. **Cài đặt các gói thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Dữ Liệu Đầu Vào (Dataset)

Đặt tệp dữ liệu gốc `Online Retail.xlsx` vào thư mục:
```text
data/raw/Online Retail.xlsx
```
- Nguồn tập dữ liệu: UCI Machine Learning Repository (Online Retail Data Set).
- Tập dữ liệu chứa 541,909 giao dịch từ 01/12/2010 đến 09/12/2011 của một nhà bán lẻ quà tặng trực tuyến tại Vương Quốc Anh.

---

## 4. Hướng Dẫn Sử Dụng (Usage)

### 4.1 Chạy Toàn Bộ Pipeline Tự Động
Thực thi toàn bộ chu trình xử lý dữ liệu, huấn luyện mô hình và lưu kết quả:
```bash
python run_pipeline.py
```

### 4.2 Khởi Chạy Web Dashboard (Streamlit)
Trực quan hóa phân khúc khách hàng, dự đoán và gợi ý sản phẩm:
```bash
streamlit run app/app.py
```

### 4.3 Chạy Kiểm Thử (Unit Tests)
```bash
pytest tests/
```

---

## 5. Quy Trình Khai Phá (Methodology)

Dự án tuân theo chuẩn **CRISP-DM** (Cross-Industry Standard Process for Data Mining):
1. **Business Understanding**: Thấu hiểu mục tiêu giảm rời bỏ, nâng cao LTV và gợi ý sản phẩm bán chéo (Cross-selling).
2. **Data Understanding**: Đánh giá chất lượng dữ liệu, thống kê mô tả, tỷ lệ đơn hủy, khách hàng vãng lai.
3. **Data Preparation**: Xử lý Null CustomerID, UnitPrice <= 0, Quantity < 0, chuẩn hóa ngày tháng.
4. **Modeling**:
   - *Phân cụm*: Phân khúc khách hàng RFM + K-Means (Elbow method & Silhouette score).
   - *Phân loại*: Dự đoán khách hàng có mua lại (Repeat Purchase) sau 30-90 ngày.
   - *Luật kết hợp*: Apriori & FP-Growth phát hiện cặp sản phẩm thường mua cùng nhau.
5. **Evaluation**: Đánh giá dựa trên Accuracy, Precision, Recall, F1-Score, ROC-AUC, Lift, Conviction.
6. **Deployment**: Xây dựng bảng điều khiển tương tác trên Streamlit phục vụ bộ phận kinh doanh/marketing.

---

## 6. Giấy Phép (License)
Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
