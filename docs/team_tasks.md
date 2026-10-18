# Phân Công Nhiệm Vụ & Quản Trị Đóng Góp Nhóm (Track B - Data Mining)

Tài liệu này xác lập khung phân công nhiệm vụ, cơ chế ghi nhận đóng góp, biểu mẫu đánh giá chéo (Peer Assessment) và tuyên bố minh bạch về việc sử dụng công cụ hỗ trợ AI theo chuẩn rubric môn học Data Mining (Cao học - Track B).

---

## 1. Thông Tin Thành Viên Nhóm (3 Thành Viên)

> **Lưu ý quan trọng:** Bảng dưới đây xác lập phân bổ trách nhiệm đồng đều giữa 3 thành viên (mỗi thành viên đóng góp ~33.3%). Nhóm học viên điền thông tin thực tế trước khi nộp báo cáo chính thức.

| STT | Họ và Tên Học Viên | Mã Số Học Viên (MSHV) | Email Học Viên | Vai Trò Chính Trong Dự Án | Tỷ Lệ Đóng Góp (%) | Điểm Đánh Giá Chéo (Peer Score /10) | Ký Tên Xác Nhận |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| 1 | `[Học viên 1 - Họ tên]` | `[MSHV 1]` | `[email1@student...]` | Trưởng nhóm / Data Preparation, Cleaning & Feature Engineering | `33.3%` | `[10/10]` | `[Chữ ký]` |
| 2 | `[Học viên 2 - Họ tên]` | `[MSHV 2]` | `[email2@student...]` | Thành viên / Customer Clustering & Classification Modeling | `33.3%` | `[10/10]` | `[Chữ ký]` |
| 3 | `[Học viên 3 - Họ tên]` | `[MSHV 3]` | `[email3@student...]` | Thành viên / Association Rules Mining, Insights & Streamlit Dashboard | `33.4%` | `[10/10]` | `[Chữ ký]` |

---

## 2. Phân Công Chi Tiết Theo Quy Trình Chuẩn CRISP-DM (Chia Đều 3 Thành Viên)

Khối lượng công việc kỹ thuật được phân chia đồng đều, cân bằng giữa khối lượng tiền xử lý kỹ thuật, nghiên cứu thuật toán học máy và triển khai ứng dụng thực tế:

| Giai đoạn CRISP-DM | Hạng mục công việc kỹ thuật | Đầu ra bàn giao cụ thể (Artifacts) | Trách nhiệm chính | Tỷ trọng |
|:---|:---|:---|:---|:---:|
| **1. Business & Data Understanding** | • Khảo sát bài toán kinh doanh bán lẻ trực tuyến, xác định mục tiêu và câu hỏi nghiên cứu.<br>• Phân tích thống kê mô tả 541,909 dòng dữ liệu gốc, phát hiện hóa đơn hủy C/nợ xấu A, phân loại StockCode phi sản phẩm, khảo sát missing values và phân phối biến. | • `docs/business_understanding.md`<br>• `docs/01_data_understanding_report.md`<br>• `notebooks/01_data_understanding.ipynb`<br>• `outputs/tables/data_understanding/` | **Thành viên 1** (Trưởng nhóm) | 16.7% |
| **2. Data Preparation & Cleaning** | • Xây dựng pipeline làm sạch dữ liệu 10 bước chuẩn hóa không rò rỉ dữ liệu (loại 9,288 dòng hủy C, 3 dòng nợ xấu A, deduplication, điền khuyết Description).<br>• Phân tích ngoại lai IQR, tạo 3 tập dữ liệu trung gian: `cleaned_transactions.csv`, `customer_transactions.csv`, `product_transactions.csv`. | • `src/preprocessing.py`<br>• `docs/02_data_preparation_report.md`<br>• `notebooks/02_eda_and_cleaning.ipynb`<br>• `data/interim/` | **Thành viên 1** (Trưởng nhóm) | 16.6% |
| **3. Feature Engineering & RFM** | • Xây dựng bộ đặc trưng RFM cấp khách hàng (4,334 khách) với quy tắc `reference_date = max_date + 1`.<br>• Phân chia thời gian quan sát (9 tháng) và kiểm chứng (90 ngày cutoff) với zero data leakage; gán nhãn `repeat_purchase_90d` (3,368 mẫu hợp lệ).<br>• Xây dựng ma trận giỏ hàng nhị phân (19,773 hóa đơn x 3,908 sản phẩm). | • `src/feature_engineering.py`<br>• `docs/03_feature_engineering_report.md`<br>• `notebooks/03_feature_engineering_rfm.ipynb`<br>• `data/processed/` | **Thành viên 2** | 16.7% |
| **4. Modeling: Customer Clustering** | • Nghiên cứu tác động biến đổi Log1p + StandardScaler so với dữ liệu gốc (tránh cụm suy biến 99.4%).<br>• Thử nghiệm và đối chiếu 26 cấu hình trên 4 thuật toán: K-Means, GMM, Agglomerative Hierarchical, DBSCAN.<br>• Đánh giá nội bộ qua Silhouette, Davies-Bouldin, Calinski-Harabasz; kiểm định độ ổn định ARI qua 5 random seeds; xây dựng hồ sơ cụm (`Best Customers` vs `Lost Customers`). | • `src/clustering.py`<br>• `docs/04_customer_clustering_report.md`<br>• `notebooks/04_customer_clustering.ipynb`<br>• `models/clustering/` | **Thành viên 2** | 16.6% |
| **5. Modeling: Repeat Purchase Classification** | • Xây dựng pipeline học máy phân loại khách hàng mua lại 90 ngày; thiết lập 5-fold Stratified Cross-Validation.<br>• Huấn luyện và so sánh mô hình cơ sở DummyClassifier với Logistic Regression, Decision Tree, Random Forest.<br>• Tối ưu hóa siêu tham số (GridSearch 40 cấu hình), kiểm soát overfitting, đồng bộ quy tắc ngưỡng quyết định xác suất $P \ge 0.50$. | • `src/classification.py`<br>• `docs/05_repeat_purchase_classification_report.md`<br>• `notebooks/05_repeat_purchase_classification.ipynb`<br>• `models/classification/` | **Thành viên 3** | 16.7% |
| **6. Market Basket Association Rules & Dashboard** | • Khai phá luật kết hợp giỏ hàng bằng 2 thuật toán Apriori và FP-Growth; kiểm toán tính tương đương (Equivalence Audit: 61 luật trùng khớp 100%).<br>• Phân tích độ nhạy (sensitivity analysis) qua lưới min_support và min_confidence; diễn giải giá trị Lift.<br>• Phát triển Streamlit Dashboard tương tác 4 tab kết nối trực tiếp với các pipeline mô hình; xây dựng Kế hoạch hành động 10 cột. | • `src/association_rules.py`<br>• `app/app.py`<br>• `docs/06_association_rules_report.md`<br>• `docs/07_model_comparison_and_insights_report.md`<br>• `notebooks/06_association_rules.ipynb`<br>• `notebooks/07_model_comparison_and_insights.ipynb` | **Thành viên 3** | 16.7% |

> **Tổng kết phân bổ:**
> - **Thành viên 1:** 33.3% (Phụ trách Giai đoạn 1 & 2: Hiểu bài toán, Thấu hiểu dữ liệu và Pipeline Tiền xử lý làm sạch).
> - **Thành viên 2:** 33.3% (Phụ trách Giai đoạn 3 & 4: Kỹ thuật đặc trưng RFM, Temporal Split và Phân cụm khách hàng).
> - **Thành viên 3:** 33.4% (Phụ trách Giai đoạn 5 & 6: Mô hình phân loại mua lại, Khai phá luật kết hợp, Action Plan và Web Dashboard).

---

## 3. Tiêu Chí Đánh Giá Chéo Nội Bộ (Peer Assessment Framework)

Các thành viên đánh giá chéo nhau theo thang điểm 1–5 cho từng tiêu chí, sau đó quy đổi về điểm tổng hợp /10:

1. **Chất lượng kỹ thuật (Technical Quality - 30%):** Code đúng logic, tái lập được, tuân thủ nguyên tắc Clean Code và cấu trúc module hóa.
2. **Tiến độ và tính kỷ luật (Timeliness & Reliability - 25%):** Hoàn thành module đúng thời hạn cam kết, tham gia đầy đủ các buổi họp kỹ thuật.
3. **Đóng góp ý tưởng & Giải quyết vấn đề (Problem Solving - 20%):** Chủ động phân tích phương án mô hình, đề xuất giải pháp khi gặp trở ngại kỹ thuật.
4. **Tài liệu & Báo cáo (Documentation - 15%):** Viết báo cáo rõ ràng, giải thích cặn kẽ số liệu thực nghiệm, chú thích mã nguồn đầy đủ.
5. **Tinh thần đồng đội & Phối hợp (Team Collaboration - 10%):** Phối hợp mượt mà trên Git, tôn trọng phản biện, hỗ trợ sửa lỗi liên module.

### Bảng Tổng Hợp Đánh Giá Chéo (3 Thành Viên)

| Thành viên được đánh giá | Điểm TB Tiêu chí 1 (30%) | Điểm TB Tiêu chí 2 (25%) | Điểm TB Tiêu chí 3 (20%) | Điểm TB Tiêu chí 4 (15%) | Điểm TB Tiêu chí 5 (10%) | Điểm Tổng Hợp (/10) | Đề xuất hệ số đóng góp ($K_i$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `[Học viên 1]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[10.0]` | `1.00` |
| `[Học viên 2]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[10.0]` | `1.00` |
| `[Học viên 3]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[5.0]` | `[10.0]` | `1.00` |

---

## 4. Tuyên Bố Minh Bạch Về Việc Sử Dụng Công Cụ Hỗ Trợ AI (AI Assistance Declaration)

Tuân thủ quy định học thuật và hướng dẫn liêm chính nghiên cứu của chương trình Cao học:

- **Công cụ hỗ trợ sử dụng:** Google Antigravity IDE / Gemini 3.8 Flash Agentic Coding Assistant.
- **Phạm vi và vai trò của công cụ AI:**
  1. Hỗ trợ rà soát mã nguồn (code review), phát hiện trùng lặp mã và tách module độc lập (`src/model_comparison_figures.py`).
  2. Hỗ trợ chuẩn hóa bộ kiểm thử tự động nhằm nâng cao độ bao phủ kiểm thử kỹ thuật.
  3. Hỗ trợ định dạng tài liệu, bảng minh chứng kiểm định (acceptance criteria matrix) và đóng gói báo cáo.
- **Trách nhiệm và kiểm soát của học viên:**
  1. Toàn bộ định hướng nghiệp vụ, lựa chọn mô hình toán học (CRISP-DM), thẩm định kết quả phân cụm/phân loại/luật kết hợp đều do 3 học viên trong nhóm trực tiếp nghiên cứu, thực hiện và chịu trách nhiệm học thuật.
  2. Mọi đoạn mã do công cụ AI hỗ trợ đều được các thành viên trong nhóm rà soát, chạy thực nghiệm kiểm chứng độc lập trên môi trường chuẩn của dự án.
  3. Không sử dụng AI để ngụy tạo số liệu, không sinh kết quả giả; toàn bộ số liệu đều được trích xuất trực tiếp từ các file kết quả thực nghiệm chuẩn xác (`outputs/tables/` và `models/`).

