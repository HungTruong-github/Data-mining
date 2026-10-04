# Phân Công Nhiệm Vụ & Quản Trị Đóng Góp Nhóm (Track B - Data Mining)

Tài liệu này xác lập khung phân công nhiệm vụ, cơ chế ghi nhận đóng góp, biểu mẫu đánh giá chéo (Peer Assessment) và tuyên bố minh bạch về việc sử dụng công cụ hỗ trợ AI theo chuẩn rubric môn học Data Mining (Cao học - Track B).

---

## 1. Thông Tin Thành Viên Nhóm (Mẫu Placeholder Cần Hoàn Thiện)

> **Lưu ý quan trọng:** Bảng dưới đây để ở định dạng mẫu trung lập (placeholder). Nhóm học viên điền thông tin thực tế trước khi nộp báo cáo chính thức. Tuyệt đối không tự ý gán ghép thông tin định danh giả mạo.

| STT | Họ và Tên Học Viên | Mã Số Học Viên (MSHV) | Email Học Viên | Vai Trò Chính Trong Dự Án | Tỷ Lệ Đóng Góp (%) | Điểm Đánh Giá Chéo (Peer Score /10) | Ký Tên Xác Nhận |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| 1 | `[Học viên 1 - Họ tên]` | `[MSHV 1]` | `[email1@student...]` | Trưởng nhóm / Data Preparation & Feature Eng. | `[25% / TBD]` | `[TBD]` | `[Chữ ký]` |
| 2 | `[Học viên 2 - Họ tên]` | `[MSHV 2]` | `[email2@student...]` | Thành viên / Customer Clustering & RFM | `[25% / TBD]` | `[TBD]` | `[Chữ ký]` |
| 3 | `[Học viên 3 - Họ tên]` | `[MSHV 3]` | `[email3@student...]` | Thành viên / Repeat Purchase Classification | `[25% / TBD]` | `[TBD]` | `[Chữ ký]` |
| 4 | `[Học viên 4 - Họ tên]` | `[MSHV 4]` | `[email4@student...]` | Thành viên / Association Rules & Streamlit App | `[25% / TBD]` | `[TBD]` | `[Chữ ký]` |
| *(5)* | `[Học viên 5 (nếu có)]` | `[MSHV 5]` | `[email5@student...]` | Thành viên / Model Governance & Báo Cáo Tổng Hợp | `[TBD]` | `[TBD]` | `[Chữ ký]` |

---

## 2. Phân Công Chi Tiết Theo Quy Trình Chuẩn CRISP-DM

| Giai đoạn CRISP-DM | Hạng mục công việc kỹ thuật | Đầu ra bàn giao cụ thể (Artifacts) | Trách nhiệm chính |
|:---|:---|:---|:---|
| **1. Business Understanding** | Xác định bài toán kinh doanh bán lẻ trực tuyến, câu hỏi nghiên cứu, định vị mục tiêu tăng AOV & giữ chân khách hàng | `docs/business_understanding.md`, mục tiêu phân tích | Thành viên 1 / Cả nhóm |
| **2. Data Understanding** | Thống kê mô tả 541,909 giao dịch, phân tích Invoice C/A, StockCode phi sản phẩm, cấu trúc phân phối | `notebooks/run_01_data_understanding.py`, `outputs/tables/data_understanding/` | Thành viên 1 |
| **3. Data Preparation** | Xây dựng pipeline làm sạch (loại hủy hàng, deduplication, IQR outlier), chuẩn hóa dữ liệu giao dịch | `src/preprocessing.py`, `data/interim/cleaned_transactions.csv` | Thành viên 1 |
| **4. Feature Engineering** | Tính toán RFM, gán nhãn repeat purchase 90 ngày (không rò rỉ dữ liệu - zero leakage), tạo ma trận giỏ hàng | `src/feature_engineering.py`, `data/processed/` | Thành viên 1 & 2 |
| **5. Modeling: Clustering** | Thử nghiệm 26 cấu hình (K-Means, GMM, Agglomerative, DBSCAN), kiểm định ổn định ARI qua đa seed, tác động chuẩn hóa | `src/clustering.py`, `notebooks/run_04_customer_clustering.py`, `models/clustering/` | Thành viên 2 |
| **6. Modeling: Classification** | Huấn luyện phân loại khách hàng mua lại, 5-fold Stratified CV, GridSearch siêu tham số, đồng bộ ngưỡng quyết định | `src/classification.py`, `notebooks/run_05_repeat_purchase_classification.py`, `models/classification/` | Thành viên 3 |
| **7. Modeling: Association Rules** | Khai phá luật kết hợp Apriori & FP-Growth, benchmark thời gian thực thi (median/IQR), phân tích độ nhạy | `src/association_rules.py`, `notebooks/run_06_association_rules.py`, `outputs/tables/association_rules/` | Thành viên 4 |
| **8. Model Evaluation & Insights** | Tổng hợp so sánh đa mô hình, xây dựng Kế hoạch hành động 10 cột, biểu đồ dashboard tổng hợp | `src/model_comparison.py`, `src/model_comparison_figures.py`, `src/insights.py` | Cả nhóm |
| **9. Deployment & Governance** | Xây dựng Streamlit App tương tác 4 tab, tài liệu rubric evidence matrix, nghiệm thu kiểm thử tự động 100% pass | `app/app.py`, `run_pipeline.py`, `docs/rubric_evidence_matrix.md` | Thành viên 4 & Trưởng nhóm |

---

## 3. Tiêu Chí Đánh Giá Chéo Nội Bộ (Peer Assessment Framework)

Các thành viên đánh giá chéo nhau theo thang điểm 1–5 cho từng tiêu chí, sau đó quy đổi về điểm tổng hợp /10:

1. **Chất lượng kỹ thuật (Technical Quality - 30%):** Code đúng logic, tái lập được, nghiệm thu kiểm định kỹ thuật đạt chuẩn clean code.
2. **Tiến độ và tính kỷ luật (Timeliness & Reliability - 25%):** Hoàn thành module đúng thời hạn cam kết, tham gia đầy đủ các buổi họp kỹ thuật.
3. **Đóng góp ý tưởng & Giải quyết vấn đề (Problem Solving - 20%):** Chủ động phân tích phương án mô hình, đề xuất giải pháp khi gặp trở ngại kỹ thuật.
4. **Tài liệu & Báo cáo (Documentation - 15%):** Viết báo cáo rõ ràng, giải thích cặn kẽ số liệu thực nghiệm, chú thích mã nguồn đầy đủ.
5. **Tinh thần đồng đội & Phối hợp (Team Collaboration - 10%):** Phối hợp mượt mà trên Git, tôn trọng phản biện, hỗ trợ sửa lỗi liên module.

### Bảng Tổng Hợp Đánh Giá Chéo (Mẫu Điền Thực Tế)

| Thành viên được đánh giá | Điểm TB Tiêu chí 1 | Điểm TB Tiêu chí 2 | Điểm TB Tiêu chí 3 | Điểm TB Tiêu chí 4 | Điểm TB Tiêu chí 5 | Điểm Tổng Hợp (/10) | Đề xuất hệ số đóng góp ($K_i$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `[Học viên 1]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `1.00` |
| `[Học viên 2]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `1.00` |
| `[Học viên 3]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `1.00` |
| `[Học viên 4]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `1.00` |

---

## 4. Tuyên Bố Minh Bạch Về Việc Sử Dụng Công Cụ Hỗ Trợ AI (AI Assistance Declaration)

Tuân thủ quy định học thuật và hướng dẫn liêm chính nghiên cứu của chương trình Cao học:

- **Công cụ hỗ trợ sử dụng:** Google Antigravity IDE / Gemini 3.8 Flash Agentic Coding Assistant.
- **Phạm vi và vai trò của công cụ AI:**
  1. Hỗ trợ rà soát mã nguồn (code review), phát hiện trùng lặp mã (duplicate functions) và tách module độc lập (`src/model_comparison_figures.py`).
  2. Hỗ trợ chuẩn hóa bộ kiểm thử tự động (pytest suite) nhằm nâng cao độ bao phủ kiểm thử kỹ thuật.
  3. Hỗ trợ định dạng tài liệu, bảng minh chứng kiểm định (acceptance criteria matrix) và đóng gói gói bàn giao (packaging script).
- **Trách nhiệm và kiểm soát của học viên:**
  1. Toàn bộ định hướng nghiệp vụ, lựa chọn mô hình toán học (CRISP-DM), thẩm định kết quả phân cụm/phân loại/luật kết hợp đều do học viên thực hiện và chịu trách nhiệm học thuật.
  2. Mọi đoạn mã do công cụ AI hỗ trợ đều được các thành viên trong nhóm rà soát, chạy thực nghiệm kiểm chứng độc lập trên môi trường chuẩn của dự án.
  3. Không sử dụng AI để ngụy tạo số liệu, không sinh kết quả giả mà toàn bộ số liệu đều trích xuất trực tiếp từ các file kết quả thực nghiệm (`outputs/tables/`).
