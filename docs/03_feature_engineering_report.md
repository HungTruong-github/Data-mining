# Báo Cáo Tổng Hợp Giai Đoạn 03: Feature Engineering & RFM Analysis

**Dự án:** Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM:** **Phase 3 - Data Preparation (Feature Engineering & Labeling)**  
**Dữ liệu đầu vào:** `data/interim/customer_transactions.csv` (391,153 dòng, 4,334 khách hàng định danh)  
**Tài liệu tham chiếu:** [Methodology](methodology.md) | [Feature Decision Dictionary](feature_decision_dictionary.md) | [Decision Log](decision_log.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Feature Engineering & RFM Analysis** chịu trách nhiệm biến đổi dữ liệu giao dịch cấp hóa đơn thành các đặc trưng hành vi cấp khách hàng (Customer-level aggregated features), thiết lập nhãn mục tiêu cho bài toán học máy có giám sát (Supervised Classification) và xây dựng ma trận giỏ hàng (Market Basket Matrix).

Toàn bộ logic nghiệp vụ được module hóa trong [`src/feature_engineering.py`](../src/feature_engineering.py), cấu hình tham số tập trung tại [`src/config.py`](../src/config.py), thực thi tự động qua [`notebooks/run_03_feature_engineering.py`](../notebooks/run_03_feature_engineering.py) và nghiên cứu trực quan trong notebook [`notebooks/03_feature_engineering_rfm.ipynb`](../notebooks/03_feature_engineering_rfm.ipynb).

### Các kết quả then chốt đạt được:
- **Tập khách hàng hợp lệ đưa vào phân tích:** **4,334** khách hàng định danh duy nhất (không có giá trị thiếu CustomerID, không có bản ghi trùng lặp).
- **Bộ đặc trưng RFM truyền thống và mở rộng:** 20 thuộc tính hành vi mua sắm được trích xuất (Recency, Frequency, Monetary, TotalItems, UniqueProducts, ActiveDays, AverageOrderValue, CustomerLifetimeDays,...).
- **Phân khúc RFM Heuristic (Mô tả ban đầu):** Phân chia 4,334 khách hàng thành 6 nhóm hành vi: Champions (21.04%), Loyal Customers (12.76%), Big Spenders (10.59%), Recent Customers (13.45%), Regular Customers (39.55%), At Risk (2.61%).
- **Cơ chế phân chia thời gian chống rò rỉ dữ liệu (Zero Data Leakage Temporal Split):**
  - Mốc thời gian tham chiếu (`reference_date`): **2011-12-10 12:50:00** (max ngày giao dịch + 1 ngày).
  - Mốc chia cửa sổ (`cutoff_date`): **2011-09-10 12:50:00** (90 ngày trước `reference_date`).
  - Cửa sổ quan sát (Observation Window): Giai đoạn 9 tháng đầu (trước cutoff) dùng để tạo biến độc lập $X$.
  - Cửa sổ dự báo (Prediction Window): Giai đoạn 90 ngày cuối dùng để tạo nhãn phụ thuộc $Y$ (`repeat_purchase_90d`).
- **Tập mẫu huấn luyện phân loại:** **3,368** khách hàng có giao dịch trước cutoff. Tỷ lệ lớp: **1,919** khách quay lại mua (nhãn 1 - **56.98%**) và **1,449** khách không quay lại (nhãn 0 - **43.02%**).
- **Ma trận giỏ hàng (Market Basket Matrix):** **19,773** hóa đơn hợp lệ chứa **3,908** mặt hàng độc nhất, phục vụ trực tiếp cho thuật toán khai phá luật kết hợp ở Bước 06.

---

## 2. Phương Pháp Luận Trích Xuất Đặc Trưng RFM

### 2.1. Định nghĩa toán học các biến RFM
Mỗi khách hàng $i$ được tổng hợp trên tập giao dịch lịch sử theo các công thức:

1. **Recency ($R_i$):** Số ngày kể từ lần giao dịch cuối cùng của khách hàng tới ngày tham chiếu mốc:
   $$R_i = \lceil \text{reference\_date} - \max(t_{i}) \rceil \quad (\text{đơn vị: ngày})$$
   *Quy ước:* `reference_date = max(InvoiceDate) + 1 ngày`, đảm bảo $R_i \ge 1$ ngày với mọi khách hàng, thuận tiện cho biến đổi $\log(1+R_i)$.

2. **Frequency ($F_i$):** Số lượng hóa đơn giao dịch độc nhất của khách hàng:
   $$F_i = \text{CountDistinct}(\text{InvoiceNo}_i)$$
   *Lưu ý cốt lõi:* Đếm số lượng hóa đơn, không đếm số dòng mặt hàng (line items), nhằm phản ánh chính xác số lần phát sinh quyết định ghé mua sắm.

3. **Monetary ($M_i$):** Tổng giá trị tiền tệ chi tiêu tích lũy của khách hàng:
   $$M_i = \sum_{j \in \text{Orders}_i} (\text{Quantity}_j \times \text{UnitPrice}_j) \quad (\text{đơn vị: GBP } £)$$

### 2.2. Các đặc trưng hành vi mở rộng
Ngoài bộ 3 RFM, hệ thống tính toán thêm các biến dẫn xuất:
- `TotalItems`: Tổng số lượng đơn vị sản phẩm đã mua ($\sum \text{Quantity}$).
- `UniqueProducts`: Số lượng mã `StockCode` phân biệt khách hàng từng lựa chọn.
- `ActiveDays`: Số ngày phân biệt có phát sinh đơn hàng.
- `AverageOrderValue (AOV)`: Giá trị trung bình trên mỗi đơn hàng ($M_i / F_i$).
- `AverageItemsPerInvoice`: Số lượng sản phẩm trung bình trong một đơn hàng ($\text{TotalItems}_i / F_i$).
- `CustomerLifetimeDays`: Khoảng thời gian từ đơn hàng đầu tiên đến đơn hàng gần nhất ($\max(t_i) - \min(t_i)$).
- `Country`: Quốc gia chính mà khách hàng đăng ký giao dịch.

---

## 3. Thống Kê Phân Phối RFM & Biến Đổi Log1p

Do đặc thù bán lẻ thương mại điện tử, các biến hành vi khách hàng có phân phối lệch phải cực mạnh (heavy right-skewed) với đuôi dài do sự hiện diện của nhóm khách sỉ B2B:

| Đặc trưng | Min | Q1 (25%) | Median (50%) | Mean | Q3 (75%) | P99 (99%) | Max | Skewness ban đầu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recency (ngày)** | 1.00 | 18.00 | 51.00 | 92.58 | 143.00 | 369.00 | 374.00 | +1.25 |
| **Frequency (đơn)** | 1.00 | 1.00 | 2.00 | 4.27 | 5.00 | 30.00 | 210.00 | +11.75 |
| **Monetary (£)** | 3.75 | 307.25 | 674.45 | 2,053.79 | 1,661.64 | 18,521.84 | 280,206.02 | +21.71 |

### Giải pháp Log Transformation:
Để đưa hình học không gian khoảng cách Euclid của các thuật toán phân cụm (K-Means, GMM) về dạng chuẩn hóa ổn định, hàm $\log(1+x)$ được áp dụng:
$$x_{\text{transformed}} = \ln(1 + x)$$
Biến đổi này nén dải phân bố của `Monetary` từ $[3.75; 280,206]$ về $[1.56; 12.54]$ và `Frequency` từ $[1; 210]$ về $[0.69; 5.35]$, giảm hệ số lệch skewness của Monetary từ **21.71 xuống 0.38** (xấp xỉ phân phối chuẩn Gauss).

---

## 4. Phân Khúc RFM Heuristic Dựa Trên Quantile (RFM Scoring)

Mỗi chiều $R, F, M$ được chia thành 5 phân vị (Quantile 1 đến 5) bằng hàm `pd.qcut`:
- Điểm **R_Score**: 5 (mua gần nhất) $\rightarrow$ 1 (đã lâu không mua).
- Điểm **F_Score**: 1 (mua 1 lần) $\rightarrow$ 5 (mua rất thường xuyên).
- Điểm **M_Score**: 1 (chi tiêu thấp nhất) $\rightarrow$ 5 (chi tiêu cao nhất).
- **RFM_Score tổng hợp:** Tổng giá trị $R + F + M \in [3; 15]$.

### Bảng tổng kết phân khúc mô tả heuristic:
*Trích xuất từ `outputs/tables/feature_engineering/rfm_segment_summary.csv`:*

| Phân khúc RFM | Số khách hàng | Tỷ lệ (%) | Recency TB (ngày) | Frequency TB (đơn) | Monetary TB (£) | Doanh số phân khúc (£) | AOV TB (£) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regular Customers** | 1,714 | 39.55% | 165.42 | 1.37 | £345.41 | £592,028.67 | £266.12 |
| **Champions** | 912 | 21.04% | 12.47 | 11.44 | £6,165.47 | £5,622,904.48 | £454.55 |
| **Recent Customers** | 583 | 13.45% | 17.64 | 1.79 | £408.33 | £238,054.92 | £239.26 |
| **Loyal Customers** | 553 | 12.76% | 70.48 | 5.71 | £2,045.08 | £1,130,926.49 | £340.35 |
| **Big Spenders** | 459 | 10.59% | 88.10 | 2.30 | £2,390.03 | £1,097,022.54 | £1,272.35 |
| **At Risk** | 113 | 2.61% | 151.90 | 3.18 | £498.15 | £56,290.54 | £162.13 |

> **Nhận định:** Nhóm **Champions** chỉ chiếm 21.04% số lượng khách hàng nhưng tạo ra **£5,622,904.48 (63.2% tổng doanh thu)** của doanh nghiệp. Đây là phân khúc cốt lõi cần chính sách chăm sóc đặc biệt.

---

## 5. Thiết Kế Nhãn Mua Lại (Repeat Purchase 90-Day Target)

### 5.1. Cơ chế phân chia thời gian (Temporal Split)
Để mô hình hóa chính xác bài toán dự báo trong kinh doanh mà không vi phạm nguyên tắc khoa học dữ liệu (tránh hoàn toàn hiện tượng Look-ahead bias / Data leakage):
- **Giai đoạn tính đặc trưng (Observation window):** Từ `2010-12-01` đến `2011-09-10 12:50:00`.
- **Giai đoạn xác định nhãn (Prediction window):** Từ `2011-09-10 12:50:00` đến `2011-12-09 12:50:00` (đúng 90 ngày).
- **Điều kiện khách hàng hợp lệ:** Khách hàng phải có tối thiểu 1 giao dịch trong giai đoạn tính đặc trưng ($N = 3,368$ khách). 966 khách hàng xuất hiện lần đầu sau ngày cutoff không tham gia tập huấn luyện vì không có dữ liệu lịch sử để dự đoán.

### 5.2. Định nghĩa biến mục tiêu:
$$Y_i = \begin{cases} 1 & \text{nếu khách hàng } i \text{ có } \ge 1 \text{ giao dịch trong } [T_{\text{cutoff}}, T_{\text{end}}] \\ 0 & \text{nếu khách hàng } i \text{ không có giao dịch nào trong } [T_{\text{cutoff}}, T_{\text{end}}] \end{cases}$$

### 5.3. Phân phối lớp nhãn mục tiêu:
- **Lớp 1 (Có mua lại):** 1,919 khách hàng (**56.98%**).
- **Lớp 0 (Không mua lại):** 1,449 khách hàng (**43.02%**).
- Tỷ lệ mất cân bằng lớp ở mức vừa phải ($1.32 : 1$), tạo điều kiện thuận lợi cho việc huấn luyện các mô hình phân loại học máy ở Bước 05.

---

## 6. Danh Mục Tệp Đầu Ra (Deliverable Artifacts)

### 6.1. Dữ liệu xử lý (`data/processed/`)
- `rfm_customer_features.csv`: 4,334 dòng × 20 cột (bảng đặc trưng RFM đầy đủ).
- `repeat_purchase_features.csv`: 3,368 dòng × 12 cột (bộ đặc trưng độc lập và nhãn `repeat_purchase_90d`).
- `rfm_clustering_features.csv`: 4,334 dòng × 4 cột (CustomerID, Recency, Frequency, Monetary).
- `association_basket_matrix.csv`: 19,773 dòng × 3,908 cột (ma trận nhị phân 0/1 giỏ hàng).

### 6.2. Bảng thống kê (`outputs/tables/feature_engineering/`)
- `feature_engineering_summary.csv`: Thống kê tổng hợp số lượng khách hàng, ngày cutoff và tỷ lệ nhãn.
- `feature_summary.csv`: Tóm tắt kiểu dữ liệu, min, max, median, mean của các biến tạo ra.
- `rfm_segment_summary.csv`: Thống kê chỉ số kinh doanh chi tiết theo 6 phân khúc RFM.
- `repeat_purchase_class_balance.csv`: Bảng phân bố tần suất và tỷ lệ phần trăm lớp nhãn 0/1.

### 6.3. Biểu đồ trực quan hóa (`outputs/figures/feature_engineering/`)
- `rfm_distribution.png`: Biểu đồ histogram và đường cong mật độ phân phối R, F, M gốc.
- `rfm_log_transform_comparison.png`: Biểu đồ so sánh phân phối trước và sau biến đổi Log1p.
- `rfm_score_distribution.png`: Biểu đồ cột phân bố điểm số R, F, M từ 1 đến 5.
- `rfm_segment_overview.png`: Biểu đồ phân bổ số lượng khách hàng và tỷ trọng doanh thu giữa các nhóm RFM.
- `repeat_purchase_label_balance.png`: Biểu đồ thể hiện tỷ lệ nhãn mua lại 90 ngày.
- `rfm_correlation_matrix.png`: Ma trận tương quan nhiệt Pearson giữa các đặc trưng khách hàng.
