# Báo Cáo Tổng Hợp Giai Đoạn 02: Data Preparation & Cleaning

**Dự án**: Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM**: **Phase 2 - Data Preparation**  
**Tập dữ liệu gốc**: `data/raw/Online Retail.csv` (541,909 dòng × 8 cột)  
**Tài liệu tham chiếu**: [Rubric Track B](file:///d:/Data-mininng/docs/rubik.md) | [Methodology](file:///d:/Data-mininng/docs/methodology.md) | [Data Dictionary](file:///d:/Data-mininng/docs/data_dictionary.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Data Preparation & Cleaning** đã hoàn thành toàn diện theo tiêu chuẩn CRISP-DM và rubric Track B. Toàn bộ mã nguồn xử lý được module hóa thành các hàm tái sử dụng trong [`src/preprocessing.py`](file:///d:/Data-mininng/src/preprocessing.py), cấu hình đường dẫn tập trung trong [`src/config.py`](file:///d:/Data-mininng/src/config.py), thực thi tự động qua script [`notebooks/run_02_eda_and_cleaning.py`](file:///d:/Data-mininng/notebooks/run_02_eda_and_cleaning.py) và thể hiện chi tiết kèm kết quả trong notebook [`notebooks/02_eda_and_cleaning.ipynb`](file:///d:/Data-mininng/notebooks/02_eda_and_cleaning.ipynb).

### Các chỉ số chính sau khi làm sạch:
- **Dữ liệu gốc ban đầu**: **541,909** dòng.
- **Tổng số dòng bị loại khỏi giao dịch mua hợp lệ**: **19,338** dòng (~3.57%).
  - Hóa đơn hủy (`C`): **9,288** dòng (3,836 hóa đơn).
  - Bút toán điều chỉnh nợ xấu kế toán (`A`): **3** dòng (3 hóa đơn).
  - Số lượng âm hoặc bằng 0 (`Quantity <= 0` ngoài C/A): **1,336** dòng.
  - Đơn giá không hợp lệ (`UnitPrice <= 0`): **1,179** dòng.
  - Bản ghi trùng lặp hoàn toàn (`duplicate`): **5,226** dòng.
  - Mã phi sản phẩm (`POST`, `DOT`, `M`,...): **2,306** dòng.
- **Giá trị khuyết Description**: Điền thành công **1,454** dòng bằng mode của từng `StockCode` (tỷ lệ khuyết giảm về **0.00%**).
- **Tập giao dịch mua hợp lệ (`cleaned_transactions.csv`)**: **522,571** dòng.
- **Tập giao dịch có khách hàng (`customer_transactions.csv`)**: **391,153** dòng (tách 131,418 dòng thiếu `CustomerID` để phục vụ riêng cho RFM và mô hình hóa).
- **Tập giao dịch sản phẩm (`product_transactions.csv`)**: **522,540** dòng (loại thêm 31 dòng phiếu quà tặng `gift_*` cho bài toán khai phá luật kết hợp).
- **Bảo toàn dữ liệu gốc**: Dữ liệu gốc `Online Retail.csv` được kiểm tra xác nhận hoàn toàn nguyên vẹn 100%.

---

## 2. Pipeline Làm Sạch Chi Tiết (10 Bước Chuẩn Hóa)

Quy trình làm sạch dữ liệu được thực hiện tuần tự và không gây rò rỉ dữ liệu (no data leakage):

| Bước | Tên bước | Hành động & Phương pháp | Số dòng loại bỏ | Số dòng còn lại | File / Bảng liên quan |
|:---:|---|---|:---:|:---:|---|
| **01** | **Raw Data** | Nạp dữ liệu gốc từ CSV qua `load_raw_data()` | — | 541,909 | [Online Retail.csv](file:///d:/Data-mininng/data/raw/Online%20Retail.csv) |
| **02** | **Standardization** | Ép kiểu dữ liệu chuẩn, tính `TotalAmount = Quantity * UnitPrice` | 0 | 541,909 | `standardize_dtypes()` |
| **03** | **Transaction Flags** | Gắn 5 cờ boolean: `IsCancelled`, `IsAdjust`, `IsDuplicate`, `IsQuantityInvalid`, `IsPriceInvalid` | 0 | 541,909 | `add_transaction_flags()` |
| **04** | **Remove Cancelled (C)** | Tách hóa đơn hủy (bắt đầu bằng `C`, Quantity âm) | -9,288 | 532,621 | [invoice_cleaning_summary.csv](file:///d:/Data-mininng/outputs/tables/data_preparation/invoice_cleaning_summary.csv) |
| **05** | **Remove Adjust (A)** | Loại bỏ bút toán nợ xấu (bắt đầu bằng `A`, UnitPrice cực lớn) | -3 | 532,618 | [invoice_cleaning_summary.csv](file:///d:/Data-mininng/outputs/tables/data_preparation/invoice_cleaning_summary.csv) |
| **06** | **Remove Invalid Quantity** | Loại bỏ các dòng có `Quantity <= 0` còn sót lại (hàng hỏng, kiểm kho) | -1,336 | 531,282 | `remove_invalid_transactions()` |
| **07** | **Remove Invalid UnitPrice** | Loại bỏ các dòng có `UnitPrice <= 0` (hàng tặng 0đ, lỗi hệ thống) | -1,179 | 530,103 | `remove_invalid_transactions()` |
| **08** | **Remove Duplicates** | Loại bỏ dòng trùng lặp 8 thuộc tính gốc, giữ bản ghi đầu tiên | -5,226 | 524,877 | `handle_duplicates()` |
| **09** | **Remove Non-Product SC** | Loại bỏ mã cước phí, dịch vụ (`POST`, `DOT`, `M`,...) | -2,306 | 522,571 | [stockcode_classification.csv](file:///d:/Data-mininng/outputs/tables/data_preparation/stockcode_classification.csv) |
| **10** | **Impute Description** | Điền khuyết Description bằng mode theo StockCode | 0 | 522,571 | [missing_values_after_cleaning.csv](file:///d:/Data-mininng/outputs/tables/data_preparation/missing_values_after_cleaning.csv) |

---

## 3. Phân Tích Chuyên Sâu Các Quyết Định Nghiệp Vụ

### 3.1. Hóa Đơn Hủy ('C') và Bút Toán Nợ Xấu ('A')
- **Hóa đơn hủy (`C`)**: Tổng cộng 3,836 hóa đơn với 9,288 dòng. Tất cả đều có số lượng âm, tổng giá trị âm lên tới -£896,812.49. Các giao dịch này không đại diện cho hành vi mua sắm tạo doanh thu thực tế. Việc tách các dòng hủy ra khỏi tập giao dịch mua giúp tránh hiện tượng làm sai lệch biến `Monetary` và `Quantity` trong phân cụm RFM.
- **Bút toán điều chỉnh nợ xấu (`A`)**: Gồm đúng 3 hóa đơn `A563185`, `A563186`, `A563187` với mã StockCode `B` và Description "Adjust bad debt". Mỗi dòng mang giá trị ±£11,062.06. Đây là bút toán cân đối tài chính nội bộ, không phải giao dịch thương mại → **Bắt buộc loại bỏ hoàn toàn**.

### 3.2. Phân Loại StockCode Phi Sản Phẩm
Tập dữ liệu chứa nhiều mã bắt đầu bằng ký tự chữ cái không đại diện cho mặt hàng vật lý:
- **Cước bưu điện và vận chuyển**: `POST` (1,256 dòng), `DOT` (710 dòng), `C2` (144 dòng).
- **Điều chỉnh thủ công và chiết khấu**: `M`/`m` (572 dòng), `D` (77 dòng), `S` (63 dòng mẫu thử).
- **Phí giao dịch tài chính & sàn**: `BANK CHARGES` (37 dòng), `AMAZONFEE` (34 dòng), `CRUK` (16 dòng).
- **Sản phẩm thật được giữ lại**: `PADS` (4 dòng - miếng đệm phụ kiện), `DCGS*` (~40 dòng - quà tặng sản phẩm đặc biệt có mô tả hợp lệ).
- **Phiếu quà tặng (`gift_*`)**: 34 dòng. Giữ lại trong tập phân tích chung nhưng loại khỏi tập phân tích luật kết hợp giỏ hàng vì không phải sản phẩm mua kèm vật lý.

### 3.3. Xử Lý Giá Trị Khuyết (Missing Values)
- **`Description` (1,454 dòng thiếu, 0.27%)**: Điền bằng mô tả xuất hiện nhiều nhất (mode) của `StockCode` tương ứng trong tập dữ liệu. Toàn bộ 1,454 dòng đều được khôi phục thành công.
- **`CustomerID` (135,080 dòng thiếu ban đầu, sau khi loại lỗi còn 131,418 dòng)**:
  - **Không điền giả tạo**: Tuyệt đối không tự ý gán giá trị trung bình hay mã ID giả vì sẽ làm méo mó bản chất từng khách hàng.
  - **Tách tập dữ liệu chuyên biệt**: Giữ các dòng này trong `cleaned_transactions.csv` để bảo toàn doanh số tổng thể của doanh nghiệp; lọc bỏ trong `customer_transactions.csv` để các bài toán phân cụm khách hàng (K-Means) và dự đoán mua lại (Classification) đạt độ chính xác cao nhất.

### 3.4. Phân Tích Giá Trị Bất Thường (Outlier Analysis - IQR)
Áp dụng công thức IQR chuẩn: $Lower = Q_1 - 1.5 \times IQR$, $Upper = Q_3 + 1.5 \times IQR$.

| Biến | $Q_1$ | Median | $Q_3$ | IQR | Ngưỡng dưới | Ngưỡng trên | P99 | Số dòng Outlier | Tỷ lệ (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Quantity** | 1.00 | 4.00 | 12.00 | 11.00 | -15.50 | 28.50 | 100.00 | 26,815 | 5.13% |
| **UnitPrice** | 1.25 | 2.08 | 4.13 | 2.88 | -3.07 | 8.45 | 16.63 | 35,772 | 6.85% |
| **TotalAmount** | 3.90 | 9.90 | 17.70 | 13.80 | -16.80 | 38.40 | 179.00 | 41,124 | 7.87% |

> [!NOTE]
> **Quyết định nghiệp vụ**: Không xóa bỏ các dòng ngoại lai này vì chúng phản ánh các đơn hàng bán buôn (wholesale / B2B) hợp lệ. Thay vào đó, hệ thống gắn các cờ cảnh báo `IsQuantityOutlier`, `IsPriceOutlier`, `IsAmountOutlier` để kiểm soát độ lệch trong bước Feature Engineering (sử dụng Log Transformation).

---

## 4. Các Tập Dữ Liệu Trung Gian Được Tạo (`data/interim/`)

| File dữ liệu | Số dòng | Số cột | Dung lượng | Mục tiêu sử dụng |
|---|:---:|:---:|:---:|---|
| [`cleaned_transactions.csv`](file:///d:/Data-mininng/data/interim/cleaned_transactions.csv) | **522,571** | 12 | ~75.7 MB | Phân tích doanh thu tổng quan, sản phẩm bán chạy, xu hướng thời gian |
| [`customer_transactions.csv`](file:///d:/Data-mininng/data/interim/customer_transactions.csv) | **391,153** | 12 | ~56.6 MB | Phân tích RFM, phân cụm K-Means, xây dựng nhãn và huấn luyện mô hình dự đoán mua lại 90 ngày |
| [`product_transactions.csv`](file:///d:/Data-mininng/data/interim/product_transactions.csv) | **522,540** | 12 | ~75.7 MB | Khai phá luật kết hợp sản phẩm (Association Rules) bằng Apriori / FP-Growth |

---

## 5. Bảng Thống Kê & Biểu Đồ So Sánh (Outputs)

Toàn bộ kết quả được lưu trữ có hệ thống theo quy ước cấu trúc project:

### 5.1. Bảng số liệu (`outputs/tables/data_preparation/`)
1. [`cleaning_summary.csv`](file:///d:/Data-mininng/outputs/tables/data_preparation/cleaning_summary.csv): Số lượng dòng qua từng giai đoạn của pipeline.
2. [`invoice_cleaning_summary.csv`](file:///d:/Data-mininng/outputs/tables/data_preparation/invoice_cleaning_summary.csv): Thống kê chi tiết đơn bình thường, đơn hủy (`C`) và nợ xấu (`A`).
3. [`stockcode_classification.csv`](file:///d:/Data-mininng/outputs/tables/data_preparation/stockcode_classification.csv): Danh mục phân loại tất cả các mã StockCode bắt đầu bằng chữ.
4. [`missing_values_after_cleaning.csv`](file:///d:/Data-mininng/outputs/tables/data_preparation/missing_values_after_cleaning.csv): Tỷ lệ dữ liệu khuyết sau bước điền Description.
5. [`outlier_summary.csv`](file:///d:/Data-mininng/outputs/tables/data_preparation/outlier_summary.csv): Thống kê chi tiết ngưỡng phân vị và số lượng outlier IQR.

### 5.2. Biểu đồ trực quan hóa (`outputs/figures/data_preparation/`)
1. [`cleaning_funnel.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/cleaning_funnel.png): Dòng chảy số lượng bản ghi qua từng công đoạn làm sạch.
2. [`removal_reasons_pie.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/removal_reasons_pie.png): Biểu đồ Donut thể hiện tỷ lệ % các lý do loại bỏ dữ liệu.
3. [`boxplot_quantity_before_after.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/boxplot_quantity_before_after.png): Boxplot so sánh phân bố số lượng trước và sau làm sạch.
4. [`boxplot_unitprice_before_after.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/boxplot_unitprice_before_after.png): Boxplot so sánh đơn giá trước và sau làm sạch.
5. [`dist_totalamount_before_after.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/dist_totalamount_before_after.png): Biểu đồ phân phối Log10 của TotalAmount.
6. [`revenue_by_month_before_after.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/revenue_by_month_before_after.png): Doanh thu theo tháng trước và sau khi làm sạch đơn hủy và nợ xấu.
7. [`outlier_detection.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/outlier_detection.png): Biểu đồ phát hiện outlier của Quantity, UnitPrice, TotalAmount.
8. [`missing_values_heatmap.png`](file:///d:/Data-mininng/outputs/figures/data_preparation/missing_values_heatmap.png): Biểu đồ so sánh mức độ khuyết dữ liệu trước và sau làm sạch.

---

## 6. Sẵn Sàng Cho Giai Đoạn Tiếp Theo

Với 3 tập dữ liệu sạch tại `data/interim/`, dự án đã sẵn sàng bước vào các giai đoạn tiếp theo:
1. **Feature Engineering & RFM Segmentation**:
   - Tính toán các chỉ số Recency, Frequency, Monetary, AOV, CancelRate trên `customer_transactions.csv`.
   - Phân cụm khách hàng bằng thuật toán K-Means kết hợp Elbow Method và Silhouette Analysis.
2. **Repeat Purchase Modeling (90 days)**:
   - Xác định ngày ngắt (`cutoff_date`) và tạo nhãn mục tiêu `repeat_purchase_90d` trên tập dữ liệu tương lai.
   - Huấn luyện và so sánh mô hình Logistic Regression, Decision Tree và Random Forest mà không gây rò rỉ dữ liệu.
3. **Association Rule Mining**:
   - Chuyển đổi `product_transactions.csv` thành dạng giỏ hàng (basket format).
   - Khai phá các tập phổ biến và luật kết hợp bằng thuật toán Apriori / FP-Growth.
