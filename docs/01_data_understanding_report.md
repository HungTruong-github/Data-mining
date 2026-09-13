# 01. Data Understanding - Báo cáo tổng hợp

> **Phase CRISP-DM**: Data Understanding  
> **Dataset**: Online Retail (UCI Machine Learning Repository)  
> **Ngày thực hiện**: 13/09/2026  

---

## 1. Môi trường & Công cụ

| Mục | Trạng thái |
|---|---|
| Môi trường ảo `.venv` | Đã tạo |
| Kernel Jupyter | Đã đăng ký (`Python (Data Mining)`) |
| Python | 3.12 |

### Thư viện đã cài

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| pandas | 3.0.5 | Xử lý dữ liệu |
| numpy | 2.5.3 | Tính toán số |
| openpyxl | 3.1.5 | Đọc file Excel |
| matplotlib | 3.11.2 | Biểu đồ |
| seaborn | 0.13.2 | Biểu đồ nâng cao |
| scikit-learn | 1.9.1 | Machine Learning |
| mlxtend | 0.25.0 | Association Rules |
| streamlit | 1.63.0 | Dashboard |
| joblib | 1.6.0 | Lưu mô hình |
| jupyter | 1.1.1 | Notebook |
| ipykernel | 7.3.0 | Kernel cho Jupyter |
| scipy | 1.18.1 | Thống kê |
| pytest | 9.1.1 | Testing |

---

## 2. Dataset

| Mục | Giá trị |
|---|---|
| Vị trí file | `data/raw/Online Retail.csv` |
| Định dạng | CSV (~45 MB) |
| Kích thước | **541,909 dòng × 8 cột** |
| Khoảng thời gian | 01/12/2010 → 09/12/2011 (**373 ngày**) |

### Các cột dữ liệu

| Cột | Kiểu thực tế | Ý nghĩa |
|---|---|---|
| `InvoiceNo` | str | Mã hóa đơn (bắt đầu bằng số, C hoặc A) |
| `StockCode` | str | Mã sản phẩm (hoặc mã phi sản phẩm) |
| `Description` | str | Tên sản phẩm |
| `Quantity` | int64 | Số lượng sản phẩm |
| `InvoiceDate` | datetime64 (sau chuyển đổi) | Ngày giờ giao dịch |
| `UnitPrice` | float64 | Đơn giá sản phẩm (£) |
| `CustomerID` | float64 (chứa NaN) | Mã khách hàng |
| `Country` | str | Quốc gia |

### Số lượng unique values

| Trường | Giá trị |
|---|---|
| InvoiceNo | 25,900 |
| StockCode | 4,070 |
| Description | 4,223 |
| CustomerID | 4,372 |
| Country | 38 |

---

## 3. Giá trị thiếu (Missing Values)

| Cột | Số lượng | Tỷ lệ |
|---|---|---|
| **CustomerID** | **135,080** | **24.93%** |
| Description | 1,454 | 0.27% |
| Các cột còn lại | 0 | 0% |

**Nhận xét**: Gần 1/4 dữ liệu thiếu `CustomerID`. Các dòng này không thể dùng cho phân tích khách hàng (RFM, classification), nhưng có thể giữ khi phân tích doanh thu tổng thể.

---

## 4. Dòng trùng lặp

| Metric | Giá trị |
|---|---|
| Số dòng trùng lặp hoàn toàn | **5,268** |
| Tỷ lệ | 0.97% |

---

## 5. Phân loại InvoiceNo theo ký tự đầu

### 5.1. Tổng quan

| Prefix | Ý nghĩa | Số dòng | Tỷ lệ |
|---|---|---|---|
| **Số** (5xx...) | Giao dịch bình thường | 532,618 | 98.29% |
| **C** | Hóa đơn bị hủy (Cancellation) | 9,288 | 1.71% |
| **A** | Adjust bad debt (điều chỉnh nợ xấu) | 3 | 0.001% |

### 5.2. Hóa đơn hủy (C)

| Metric | Giá trị |
|---|---|
| Số dòng hủy | 9,288 (1.71%) |
| Số hóa đơn hủy unique | 3,836 |
| Tổng số hóa đơn | 25,900 |
| Tỷ lệ hóa đơn hủy | 14.81% |

### 5.3. Adjust bad debt (A) ⚠️

| Metric | Giá trị |
|---|---|
| Số dòng | **3** |
| Số InvoiceNo unique | 3 |
| StockCode | `B` |
| Description | "Adjust bad debt" |
| UnitPrice | ±£11,062.06 |
| CustomerID | Tất cả NaN |

**Chi tiết 3 dòng Adjust bad debt:**

| InvoiceNo | StockCode | Description | Quantity | UnitPrice |
|---|---|---|---|---|
| A563185 | B | Adjust bad debt | 1 | £11,062.06 |
| A563186 | B | Adjust bad debt | 1 | -£11,062.06 |
| A563187 | B | Adjust bad debt | 1 | -£11,062.06 |

**Nhận xét**: Đây là **bút toán kế toán điều chỉnh nợ xấu**, không phải giao dịch mua bán thực tế. Giá trị UnitPrice rất lớn (±£11,062.06) sẽ gây méo thống kê nếu không loại bỏ.

---

## 6. Dòng Description chứa "adjust" hoặc "bad debt"

| Metric | Giá trị |
|---|---|
| Tổng dòng | **34** |
| UnitPrice = 0 (chỉ điều chỉnh số lượng) | 31 |
| UnitPrice ≠ 0 (ảnh hưởng doanh thu) | 3 |
| Tất cả thiếu CustomerID | **Đúng** |

**Các loại adjustment tìm thấy:**
- `Adjustment` / `adjustment` (điều chỉnh tồn kho)
- `reverse 21/5/10 adjustment` (đảo ngược điều chỉnh)
- `Adjust bad debt` (điều chỉnh nợ xấu - 3 dòng có prefix A)
- `taig adjust` / `taig adjust no stock` (điều chỉnh do hết hàng)
- `OOPS ! adjustment` (sửa lỗi)
- `temp adjustment` (tạm thời)
- `amazon adjust` / `Amazon Adjustment` / `dotcom adjust`
- `reverse previous adjustment`
- `re-adjustment`

---

## 7. StockCode phi sản phẩm

Tổng: **2,995 dòng** (0.55%) có StockCode bắt đầu bằng chữ cái.

| StockCode | Số dòng | Ý nghĩa | Xử lý |
|---|---|---|---|
| `POST` | 1,256 | Phí vận chuyển (Postage) | Loại khi phân tích sản phẩm |
| `DOT` | 710 | Phí vận chuyển Dotcom | Loại khi phân tích sản phẩm |
| `M` / `m` | 572 | Điều chỉnh thủ công (Manual) | Loại |
| `C2` | 144 | Phí vận chuyển (Carriage) | Loại khi phân tích sản phẩm |
| `D` | 77 | Giảm giá (Discount) | Loại |
| `S` | 63 | Mẫu thử (Samples) | Loại |
| `BANK CHARGES` | 37 | Phí ngân hàng | Loại |
| `AMAZONFEE` | 34 | Phí Amazon | Loại |
| `CRUK` | 16 | Hoa hồng CRUK | Loại |
| `B` | 3 | Adjust bad debt | Loại |
| `PADS` | 4 | Phụ kiện đệm | Giữ (sản phẩm thật) |
| `gift_*` | ~34 | Gift voucher | Cân nhắc giữ/loại |
| `DCGS*` | ~40 | Sản phẩm đặc biệt | Giữ (sản phẩm thật) |

---

## 8. Quantity ≤ 0 và UnitPrice ≤ 0

### Quantity ≤ 0

| Phân loại | Số dòng |
|---|---|
| **Tổng Quantity ≤ 0** | **10,624** |
| Là hóa đơn hủy (C) | 9,288 |
| Là adjust bad debt (A) | 0 |
| Khác (không C, không A) | **1,336** |

**Nhận xét**: 9,288/10,624 dòng Quantity ≤ 0 thuộc hóa đơn hủy (C) → sẽ tự được loại khi loại prefix C. Còn 1,336 dòng cần điều tra thêm.

### UnitPrice ≤ 0

| Metric | Giá trị |
|---|---|
| Tổng dòng | **2,517** |
| Min UnitPrice | -£11,062.06 |
| Max UnitPrice | £38,970.00 |

---

## 9. TotalAmount và Thống kê mô tả

| Metric | Giá trị |
|---|---|
| TotalAmount = Quantity × UnitPrice | |
| Min | -£168,469.60 |
| Max | £168,469.60 |
| Mean | £17.99 |
| Median | £9.75 |
| Tổng doanh thu (toàn bộ) | £9,747,747.93 |

### Thống kê mô tả các cột số

| | Quantity | UnitPrice | CustomerID | TotalAmount |
|---|---|---|---|---|
| count | 541,909 | 541,909 | 406,829 | 541,909 |
| mean | 9.55 | 4.61 | 15,287.69 | 17.99 |
| min | -80,995 | -11,062.06 | 12,346 | -168,469.60 |
| 25% | 1 | 1.25 | 13,953 | 3.40 |
| 50% | 3 | 2.08 | 15,152 | 9.75 |
| 75% | 10 | 4.13 | 16,791 | 17.40 |
| max | 80,995 | 38,970 | 18,287 | 168,469.60 |
| std | 218.08 | 96.76 | 1,713.60 | 378.81 |

---

## 10. Biểu đồ đã tạo

| File | Nội dung |
|---|---|
| `outputs/figures/dist_quantity.png` | Phân phối Quantity (toàn bộ + zoom 1-50) |
| `outputs/figures/dist_unitprice.png` | Phân phối UnitPrice (toàn bộ + zoom 0-10£) |
| `outputs/figures/revenue_by_month.png` | Doanh thu theo tháng (loại C và A) |
| `outputs/figures/top10_products_revenue.png` | Top 10 sản phẩm theo doanh thu |
| `outputs/figures/top_countries_revenue.png` | Top 15 quốc gia + pie chart ngoài UK |
| `outputs/figures/invoice_type_distribution.png` | Phân bổ loại InvoiceNo (Normal/C/A) |

---

## 11. Tất cả file đã tạo/cập nhật

### File mới tạo

| File | Mô tả |
|---|---|
| `src/config.py` | Cấu hình đường dẫn tập trung, relative path |
| `src/data_loader.py` | Module tải dữ liệu, hỗ trợ CSV + XLSX |
| `notebooks/01_data_understanding.ipynb` | Notebook Data Understanding hoàn chỉnh |
| `notebooks/run_01_data_understanding.py` | Script chạy tự động (tương đương notebook) |
| `outputs/tables/raw_summary.csv` | Bảng tổng hợp dữ liệu gốc (27 metrics) |
| `outputs/tables/descriptive_statistics.csv` | Thống kê mô tả |
| `outputs/figures/dist_quantity.png` | Biểu đồ phân phối Quantity |
| `outputs/figures/dist_unitprice.png` | Biểu đồ phân phối UnitPrice |
| `outputs/figures/revenue_by_month.png` | Biểu đồ doanh thu theo tháng |
| `outputs/figures/top10_products_revenue.png` | Biểu đồ Top 10 sản phẩm |
| `outputs/figures/top_countries_revenue.png` | Biểu đồ Top quốc gia |
| `outputs/figures/invoice_type_distribution.png` | Biểu đồ phân bổ loại InvoiceNo |
| `docs/01_data_understanding_report.md` | File tổng hợp này |

### File đã cập nhật

| File | Thay đổi |
|---|---|
| `docs/data_dictionary.md` | Thêm thống kê thực tế, phân loại InvoiceNo (C/A), catalog StockCode phi sản phẩm, sửa kiểu CustomerID |

---

## 12. Tổng hợp các vấn đề dữ liệu

| # | Vấn đề | Số lượng | Mức nghiêm trọng | Hướng xử lý |
|---|---|---|---|---|
| 1 | CustomerID thiếu | 135,080 (24.93%) | **Cao** | Loại khi phân tích KH; giữ khi phân tích doanh thu |
| 2 | Hóa đơn hủy (C) | 9,288 dòng | Trung bình | Loại khỏi tập phân tích chính |
| 3 | Adjust bad debt (A) | 3 dòng | **Cao** (giá trị lớn) | Loại - bút toán kế toán, UnitPrice ±£11,062 |
| 4 | Description chứa "adjust"/"bad debt" | 34 dòng | Trung bình | 31 dòng UnitPrice=0 (loại); 3 dòng đã tính ở #3 |
| 5 | StockCode phi sản phẩm | 2,995 dòng (0.55%) | Trung bình | Loại khi phân tích sản phẩm & luật kết hợp |
| 6 | Quantity ≤ 0 (không C, không A) | 1,336 dòng | Trung bình | Điều tra & loại bỏ |
| 7 | UnitPrice ≤ 0 | 2,517 dòng | Trung bình | Điều tra; có thể là miễn phí/điều chỉnh |
| 8 | Dòng trùng lặp | 5,268 (0.97%) | Thấp | Cân nhắc loại bỏ |
| 9 | Outlier cực đoan | Qty: -80,995→80,995 | Trung bình | Kiểm tra IQR, xử lý outlier |
| 10 | CustomerID kiểu float | - | Thấp | Chuyển sang int sau dropna |
| 11 | Description thiếu | 1,454 dòng | Thấp | Kiểm tra mối liên hệ với CustomerID thiếu |

---

## 13. Bước tiếp theo: `02_eda_and_cleaning`

Bước tiếp theo trong CRISP-DM là **Data Preparation**:

1. **Loại bỏ dòng không hợp lệ**:
   - Hóa đơn hủy (InvoiceNo bắt đầu bằng `C`)
   - Adjust bad debt (InvoiceNo bắt đầu bằng `A`)
   - StockCode phi sản phẩm (`POST`, `DOT`, `M`, `D`, `S`, `B`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`)
   - Quantity ≤ 0 (ngoài hóa đơn hủy)
   - UnitPrice ≤ 0

2. **Xử lý CustomerID thiếu**: Loại khi phân tích khách hàng

3. **Xử lý dòng trùng lặp**: Quyết định giữ/loại

4. **Phát hiện và xử lý outlier**: IQR, percentile

5. **EDA chuyên sâu**:
   - Correlation matrix
   - Phân tích theo giờ/ngày/tháng
   - Phân bố khách hàng

6. **Lưu dữ liệu sạch** vào `data/interim/`

> **Lưu ý**: Chưa tạo nhãn `repeat_purchase_90d` và chưa huấn luyện mô hình ở bước này.
