# Data Dictionary

## 1. Nguồn dữ liệu

Dataset sử dụng: **Online Retail** từ UCI Machine Learning Repository.

File gốc: `data/raw/Online Retail.csv` (CSV, ~45 MB).

Dữ liệu gồm các giao dịch bán lẻ trực tuyến trong giai đoạn từ **01/12/2010** đến **09/12/2011** (373 ngày). Mỗi dòng là một sản phẩm thuộc một hóa đơn.

### Thống kê tổng quan (Data Understanding)

| Metric | Giá trị |
|---|---|
| Tổng số dòng | 541,909 |
| Tổng số cột | 8 |
| Số hóa đơn (InvoiceNo) | 25,900 |
| Số sản phẩm (StockCode) | 4,070 |
| Số khách hàng (CustomerID) | 4,372 |
| Số quốc gia (Country) | 38 |
| Dòng thiếu CustomerID | 135,080 (24.93%) |
| Dòng thiếu Description | 1,454 (0.27%) |
| Dòng trùng lặp hoàn toàn | 5,268 (0.97%) |
| Hóa đơn hủy (bắt đầu bằng C) | 3,836 hóa đơn / 9,288 dòng |
| Adjust bad debt (bắt đầu bằng A) | 3 hóa đơn / 3 dòng |
| Dòng Description chứa "adjust" hoặc "bad debt" | 34 |
| Dòng StockCode phi sản phẩm (bắt đầu bằng chữ) | 2,995 (0.55%) |
| Dòng Quantity ≤ 0 | 10,624 |
| Dòng UnitPrice ≤ 0 | 2,517 |

## 2. Các cột dữ liệu gốc

| Cột | Kiểu dữ liệu | Ý nghĩa | Vai trò xử lý |
|---|---|---|---|
| `InvoiceNo` | String | Mã hóa đơn | Gom nhóm giao dịch; không dùng trực tiếp để huấn luyện |
| `StockCode` | String | Mã sản phẩm | Phân tích sản phẩm và luật kết hợp |
| `Description` | String | Tên sản phẩm | Hiển thị và phân tích mô tả |
| `Quantity` | Integer | Số lượng sản phẩm | Tạo tổng số lượng và doanh thu |
| `InvoiceDate` | Datetime | Ngày và giờ phát sinh giao dịch | Tạo đặc trưng thời gian |
| `UnitPrice` | Float | Giá một sản phẩm | Tính tổng giá trị giao dịch |
| `CustomerID` | Float64 (chứa NaN) | Mã khách hàng | Gom nhóm và tạo đặc trưng khách hàng; 24.93% bị thiếu |
| `Country` | String | Quốc gia của khách hàng | Phân tích theo khu vực |

## 3. Phân loại InvoiceNo theo ký tự đầu

| Prefix | Ý nghĩa | Số dòng | Xử lý |
|---|---|---|---|
| Số (5xx...) | Giao dịch bình thường | 532,618 | Giữ lại |
| `C` | Hóa đơn bị hủy (Cancellation) | 9,288 | Loại khỏi tập phân tích chính |
| `A` | Adjust bad debt (điều chỉnh nợ xấu) | 3 | Loại - bút toán kế toán, không phải giao dịch |

### Chi tiết Adjust bad debt (A)

3 dòng có InvoiceNo bắt đầu bằng `A`, StockCode = `B`, Description = "Adjust bad debt".
Giá trị UnitPrice rất lớn (±£11,062.06). Đây là bút toán điều chỉnh nợ xấu,
**không phải giao dịch mua bán** → phải loại bỏ.

Ngoài ra, có **34 dòng** trong dataset chứa từ "adjust" hoặc "bad debt" trong Description.
Trong đó 31 dòng có UnitPrice = 0 (chỉ điều chỉnh số lượng), 3 dòng có UnitPrice ≠ 0.
Tất cả đều thiếu CustomerID.

## 4. StockCode phi sản phẩm

Một số StockCode không phải mã sản phẩm thực mà là phí dịch vụ, điều chỉnh:

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

## 5. Quy tắc làm sạch

### Hóa đơn hủy

### Adjust bad debt

Nếu `InvoiceNo` bắt đầu bằng chữ `A`, xem đó là bút toán điều chỉnh nợ xấu. Loại bỏ khi phân tích.

### StockCode phi sản phẩm

Loại các mã `POST`, `DOT`, `M`, `m`, `C2`, `D`, `S`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, `B` khi phân tích sản phẩm và luật kết hợp.

### Giá trị không hợp lệ

- Loại bỏ bản ghi có `Quantity <= 0`.
- Loại bỏ bản ghi có `UnitPrice <= 0`.
- Kiểm tra các giá trị cực lớn bằng IQR hoặc percentile.

### Khách hàng không có mã

Các bản ghi thiếu `CustomerID` không dùng cho phân tích khách hàng, vì không thể gán giao dịch cho một khách hàng cụ thể. Có thể giữ lại chúng khi phân tích doanh thu tổng thể.

### Thời gian

Từ `InvoiceDate` tạo thêm:

- `invoice_year`
- `invoice_month`
- `invoice_day`
- `invoice_hour`
- `invoice_weekday`

## 6. Cột dẫn xuất ở cấp giao dịch

| Cột | Công thức hoặc cách tạo | Ý nghĩa |
|---|---|---|
| `TotalAmount` | `Quantity * UnitPrice` | Giá trị dòng sản phẩm |
| `IsCancelled` | Dựa vào `InvoiceNo` | Xác định giao dịch hủy |
| `InvoiceDateOnly` | Chỉ lấy phần ngày | Tính số ngày hoạt động |
| `InvoiceMonth` | Năm-tháng của hóa đơn | Phân tích doanh thu theo tháng |
| `InvoiceHour` | Giờ trong ngày | Phân tích giờ mua hàng |

## 7. Cột dẫn xuất ở cấp hóa đơn

Sau khi gom theo `InvoiceNo`, tạo:

| Cột | Ý nghĩa |
|---|---|
| `InvoiceTotalAmount` | Tổng giá trị hóa đơn |
| `InvoiceTotalQuantity` | Tổng số sản phẩm trong hóa đơn |
| `UniqueProductCount` | Số sản phẩm khác nhau trong hóa đơn |
| `InvoiceCountry` | Quốc gia của hóa đơn |
| `InvoiceHour` | Giờ phát sinh hóa đơn |

## 8. Cột dẫn xuất ở cấp khách hàng

| Cột | Cách tính | Ý nghĩa |
|---|---|---|
| `Recency` | Ngày tham chiếu - ngày mua cuối | Số ngày khách chưa mua |
| `Frequency` | Số hóa đơn khác nhau | Mức độ thường xuyên mua |
| `Monetary` | Tổng `TotalAmount` | Tổng giá trị mua |
| `TotalItems` | Tổng `Quantity` | Tổng số sản phẩm đã mua |
| `UniqueProducts` | Số `StockCode` khác nhau | Mức độ đa dạng sản phẩm |
| `AverageOrderValue` | `Monetary / Frequency` | Giá trị hóa đơn trung bình |
| `ActiveDays` | Số ngày có giao dịch | Số ngày khách hoạt động |
| `CancellationRate` | Tỷ lệ hóa đơn hủy | Mức độ hủy đơn |
| `AveragePurchaseInterval` | Khoảng cách trung bình giữa các lần mua | Chu kỳ mua hàng |
| `Country` | Quốc gia của khách hàng | Phân tích theo khu vực |

## 9. Nhãn classification

Nhãn `repeat_purchase_90d` được tạo theo quy trình:

1. Chọn ngày kết thúc quan sát, gọi là `cutoff_date`.
2. Tính các đặc trưng khách hàng chỉ từ giao dịch trước hoặc bằng `cutoff_date`.
3. Kiểm tra giao dịch của khách hàng trong 90 ngày sau `cutoff_date`.
4. Nếu có giao dịch hợp lệ thì gán nhãn 1, ngược lại gán nhãn 0.

Không được dùng giao dịch trong 90 ngày tương lai để tạo các đặc trưng như `Frequency`, `Monetary` hoặc `Recency`.

## 10. Dữ liệu cho association rules

Tạo bảng dạng giỏ hàng:

```text
InvoiceNo | Product_A | Product_B | Product_C | ...
```

Mỗi hóa đơn được biểu diễn bằng tập sản phẩm xuất hiện trong hóa đơn đó. Một sản phẩm chỉ xuất hiện một lần trong mỗi giỏ hàng, không phụ thuộc số lượng.
