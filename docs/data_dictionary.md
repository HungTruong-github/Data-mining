# Data Dictionary

## 1. Nguồn dữ liệu

Dataset sử dụng: **Online Retail** từ UCI Machine Learning Repository.

Dữ liệu gồm các giao dịch bán lẻ trực tuyến trong giai đoạn từ tháng 12/2010 đến tháng 12/2011. Mỗi dòng là một sản phẩm thuộc một hóa đơn.

## 2. Các cột dữ liệu gốc

| Cột | Kiểu dữ liệu | Ý nghĩa | Vai trò xử lý |
|---|---|---|---|
| `InvoiceNo` | String | Mã hóa đơn | Gom nhóm giao dịch; không dùng trực tiếp để huấn luyện |
| `StockCode` | String | Mã sản phẩm | Phân tích sản phẩm và luật kết hợp |
| `Description` | String | Tên sản phẩm | Hiển thị và phân tích mô tả |
| `Quantity` | Integer | Số lượng sản phẩm | Tạo tổng số lượng và doanh thu |
| `InvoiceDate` | Datetime | Ngày và giờ phát sinh giao dịch | Tạo đặc trưng thời gian |
| `UnitPrice` | Float | Giá một sản phẩm | Tính tổng giá trị giao dịch |
| `CustomerID` | String | Mã khách hàng | Gom nhóm và tạo đặc trưng khách hàng |
| `Country` | String | Quốc gia của khách hàng | Phân tích theo khu vực |

## 3. Quy tắc làm sạch

### Hóa đơn hủy

Nếu `InvoiceNo` bắt đầu bằng chữ `C`, xem đó là hóa đơn bị hủy. Các dòng này được loại khỏi tập giao dịch mua hàng chính.

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

## 4. Cột dẫn xuất ở cấp giao dịch

| Cột | Công thức hoặc cách tạo | Ý nghĩa |
|---|---|---|
| `TotalAmount` | `Quantity * UnitPrice` | Giá trị dòng sản phẩm |
| `IsCancelled` | Dựa vào `InvoiceNo` | Xác định giao dịch hủy |
| `InvoiceDateOnly` | Chỉ lấy phần ngày | Tính số ngày hoạt động |
| `InvoiceMonth` | Năm-tháng của hóa đơn | Phân tích doanh thu theo tháng |
| `InvoiceHour` | Giờ trong ngày | Phân tích giờ mua hàng |

## 5. Cột dẫn xuất ở cấp hóa đơn

Sau khi gom theo `InvoiceNo`, tạo:

| Cột | Ý nghĩa |
|---|---|
| `InvoiceTotalAmount` | Tổng giá trị hóa đơn |
| `InvoiceTotalQuantity` | Tổng số sản phẩm trong hóa đơn |
| `UniqueProductCount` | Số sản phẩm khác nhau trong hóa đơn |
| `InvoiceCountry` | Quốc gia của hóa đơn |
| `InvoiceHour` | Giờ phát sinh hóa đơn |

## 6. Cột dẫn xuất ở cấp khách hàng

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

## 7. Nhãn classification

Nhãn `repeat_purchase_90d` được tạo theo quy trình:

1. Chọn ngày kết thúc quan sát, gọi là `cutoff_date`.
2. Tính các đặc trưng khách hàng chỉ từ giao dịch trước hoặc bằng `cutoff_date`.
3. Kiểm tra giao dịch của khách hàng trong 90 ngày sau `cutoff_date`.
4. Nếu có giao dịch hợp lệ thì gán nhãn 1, ngược lại gán nhãn 0.

Không được dùng giao dịch trong 90 ngày tương lai để tạo các đặc trưng như `Frequency`, `Monetary` hoặc `Recency`.

## 8. Dữ liệu cho association rules

Tạo bảng dạng giỏ hàng:

```text
InvoiceNo | Product_A | Product_B | Product_C | ...
```

Mỗi hóa đơn được biểu diễn bằng tập sản phẩm xuất hiện trong hóa đơn đó. Một sản phẩm chỉ xuất hiện một lần trong mỗi giỏ hàng, không phụ thuộc số lượng.
