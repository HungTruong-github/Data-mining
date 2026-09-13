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

## 5. Quy tắc làm sạch và Thống kê thực tế (Phase 02)

| Bước | Vấn đề | Tiêu chí nhận diện | Số dòng loại bỏ | Số dòng còn lại | Quyết định & Lý do |
|---|---|---|---|---|---|
| **01** | Dữ liệu gốc | — | — | 541,909 | Toàn bộ dữ liệu raw từ CSV |
| **02** | Hóa đơn hủy | `InvoiceNo` bắt đầu bằng `C` | 9,288 dòng (3,836 HĐ) | 532,621 | Loại bỏ: Giao dịch hủy có số lượng âm, không phản ánh mua hàng thực tế |
| **03** | Bút toán nợ xấu | `InvoiceNo` bắt đầu bằng `A` | 3 dòng (3 HĐ) | 532,618 | Loại bỏ: Bút toán kế toán nội bộ điều chỉnh nợ xấu, giá trị lớn (±£11,062.06) |
| **04** | Số lượng không hợp lệ | `Quantity <= 0` (ngoài C/A) | 1,336 dòng | 531,282 | Loại bỏ: Hàng hư hỏng, kiểm kê kho, thất thoát |
| **05** | Đơn giá không hợp lệ | `UnitPrice <= 0` | 1,179 dòng | 530,103 | Loại bỏ: Quà tặng mẫu 0 đồng hoặc bản ghi lỗi hệ thống |
| **06** | Bản ghi trùng lặp | Trùng hoàn toàn 8 thuộc tính gốc | 5,226 dòng | 524,877 | Loại bỏ: Giữ lại bản ghi đầu tiên (`keep='first'`), khắc phục lỗi gửi trùng |
| **07** | StockCode phi sản phẩm | Mã thuộc `NON_PRODUCT_STOCK_CODES` | 2,306 dòng | 522,571 | Loại bỏ: Cước bưu chính (`POST`, `DOT`), phí điều chỉnh (`M`), ngân hàng,... |
| **08** | **Tập giao dịch sạch** | Toàn bộ giao dịch hợp lệ | — | **522,571** | Lưu tại `data/interim/cleaned_transactions.csv` |
| **09** | **Tập khách hàng** | Thiếu `CustomerID` (NaN) | 131,418 dòng | **391,153** | Lưu tại `data/interim/customer_transactions.csv` (cho RFM, K-Means) |
| **10** | **Tập sản phẩm** | Mã phiếu quà tặng (`gift_*`) | 31 dòng | **522,540** | Lưu tại `data/interim/product_transactions.csv` (cho Association Rules) |

### Xử lý Missing Values
- `Description`: Thiếu 1,454 dòng (0.27%) ban đầu. Đã được điền hoàn toàn bằng mode của từng `StockCode` (100% được khôi phục, không còn missing).
- `CustomerID`: Thiếu 135,080 dòng (24.93%) ở dữ liệu raw, sau khi loại các dòng không hợp lệ còn 131,418 dòng. Không điền nhân tạo mà giữ trong `cleaned_transactions.csv` và chỉ lọc khi phân tích khách hàng.

## 6. Cột dẫn xuất & Cờ giao dịch (Transaction Flags)

| Cột | Kiểu dữ liệu | Cách tạo / Công thức | Ý nghĩa & Ứng dụng |
|---|---|---|---|
| `TotalAmount` | Float64 | `Quantity * UnitPrice` | Tổng giá trị tiền tệ của dòng sản phẩm |
| `IsCancelled` | Boolean | `InvoiceNo.str.startswith('C')` | Đánh dấu hóa đơn hủy |
| `IsAdjust` | Boolean | `InvoiceNo.str.startswith('A')` | Đánh dấu hóa đơn điều chỉnh nợ xấu |
| `IsDuplicate` | Boolean | `df.duplicated(subset=original_cols)` | Đánh dấu dòng trùng lặp hoàn toàn |
| `IsQuantityInvalid` | Boolean | `Quantity <= 0` | Đánh dấu số lượng không hợp lệ |
| `IsPriceInvalid` | Boolean | `UnitPrice <= 0` | Đánh dấu đơn giá không hợp lệ |
| `IsQuantityOutlier` | Boolean | $Quantity < Q_1 - 1.5 \times IQR$ hoặc $> Q_3 + 1.5 \times IQR$ | Đánh dấu ngoại lai số lượng (26,815 dòng, 5.13%) |
| `IsPriceOutlier` | Boolean | $UnitPrice < Q_1 - 1.5 \times IQR$ hoặc $> Q_3 + 1.5 \times IQR$ | Đánh dấu ngoại lai đơn giá (35,772 dòng, 6.85%) |
| `IsAmountOutlier` | Boolean | $TotalAmount < Q_1 - 1.5 \times IQR$ hoặc $> Q_3 + 1.5 \times IQR$ | Đánh dấu ngoại lai tổng tiền (41,124 dòng, 7.87%) |

## 7. Các tập dữ liệu trung gian (Interim Datasets)

| File | Đường dẫn | Số dòng | Số cột | Mục đích sử dụng |
|---|---|---|---|---|
| `cleaned_transactions.csv` | `data/interim/` | 522,571 | 12 | Toàn bộ giao dịch mua hợp lệ; phân tích doanh thu tổng thể, xu hướng bán hàng |
| `customer_transactions.csv` | `data/interim/` | 391,153 | 12 | Chỉ giao dịch có `CustomerID`; phân khúc khách hàng (RFM, K-Means), dự đoán mua lại |
| `product_transactions.csv` | `data/interim/` | 522,540 | 12 | Loại bỏ non-product và voucher; khai phá luật kết hợp sản phẩm (Association Rules) |

## 8. Cột dẫn xuất ở cấp hóa đơn

Sau khi gom theo `InvoiceNo`, tạo:

| Cột | Ý nghĩa |
|---|---|
| `InvoiceTotalAmount` | Tổng giá trị hóa đơn |
| `InvoiceTotalQuantity` | Tổng số sản phẩm trong hóa đơn |
| `UniqueProductCount` | Số sản phẩm khác nhau trong hóa đơn |
| `InvoiceCountry` | Quốc gia của hóa đơn |
| `InvoiceHour` | Giờ phát sinh hóa đơn |

## 9. Cột dẫn xuất ở cấp khách hàng

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

## 10. Nhãn classification

Nhãn `repeat_purchase_90d` được tạo theo quy trình:

1. Chọn ngày kết thúc quan sát, gọi là `cutoff_date`.
2. Tính các đặc trưng khách hàng chỉ từ giao dịch trước hoặc bằng `cutoff_date`.
3. Kiểm tra giao dịch của khách hàng trong 90 ngày sau `cutoff_date`.
4. Nếu có giao dịch hợp lệ thì gán nhãn 1, ngược lại gán nhãn 0.

Không được dùng giao dịch trong 90 ngày tương lai để tạo các đặc trưng như `Frequency`, `Monetary` hoặc `Recency`.

## 11. Dữ liệu cho association rules

Tạo bảng dạng giỏ hàng:

```text
InvoiceNo | Product_A | Product_B | Product_C | ...
```

Mỗi hóa đơn được biểu diễn bằng tập sản phẩm xuất hiện trong hóa đơn đó. Một sản phẩm chỉ xuất hiện một lần trong mỗi giỏ hàng, không phụ thuộc số lượng.
