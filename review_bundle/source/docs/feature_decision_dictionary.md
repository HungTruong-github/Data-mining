# Feature Decision Dictionary

Tài liệu mô tả toàn bộ các cột/features trong project, phân loại theo mục đích sử dụng.

## 1. Cột gốc (Raw — 8 cột)

| Cột | Kiểu | Đơn vị | Mô tả | Vai trò |
|---|---|---|---|---|
| InvoiceNo | str | — | Mã hóa đơn; prefix "C" = hủy, prefix "A" = adjust bad debt | Khóa join, EDA |
| StockCode | object | — | Mã sản phẩm; một số mã phi sản phẩm (POST, DOT, M, C2, D, S...) | Khóa join, tạo basket |
| Description | object | — | Mô tả sản phẩm; có 1,454 dòng thiếu (0.27%) | Display only |
| Quantity | int64 | items | Số lượng; có giá trị âm (hủy đơn, điều chỉnh) | Tính TotalAmount |
| InvoiceDate | datetime64 | — | Ngày giờ giao dịch; 2010-12-01 → 2011-12-09 | Tính Recency, temporal split |
| UnitPrice | float64 | GBP (£) | Đơn giá; có giá trị ≤ 0 (điều chỉnh) | Tính TotalAmount |
| CustomerID | float64 | — | Mã khách hàng; 135,080 dòng thiếu (24.93%) | Aggregate cấp khách hàng |
| Country | str | — | Quốc gia; 38 giá trị unique | Feature phân loại |

## 2. Cột tạo trong EDA (Notebook 01–02)

| Cột | Công thức | Cấp độ | Mục đích | Đưa vào model? |
|---|---|---|---|---|
| TotalAmount | Quantity × UnitPrice | Transaction line | EDA, tính Monetary | Không trực tiếp — aggregate thành Monetary |
| InvoicePrefix | InvoiceNo[0] if alpha else '' | Transaction line | Phân loại hóa đơn (bình thường/hủy/adjust) | Không — QA flag |
| IsCancelled | InvoiceNo.startswith('C') | Transaction line | Flag hủy đơn | Không — QA flag |

## 3. Features cấp khách hàng — RFM (Notebook 03)

Tất cả được tính trên **feature_period** (giao dịch trước cutoff_date), không bao gồm label period.

| Feature | Định nghĩa | Đơn vị | Rủi ro leakage | Model sử dụng |
|---|---|---|---|---|
| Recency | (reference_date − ngày mua cuối) | ngày | Thấp — dùng feature_period | Clustering, Classification |
| Frequency | Số hóa đơn unique (InvoiceNo) | count | Thấp | Clustering, Classification |
| Monetary | Tổng TotalAmount | GBP (£) | Thấp | Clustering, Classification |
| TotalItems | Tổng Quantity | items | Thấp | Classification |
| UniqueProducts | Số StockCode unique | count | Thấp | Classification |
| UniqueInvoices | Số InvoiceNo unique | count | Trùng với Frequency — không dùng cùng lúc | Không sử dụng (redundant) |
| ActiveDays | Số ngày unique có giao dịch | ngày | Thấp | Classification |
| FirstPurchaseDate | Ngày mua đầu tiên | datetime | Audit only | Không |
| LastPurchaseDate | Ngày mua cuối cùng | datetime | Audit only | Không |
| AverageOrderValue | Monetary / Frequency | GBP/order | Thấp | Classification |
| AverageItemsPerInvoice | TotalItems / Frequency | items/order | Thấp | Classification |
| CustomerLifetimeDays | (LastPurchaseDate − FirstPurchaseDate).days | ngày | Thấp — thời gian quan sát, không phải CLV dự đoán | Classification |
| Country | Quốc gia xuất hiện nhiều nhất | category | Thấp | Classification (One-Hot Encoded) |

### Quy ước

- `reference_date = last_observation_date + 1 day` — quy ước để Recency tối thiểu = 1 ngày
- `Frequency` = số hóa đơn unique (không phải số dòng giao dịch)
- `Monetary` là giá trị mua hàng brutto (Quantity × UnitPrice), chưa đối soát hoàn/hủy → không phải net customer value
- `UniqueInvoices` về bản chất trùng với `Frequency` — được giữ lại để kiểm tra nhưng KHÔNG đưa cả hai vào cùng model

## 4. RFM Scoring (Notebook 03)

| Cột | Công thức | Thang | Ghi chú |
|---|---|---|---|
| R_Score | qcut(Recency, 5, labels=[5,4,3,2,1]) | 1–5 | Recency nhỏ → điểm cao |
| F_Score | qcut(Frequency, 5, labels=[1,2,3,4,5]) | 1–5 | Frequency lớn → điểm cao |
| M_Score | qcut(Monetary, 5, labels=[1,2,3,4,5]) | 1–5 | Monetary lớn → điểm cao |
| RFM_Score | R_Score + F_Score + M_Score | 3–15 | Tổng hợp |
| RFM_Code | concat(R_Score, F_Score, M_Score) | str | Mã phân khúc |
| RFM_Segment | Rule-based mapping từ RFM_Code | str | Champions, Loyal, At Risk... |

## 5. Features cho Clustering (Notebook 04)

| Feature dùng | Transformation | Lý do |
|---|---|---|
| Recency, Frequency, Monetary | log1p → StandardScaler | Giảm ảnh hưởng skew; chuẩn hóa khoảng cách cho K-Means |

**Ghi chú**: Clustering chỉ dùng 3 features RFM (log1p scaled). Các features mở rộng tồn tại trong CSV nhưng KHÔNG đưa vào model clustering.

## 6. Target cho Classification (Notebook 03 → 05)

| Cột | Định nghĩa | Giá trị | Window |
|---|---|---|---|
| repeat_purchase_90d | Có ≥ 1 giao dịch trong label_period (cutoff → cutoff + 90 ngày) | 0 / 1 | 90 ngày |

**Leakage safeguard**: Target từ label period; features từ feature period. Hai window không chồng lấn.

## 7. Basket cho Association Rules (Notebook 03/06)

| Cột | Định nghĩa | Mục đích |
|---|---|---|
| InvoiceNo (index) | Mã hóa đơn | Basket ID |
| StockCode columns | 0/1 (boolean) | Sản phẩm có mặt trong basket |

**Quy ước**: 1 basket = 1 InvoiceNo, 1 item = 1 StockCode (không dùng Description làm khóa).

## 8. Cột output — Cluster labels

| Cột | Source | Mô tả |
|---|---|---|
| Cluster | K-Means model | Label cụm (0, 1, ...) |
| Algorithm | Config | Thuật toán đã chọn |
| FeatureVersion | Config | "log1p_scaled" |
| BusinessSegment | Rule-based mapping | Tên mô tả dựa trên profile RFM |