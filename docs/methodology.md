# Methodology

## 1. Khung phương pháp

Project được triển khai theo quy trình CRISP-DM:

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment / Demo
```

## 2. Business Understanding

Xác định ba mục tiêu:

1. Phân khúc khách hàng.
2. Dự đoán khả năng mua lại trong 90 ngày.
3. Tìm sản phẩm thường được mua cùng nhau.

Các kết quả cần chuyển thành quyết định nghiệp vụ như chăm sóc khách hàng, gợi ý sản phẩm và bán hàng kèm.

## 3. Data Understanding

Thực hiện các bước:

- Đọc file Excel và kiểm tra kích thước dữ liệu.
- Kiểm tra kiểu dữ liệu từng cột.
- Kiểm tra missing values.
- Kiểm tra hóa đơn hủy.
- Kiểm tra giá trị âm và giá trị bất thường.
- Thống kê số khách hàng, số hóa đơn, số sản phẩm và số quốc gia.
- Vẽ phân phối doanh thu, số lượng và giá sản phẩm.
- Phân tích doanh thu theo tháng, quốc gia và sản phẩm.

Kết quả của giai đoạn này được lưu tại `outputs/tables/data_understanding/` và `outputs/figures/data_understanding/`.

## 4. Data Preparation

### 4.1 Làm sạch (Data Preparation & Cleaning)

Pipeline làm sạch được triển khai theo quy trình 10 bước chuẩn hóa, đảm bảo tính tái lập (reproducible), bảo toàn dữ liệu gốc (raw data integrity) và không gây rò rỉ dữ liệu (no data leakage):

1. **Chuẩn hóa kiểu dữ liệu & Thuộc tính dẫn xuất cấp giao dịch**:
   - Ép kiểu `InvoiceNo` và `StockCode` về dạng chuỗi ký tự (`str`), xóa bỏ khoảng trắng.
   - Chuyển đổi `InvoiceDate` sang kiểu thời gian `datetime64[ns]`.
   - Ép kiểu số cho `Quantity` (int64) và `UnitPrice` (float64).
   - Tạo thuộc tính `TotalAmount = Quantity * UnitPrice`.

2. **Gắn cờ giao dịch (Transaction Flags)**:
   - Tạo 5 cờ logic: `IsCancelled` (hóa đơn 'C'), `IsAdjust` (hóa đơn 'A'), `IsDuplicate` (dòng trùng lặp hoàn toàn), `IsQuantityInvalid` (Quantity <= 0), `IsPriceInvalid` (UnitPrice <= 0).

3. **Phân tích và loại bỏ hóa đơn hủy (`C`) và điều chỉnh nợ xấu (`A`)**:
   - Hóa đơn hủy (`C`): 9,288 dòng (3,836 hóa đơn) với `Quantity < 0`, doanh số âm -£896,812.49. Tách riêng khỏi tập giao dịch mua để không làm lệch phân tích hành vi và RFM.
   - Bút toán nợ xấu (`A`): 3 dòng mang mã `B` (Adjust bad debt), UnitPrice lên tới ±£11,062.06. Đây là nghiệp vụ kế toán nội bộ, loại bỏ hoàn toàn.

4. **Xử lý giá trị thiếu (Missing Values)**:
   - `Description`: Thiếu 1,454 dòng (0.27%). Điền tự động bằng mode (mô tả phổ biến nhất) của từng `StockCode`. Nếu không có, điền `"Unknown Product"`.
   - `CustomerID`: Thiếu 135,080 dòng (24.93%). Đây là khách vãng lai. Giữ lại trong tập dữ liệu chung để tính doanh số tổng thể, nhưng lọc bỏ khi phân tích hành vi cấp khách hàng.

5. **Loại bỏ giao dịch không hợp lệ**:
   - Loại bỏ các dòng có `Quantity <= 0` còn lại (1,336 dòng - hàng hỏng, điều chỉnh kho).
   - Loại bỏ các dòng có `UnitPrice <= 0` (1,179 dòng - hàng khuyến mãi 0 đồng hoặc lỗi hệ thống).

6. **Loại bỏ bản ghi trùng lặp (Duplicates)**:
   - Loại bỏ 5,226 dòng trùng lặp hoàn toàn trên 8 thuộc tính gốc (lỗi truyền dữ liệu hoặc người dùng bấm gửi nhiều lần), chỉ giữ bản ghi đầu tiên.

7. **Phân loại và loại bỏ `StockCode` phi sản phẩm**:
   - Phân loại các mã dịch vụ/phí/điều chỉnh: `POST`, `DOT`, `M`, `C2`, `D`, `S`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, `B` (loại 2,306 dòng).
   - Giữ lại các mã sản phẩm vật lý đặc biệt có Description hợp lệ (`PADS`, `DCGS*`).
   - Tách riêng mã phiếu quà tặng (`gift_*`) khi phân tích sản phẩm.

8. **Phát hiện và gắn cờ giá trị bất thường (Outliers)**:
   - Áp dụng phương pháp IQR (Interquartile Range) cho `Quantity`, `UnitPrice`, và `TotalAmount`.
   - Không xóa bỏ cứng các dòng này (vì đại diện cho giao dịch mua buôn sỉ thực tế), mà gắn cờ `IsQuantityOutlier`, `IsPriceOutlier`, `IsAmountOutlier`.

9. **Tạo 3 tập dữ liệu trung gian (`data/interim/`)**:
   - `cleaned_transactions.csv` (522,571 dòng): Toàn bộ giao dịch mua bán hợp lệ.
   - `customer_transactions.csv` (391,153 dòng): Chỉ giữ giao dịch có định danh `CustomerID`, phục vụ tính RFM, phân cụm K-Means và dự báo mua lại.
   - `product_transactions.csv` (522,540 dòng): Loại bỏ thêm mã phiếu quà tặng `gift_*`, phục vụ khai phá luật kết hợp sản phẩm (Association Rules).

10. **Lưu vết và trực quan hóa so sánh Trước - Sau**:
    - Lưu toàn bộ bảng thống kê vào `outputs/tables/data_preparation/`.
    - Lưu 8 biểu đồ so sánh vào `outputs/figures/data_preparation/`.

### 4.2 Feature engineering (Phase 03)

Tạo đặc trưng ở ba cấp độ phục vụ ba bài toán chính:

#### 4.2.1 Đặc trưng RFM cấp khách hàng

Từ `data/interim/customer_transactions.csv` (391,153 dòng, 4,334 khách hàng), tạo bảng RFM mỗi khách hàng 1 dòng:

**Định nghĩa RFM:**
- **Recency** = `reference_date − LastPurchaseDate` (đơn vị: ngày). `reference_date = max(InvoiceDate) + 1 ngày`. Recency nhỏ → khách mua gần đây.
- **Frequency** = Số hóa đơn (`InvoiceNo`) khác nhau bằng `nunique()`. Không đếm số dòng giao dịch vì một hóa đơn có thể chứa nhiều sản phẩm.
- **Monetary** = Tổng `TotalAmount = Quantity × UnitPrice`.

**Lý do chọn `reference_date = max_date + 1`**: Đảm bảo khách mua vào ngày cuối cùng vẫn có Recency ≥ 1 (tránh Recency = 0).

**Các biến mở rộng:**
- `TotalItems`: Tổng Quantity mua.
- `UniqueProducts`: Số StockCode khác nhau.
- `UniqueInvoices`: Số InvoiceNo khác nhau (= Frequency).
- `ActiveDays`: Số ngày có giao dịch.
- `AverageOrderValue`: Monetary / Frequency.
- `AverageItemsPerInvoice`: TotalItems / Frequency.
- `CustomerLifetimeDays`: Số ngày từ lần mua đầu đến lần mua cuối.
- `Country`: Quốc gia xuất hiện nhiều nhất.

**Lý do không dùng `CustomerID` làm feature**: CustomerID là định danh, không mang thông tin hành vi. Đưa vào mô hình sẽ gây overfitting.

#### 4.2.2 Điểm RFM và phân khúc mô tả

Chia mỗi biến R, F, M thành 5 nhóm (1–5) bằng `pd.qcut` với xử lý quantile trùng (`duplicates='drop'` và fallback `rank(method='first')`).

Phân khúc mô tả dựa trên logic if-else:
- Champions: R ≥ 4 AND F ≥ 4 AND M ≥ 4.
- Loyal Customers: F ≥ 4 AND M ≥ 3.
- Big Spenders: M ≥ 4.
- Recent Customers: R ≥ 4 AND F ≤ 2.
- At Risk: R ≤ 2 AND F ≥ 2.
- Regular Customers: Tất cả trường hợp còn lại.

Đây là phân khúc heuristic, sẽ được kiểm chứng bằng K-Means clustering ở Phase 05.

### 4.3 Chuẩn bị cho clustering

- Chọn các đặc trưng: Recency, Frequency, Monetary, TotalItems, UniqueProducts, AverageOrderValue, CustomerLifetimeDays.
- Log-transform (`log1p`) các biến lệch phải: Recency, Frequency, Monetary. Giúp phân phối gần chuẩn hơn, cải thiện hiệu quả của K-Means (vốn nhạy cảm với scale).
- Chuẩn hóa bằng StandardScaler (thực hiện ở notebook 04).
- Xác định số cụm bằng Elbow Method và Silhouette Score (thực hiện ở notebook 04).
- Giữ CustomerID trong file để nối kết quả, nhưng **không** dùng làm biến đầu vào.

Kết quả lưu: `data/processed/rfm_clustering_features.csv` (4,334 khách hàng × 11 cột).

### 4.4 Chuẩn bị cho classification (repeat_purchase_90d)

#### 4.4.1 Chiến lược tránh data leakage

Đây là bước **bắt buộc** phải tách thời gian chặt chẽ:

```
|←── Feature period ──→|←── Label window (90 ngày) ──→|
                    cutoff_date                    max_date
```

- `max_date = max(InvoiceDate)` trong customer_transactions.
- `cutoff_date = max_date − 90 ngày`.
- Feature chỉ được tính từ giao dịch **≤ cutoff_date**.
- Label kiểm tra giao dịch trong **(cutoff_date, cutoff_date + 90 ngày]**.
- `repeat_purchase_90d = 1` nếu khách có ≥ 1 InvoiceNo khác nhau trong cửa sổ label.

**Tuyệt đối không được:**
- Dùng toàn bộ lịch sử đến max_date để tạo feature.
- Đưa future_invoice_count hoặc future_revenue vào file modeling.
- Dùng ngày mua trong tương lai làm Recency.

#### 4.4.2 Giới hạn của nhãn repeat purchase 90 ngày

- Chỉ đánh giá được khả năng mua lại trong 1 cửa sổ thời gian cố định.
- Khách hàng mới xuất hiện sau cutoff_date không có lịch sử → không được gán nhãn.
- Cửa sổ 90 ngày có thể không phù hợp với mọi ngành hàng.
#### 4.4.3 Xác thực chống Data Leakage và Cutoff Date

Hàm `validate_feature_data()` thực hiện kiểm tra chặt chẽ tính hợp lệ về mặt thời gian:
1. **Kiểm tra Recency**: Recency tính so với `feature_ref_date = cutoff_date + 1 day` phải ≥ 1 ngày (chứng minh không có giao dịch nào từ hoặc sau `feature_ref_date` bị đưa vào tính feature).
2. **Kiểm tra lịch sử khách hàng**: 100% khách hàng trong `repeat_purchase_features.csv` (3,368 khách) có lịch sử giao dịch `InvoiceDate <= cutoff_date`.
3. **Chặn rò rỉ khách hàng mới**: 0 khách hàng mới xuất hiện sau `cutoff_date` bị lọt vào tập feature.
4. **Kiểm tra tính toán không cộng dồn tương lai**: Đối chiếu `Frequency` của `repeat_purchase_features` khớp 100% với số hóa đơn tính riêng trên tập giao dịch `InvoiceDate <= cutoff_date`.
5. **Xác thực nhãn mục tiêu**: 100% khách mang nhãn 1 thực sự có giao dịch trong `(cutoff_date, cutoff_date + 90 days]` và 100% khách mang nhãn 0 không có giao dịch nào trong khoảng thời gian này.
6. **Không chứa cột tương lai**: `future_invoice_count` và `future_revenue` hoàn toàn bị loại bỏ khỏi dữ liệu đưa vào mô hình.

Kết quả: `data/processed/repeat_purchase_features.csv` — chứa feature + label, **không** chứa future info.

### 4.5 Chuẩn bị basket data cho Association Rules

- Sử dụng `data/interim/product_transactions.csv` (đã loại mã phi sản phẩm và gift voucher).
- Mỗi `InvoiceNo` = 1 giao dịch (basket).
- Mỗi `StockCode` chỉ xuất hiện 1 lần trong mỗi hóa đơn.
- Dùng StockCode làm item ID; giữ Description để tra cứu tên sản phẩm.
- Loại duplicate theo (InvoiceNo, StockCode).

Kết quả:
- `data/processed/association_basket_long.csv` — dạng dài (InvoiceNo, StockCode, Description).
- `data/processed/association_basket_matrix.csv` — ma trận one-hot nếu kích thước hợp lý.


## 5. Modeling

### 5.1 Customer clustering

Thuật toán chính:

- K-Means.

Kết quả cần lưu:

- Nhãn cụm của từng khách hàng.
- Quy mô từng cụm.
- Giá trị RFM trung bình của từng cụm.
- Tên nghiệp vụ gợi ý cho từng cụm.

Ví dụ tên cụm:

- VIP Customers.
- Loyal Customers.
- New Customers.
- At-Risk Customers.

### 5.2 Repeat-purchase classification

So sánh tối thiểu ba thuật toán:

1. Logistic Regression: mô hình baseline dễ giải thích.
2. Decision Tree: dễ trực quan hóa luật quyết định.
3. Random Forest: mô hình ensemble có khả năng xử lý quan hệ phi tuyến.

Có thể bổ sung XGBoost nếu thời gian cho phép.

Thực hiện tuning cơ bản bằng GridSearchCV hoặc RandomizedSearchCV.

### 5.3 Association rule mining

Sử dụng Apriori hoặc FP-Growth.

Các tham số cần theo dõi:

- `min_support`.
- `min_confidence`.
- `min_lift`.

Chỉ giữ các luật có ý nghĩa và loại bỏ luật chứa sản phẩm xuất hiện quá ít.

## 6. Evaluation

### 6.1 Đánh giá classification

Sử dụng:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion Matrix.

Do mục tiêu là phát hiện khách hàng có khả năng mua lại, cần phân tích thêm Recall và F1-score thay vì chỉ dùng Accuracy.

### 6.2 Đánh giá clustering

Sử dụng:

- Silhouette Score.
- Elbow Method.
- So sánh giá trị RFM trung bình giữa các cụm.
- Kiểm tra khả năng giải thích nghiệp vụ của từng cụm.

### 6.3 Đánh giá association rules

Sử dụng:

- Support: mức độ phổ biến của luật.
- Confidence: xác suất xuất hiện vế phải khi có vế trái.
- Lift: mức độ liên hệ giữa hai nhóm sản phẩm.

## 7. Knowledge Discovery

Các phát hiện cần trình bày trong báo cáo:

- Nhóm khách hàng đem lại doanh thu cao.
- Nhóm khách hàng lâu chưa quay lại.
- Đặc trưng ảnh hưởng đến khả năng mua lại.
- Sản phẩm thường được mua chung.
- Thời điểm có doanh thu hoặc số lượng giao dịch cao.

Các phát hiện phải được diễn giải dưới dạng kết luận có số liệu, không chỉ chụp biểu đồ.

## 8. Deployment và demo

Dashboard Streamlit dự kiến có các chức năng:

1. Tổng quan dữ liệu.
2. Tra cứu thông tin và nhóm của khách hàng.
3. Dự đoán khả năng mua lại.
4. Hiển thị các sản phẩm thường mua cùng nhau.
5. Hiển thị biểu đồ so sánh mô hình.

## 9. Khả năng tái lập

Project phải có:

- `requirements.txt`.
- Hướng dẫn cài đặt trong `README.md`.
- File `run_pipeline.py` để chạy pipeline.
- Seed cố định cho các thuật toán có yếu tố ngẫu nhiên.
- Thư mục dữ liệu và output được mô tả rõ.
- Không hard-code đường dẫn máy cá nhân.

## 10. Rủi ro phương pháp luận

- Dùng giao dịch tương lai để tạo đặc trưng gây data leakage.
- Chia dữ liệu ngẫu nhiên trước khi tạo nhãn theo thời gian.
- Đánh giá clustering chỉ bằng hình ảnh mà không có chỉ số.
- Chọn luật kết hợp chỉ dựa vào Confidence mà bỏ qua Lift.
- Dùng Accuracy duy nhất khi nhãn bị mất cân bằng.
