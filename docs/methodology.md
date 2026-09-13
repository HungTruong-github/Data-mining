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

Kết quả của giai đoạn này được lưu tại `outputs/tables/` và `outputs/figures/`.

## 4. Data Preparation

### 4.1 Làm sạch

- Chuyển `InvoiceDate` sang kiểu datetime.
- Loại bỏ hóa đơn hủy khỏi tập mua hàng chính.
- Loại bỏ `Quantity <= 0` và `UnitPrice <= 0`.
- Xử lý bản ghi thiếu `CustomerID`.
- Kiểm tra và xử lý outlier.

### 4.2 Feature engineering

Tạo bảng ở ba cấp độ:

- Cấp giao dịch.
- Cấp hóa đơn.
- Cấp khách hàng.

Các đặc trưng chính ở cấp khách hàng là Recency, Frequency, Monetary, TotalItems, UniqueProducts, AverageOrderValue và CancellationRate.

### 4.3 Chuẩn bị cho clustering

- Chọn các đặc trưng RFM.
- Có thể log-transform các biến lệch phải như `Monetary` và `Frequency`.
- Chuẩn hóa bằng StandardScaler.
- Xác định số cụm bằng Elbow Method và Silhouette Score.

### 4.4 Chuẩn bị cho classification

- Tạo nhãn `repeat_purchase_90d`.
- Loại bỏ ID khỏi tập đặc trưng.
- One-hot encoding cho biến phân loại nếu sử dụng `Country`.
- Chia dữ liệu thành train/test.
- Dùng StratifiedKFold trên tập train.
- Chỉ fit scaler và encoder trên tập train để tránh data leakage.

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
