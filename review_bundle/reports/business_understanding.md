# Business Understanding

## 1. Tên đề tài

**Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến**

Tên tiếng Anh đề xuất:

**Customer Segmentation, Repeat-Purchase Prediction and Association Rule Mining for Online Retail Data**

## 2. Bối cảnh nghiệp vụ

Doanh nghiệp bán lẻ trực tuyến cần hiểu hành vi mua hàng để:

- Nhận diện nhóm khách hàng có giá trị cao.
- Phát hiện khách hàng có nguy cơ không mua lại.
- Đề xuất sản phẩm phù hợp.
- Thiết kế chương trình khuyến mãi và bán kèm.

Dữ liệu gốc là dữ liệu giao dịch của một cửa hàng bán lẻ trực tuyến tại Vương quốc Anh. Mỗi dòng biểu diễn một sản phẩm trong một hóa đơn.

## 3. Phát biểu bài toán

Project giải quyết ba bài toán khai phá dữ liệu:

### Bài toán 1: Phân khúc khách hàng

Phân nhóm khách hàng dựa trên mức độ gần đây của giao dịch, tần suất mua và tổng giá trị mua hàng.

### Bài toán 2: Dự đoán mua lại

Dự đoán khách hàng có phát sinh giao dịch trong vòng 90 ngày sau thời điểm quan sát hay không.

Nhãn mục tiêu:

```text
repeat_purchase_90d = 1: khách hàng có mua lại trong 90 ngày tiếp theo
repeat_purchase_90d = 0: khách hàng không mua lại trong 90 ngày tiếp theo
```

### Bài toán 3: Khai phá luật kết hợp

Tìm các sản phẩm thường xuất hiện trong cùng một hóa đơn, ví dụ:

```text
Sản phẩm A -> Sản phẩm B
```

Các luật này có thể hỗ trợ gợi ý sản phẩm và bán hàng kèm.

## 4. Mục tiêu project

1. Làm sạch và chuẩn hóa dữ liệu giao dịch.
2. Xây dựng bộ đặc trưng khách hàng từ dữ liệu giao dịch.
3. Phân khúc khách hàng bằng K-Means.
4. So sánh tối thiểu ba mô hình dự đoán khả năng mua lại.
5. Khai phá luật kết hợp bằng Apriori hoặc FP-Growth.
6. Đánh giá mô hình bằng các chỉ số phù hợp.
7. Trình bày kết quả bằng dashboard Streamlit.
8. Đưa ra các phát hiện có ý nghĩa nghiệp vụ.

## 5. Câu hỏi nghiệp vụ

- Nhóm khách hàng nào tạo ra doanh thu cao nhất?
- Nhóm nào có nguy cơ không mua lại?
- Tần suất mua và giá trị đơn hàng ảnh hưởng thế nào đến khả năng mua lại?
- Sản phẩm nào thường được mua cùng nhau?
- Có thể ưu tiên chương trình chăm sóc cho nhóm khách hàng nào?

## 6. Phạm vi project

### Trong phạm vi

- Sử dụng dữ liệu Online Retail công khai.
- Phân tích khách hàng, hóa đơn và sản phẩm.
- Sử dụng K-Means cho clustering.
- Sử dụng Logistic Regression, Decision Tree và Random Forest cho classification.
- Sử dụng Apriori hoặc FP-Growth cho association rules.
- Xây dựng dashboard minh họa kết quả.

### Ngoài phạm vi

- Không xây dựng hệ thống thương mại điện tử thật.
- Không tự động gửi email hoặc khuyến mãi cho khách hàng.
- Không khẳng định kết quả là quan hệ nhân quả.
- Không sử dụng dữ liệu mô phỏng làm dữ liệu đánh giá chính.

## 7. Kết quả đầu ra

- File dữ liệu giao dịch đã làm sạch.
- Bảng đặc trưng khách hàng.
- Bảng phân nhóm khách hàng.
- Bảng so sánh các mô hình classification.
- File luật kết hợp sản phẩm.
- Biểu đồ EDA và đánh giá mô hình.
- Mô hình tốt nhất được lưu lại.
- Dashboard Streamlit.
- Báo cáo kỹ thuật theo cấu trúc CRISP-DM.

## 8. Tiêu chí thành công

- Pipeline có thể chạy lại từ dữ liệu gốc.
- Không sử dụng thông tin sau thời điểm dự đoán để tạo đặc trưng.
- Có ít nhất ba mô hình classification được so sánh.
- Có đánh giá bằng F1-score, Recall và ROC-AUC.
- Các cụm khách hàng có mô tả dễ hiểu.
- Luật kết hợp có Support, Confidence và Lift rõ ràng.
- Dashboard hiển thị được kết quả chính.

## 9. Hạn chế dự kiến

- Dataset chỉ phản ánh một doanh nghiệp và một giai đoạn lịch sử.
- Không có thông tin chi phí marketing, lợi nhuận hoặc phản hồi khách hàng.
- Nhãn mua lại được xây dựng từ lịch sử giao dịch, không phải nhãn do doanh nghiệp cung cấp trực tiếp.
- Kết quả không nên áp dụng trực tiếp cho thị trường Việt Nam nếu chưa kiểm định lại trên dữ liệu địa phương.
