# Team Tasks

## 1. Nguyên tắc phân công

Project phù hợp nhóm 4–5 người theo Track B. Mỗi thành viên cần có đóng góp cụ thể ở ít nhất một trong các phần:

- Thiết kế bài toán.
- Phân tích và tiền xử lý dữ liệu.
- Xây dựng mô hình.
- Đánh giá và trực quan hóa.
- Viết báo cáo và chuẩn bị demo.

Tên thành viên được cập nhật sau khi nhóm chốt danh sách.

## 2. Phân công đề xuất cho nhóm 4 người

| Thành viên | Phần việc chính | Sản phẩm bàn giao |
|---|---|---|
| Thành viên 1 | Business understanding và data understanding | `business_understanding.md`, EDA, data dictionary |
| Thành viên 2 | Làm sạch và feature engineering | `preprocessing.py`, `feature_engineering.py`, dữ liệu processed |
| Thành viên 3 | Clustering và association rules | K-Means, Apriori/FP-Growth, bảng kết quả |
| Thành viên 4 | Classification, evaluation và dashboard | Các mô hình classification, đánh giá, `app.py` |

Tất cả thành viên cùng tham gia kiểm tra code, thảo luận kết quả và hoàn thiện báo cáo.

## 3. Phân công đề xuất cho nhóm 5 người

| Thành viên | Phần việc chính | Sản phẩm bàn giao |
|---|---|---|
| Thành viên 1 | Business understanding và quản lý tiến độ | Tài liệu mục tiêu, câu hỏi nghiệp vụ, timeline |
| Thành viên 2 | Data understanding và EDA | Notebook EDA, biểu đồ, thống kê mô tả |
| Thành viên 3 | Data preparation và feature engineering | Pipeline làm sạch, RFM, dữ liệu processed |
| Thành viên 4 | Modeling và evaluation | Clustering, classification, model comparison |
| Thành viên 5 | Association rules, dashboard và báo cáo | Luật kết hợp, Streamlit, biểu đồ và tài liệu |

## 4. Quy ước Git

### Nhánh đề xuất

```text
main
develop
feature/data-preprocessing
feature/rfm-clustering
feature/classification
feature/association-rules
feature/dashboard
```

### Quy tắc commit

```text
feat: add RFM feature engineering
fix: handle cancelled invoices
docs: update methodology
test: add preprocessing tests
```

### Pull request

Mỗi thay đổi lớn nên được tạo thành một pull request để thành viên khác kiểm tra trước khi merge vào `develop`.

## 5. Timeline đề xuất

| Giai đoạn | Công việc | Kết quả |
|---|---|---|
| Tuần 1 | Chốt đề tài, nguồn dữ liệu và mục tiêu | Business understanding |
| Tuần 2 | Đọc dữ liệu và EDA | Thống kê, biểu đồ, data dictionary |
| Tuần 3 | Làm sạch và tạo đặc trưng | Các file trong `data/processed/` |
| Tuần 4 | RFM và K-Means | Phân nhóm khách hàng |
| Tuần 5 | Classification | So sánh ít nhất ba mô hình |
| Tuần 6 | Association rules | Danh sách luật kết hợp |
| Tuần 7 | Dashboard và kiểm thử | Demo chạy được |
| Tuần 8 | Viết, kiểm tra và hoàn thiện báo cáo | Báo cáo cuối kỳ và mã nguồn |

## 6. Peer assessment

Nhóm cần ghi nhận đóng góp của từng thành viên theo các tiêu chí:

- Đóng góp ý tưởng và thiết kế giải pháp.
- Thực hiện code và thực nghiệm.
- Viết báo cáo và chuẩn bị tài liệu.
- Hợp tác và tuân thủ tiến độ.
- Trình bày và trả lời phản biện ở checkpoint giữa kỳ.

Mỗi thành viên nên lưu lại commit, notebook, báo cáo phần việc và biên bản họp nhóm để làm minh chứng.

## 7. Quy tắc phối hợp

- Không sửa trực tiếp code của thành viên khác mà không trao đổi.
- Không đưa file dữ liệu tạm và model lớn vào commit nếu không cần thiết.
- Mọi kết quả trong báo cáo phải tái tạo được từ code.
- Ghi rõ phần sử dụng công cụ AI nếu có.
- Kiểm tra lại toàn bộ pipeline trước khi nộp.
