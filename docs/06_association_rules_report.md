# Báo Cáo Tổng Hợp Giai Đoạn 06: Market Basket Association Rules (Khai Phá Luật Kết Hợp)

**Dự án:** Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM:** **Phase 4 - Modeling (Association Rule Mining)**  
**Dữ liệu đầu vào:** `data/processed/association_basket_matrix.csv` (19,773 hóa đơn × 3,908 sản phẩm)  
**Tài liệu tham chiếu:** [Methodology](methodology.md) | [References](references.md) | [Decision Log](decision_log.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Market Basket Association Rules** khai thác các quy luật tương quan ẩn giấu trong giỏ hàng mua sắm của khách hàng, nhằm trả lời câu hỏi: *“Khách hàng khi đã mua sản phẩm A (hoặc nhóm sản phẩm A) thì có xu hướng đồng thời mua thêm sản phẩm B nào trong cùng một hóa đơn?”*. Kết quả trực tiếp định hình các chiến lược sắp xếp danh mục (merchandising), gói sản phẩm khuyến mãi (product bundle) và thuật toán gợi ý bán chéo (cross-sell recommendation) trên Web Dashboard.

Toàn bộ quy trình được module hóa trong [`src/association_rules.py`](../src/association_rules.py), thực thi qua [`notebooks/run_06_association_rules.py`](../notebooks/run_06_association_rules.py) và nghiên cứu trực quan tại [`notebooks/06_association_rules.ipynb`](../notebooks/06_association_rules.ipynb).

### Các kết quả khoa học chính:
- **Quy mô không gian giỏ hàng:** Xử lý ma trận nhị phân gồm **19,773 hóa đơn giao dịch** độc lập và **3,908 mã mặt hàng** thương mại.
- **Thiết lập ngưỡng ràng buộc nghiệp vụ:**
  - `min_support = 0.02` (2.0%): Một tập sản phẩm phải xuất hiện đồng thời trong tối thiểu **396 hóa đơn** để loại trừ các biến cố ngẫu nhiên.
  - `min_confidence = 0.50` (50.0%): Xác suất có điều kiện $P(\text{Consequent} \mid \text{Antecedent})$ phải đạt ít nhất 50%.
  - `lift > 1.0`: Bắt buộc luật phải thể hiện mối tương quan dương (sự xuất hiện của vế trái làm tăng khả năng xuất hiện vế phải).
- **Kiểm toán tính tương đương khoa học (Equivalence Audit):**
  - Thực nghiệm song song 2 thuật toán kinh điển: **Apriori** (dựa trên nguyên lý sinh tập ứng viên Candidate Generation) và **FP-Growth** (dựa trên cấu trúc cây mẫu thường xuyên Frequent Pattern Tree).
  - Cả 2 thuật toán đều khai phá được đúng **389 tập mục phổ biến (frequent itemsets)** và sinh ra đúng **61 luật kết hợp hợp lệ**.
  - Sai khác tuyệt đối về chỉ số giữa 2 thuật toán: $\Delta_{\text{support}} = 0.0$, $\Delta_{\text{confidence}} = 0.0$, $\Delta_{\text{lift}} = 0.0$ (tương đương 100%).
- **Thời gian thực thi (Runtime Benchmark):** Đo lường trung vị qua 3 lượt chạy độc lập: Apriori đạt **1.78 giây**, FP-Growth đạt **2.11 giây** (trên tập dữ liệu mật độ thưa của Online Retail).
- **Phát hiện quy luật bán chéo nổi bật:**
  - Bộ đồ sứ quý tộc *Regency Teacup and Saucer* (các màu hồng, xanh lá, hoa hồng) đạt độ nâng cao nhất toàn hệ thống với **Lift = 18.23** (Confidence 70.72%, Support 2.74%).
  - Luật có độ tin cậy cao nhất đạt **Confidence = 90.47%** (`PINK + ROSES REGENCY TEACUP` $\rightarrow$ `GREEN REGENCY TEACUP`, Lift = 17.66).
  - Dòng sản phẩm làm vườn *Gardeners Kneeling Pad* đạt Lift = 15.60 (Confidence 72.03%).

---

## 2. Cơ Sở Lý Thuyết & Các Chỉ Số Đo Lường

Với một luật kết hợp có dạng $X \Rightarrow Y$ (trong đó $X$ là vế trước - Antecedent, $Y$ là vế sau - Consequent, $X \cap Y = \emptyset$):

1. **Độ hỗ trợ (Support):** Tỷ lệ các giao dịch chứa đồng thời cả $X$ và $Y$:
   $$\text{Support}(X \Rightarrow Y) = P(X \cup Y) = \frac{\sigma(X \cup Y)}{N}$$

2. **Độ tin cậy (Confidence):** Xác suất có điều kiện xuất hiện $Y$ khi đã biết giao dịch có $X$:
   $$\text{Confidence}(X \Rightarrow Y) = P(Y \mid X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$

3. **Độ nâng (Lift):** Tỷ số giữa độ tin cậy thực tế so với độ tin cậy kỳ vọng nếu $X$ và $Y$ hoàn toàn độc lập thống kê:
   $$\text{Lift}(X \Rightarrow Y) = \frac{P(X \cup Y)}{P(X) \times P(Y)} = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$$
   - $\text{Lift} = 1$: $X$ và $Y$ độc lập (không có giá trị kinh doanh).
   - $\text{Lift} > 1$: $X$ và $Y$ có quan hệ tương quan đồng xuất hiện tích cực.
   - $\text{Lift} < 1$: $X$ và $Y$ mang tính thay thế hoặc loại trừ lẫn nhau.

---

## 3. Kiểm Toán Tính Tương Đương & Benchmark Thuật Toán

*Trích xuất từ `outputs/tables/association_rules/association_algorithm_comparison.csv` và `association_rules_equivalence_audit.csv`:*

| Tiêu chí đối chứng | Thuật toán Apriori | Thuật toán FP-Growth | Mức độ chênh lệch | Đánh giá |
|:---|:---:|:---:|:---:|:---|
| **Số tập mục phổ biến (Itemsets)** | 389 | 389 | 0 | Đồng nhất tuyệt đối |
| **Kích thước tập mục lớn nhất** | 3 sản phẩm | 3 sản phẩm | 0 | Đồng nhất |
| **Tổng số luật sinh ra** | 61 | 61 | 0 | Đồng nhất |
| **Số luật hợp lệ (Lift > 1)** | 61 | 61 | 0 | Đồng nhất |
| **Giá trị Lift cực đại** | 18.2311 | 18.2311 | 0.0000 | Trùng khớp 100% |
| **Sai khác Support lớn nhất** | — | — | **0.0000** | Trùng khớp |
| **Sai khác Confidence lớn nhất** | — | — | **0.0000** | Trùng khớp |
| **Thời gian thực thi trung vị** | **1.78 giây** | 2.11 giây | +0.33s | Apriori nhỉnh hơn trên tập thưa |

---

## 4. Phân Tích Độ Nhạy Của Tham Số (Sensitivity Analysis)

*Trích xuất từ `outputs/tables/association_rules/rule_sensitivity_analysis.csv` qua lưới 12 cấu hình:*

| min_support | min_confidence | Số lượng luật sinh ra | Lift trung bình | Lift cực đại | Đánh giá khả thi ứng dụng |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0.01 (1%) | 0.30 (30%) | 1,842 | 8.45 | 32.40 | Quá nhiều luật loãng, chi phí lưu trữ và tra cứu cao. |
| 0.01 (1%) | 0.50 (50%) | 812 | 10.12 | 28.15 | Nhiều luật có support thấp (giao dịch ngẫu nhiên). |
| 0.01 (1%) | 0.70 (70%) | 345 | 12.30 | 25.60 | Giữ được luật mạnh nhưng chứa sản phẩm ngách. |
| **0.02 (2%)** | **0.50 (50%)** | **61** | **12.65** | **18.23** | **Cấu hình tối ưu được lựa chọn:** Số lượng luật gọn, các sản phẩm đều là mặt hàng chủ lực, độ tin cậy vững chắc. |
| 0.02 (2%) | 0.70 (70%) | 38 | 13.80 | 18.23 | Thu hẹp danh mục gợi ý. |
| 0.03 (3%) | 0.50 (50%) | 18 | 9.42 | 16.13 | Bỏ sót nhiều cơ hội bán chéo các cặp sản phẩm tốt. |
| 0.05 (5%) | 0.50 (50%) | 2 | 4.12 | 5.30 | Quá chặt; chỉ còn 2 luật cơ bản của các mặt hàng siêu phổ biến. |

---

## 5. Top Các Luật Kết Hợp Giá Trị Nhất & Khuyến Nghị Kinh Doanh

*Trích xuất từ `outputs/tables/association_rules/selected_association_rules.csv`:*

| STT | Vế trước (Antecedent - Đang xem/Trong giỏ) | Vế sau (Consequent - Đề xuất mua kèm) | Support | Confidence | Lift | Hành động kinh doanh khuyến nghị |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| **1** | `PINK REGENCY TEACUP AND SAUCER` | `GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER` | 2.74% | **70.72%** | **18.23x** | Đóng gói combo 3 món đồ sứ cao cấp Regency; gợi ý mua kèm trọn bộ khi khách chọn cốc màu hồng. |
| **2** | `GREEN REGENCY + ROSES REGENCY TEACUP` | `PINK REGENCY TEACUP AND SAUCER` | 2.74% | **70.53%** | **18.23x** | Gợi ý bổ sung cốc màu hồng để hoàn thiện bộ sưu tập 3 màu. |
| **3** | `PINK REGENCY + ROSES REGENCY TEACUP` | `GREEN REGENCY TEACUP AND SAUCER` | 2.74% | **90.47%** | **17.66x** | Gần như chắc chắn (90.5%) khách hàng mua thêm màu xanh lá nếu đã có màu hồng và hoa hồng. |
| **4** | `PINK REGENCY TEACUP AND SAUCER` | `GREEN REGENCY TEACUP AND SAUCER` | 3.20% | **82.61%** | **16.13x** | Ghép đôi 2 sản phẩm bán kèm với ưu đãi giảm giá 10% cho chiếc thứ hai. |
| **5** | `GARDENERS KNEELING PAD CUP OF TEA` | `GARDENERS KNEELING PAD KEEP CALM` | 2.76% | **72.03%** | **15.60x** | Thiết kế bộ quà tặng đôi miếng đệm làm vườn văn phòng/gia đình. |
| **6** | `REGENCY CAKESTAND 3 TIER + GREEN REGENCY` | `ROSES REGENCY TEACUP AND SAUCER` | 2.06% | **80.16%** | **14.88x** | Bán chéo tách trà kèm theo khay bánh 3 tầng Regency cho tiệc trà chiều. |

> **Nguyên tắc nghiệp vụ quan trọng:**
> Các luật kết hợp (Lift > 1) phản ánh mối quan hệ tương quan đồng xuất hiện trong hành vi tiêu dùng, **chưa chứng minh quan hệ nhân quả (causality)**. Vì vậy, các đề xuất combo cần được triển khai thử nghiệm A/B Testing trên kênh bán hàng trước khi áp dụng trên quy mô toàn hệ thống.

---

## 6. Danh Mục Tệp Đầu Ra (Deliverable Artifacts)

### 6.1. Tệp dữ liệu luật (`outputs/tables/association_rules/` & `outputs/rules/`)
- `selected_association_rules.csv`: Toàn bộ 61 luật kết hợp đã chọn kèm đầy đủ chỉ số Support, Confidence, Lift, Leverage, Conviction.
- `apriori_rules.csv` & `fpgrowth_rules.csv`: Tập luật khai phá độc lập từ 2 thuật toán.
- `association_algorithm_comparison.csv`: Bảng so sánh hiệu năng và số lượng tập mục.
- `association_rules_equivalence_audit.csv`: Biên bản kiểm toán tương đương sai khác bằng 0.0.
- `association_business_insights.csv`: Bảng chuyển đổi 10 luật hàng đầu thành các hành động kinh doanh cụ thể.
- `rule_sensitivity_analysis.csv`: Bảng thực nghiệm 12 điểm lưới độ nhạy tham số.

### 6.2. Biểu đồ trực quan hóa (`outputs/figures/association_rules/`)
- `top_products_frequency.png`: Biểu đồ cột ngang Top 20 sản phẩm xuất hiện nhiều nhất trong giỏ hàng.
- `algorithm_comparison.png`: Biểu đồ so sánh thời gian thực thi và số luật giữa Apriori và FP-Growth.
- `scatter_support_confidence_lift.png`: Biểu đồ tán xạ 3 chiều (Support trục X, Confidence trục Y, Lift thể hiện qua màu sắc).
- `top_rules_lift.png`: Biểu đồ thanh ngang trực quan hóa Top 15 luật có Lift cao nhất.
