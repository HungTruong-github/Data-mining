# Báo Cáo Tổng Hợp Giai Đoạn 07: Model Comparison, Business Insights & Strategic Action Plan

**Dự án:** Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM:** **Phase 5 & 6 - Evaluation & Deployment Preparation**  
**Tài liệu tham chiếu:** [Methodology](methodology.md) | [Business Understanding](business_understanding.md) | [Rubric Evidence Matrix](rubric_evidence_matrix.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Model Comparison & Business Insights** đóng vai trò là giai đoạn tổng kết học thuật và ứng dụng thực tiễn của toàn bộ đồ án theo quy trình chuẩn hóa CRISP-DM. Tại giai đoạn này, các kết quả độc lập từ 3 luồng khai phá dữ liệu cốt lõi (**Phân cụm khách hàng**, **Dự đoán mua lại** và **Khai phá luật kết hợp giỏ hàng**) được hợp nhất thành một bức tranh toàn diện về hành vi người tiêu dùng, từ đó thiết lập **Kế hoạch hành động chiến lược 10 cột** và hệ thống quản trị mô hình (Model Governance & Lineage).

Toàn bộ logic so sánh được module hóa trong [`src/model_comparison.py`](../src/model_comparison.py), mã tạo biểu đồ tổng hợp trong [`src/model_comparison_figures.py`](../src/model_comparison_figures.py), các hàm sinh insight nghiệp vụ trong [`src/insights.py`](../src/insights.py), thực thi tự động qua [`notebooks/run_07_model_comparison_and_insights.py`](../notebooks/run_07_model_comparison_and_insights.py) và thể hiện trong [`notebooks/07_model_comparison_and_insights.ipynb`](../notebooks/07_model_comparison_and_insights.ipynb).

### Các kết quả hợp nhất then chốt:
1. **Tổng hợp đa mô hình (Cross-Model Synthesis):**
   - **Phân cụm RFM (Clustering):** **K-Means ($K=2$)** đạt Silhouette = **0.4330**, Davies-Bouldin = **0.8917**. Phân chia 4,334 khách hàng thành nhóm *Best Customers* (38.35%, đóng góp 84.9% doanh số) và *Lost Customers* (61.65%, đóng góp 15.1% doanh số).
   - **Phân loại mua lại (Classification):** **Random Forest** đạt Test ROC-AUC = **0.7756**, Test F1 = **0.7375**, PR-AUC = **0.8396**, Accuracy = **70.33%**, Specificity = **66.55%**.
   - **Luật kết hợp (Association Rules):** **61 luật kết hợp hợp lệ** với Lift cực đại **18.23x**, Confidence đạt tới **90.47%**.
2. **Giao thoa giữa Phân cụm và Dự đoán mua lại:**
   - Khách hàng thuộc nhóm **Best Customers** có tỷ lệ thực tế mua lại trong 90 ngày lên tới **95.7%**.
   - Khách hàng thuộc nhóm **Lost Customers** có tỷ lệ quay lại mua sắm chỉ đạt **26.4%**.
   - Điều này khẳng định sự hội tụ nhất quán giữa thuật toán học không giám sát (K-Means) và bài toán học có giám sát (Random Forest).
3. **Kế hoạch hành động chiến lược 10 cột (10-column Action Plan):** Chuyển hóa số liệu toán học thành 4 chiến dịch kinh doanh cụ thể, có mục tiêu KPI định lượng và phương pháp kiểm chứng độc lập.
4. **Hệ thống truy xuất nguồn gốc và quản trị (Data & Model Lineage):** Lưu trữ mã băm SHA-256 cho toàn bộ các tệp đầu ra trong `outputs/evidence/pipeline_manifest.json` nhằm bảo đảm tính tái lập 100%.

---

## 2. Bảng Tổng Hợp So Sánh Đa Mô Hình (Multi-Model Evaluation)

### 2.1. Phân cụm khách hàng (Customer Clustering)
*So sánh 4 thuật toán đại diện trên không gian $\log(1+\text{RFM}) \rightarrow \text{StandardScaler}$:*

| Thuật toán | Cấu hình tối ưu | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cụm min / max (%) | Đánh giá khoa học |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **K-Means** | **K=2** | **0.4330** | **0.8917** | **4,364.61** | **38.35% / 61.65%** | **Được chọn:** Cân bằng tốt nhất giữa độ phân tách hình học và ý nghĩa kinh doanh. |
| Agglomerative | Ward, K=2 | 0.4232 | 0.9064 | 4,227.37 | 39.82% / 60.18% | Xếp hạng 2; tiệm cận K-Means nhưng runtime cao hơn gấp 3 lần. |
| GMM | K=2 | 0.2874 | 1.0662 | 2,307.42 | 34.73% / 65.27% | Phân vùng kém rõ nét do ranh giới cụm mềm (soft boundaries). |
| DBSCAN | eps=0.5, min=5 | 0.2946 | 1.0626 | 2,408.38 | 34.51% / 64.17% | Tồn tại 1.32% điểm nhiễu ngoại lai; không thích hợp cho dữ liệu RFM thưa. |

### 2.2. Phân loại dự đoán mua lại 90 ngày (Repeat Purchase Classification)
*Đánh giá trên tập kiểm thử độc lập (Test Set: 674 mẫu):*

| Mô hình | CV F1 (Train) | Test F1 | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test Precision | Test Recall | Test Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DummyClassifier** | 0.7259 | 0.7259 | 0.5000 | 0.5697 | 56.97% | 56.97% | 100.00% | 0.00% |
| **Decision Tree** | 0.6978 | 0.6898 | 0.7424 | 0.7914 | 68.10% | 77.35% | 62.24% | 75.86% |
| **Logistic Regression** | 0.7069 | **0.7471** | **0.7761** | 0.8362 | **71.36%** | 75.20% | **74.22%** | **67.59%** |
| **Random Forest (Chọn)** | **0.7102** | 0.7375 | 0.7756 | **0.8396** | 70.33% | 74.34% | 73.18% | 66.55% |

> **Biện luận lựa chọn mô hình:**
> - `DummyClassifier` đạt F1 = 0.7259 nhưng hoàn toàn vô giá trị (ROC-AUC = 0.5000, Specificity = 0.0) do chỉ đoán lớp đa số 1.
> - `RandomForest` được chọn làm mô hình triển khai chính thức vì có **CV F1 cao nhất trên tập Train (0.7102)** với độ lệch chuẩn thấp nhất ($\pm 0.0175$), đạt PR-AUC cao nhất trên tập Test (**0.8396**) và kiểm soát rủi ro overfitting vượt trội so với Decision Tree đơn lẻ.
> - `LogisticRegression` cũng là một lựa chọn xuất sắc với khả năng diễn giải trọng số tuyến tính trực quan.

### 2.3. Khai phá luật kết hợp giỏ hàng (Market Basket Rules)
*Đối chiếu tính tương đương giữa Apriori và FP-Growth:*
- Số tập phổ biến: **389 itemsets**.
- Số luật hợp lệ: **61 rules** (Lift > 1, Support $\ge 0.02$, Confidence $\ge 0.50$).
- Độ lệch chỉ số: $\Delta = 0.0000$ (kết quả trùng khớp hoàn toàn).
- Thời gian chạy: Apriori 1.78s vs FP-Growth 2.11s.

---

## 3. Kế Hoạch Hành Động Chiến Lược 10 Cột (Strategic Action Plan)

*Trích xuất từ `outputs/tables/insights/customer_segment_action_plan.csv`:*

| Cụm | Tên phân khúc | Số khách | Tỷ lệ (%) | Chiến lược cốt lõi | Hành động cụ thể đề xuất | Bằng chứng từ dữ liệu | Mục tiêu KPI | Phương pháp kiểm chứng | Giới hạn & Rủi ro |
|:---:|:---|:---:|:---:|:---|:---|:---|:---|:---|:---|
| **0** | **Best Customers** | 1,662 | 38.35% | **Retention & VIP Loyalty** | Triển khai đặc quyền hội viên VIP: ưu tiên giao hàng miễn phí, quyền truy cập sớm bộ sưu tập mới, dịch vụ chăm sóc khách hàng 1:1. | Recency TB = 25.8 ngày, Frequency = 8.4 đơn, Doanh số = £4,464/khách. | Tỷ lệ duy trì đơn hàng (Retention Rate) $\ge 80\%$ trong 6 tháng kế tiếp. | Theo dõi tỷ lệ quay lại tự nhiên so với các quý trước qua Cohort Analysis. | Chi phí vận hành chương trình tích điểm / quà tặng VIP cần kiểm soát dưới 5% doanh thu của nhóm. |
| **0** | **Best Customers** | 1,662 | 38.35% | **Premium Upsell & Cross-sell** | Đề xuất các bộ sưu tập cao cấp hoặc combo quà tặng giá trị cao (Regency Teacup 3 màu, khay bánh 3 tầng). | Chi tiêu trung bình cao (£4,464), sức mua lớn, độ nhạy giá thấp. | Nâng giá trị đơn hàng trung bình (AOV) thêm 10%. | Thử nghiệm A/B Testing: Đo lường AOV của nhóm nhận gợi ý combo so với nhóm đối chứng không nhận. | Cần đảm bảo tồn kho sẵn có cho các dòng sản phẩm cao cấp trước khi đẩy mạnh chiến dịch. |
| **0** | **Best Customers** | 1,662 | 38.35% | **Advocacy & Referral Campaign** | Mời viết đánh giá sản phẩm (review) và giới thiệu bạn bè nhận thưởng hai chiều (Referral Program). | Tỷ lệ khách hàng mua lại cực cao: 95.7%. Mức độ hài lòng và gắn bó thương hiệu lớn. | Tỷ lệ khách hàng giới thiệu thêm người dùng mới $\ge 8\%$. | Gắn mã giới thiệu cá nhân (Referral code) và đối chiếu số lượt đăng ký mới qua mã. | Nguy cơ gian lận tự tạo tài khoản phụ để trục lợi khuyến mãi. |
| **1** | **Lost Customers** | 2,672 | 61.65% | **First-to-Second Purchase Nurturing** | Chiến dịch kích hoạt sau đơn hàng đầu tiên: gửi email hướng dẫn sử dụng, tặng voucher giảm giá 15% cho đơn thứ hai có hạn dùng 30 ngày. | Tần suất trung vị chỉ có 1 đơn, Recency trung vị 96 ngày, tỷ lệ mua lại chỉ đạt 26.4%. | Nâng tỷ lệ khách hàng mua lại lần 2 từ 26.4% lên 35.0%. | Thử nghiệm tự động gửi email kích hoạt vào ngày thứ 14 và ngày thứ 30 sau đơn đầu. | Rủi ro gây phiền toái (spam) khiến khách hàng hủy đăng ký nhận email (unsubscribe). |

---

## 4. Quản Trị Mô Hình & Tính Tái Lập (Model Governance & Reproducibility)

Nhằm đáp ứng tiêu chí khắt khe nhất của chuẩn nghiên cứu khoa học và rubric Track B:
- **Kiểm tra tính toàn vẹn qua mã băm SHA-256:** Mỗi tệp đầu ra trong pipeline đều được tự động tính toán mã băm SHA-256 lưu trong [`pipeline_manifest.json`](../outputs/evidence/pipeline_manifest.json). Khi tái thực thi, hệ thống sẽ đối chiếu mã băm để đảm bảo tính bất biến của kết quả.
- **Thống nhất quy tắc quyết định (Threshold Synchrony):** Ngưỡng $P(\text{repeat}) \ge 0.50$ được mã hóa cứng trong `classification_metadata.json` và hàm [`predict_customers`](../src/classification.py) dùng chung cho cả offline testing và Web Dashboard.
- **Tách biệt vai trò mã nguồn:** Mô-đun hóa độc lập giữa tầng tính toán khoa học ([`src/model_comparison.py`](../src/model_comparison.py)) và tầng đồ họa trực quan ([`src/model_comparison_figures.py`](../src/model_comparison_figures.py)), đảm bảo khả năng bảo trì và kiểm thử tự động.

---

## 5. Danh Mục Tệp Đầu Ra (Deliverable Artifacts)

### 5.1. Báo cáo và siêu dữ liệu tổng kết (`outputs/reports/` & `outputs/evidence/`)
- `07_model_comparison_and_insights.md`: Báo cáo khoa học tổng hợp markdown chi tiết.
- `07_model_comparison_and_insights_summary.json`: Bản tóm tắt JSON chứa toàn bộ chỉ số thực nghiệm và trạng thái validation PASS.
- `pipeline_manifest.json`: Siêu dữ liệu phiên bản, SHA-256 checksums và môi trường thực thi.
- `pipeline_execution.log`: Nhật ký ghi nhận thời gian thực thi tuần tự từ bước 01 đến 07.

### 5.2. Bảng số liệu tổng hợp (`outputs/tables/model_comparison/` & `outputs/tables/insights/`)
- `clustering_model_comparison.csv`: Bảng xếp hạng chi tiết 26 cấu hình phân cụm.
- `classification_model_comparison.csv`: Bảng so sánh 4 mô hình phân loại trên cả tập CV và Test.
- `association_rules_comparison.csv`: Bảng đối chiếu hiệu năng Apriori vs FP-Growth.
- `customer_segment_action_plan.csv`: Kế hoạch hành động chiến lược 10 cột.
- `customer_segment_insights.csv`: Bảng so sánh hồ sơ và tỷ lệ mua lại giữa các cụm.
- `product_association_insights.csv`: Danh mục 10 gợi ý sản phẩm bán chéo hàng đầu.
- `classification_feature_insights.csv`: Tầm quan trọng và ý nghĩa nghiệp vụ của các đặc trưng.

### 5.3. Biểu đồ trực quan hóa tổng hợp (`outputs/figures/model_comparison/`)
- `insight_summary_dashboard.png`: Dashboard đồ họa 4 góc tích hợp toàn bộ các kết quả quan trọng nhất của đồ án.
- `clustering_model_comparison.png`: Biểu đồ so sánh Silhouette Score giữa 26 cấu hình.
- `classification_model_comparison.png`: Biểu đồ so sánh Accuracy, F1, ROC-AUC trên tập Test.
- `classification_cv_vs_test.png`: Đồ thị đối chiếu điểm F1 giữa tập huấn luyện CV và tập kiểm thử Test.
- `segment_rfm_profile.png`: Biểu đồ radar và cột so sánh giá trị RFM giữa 2 phân khúc.
- `segment_repeat_purchase_rate.png`: Biểu đồ so sánh tỷ lệ mua lại giữa Best Customers (95.7%) và Lost Customers (26.4%).
- `top_association_rules.png`: Trực quan hóa giá trị Lift của các luật kết hợp hàng đầu.
- `feature_importance_top20.png`: Biểu đồ xếp hạng mức độ ảnh hưởng của các đặc trưng lên quyết định mua lại.
