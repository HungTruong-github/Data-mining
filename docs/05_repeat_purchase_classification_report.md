# Báo Cáo Tổng Hợp Giai Đoạn 05: Repeat Purchase Classification (Dự Đoán Mua Lại)

**Dự án:** Phân khúc khách hàng, dự đoán khả năng mua lại và khai phá luật kết hợp trong dữ liệu bán lẻ trực tuyến  
**Giai đoạn CRISP-DM:** **Phase 4 & 5 - Modeling & Evaluation (Supervised Learning - Classification)**  
**Dữ liệu đầu vào:** `data/processed/repeat_purchase_features.csv` (3,368 mẫu khách hàng, 10 đặc trưng dự báo)  
**Tài liệu tham chiếu:** [Methodology](methodology.md) | [Feature Decision Dictionary](feature_decision_dictionary.md) | [Decision Log](decision_log.md)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Giai đoạn **Repeat Purchase Classification** giải quyết bài toán dự báo khả năng một khách hàng có tiếp tục quay lại phát sinh đơn hàng trong vòng **90 ngày tiếp theo** hay không. Mô hình đóng vai trò then chốt trong việc xếp hạng ưu tiên khách hàng để tự động hóa các chiến dịch tiếp thị và giữ chân khách hàng (retention marketing).

Toàn bộ pipeline phân loại được module hóa trong [`src/classification.py`](../src/classification.py), các hàm đánh giá đo lường trong [`src/evaluation.py`](../src/evaluation.py), thực thi tự động qua [`notebooks/run_05_repeat_purchase_classification.py`](../notebooks/run_05_repeat_purchase_classification.py) và nghiên cứu chi tiết tại [`notebooks/05_repeat_purchase_classification.ipynb`](../notebooks/05_repeat_purchase_classification.ipynb).

### Các kết quả khoa học chính:
- **Nguyên tắc phân chia dữ liệu không rò rỉ (Strict Train/Test Isolation):**
  - Tập dữ liệu 3,368 khách hàng được phân chia theo tỷ lệ **80% Train (2,694 mẫu)** và **20% Test (674 mẫu)** qua phương pháp `StratifiedShuffleSplit` để bảo toàn nguyên vẹn tỷ lệ phân bố nhãn.
  - Quá trình chuẩn hóa đặc trưng số (`StandardScaler`) và mã hóa đặc trưng danh mục (`OneHotEncoder`) được đóng gói thành `ColumnTransformer` tích hợp trực tiếp vào trong `Pipeline` của Scikit-Learn. Bộ biến đổi chỉ được `fit` trên tập Train và chuyển tiếp (`transform`) sang tập Test, đảm bảo 100% không rò rỉ dữ liệu.
- **Quy trình kiểm chứng chéo (5-fold Stratified Cross-Validation):** Toàn bộ việc tinh chỉnh siêu tham số (GridSearchCV qua 40 cấu hình) và lựa chọn mô hình chỉ được thực hiện trên tập Train. Tập Test được giữ nguyên độc lập làm tập đánh giá kiểm chứng cuối cùng (Holdout evaluation).
- **Mô hình tối ưu được lựa chọn:** **Random Forest Classifier** (`n_estimators: 100, max_depth: 5, min_samples_leaf: 5, criterion: gini`) đạt hiệu năng cân bằng toàn diện nhất:
  - 5-fold Stratified CV trên Train: CV F1 = **0.7102** $\pm$ **0.0175**, CV ROC-AUC = **0.7372**, CV Average Precision (PR-AUC) = **0.8099**.
  - Đánh giá trên tập Test Holdout: Test F1 = **0.7375**, Test ROC-AUC = **0.7756**, Test PR-AUC = **0.8396**, Test Accuracy = **70.33%**, Test Precision = **74.34%**, Test Recall = **73.18%**, Test Specificity = **66.55%**.
- **Đối chiếu khoa học với Baseline Dummy Classifier:**
  - Mô hình ngây thơ `DummyClassifier` (đoán toàn bộ lớp đa số 1) đạt F1 trên Test = 0.7259 chỉ nhờ hiện tượng mất cân bằng lớp (lớp 1 chiếm 56.97%), nhưng có ROC-AUC = **0.5000** và Specificity = **0.0%** (hoàn toàn không nhận diện được bất kỳ khách hàng nào không quay lại).
  - Random Forest vượt trội với ROC-AUC đạt **0.7756** và Specificity đạt **66.55%** (nhận diện chính xác 193/290 khách hàng rời bỏ).
- **Ngưỡng quyết định thống nhất (Unified Threshold):** Xác suất $P(\text{repeat}) \ge 0.50$ được đồng bộ tuyệt đối giữa offline evaluation, các hàm dự báo kiểm thử và giao diện Web Dashboard.

---

## 2. Thiết Kế Bộ Đặc Trưng Đầu Vào (Feature Set)

Bài toán sử dụng 10 đặc trưng hành vi được trích xuất hoàn toàn trong giai đoạn quan sát lịch sử (trước ngày mốc cutoff):

| Tên biến đặc trưng | Kiểu biến | Vai trò tiền xử lý trong Pipeline | Ý nghĩa nghiệp vụ |
|:---|:---:|:---:|:---|
| **Recency** | Numeric | StandardScaler | Số ngày kể từ đơn hàng cuối đến ngày cutoff. |
| **Frequency** | Numeric | StandardScaler | Tổng số đơn hàng độc nhất đã thực hiện. |
| **Monetary** | Numeric | StandardScaler | Tổng số tiền đã chi tiêu tích lũy (£). |
| **TotalItems** | Numeric | StandardScaler | Tổng số lượng mặt hàng đã mua. |
| **UniqueProducts** | Numeric | StandardScaler | Số chủng loại sản phẩm phân biệt đã mua. |
| **ActiveDays** | Numeric | StandardScaler | Số ngày khác nhau có phát sinh đơn hàng. |
| **AverageOrderValue** | Numeric | StandardScaler | Giá trị trung bình trên một đơn hàng (£). |
| **AverageItemsPerInvoice** | Numeric | StandardScaler | Số mặt hàng trung bình trên một đơn hàng. |
| **CustomerLifetimeDays** | Numeric | StandardScaler | Số ngày từ lần mua đầu tiên đến lần mua cuối. |
| **Country** | Categorical | OneHotEncoder (handle_unknown='ignore') | Quốc gia đăng ký của khách hàng (38 nước). |

---

## 3. Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning)

Quá trình Grid Search qua 40 cấu hình trên 5-fold Stratified Cross-Validation xác định không gian tham số tối ưu:

| Thuật toán | Không gian tìm kiếm (Search Space) | Siêu tham số tối ưu được chọn | Điểm CV F1 trung bình | Điểm CV ROC-AUC |
|:---|:---|:---|:---:|:---:|
| **DummyClassifier** | `strategy`: ['most_frequent'] | `strategy: 'most_frequent'` | 0.7259 $\pm$ 0.0003 | 0.5000 |
| **LogisticRegression** | `C`: [0.1, 1.0, 10.0]<br>`class_weight`: [None, 'balanced'] | `C: 1.0`, `class_weight: None` | 0.7069 $\pm$ 0.0206 | 0.7324 |
| **DecisionTree** | `max_depth`: [3, 5, 10]<br>`min_samples_leaf`: [1, 5, 20]<br>`class_weight`: [None, 'balanced'] | `max_depth: 3`, `min_samples_leaf: 1`, `class_weight: None` | 0.6978 $\pm$ 0.0222 | 0.7181 |
| **RandomForest** | `n_estimators`: [100, 200]<br>`max_depth`: [5, 10]<br>`min_samples_leaf`: [1, 5]<br>`class_weight`: [None, 'balanced'] | `n_estimators: 100`, `max_depth: 5`, `min_samples_leaf: 5`, `class_weight: None` | **0.7102 $\pm$ 0.0175** | **0.7372** |

> **Quyết định lựa chọn:** `RandomForest` đạt điểm CV F1 cao nhất trong nhóm các mô hình học máy thực thụ (0.7102), có độ lệch chuẩn nhỏ nhất ($\pm$ 0.0175) chứng minh tính bền vững cao trước các tập fold khác nhau. Việc đặt `max_depth=5` và `min_samples_leaf=5` giúp ngăn chặn hiện tượng học vẹt (overfitting).

---

## 4. Kết Quả Đánh Giá Trên Tập Test Độc Lập (Holdout Evaluation)

*Trích xuất từ `outputs/tables/classification/model_comparison.csv`:*

| Chỉ số đánh giá | Dummy Classifier (Baseline) | Decision Tree | Logistic Regression | Random Forest (Selected Model) |
|:---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 56.97% | 68.10% | 71.36% | **70.33%** |
| **Test Balanced Accuracy** | 50.00% | 69.05% | 70.90% | **69.86%** |
| **Test Precision** | 56.97% | 77.35% | 75.20% | **74.34%** |
| **Test Recall** | 100.00% | 62.24% | 74.22% | **73.18%** |
| **Test F1-Score** | 0.7259 | 0.6898 | 0.7471 | **0.7375** |
| **Test ROC-AUC** | **0.5000** | **0.7424** | **0.7761** | **0.7756** |
| **Test Average Precision (PR-AUC)** | 0.5697 | 0.7914 | 0.8362 | **0.8396** |
| **Test Specificity** | **0.00%** | **75.86%** | **67.59%** | **66.55%** |
| **True Negatives (TN)** | 0 | 220 | 196 | **193** |
| **False Positives (FP)** | 290 | 70 | 94 | **97** |
| **False Negatives (FN)** | 0 | 145 | 99 | **103** |
| **True Positives (TP)** | 384 | 239 | 285 | **281** |
| **Thời gian thực thi (s)** | 0.019s | 0.009s | 0.019s | 0.050s |

### Phân tích Ma trận nhầm lẫn (Confusion Matrix của Random Forest):
- Số lượng dự đoán đúng: **474 / 674 khách hàng** (Accuracy = 70.33%).
- Trong 384 khách hàng thực sự quay lại: Dự đoán chính xác **281 khách** (Recall = 73.18%).
- Trong 378 khách hàng được mô hình gắn nhãn sẽ quay lại: Có **281 khách thực tế quay lại** (Precision = 74.34%).
- Trong 290 khách hàng rời bỏ: Nhận diện chính xác **193 khách** (Specificity = 66.55%).

---

## 5. Tầm Quan Trọng Của Các Đặc Trưng (Feature Importance)

*Trích xuất từ `outputs/tables/classification/feature_importance.csv`:*

| Thứ hạng | Đặc trưng | Tỷ trọng đóng góp (Gini Importance) | Nhận xét phân tích hành vi |
|:---:|:---|:---:|:---|
| **1** | **Frequency** | **18.90%** | Tần suất mua hàng trong quá khứ là biến dự báo mạnh nhất về hành vi lặp lại. |
| **2** | **ActiveDays** | **17.79%** | Khách hàng phân bổ mua sắm qua nhiều ngày có thói quen gắn bó thương hiệu cao. |
| **3** | **CustomerLifetimeDays** | **12.92%** | Thâm niên quan hệ khách hàng; khách mới có rủi ro rời bỏ cao hơn khách kỳ cựu. |
| **4** | **UniqueProducts** | **12.24%** | Khách hàng khám phá nhiều danh mục sản phẩm có mức độ gắn kết giỏ hàng lớn. |
| **5** | **Monetary** | **11.01%** | Giá trị chi tiêu tích lũy phản ánh độ tin cậy và khả năng tài chính. |
| **6** | **TotalItems** | **9.59%** | Khối lượng hàng hóa mua sắm. |
| **7** | **Recency** | **7.83%** | Khoảng cách từ lần mua cuối; càng lâu chưa mua thì xác suất quay lại càng giảm. |
| **8** | **AverageOrderValue** | **5.35%** | Quy mô trung bình của một đơn hàng. |
| **9** | **AverageItemsPerInvoice** | **3.71%** | Độ dày giỏ hàng trung bình. |
| **10** | **Country (Tất cả quốc gia)** | **< 0.66%** | Vị trí địa lý đóng góp rất thấp vào quyết định mua sắm trực tuyến. |

---

## 6. Danh Mục Tệp Đầu Ra (Deliverable Artifacts)

### 6.1. Pipeline mô hình đã serialize (`models/classification/`)
- `best_classifier_pipeline.joblib`: Pipeline Scikit-Learn đầy đủ (Preprocessor + RandomForestClassifier).
- `classification_metadata.json`: Bản ghi metadata hoàn chỉnh (ngưỡng 0.50, feature_columns, hyperparams, train/test rows, class distribution).
- `decision_tree_pipeline.joblib`, `logistic_regression_pipeline.joblib`, `dummy_classifier_pipeline.joblib`: Các pipeline đối chứng đã fit.

### 6.2. Bảng số liệu (`outputs/tables/classification/`)
- `model_comparison.csv`: Bảng so sánh 4 mô hình với 20 chỉ số chi tiết.
- `cv_results.csv`: Kết quả 5-fold Cross-Validation chi tiết từng fold.
- `cv_tuning_history.csv`: Lịch sử duyệt qua 40 cấu hình siêu tham số.
- `test_predictions.csv`: 674 dòng kết quả dự đoán trên tập test (nhãn thực tế, nhãn dự đoán, xác suất $P$).
- `feature_importance.csv`: Bảng xếp hạng độ quan trọng của toàn bộ các biến.
- `classification_report.csv`: Báo cáo chi tiết Precision, Recall, F1 theo từng lớp.

### 6.3. Biểu đồ trực quan hóa (`outputs/figures/classification/`)
- `confusion_matrix_best_model.png`: Ma trận nhầm lẫn chuẩn hóa của Random Forest.
- `roc_curves.png`: Đường cong ROC-AUC so sánh 4 mô hình trên tập Test.
- `precision_recall_curves.png`: Đường cong Precision-Recall so sánh khả năng phát hiện lớp dương.
- `feature_importance_best_model.png`: Biểu đồ thanh ngang Top 10 đặc trưng quan trọng nhất.
- `model_metric_comparison.png`: Biểu đồ so sánh trực quan Accuracy, Precision, Recall, F1, AUC giữa các mô hình.
