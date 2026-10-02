# Rubric Evidence Matrix

Ma trận minh chứng cho 6 tiêu chí đánh giá, dẫn tới notebook/section/output cụ thể.

## 1. Business Understanding (10%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| Bài toán thực tế & người dùng | docs/business_understanding.md | — | Phân tích hành vi khách hàng UK retailer | PASS |
| Mục tiêu khai phá | NB01 §1.1 | — | Segmentation + Prediction + Association | PASS |
| Quyết định được hỗ trợ | docs/business_understanding.md | — | Marketing targeting, cross-sell, retention | PASS |
| Đối chiếu góp ý tiến độ | — | — | NEEDS_USER_INPUT: cần dữ liệu thật từ nhóm | BLOCKED |

## 2. Data Understanding & Preprocessing (20%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| Thống kê mô tả | NB01 | run_01 | raw_summary.csv, descriptive_statistics.csv | PASS |
| Missing analysis | NB01, NB02 | — | missing_values_heatmap.png, cleaning_summary.csv | PASS |
| Outlier analysis | NB02 | preprocessing.py | outlier_summary.csv, outlier_detection.png | PASS |
| Duplicate analysis | NB01, NB02 | — | 5,268 duplicates documented | PASS |
| Cleaning pipeline | NB02 | run_02 | cleaned_transactions.csv (72.17 MB) | PASS |
| Tác động preprocessing lên model | NB02→NB04,NB05 | — | So sánh ảnh hưởng missing ID & log-transform | PASS |

## 3. Lựa chọn & Xây dựng Mô hình (25%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| So sánh ≥ 2 phương pháp clustering | NB04 | clustering.py | K-Means, GMM, Agglomerative, DBSCAN (26 configs) | PASS |
| So sánh ≥ 2 classifiers | NB05 | classification.py | Dummy, LR, DT, RF (5-fold Stratified CV) | PASS |
| So sánh Apriori vs FP-Growth | NB06 | association_rules.py | association_algorithm_comparison.csv (61 rules) | PASS |
| Tuning quy trình | NB04: K=2..8, NB05: class_weight/hyperparams | — | clustering_algorithm_comparison.csv, model_comparison.csv | PASS |
| Giải thích ưu/nhược | NB04, NB05, NB07 | — | Narrative trong notebooks và report | PASS |

## 4. Đánh giá & Phân tích Kết quả (20%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| Internal metrics clustering | NB04 | clustering.py | Silhouette, DB, CH scores (K=2 selected, sil=0.4330) | PASS |
| CV/validation đúng | NB05 | classification.py | 5-fold StratifiedKFold trên train set, test holdout | PASS |
| Baseline comparison | NB05 | — | DummyClassifier CV F1=0.7259 vs RF CV F1=0.6726 | PASS |
| Confusion matrix & reports | NB05 | evaluation.py | confusion_matrix.png, classification_report.csv | PASS |
| Sai số & giới hạn | NB07 | insights.py | 07_model_comparison_and_insights.md limitations | PASS |
| Lift > 1 validation | NB06 | association_rules.py | validate_rules() strictly enforces lift > 1 | PASS |

## 5. Kỹ thuật & Tái lập (15%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| Code module hóa | src/*.py | — | 12 modules trong src/, compileall pass | PASS |
| Pipeline chạy lại | run_pipeline.py | — | 7 steps sequential executed in 495s | PASS |
| Tests | tests/*.py | pytest | 34/34 tests passed (unit, integration, invariance, smoke) | PASS |
| Notebook executed | NB01-07 | nbclient | 7/7 notebooks executed (0 errors), exported HTML | PASS |
| Demo dashboard | app/app.py | streamlit | Smoke tested via test_app_smoke.py (2/2 pass) | PASS |
| Runners synchronized | notebooks/run_*.py | — | Shared src modules, identical logic and outputs | PASS |

## 6. Báo cáo & Tổng kết (10%)

| Sub-requirement | Notebook/Section | Code/Function | Output/Evidence | Status |
|---|---|---|---|---|
| Report CRISP-DM | NB07, outputs/reports/ | model_comparison.py | 07_model_comparison_and_insights.md & HTML | PASS |
| Đóng góp/peer assessment | — | — | docs/team_tasks.md (template ready) | NEEDS_USER_INPUT |
| Hạn chế trung thực | NB07 | — | Limitations section minh bạch | PASS |
| AI disclosure | — | — | docs/references.md & report (AI assisted prompt) | NEEDS_USER_INPUT |
| README cập nhật | README.md | — | Cập nhật cấu trúc thực tế và hướng dẫn chạy | PASS |

## Summary

| Tiêu chí | Trọng số | Status |
|---|---|---|
| Business Understanding | 10% | PASS |
| Data Understanding & Preprocessing | 20% | PASS |
| Lựa chọn & Xây dựng Mô hình | 25% | PASS |
| Đánh giá & Phân tích Kết quả | 20% | PASS |
| Kỹ thuật & Tái lập | 15% | PASS (34 tests, 7 notebooks executed 0 errors, pipeline 100%) |
| Báo cáo & Tổng kết | 10% | PARTIAL — kỹ thuật hoàn tất, cần thông tin thành viên nhóm |

### Items requiring user input (NEEDS_USER_INPUT)

1. Tên, MSSV, phân công đóng góp và phiếu peer assessment của từng thành viên nhóm (trong `docs/team_tasks.md`).
2. Biên bản / ghi nhận góp ý tiến độ từ giảng viên hướng dẫn (nếu có).
3. Xác nhận và ký duyệt tuyên bố sử dụng công cụ AI (AI-use disclosure) từ các thành viên.
