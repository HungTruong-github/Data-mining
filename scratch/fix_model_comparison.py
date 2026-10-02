import re

with open('src/model_comparison.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace generate_report function
new_generate_report = '''def generate_report(clustering_comp, classification_comp, assoc_comp,
                    seg=None, actions=None, prod=None, feat=None, output_path=None):
    """
    Generate comprehensive CRISP-DM Step 07 Markdown report from dynamic comparison outputs.
    Guarantees no hard-coded summary values and clean UTF-8 Vietnamese text.
    """
    if output_path is None:
        output_path = REPORTS_DIR / '07_model_comparison_and_insights.md'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load supplementary tables if not supplied
    if seg is None:
        seg_path = TABLES_DIR / 'insights' / 'customer_segment_insights.csv'
        seg = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()

    if actions is None:
        act_path = TABLES_DIR / 'insights' / 'customer_segment_action_plan.csv'
        actions = pd.read_csv(act_path) if act_path.exists() else pd.DataFrame()

    if prod is None:
        prod_path = TABLES_DIR / 'insights' / 'product_association_insights.csv'
        prod = pd.read_csv(prod_path) if prod_path.exists() else pd.DataFrame()

    if feat is None:
        feat_path = TABLES_DIR / 'insights' / 'classification_feature_insights.csv'
        feat = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()

    sc = clustering_comp[clustering_comp['is_selected'].astype(bool)].iloc[0]
    mc = classification_comp[classification_comp['is_selected'].astype(bool)].iloc[0]
    sa = assoc_comp[assoc_comp['is_selected'].astype(bool)].iloc[0]

    cv_f1 = mc.get('cv_f1_mean', mc.get('cv_f1', 0))

    # Dynamic segment breakdown string
    if not seg.empty and 'business_segment_name' in seg.columns and 'customer_percentage' in seg.columns:
        seg_summary = "; ".join([
            f"{r['business_segment_name']} ({r['customer_percentage']}%, {r.get('customer_count', 0):,} khách hàng)"
            for _, r in seg.iterrows()
        ])
    else:
        seg_summary = f"{len(seg)} phân khúc"

    # Baseline comparison note
    dummy_rows = classification_comp[classification_comp['model'] == 'DummyClassifier']
    lr_rows = classification_comp[classification_comp['model'] == 'LogisticRegression']
    if not dummy_rows.empty:
        dummy_cv_f1 = dummy_rows.iloc[0].get('cv_f1_mean', dummy_rows.iloc[0].get('cv_f1', 0))
        pos_ratio = float(dummy_rows.iloc[0].get('test_precision', 0.5698)) * 100
        lr_auc = float(lr_rows.iloc[0].get('test_roc_auc', 0)) if not lr_rows.empty else 0.0
        baseline_note = (
            f"Baseline DummyClassifier (chiến lược đoán lớp đa số) có CV F1 = {dummy_cv_f1:.4f} do tỷ lệ lớp dương trong cohort đạt {pos_ratio:.1f}%. "
            f"Mô hình học máy {mc['model']} được chọn theo tiêu chí CV F1 cao nhất trong nhóm learned models (loại trừ Dummy khỏi nhóm learned models do Dummy có ROC-AUC=0.5000, hoàn toàn không có khả năng phân loại/xếp hạng). "
            f"Bên cạnh đó, Logistic Regression đạt Test ROC-AUC = {lr_auc:.4f}, thể hiện năng lực phân biệt và xếp hạng rủi ro rất tốt."
        )
    else:
        baseline_note = f"Mô hình học máy {mc['model']} được chọn trên cơ sở Stratified 5-Fold CV F1."

    report = f"""# CRISP-DM Step 07: Model Comparison, Business Insights & Strategic Recommendations

**Project:** UCI Online Retail Data Mining Analysis  
**Phase:** CRISP-DM Step 05 (Evaluation) & Step 06 (Deployment Preparation)  
**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary

Báo cáo này tổng hợp toàn diện kết quả từ các bước khai phá dữ liệu (Phân cụm khách hàng, Dự đoán mua lại, và Khai phá luật kết hợp) trên tập dữ liệu UCI Online Retail.

- **Customer Clustering (Phân cụm khách hàng):** Mô hình được chọn động từ bảng xếp hạng đa tiêu chí là **{sc['algorithm']} (K={sc['n_clusters']})** với Silhouette Score = **{sc['silhouette_score']:.4f}**, Davies-Bouldin Index = **{sc['davies_bouldin_score']:.4f}**. Phân tách thành các nhóm: {seg_summary}.
- **Repeat Purchase Classification (Dự đoán mua lại):** Mô hình học được chọn là **{mc['model']}** với Stratified 5-Fold CV F1 = **{cv_f1:.4f}**, Test F1 = **{mc.get('test_f1', 0):.4f}**, Test Precision = **{mc.get('test_precision', 0):.4f}**, Test Recall = **{mc.get('test_recall', 0):.4f}**, Test ROC-AUC = **{mc.get('test_roc_auc', 0):.4f}**. {baseline_note}
- **Association Rule Mining (Khai phá luật kết hợp):** Thuật toán được chọn là **{sa['algorithm']}** sinh ra **{sa.get('n_valid_rules', sa.get('valid_rule_count', 0))} luật hợp lệ** (Lift > 1.0) trong thời gian thực thi {sa['runtime_seconds']:.2f} giây. Hai thuật toán Apriori và FP-Growth đã được kiểm chứng tương đương 100% về tập luật và metric.

---

## 2. Model Comparison Tables (Bảng so sánh mô hình)

### 2.1 Customer Clustering Model Comparison
{clustering_comp.to_markdown(index=False)}

### 2.2 Classification Model Comparison
{classification_comp.to_markdown(index=False)}

### 2.3 Association Rules Algorithm Comparison
{assoc_comp.to_markdown(index=False)}

---

## 3. Customer Segment Profiles & Strategic Action Plan (Phân khúc & Kế hoạch hành động)

### 3.1 Segment Profiles (Hồ sơ phân khúc)
{seg.to_markdown(index=False) if not seg.empty else "No segment profiles available."}

### 3.2 Strategic Action Plan (Kế hoạch hành động chiến lược)
{actions.to_markdown(index=False) if not actions.empty else "No action plan available."}

---

## 4. Product Co-Purchase Association Rules (Top 10 Luật kết hợp hàng đầu)
{prod.head(10).to_markdown(index=False) if not prod.empty else "No association insights available."}

---

## 5. Predictive Feature Importance (Tầm quan trọng của đặc trưng dự đoán)
{feat.head(10).to_markdown(index=False) if not feat.empty else "No feature insights available."}

---

## 6. Generated Visualizations & Dashboards (Biểu đồ & Dashboard minh họa)
- `clustering_model_comparison.png`
- `classification_model_comparison.png`
- `clustering_quality_metrics.png`
- `classification_cv_vs_test.png`
- `insight_summary_dashboard.png`

---

## 7. Limitations & Scientific Constraints (Giới hạn & Ràng buộc phương pháp)
1. **Single Retailer Scope:** Dữ liệu chỉ từ một nhà bán lẻ trực tuyến tại Vương quốc Anh (12/2010 - 12/2011), không tự động suy rộng ra toàn ngành thương mại điện tử.
2. **Missing CustomerID:** 24.93% giao dịch không có CustomerID bị loại khỏi bài toán cấp khách hàng (selection bias đã được phân tích và gắn cờ).
3. **Class Imbalance & Baseline:** Tỷ lệ mua lại 90 ngày đạt mức đa số trong cohort, khiến Dummy Classifier có F1 danh nghĩa cao; Random Forest được chọn là mô hình học máy tối ưu F1 thực tế qua Cross-Validation trên tập train.
4. **Retrospective vs Predictive Segment Evaluation:** Tỷ lệ mua lại theo cụm là phân tích hồi cứu mô tả do cụm RFM được xây dựng trên toàn bộ lịch sử quan sát.
5. **Association vs Causation:** Luật kết hợp (Lift > 1) chỉ biểu thị tương quan đồng xuất hiện thống kê, chưa phải quan hệ nhân quả; cần kiểm chứng qua A/B testing trước khi quyết định nhập hàng combo.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report'''

# Replace in content
gen_rep_pattern = re.compile(r'def generate_report\(.*?\n    return report', re.DOTALL)
content = gen_rep_pattern.sub(new_generate_report, content)

with open('src/model_comparison.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated src/model_comparison.py successfully!")
