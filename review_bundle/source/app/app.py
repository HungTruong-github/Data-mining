"""
Interactive Dashboard for Online Retail Customer Analytics & Data Mining.
Provides 4 functional tabs:
1. Repeat Purchase Prediction (Classification) with probability scores
2. Customer Segmentation & Strategic Action Plan (Clustering)
3. Product Recommendation & Cross-Sell (Association Rules)
4. Model Governance, Methodology & Manifest (CRISP-DM Metadata)
"""
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Online Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import predict_with_threshold


@st.cache_resource
def load_all_artifacts():
    """Load model pipelines, metadata, and insight tables."""
    artifacts = {}

    # 1. Classification
    clf_pipe_path = PROJECT_ROOT / 'models' / 'classification' / 'best_classifier_pipeline.joblib'
    clf_meta_path = PROJECT_ROOT / 'models' / 'classification' / 'classification_metadata.json'
    if clf_pipe_path.exists() and clf_meta_path.exists():
        artifacts['clf_pipeline'] = joblib.load(clf_pipe_path)
        with open(clf_meta_path, 'r', encoding='utf-8') as f:
            artifacts['clf_metadata'] = json.load(f)

    # 2. Clustering
    cl_model_path = PROJECT_ROOT / 'models' / 'clustering' / 'clustering_model.pkl'
    cl_cfg_path = PROJECT_ROOT / 'models' / 'clustering' / 'clustering_config.pkl'
    cl_prof_path = PROJECT_ROOT / 'outputs' / 'tables' / 'clustering' / 'cluster_profiles.csv'
    if cl_model_path.exists() and cl_cfg_path.exists():
        artifacts['cl_model'] = joblib.load(cl_model_path)
        artifacts['cl_config'] = joblib.load(cl_cfg_path)
    if cl_prof_path.exists():
        artifacts['cl_profiles'] = pd.read_csv(cl_prof_path)

    # 3. Association Rules
    rules_p1 = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'selected_association_rules.csv'
    rules_p2 = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_rules' / 'selected_association_rules.csv'
    if rules_p1.exists():
        artifacts['rules'] = pd.read_csv(rules_p1)
    elif rules_p2.exists():
        artifacts['rules'] = pd.read_csv(rules_p2)

    # 4. Action plan
    act_path = PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'customer_segment_action_plan.csv'
    if act_path.exists():
        artifacts['action_plan'] = pd.read_csv(act_path)

    # 5. Manifest
    man_path = PROJECT_ROOT / 'outputs' / 'evidence' / 'pipeline_manifest.json'
    if man_path.exists():
        with open(man_path, 'r', encoding='utf-8') as f:
            artifacts['manifest'] = json.load(f)

    return artifacts


try:
    artifacts = load_all_artifacts()
except Exception as e:
    st.error(f"Lỗi tải tài nguyên pipeline: {e}")
    st.stop()

# Title and Header
st.title("🛍️ Online Retail Customer Analytics & Repeat Purchase Prediction Dashboard")
st.caption("Dự án Khai phá Dữ liệu Khách hàng — Khoa học Dữ liệu & Học máy (CRISP-DM Framework)")

# Governance Warning Banner
clf_meta_banner = artifacts.get('clf_metadata', {})
train_0 = clf_meta_banner.get('class_distribution', {}).get('train_0', 1159)
train_1 = clf_meta_banner.get('class_distribution', {}).get('train_1', 1535)
total_train = max(1, train_0 + train_1)
pos_pct = (train_1 / total_train) * 100
sel_model_name = clf_meta_banner.get('selected_model', 'RandomForest')
applied_th = float(clf_meta_banner.get('threshold', 0.5))

st.warning(
    f"⚠️ **LƯU Ý NGHIỆP VỤ & PHƯƠNG PHÁP LUẬN:** "
    f"Mô hình phân loại `{sel_model_name}` áp dụng ngưỡng xác suất decision threshold = {applied_th:.2f}. "
    f"Do lớp dương chiếm {pos_pct:.1f}% tập huấn luyện, mô hình Dummy Classifier (đoán toàn bộ lớp đa số) đạt F1 cao trên tập dữ liệu này "
    f"nhưng có ROC-AUC = 0.50 (hoàn toàn không có năng lực phân biệt). Mô hình học máy được dùng để **xếp hạng xác suất ưu tiên chăm sóc khách hàng**, "
    f"không thay thế hoàn toàn quyết định kinh doanh tự động. Các luật kết hợp (Lift > 1) phản ánh tương quan đồng xuất hiện, "
    f"chưa chứng minh quan hệ nhân quả (causality). Các KPI đề xuất (ví dụ tăng AOV 10%) là giả định định hướng cần A/B test kiểm chứng."
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Dự đoán Khách hàng Mua lại",
    "👥 Phân khúc Khách hàng & Kế hoạch Hành động",
    "🛒 Gợi ý Bán chéo Sản phẩm (Cross-sell)",
    "📑 Báo cáo Kỹ thuật & Quản trị Mô hình"
])

# -------------------------------------------------------------
# TAB 1: CLASSIFICATION
# -------------------------------------------------------------
with tab1:
    st.header("🎯 Dự đoán Khả năng Khách hàng Mua lại trong 90 ngày")

    clf_pipeline = artifacts.get('clf_pipeline')
    clf_meta = artifacts.get('clf_metadata', {})

    if clf_pipeline is None:
        st.error("Chưa tìm thấy mô hình phân loại. Vui lòng chạy pipeline bước 05.")
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mô hình được chọn", clf_meta.get('selected_model', 'RandomForest'))
        col_m2.metric("Ngưỡng phân loại", f"{clf_meta.get('threshold', 0.5):.2f}")
        col_m3.metric("Số lượng biến đặc trưng", len(clf_meta.get('feature_columns', [])))
        col_m4.metric("Kích thước tập huấn luyện", f"{clf_meta.get('training_row_count', 0):,} mẫu")

        st.write("---")
        st.subheader("1. Tải lên tập dữ liệu khách hàng (CSV)")
        uploaded_file = st.file_uploader("Chọn file CSV đặc trưng khách hàng:", type=["csv"])

        # Sample data button
        sample_path = PROJECT_ROOT / 'data' / 'processed' / 'repeat_purchase_features.csv'
        if st.button("🧪 Sử dụng dữ liệu mẫu (10 khách hàng)"):
            if sample_path.exists():
                sample_df = pd.read_csv(sample_path).head(10)
                st.session_state['input_df'] = sample_df
            else:
                st.error("Không tìm thấy file repeat_purchase_features.csv mẫu.")

        input_df = None
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.session_state['input_df'] = input_df
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")
        elif 'input_df' in st.session_state:
            input_df = st.session_state['input_df']

        if input_df is not None:
            st.write(f"Dữ liệu đầu vào ({len(input_df)} dòng):")
            st.dataframe(input_df.head(), use_container_width=True)

            if st.button("🚀 Thực hiện dự đoán (Run Prediction)", type="primary"):
                feat_cols = clf_meta.get('feature_columns', [])
                missing = [c for c in feat_cols if c not in input_df.columns]

                if missing:
                    st.error(f"Dữ liệu thiếu các cột đặc trưng bắt buộc: {missing}")
                elif len(input_df) == 0:
                    st.error("Dữ liệu đầu vào rỗng (0 dòng)!")
                else:
                    with st.spinner("Đang chạy dự báo qua Scikit-Learn Pipeline..."):
                        X_in = input_df[feat_cols].copy()
                        applied_threshold = float(clf_meta.get('threshold', 0.5))
                        pos_label = clf_meta.get('positive_class', 1)
                        preds, probs = predict_with_threshold(clf_pipeline, X_in, threshold=applied_threshold, pos_label=pos_label)

                        out_df = input_df.copy()
                        if 'CustomerID' in input_df.columns:
                            cols = ['CustomerID'] + [c for c in out_df.columns if c != 'CustomerID']
                            out_df = out_df[cols]

                        out_df['Du_doan_Mua_lai'] = preds
                        out_df['Xac_suat_Mua_lai'] = probs.round(4)

                        st.success("✅ Dự đoán thành công!")
                        st.subheader("2. Kết quả Dự báo & Phân loại")
                        st.dataframe(out_df, use_container_width=True)

                        n_rep = int((preds == 1).sum())
                        n_non = int((preds == 0).sum())
                        pct_rep = n_rep / len(preds) * 100

                        c1, c2 = st.columns(2)
                        c1.metric("Dự đoán Mua lại (Repeat)", f"{n_rep} ({pct_rep:.1f}%)")
                        c2.metric("Dự đoán Không mua lại (No-Repeat)", f"{n_non} ({100 - pct_rep:.1f}%)")

                        # Download button
                        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Tải về file kết quả dự đoán (CSV)",
                            data=csv_bytes,
                            file_name="predicted_repeat_purchase_results.csv",
                            mime="text/csv"
                        )

# -------------------------------------------------------------
# TAB 2: CLUSTERING & ACTION PLAN
# -------------------------------------------------------------
with tab2:
    st.header("👥 Phân khúc Khách hàng & Kế hoạch Hành động Chiến lược")

    cl_cfg = artifacts.get('cl_config', {})
    cl_prof = artifacts.get('cl_profiles')
    act_plan = artifacts.get('action_plan')

    if cl_cfg:
        st.info(
            f"**Cấu hình Phân cụm được chọn:** Thuật toán **{cl_cfg.get('algorithm', 'K-Means')}** "
            f"(K={cl_cfg.get('n_clusters', 2)}) — Phiên bản đặc trưng: `{cl_cfg.get('feature_version', 'log1p_scaled')}` — "
            f"Số khách hàng phân tích: {cl_cfg.get('n_customers', 0):,}."
        )

    if cl_prof is not None and not cl_prof.empty:
        st.subheader("1. Hồ sơ Đặc trưng các Phân khúc (Segment Profiles)")
        st.dataframe(cl_prof, use_container_width=True)

    if act_plan is not None and not act_plan.empty:
        st.write("---")
        st.subheader("2. Kế hoạch Hành động Chiến lược (Strategic Action Plan)")
        st.dataframe(act_plan, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: ASSOCIATION RULES
# -------------------------------------------------------------
with tab3:
    st.header("🛒 Khai phá Luật Kết hợp Sản phẩm & Gợi ý Bán chéo (Cross-Sell)")

    rules_df = artifacts.get('rules')
    if rules_df is None or rules_df.empty:
        st.info("Chưa tìm thấy luật kết hợp đã chọn. Vui lòng chạy pipeline bước 06.")
    else:
        st.subheader("1. Tra cứu Gợi ý Sản phẩm đồng mua")
        all_antecedents = sorted(list(rules_df['antecedents_str'].dropna().unique()))
        selected_item = st.selectbox("Chọn sản phẩm khách hàng đang xem / có trong giỏ hàng:", all_antecedents)

        matched_rules = rules_df[rules_df['antecedents_str'] == selected_item].sort_values('lift', ascending=False)
        if not matched_rules.empty:
            st.write(f"Tìm thấy **{len(matched_rules)}** gợi ý sản phẩm đi kèm:")
            for _, r in matched_rules.iterrows():
                st.markdown(
                    f"- 🎯 **Gợi ý mua kèm:** `{r['consequents_str']}` | "
                    f"**Độ tin cậy:** {r['confidence']:.1%} | "
                    f"**Độ nâng (Lift):** {r['lift']:.2f}x | "
                    f"**Độ hỗ trợ:** {r['support']:.2%}"
                )

        st.write("---")
        st.subheader("2. Bảng 10 Luật Kết hợp có Lift cao nhất toàn bộ hệ thống")
        disp_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
        if 'algorithm' in rules_df.columns:
            disp_cols.append('algorithm')
        st.dataframe(rules_df[disp_cols].head(10), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: GOVERNANCE & MANIFEST
# -------------------------------------------------------------
with tab4:
    st.header("📑 Báo cáo Kỹ thuật & Quản trị Mô hình (Model Governance & Lineage)")

    manifest = artifacts.get('manifest', {})
    if manifest:
        st.subheader("1. Thông tin Phiên bản Pipeline (Provenance)")
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        col_g1.metric("Run ID", manifest.get('run_id', 'N/A'))
        col_g2.metric("Trạng thái Xác thực", manifest.get('validation_status', 'PASS'))
        col_g3.metric("Git Commit", str(manifest.get('git_commit', 'unknown'))[:8])
        col_g4.metric("Phiên bản Python", manifest.get('python_version', '3.12'))

        st.subheader("2. Kiểm định Tính toàn vẹn Dữ liệu (SHA-256 Checksums)")
        steps = manifest.get('steps', {})
        step_names = list(steps.keys())
        selected_step = st.selectbox("Chọn bước pipeline để xem file outputs & checksums:", step_names)

        if selected_step:
            file_entries = steps[selected_step]
            manifest_table = pd.DataFrame(file_entries)
            if not manifest_table.empty:
                manifest_table['size_kb'] = (manifest_table['size_bytes'] / 1024).round(1)
                st.dataframe(manifest_table[['file', 'size_kb', 'sha256']], use_container_width=True)

    st.write("---")
    st.subheader("3. Đường dẫn Báo cáo Tổng kết CRISP-DM")
    report_file = PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights.md'
    if report_file.exists():
        st.success(f"Báo cáo Markdown đầy đủ sẵn sàng tại: `{report_file.relative_to(PROJECT_ROOT)}`")
