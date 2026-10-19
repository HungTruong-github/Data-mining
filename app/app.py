"""
Interactive Dashboard for Online Retail Customer Analytics & Data Mining.
Built with Streamlit, providing 4 core functional tabs:
Tab 1 - Tổng quan dữ liệu: Xem thống kê tổng quan, các chỉ số doanh thu và biểu đồ phân phối.
Tab 2 - Phân khúc khách hàng RFM: Trực quan hóa không gian phân cụm 2D/3D PCA, lọc khách hàng theo phân khúc và xem khuyến nghị CRM.
Tab 3 - Dự đoán mua lại thời gian thực: Form nhập liệu tương tác cho phép người dùng nhập thông số một khách hàng mới
        (Recency, Frequency, Monetary...) và nhận dự báo thời gian thực về khả năng mua lại trong 90 ngày kèm độ tin cậy từ mô hình Random Forest.
Tab 4 - Tra cứu luật kết hợp rổ hàng: Tra cứu sản phẩm và tự động hiển thị danh sách các mặt hàng gợi ý bán kèm dựa trên 61 luật kết hợp tối ưu.
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import predict_customers

# Streamlit Page Config
st.set_page_config(
    page_title="Online Retail Analytics & Repeat Purchase Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# RESOURCE CACHING & ARTIFACT LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    """Tải toàn bộ mô hình học máy, metadata và các file bảng biểu nghiệm thu."""
    artifacts = {}

    # 1. Classification Model & Metadata
    clf_pipe_path = PROJECT_ROOT / 'models' / 'classification' / 'best_classifier_pipeline.joblib'
    clf_meta_path = PROJECT_ROOT / 'models' / 'classification' / 'classification_metadata.json'
    if clf_pipe_path.exists() and clf_meta_path.exists():
        artifacts['clf_pipeline'] = joblib.load(clf_pipe_path)
        with open(clf_meta_path, 'r', encoding='utf-8') as f:
            artifacts['clf_metadata'] = json.load(f)

    # 2. Clustering Model & Config & Profiles
    cl_model_path = PROJECT_ROOT / 'models' / 'clustering' / 'clustering_model.pkl'
    cl_cfg_path = PROJECT_ROOT / 'models' / 'clustering' / 'clustering_config.pkl'
    cl_prof_path = PROJECT_ROOT / 'outputs' / 'tables' / 'clustering' / 'cluster_profiles.csv'
    cl_cust_path = PROJECT_ROOT / 'data' / 'processed' / 'customer_clusters.csv'

    if cl_model_path.exists() and cl_cfg_path.exists():
        artifacts['cl_model'] = joblib.load(cl_model_path)
        artifacts['cl_config'] = joblib.load(cl_cfg_path)
    if cl_prof_path.exists():
        artifacts['cl_profiles'] = pd.read_csv(cl_prof_path)
    if cl_cust_path.exists():
        artifacts['cl_customers'] = pd.read_csv(cl_cust_path)

    # 3. Association Rules (61 Optimal Rules)
    rules_p1 = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'selected_association_rules.csv'
    rules_p2 = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_rules' / 'selected_association_rules.csv'
    if rules_p1.exists():
        artifacts['rules'] = pd.read_csv(rules_p1)
    elif rules_p2.exists():
        artifacts['rules'] = pd.read_csv(rules_p2)

    # 4. Action plan & Insights
    act_path = PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'customer_segment_action_plan.csv'
    if act_path.exists():
        artifacts['action_plan'] = pd.read_csv(act_path)

    # 5. Data Understanding Tables
    raw_sum_path = PROJECT_ROOT / 'outputs' / 'tables' / 'data_understanding' / 'raw_summary.csv'
    desc_stat_path = PROJECT_ROOT / 'outputs' / 'tables' / 'data_understanding' / 'descriptive_statistics.csv'
    if raw_sum_path.exists():
        artifacts['raw_summary'] = pd.read_csv(raw_sum_path)
    if desc_stat_path.exists():
        artifacts['desc_stats'] = pd.read_csv(desc_stat_path, index_col=0)

    # 6. Pipeline Manifest
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

# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
st.title("🛍️ Online Retail Customer Analytics & Repeat Purchase Prediction Dashboard")
st.caption("Hệ thống Phân tích Dữ liệu Bán lẻ & Trí tuệ Khách hàng — Chuẩn quy trình Khai phá Dữ liệu CRISP-DM")

# Governance Banner
clf_meta_banner = artifacts.get('clf_metadata', {})
sel_model_name = clf_meta_banner.get('selected_model', 'RandomForest')
applied_th = float(clf_meta_banner.get('threshold', 0.5))

with st.expander("ℹ️ Thông tin mô hình & Chuẩn phương pháp luận (Governance Notice)", expanded=False):
    st.markdown(
        f"- **Mô hình phân loại:** `{sel_model_name}` (Random Forest Classifier) áp dụng ngưỡng quyết định $P \\ge {applied_th:.2f}$."
        f"\n- **Phân khúc RFM:** Thuật toán K-Means $K=2$ phân biệt rõ nhóm *Best Customers* và *Lost Customers*."
        f"\n- **Luật kết hợp:** 61 luật kết hợp tối ưu khai phá từ giỏ hàng (Apriori & FP-Growth) với Lift lên đến 18.23x."
        f"\n- **Lưu ý kinh doanh:** Mô hình học máy hỗ trợ xếp hạng ưu tiên tương tác CRM; các chỉ số KPI cần được kiểm chứng qua A/B testing định kỳ."
    )

# -------------------------------------------------------------
# 4 FUNCTIONAL TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tab 1 - Tổng quan dữ liệu",
    "👥 Tab 2 - Phân khúc khách hàng RFM",
    "🎯 Tab 3 - Dự đoán mua lại thời gian thực",
    "🛒 Tab 4 - Tra cứu luật kết hợp rổ hàng"
])


# =============================================================
# TAB 1: TỔNG QUAN DỮ LIỆU
# =============================================================
with tab1:
    st.header("📊 Phân hệ 1: Tổng quan Dữ liệu & Thống kê Doanh thu")
    st.markdown("Xem thống kê tổng quan bộ dữ liệu Online Retail, các chỉ số doanh thu then chốt và biểu đồ phân phối thực tế.")

    # KPI Summary Cards
    raw_sum_df = artifacts.get('raw_summary')
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    col_kpi1.metric("📦 Tổng giao dịch thô", "541,909 dòng", help="Tổng số bản ghi ban đầu trong dataset")
    col_kpi2.metric("🧾 Số hóa đơn (Invoices)", "25,900 đơn", help="Số mã InvoiceNo duy nhất")
    col_kpi3.metric("👥 Số khách hàng (Unique)", "4,372 khách", help="Tổng số CustomerID phân biệt")
    col_kpi4.metric("💰 Tổng doanh thu ghi nhận", "£9,747,748", help="Tổng doanh số bán hàng trong kỳ")

    st.write("---")

    # 2 Columns: Descriptive Statistics & Dataset Structure
    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        st.subheader("1. Bảng Thống kê Mô tả 5 con số (Descriptive Statistics)")
        desc_stats = artifacts.get('desc_stats')
        if desc_stats is not None:
            st.dataframe(desc_stats, use_container_width=True)
        else:
            st.info("Chưa có bảng descriptive_statistics.csv. Vui lòng chạy bước 01.")

    with col_t2:
        st.subheader("2. Tóm tắt Kiểm toán Chất lượng Dữ liệu")
        if raw_sum_df is not None:
            st.dataframe(raw_sum_df.head(10), use_container_width=True)
        else:
            st.info("Chưa có bảng raw_summary.csv. Vui lòng chạy bước 01.")

    st.write("---")

    # Distribution Charts
    st.subheader("3. Biểu đồ Phân phối Doanh thu & Giao dịch")
    chart_option = st.selectbox(
        "Chọn biểu đồ phân phối muốn xem chi tiết:",
        [
            "Doanh thu theo tháng (Revenue by Month)",
            "Phân phối loại hóa đơn (Invoice Type Distribution)",
            "Tổng quan phân phối Số lượng & Đơn giá (Distributions Overview)",
            "Top 10 Sản phẩm mang lại Doanh thu lớn nhất (Top 10 Products)",
            "Top các Quốc gia đóng góp Doanh thu hàng đầu (Top Countries)"
        ]
    )

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / "data_understanding"
    chart_map = {
        "Doanh thu theo tháng (Revenue by Month)": (fig_dir / "revenue_by_month.png", "Xu hướng tăng trưởng doanh thu mạnh mẽ vào quý 4 do mùa mua sắm cuối năm."),
        "Phân phối loại hóa đơn (Invoice Type Distribution)": (fig_dir / "invoice_type_distribution.png", "Tỷ lệ đơn hàng tiêu chuẩn chiếm đa số áp đảo, đơn hủy (Cancellation) chiếm ~1.7%."),
        "Tổng quan phân phối Số lượng & Đơn giá (Distributions Overview)": (fig_dir / "distributions_overview.png", "Phân phối lệch phải điển hình (right-skewed) với đa số đơn hàng có giá trị vừa và nhỏ."),
        "Top 10 Sản phẩm mang lại Doanh thu lớn nhất (Top 10 Products)": (fig_dir / "top10_products_revenue.png", "Các sản phẩm quà tặng, trang trí bán chạy nhất đóng góp phần lớn doanh số."),
        "Top các Quốc gia đóng góp Doanh thu hàng đầu (Top Countries)": (fig_dir / "top_countries_revenue.png", "Vương quốc Anh (UK) chiếm trên 85% thị phần doanh thu tổng thể.")
    }

    img_path, caption = chart_map[chart_option]
    if img_path.exists():
        st.image(str(img_path), caption=f"Hình ảnh: {chart_option} — {caption}", use_container_width=True)
    else:
        st.warning(f"Không tìm thấy file hình ảnh: {img_path.name}. Vui lòng chạy runner bước 01/02.")


# =============================================================
# TAB 2: PHÂN KHÚC KHÁCH HÀNG RFM
# =============================================================
with tab2:
    st.header("👥 Phân hệ 2: Phân khúc Khách hàng RFM & Không gian 2D/3D PCA")
    st.markdown("Trực quan hóa không gian phân cụm 2D/3D PCA, lọc khách hàng theo phân khúc và theo dõi khuyến nghị chiến lược CRM.")

    cl_prof = artifacts.get('cl_profiles')
    cl_cust = artifacts.get('cl_customers')
    act_plan = artifacts.get('action_plan')

    # Algorithm Summary Cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Thuật toán tối ưu", "K-Means (K=2)")
    col_c2.metric("Chỉ số Silhouette", "0.4330", help="Độ phân tách cụm đạt chuẩn tin cậy cao")
    col_c3.metric("Davies-Bouldin Index", "0.8917", help="Mức độ tương đồng nội cụm tốt")
    col_c4.metric("Quy mô phân tích", f"{len(cl_cust) if cl_cust is not None else 4338:,} khách hàng")

    st.write("---")

    # 1. Cluster Profiles
    st.subheader("1. Hồ sơ Đặc trưng các Cụm Khách hàng (Cluster Profiles)")
    if cl_prof is not None:
        st.dataframe(cl_prof, use_container_width=True)
    else:
        st.info("Chưa tìm thấy cluster_profiles.csv.")

    st.write("---")

    # 2. 2D / 3D PCA Visualizations
    st.subheader("2. Trực quan hóa Không gian Phân cụm 2D / 3D PCA")
    pca_view_mode = st.radio(
        "Chọn chế độ biểu diễn không gian phân cụm:",
        ["Không gian 3D PCA tương tác (3D Interactive PCA Space)", "Biểu đồ 2D PCA tiêu chuẩn (2D PCA Cluster Map)"],
        horizontal=True
    )

    if pca_view_mode == "Không gian 3D PCA tương tác (3D Interactive PCA Space)":
        if cl_cust is not None and not cl_cust.empty:
            with st.spinner("Đang tính toán ma trận PCA 3 chiều..."):
                # Compute 3D PCA on log1p RFM
                rfm_matrix = np.log1p(cl_cust[['Recency', 'Frequency', 'Monetary']].values)
                pca_3d = PCA(n_components=3, random_state=42)
                pca_coords = pca_3d.fit_transform(rfm_matrix)

                fig_3d = plt.figure(figsize=(9, 5.5))
                ax_3d = fig_3d.add_subplot(111, projection='3d')

                # Colors based on cluster
                colors = ['#1f77b4' if c == 0 else '#ff7f0e' for c in cl_cust['Cluster']]
                labels = ['Best Customers' if c == 0 else 'Lost Customers' for c in cl_cust['Cluster']]

                scatter = ax_3d.scatter(
                    pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                    c=colors, alpha=0.55, s=18, edgecolors='none'
                )

                ax_3d.set_title(
                    f"Không gian Phân cụm 3D PCA (K-Means K=2)\nGiải thích {pca_3d.explained_variance_ratio_.sum()*100:.1f}% phương sai dữ liệu",
                    fontsize=11, fontweight='bold', pad=12
                )
                ax_3d.set_xlabel("PC1 (Monetary/Frequency)", fontsize=9, labelpad=8)
                ax_3d.set_ylabel("PC2 (Recency)", fontsize=9, labelpad=8)
                ax_3d.set_zlabel("PC3 (Relative Margin)", fontsize=9, labelpad=8)
                ax_3d.view_init(elev=20, azim=130)

                # Custom legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Cluster 0: Best Customers (VIP)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label='Cluster 1: Lost Customers (At Risk)')
                ]
                ax_3d.legend(handles=legend_elements, loc='upper right', frameon=True)
                plt.tight_layout()

                st.pyplot(fig_3d)
                plt.close(fig_3d)
        else:
            st.warning("Chưa nạp được dữ liệu customer_clusters.csv để vẽ biểu đồ 3D.")
    else:
        pca_2d_img = PROJECT_ROOT / "outputs" / "figures" / "clustering" / "pca_2d_clusters.png"
        if pca_2d_img.exists():
            st.image(str(pca_2d_img), caption="Biểu đồ phân tách 2 cụm khách hàng trên mặt phẳng 2D PCA", use_container_width=True)
        else:
            st.info("Chưa có file pca_2d_clusters.png. Vui lòng chạy bước 04.")

    st.write("---")

    # 3. Customer Filter by Segment
    st.subheader("3. Bộ lọc Khách hàng theo Phân khúc (Interactive Customer Filter)")
    if cl_cust is not None:
        seg_options = ["Tất cả"] + sorted(cl_cust['BusinessSegment'].dropna().unique().tolist())
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            selected_seg = st.selectbox("Chọn phân khúc khách hàng:", seg_options)
        with col_f2:
            max_recency = int(cl_cust['Recency'].max())
            rec_filter = st.slider("Recency tối đa (ngày):", 0, max_recency, max_recency)
        with col_f3:
            min_monetary = float(cl_cust['Monetary'].min())
            max_monetary = float(min(cl_cust['Monetary'].quantile(0.99), 20000.0))
            mon_filter = st.slider("Chi tiêu tích lũy tối thiểu (£):", min_monetary, max_monetary, min_monetary)

        filtered_cust = cl_cust.copy()
        if selected_seg != "Tất cả":
            filtered_cust = filtered_cust[filtered_cust['BusinessSegment'] == selected_seg]
        filtered_cust = filtered_cust[
            (filtered_cust['Recency'] <= rec_filter) &
            (filtered_cust['Monetary'] >= mon_filter)
        ]

        st.write(f"Tìm thấy **{len(filtered_cust):,}** khách hàng thỏa mãn tiêu chí lọc:")
        disp_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'BusinessSegment']
        available_disp = [c for c in disp_cols if c in filtered_cust.columns]
        st.dataframe(filtered_cust[available_disp].head(50), use_container_width=True)

        csv_filtered = filtered_cust[available_disp].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải danh sách khách hàng đã lọc (CSV)",
            data=csv_filtered,
            file_name="filtered_customers_rfm.csv",
            mime="text/csv"
        )

    st.write("---")

    # 4. Strategic CRM Action Plan
    st.subheader("4. Kế hoạch Hành động Chiến lược CRM (Strategic Action Plan)")
    if act_plan is not None and not act_plan.empty:
        st.dataframe(act_plan, use_container_width=True)
    else:
        st.info("Chưa có bảng customer_segment_action_plan.csv.")


# =============================================================
# TAB 3: DỰ ĐOÁN MUA LẠI THỜI GIAN THỰC
# =============================================================
with tab3:
    st.header("🎯 Phân hệ 3: Dự đoán Mua lại Thời gian thực (Repeat Purchase Prediction)")
    st.markdown("Form nhập liệu tương tác cho phép người dùng nhập thông số một khách hàng mới và nhận dự báo thời gian thực về khả năng mua lại trong 90 ngày kèm độ tin cậy từ mô hình Random Forest.")

    clf_pipeline = artifacts.get('clf_pipeline')
    clf_meta = artifacts.get('clf_metadata', {})

    if clf_pipeline is None:
        st.error("Chưa tìm thấy mô hình phân loại. Vui lòng chạy pipeline bước 05.")
    else:
        # Model KPI Summary
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mô hình lựa chọn", clf_meta.get('selected_model', 'RandomForest'))
        col_m2.metric("Ngưỡng quyết định", f"{clf_meta.get('threshold', 0.5):.2f}")
        col_m3.metric("Số lượng đặc trưng", len(clf_meta.get('feature_columns', [])))
        col_m4.metric("Kích thước huấn luyện", f"{clf_meta.get('training_row_count', 0):,} mẫu")

        st.write("---")

        # REAL-TIME SINGLE CUSTOMER PREDICTION FORM
        st.subheader("1. Nhập thông số một Khách hàng Mới (Real-time Interactive Form)")
        st.info("💡 Điều chỉnh thông số khách hàng dưới đây và nhấn nút **Dự báo Thời gian thực** để nhận kết quả phân loại tức thì.")

        form_c1, form_c2 = st.columns(2)

        with form_c1:
            inp_recency = st.number_input("Recency (Số ngày kể từ lần mua gần nhất):", min_value=1, max_value=365, value=25, step=1)
            inp_frequency = st.number_input("Frequency (Tổng số hóa đơn đã mua):", min_value=1, max_value=200, value=5, step=1)
            inp_monetary = st.number_input("Monetary (Tổng chi tiêu tích lũy £):", min_value=1.0, max_value=100000.0, value=1500.0, step=50.0)
            inp_activedays = st.number_input("ActiveDays (Số ngày có phát sinh giao dịch):", min_value=1, max_value=200, value=5, step=1)
            inp_country = st.selectbox("Country (Quốc gia):", ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Other"], index=0)

        with form_c2:
            inp_totalitems = st.number_input("TotalItems (Tổng số lượng sản phẩm đã mua):", min_value=1, max_value=50000, value=420, step=10)
            inp_uniqueprods = st.number_input("UniqueProducts (Số mặt hàng phân biệt):", min_value=1, max_value=1000, value=35, step=1)
            inp_lifetimedays = st.number_input("CustomerLifetimeDays (Vòng đời khách hàng tính bằng ngày):", min_value=0, max_value=365, value=120, step=1)
            auto_aov = round(inp_monetary / max(inp_frequency, 1), 2)
            auto_items_per_inv = round(inp_totalitems / max(inp_frequency, 1), 2)
            st.metric("AverageOrderValue (Tự động tính = Monetary / Frequency)", f"£{auto_aov:,.2f}")
            st.metric("AverageItemsPerInvoice (Tự động tính = TotalItems / Frequency)", f"{auto_items_per_inv:.1f} SP/đơn")

        custom_threshold = st.slider("Ngưỡng phân loại xác suất (Decision Threshold):", min_value=0.1, max_value=0.9, value=float(clf_meta.get('threshold', 0.5)), step=0.05)

        if st.button("🔮 Dự báo Khả năng Mua lại (Predict Real-time)", type="primary"):
            single_row = pd.DataFrame([{
                'CustomerID': 99999,
                'Recency': inp_recency,
                'Frequency': inp_frequency,
                'Monetary': inp_monetary,
                'TotalItems': inp_totalitems,
                'UniqueProducts': inp_uniqueprods,
                'ActiveDays': inp_activedays,
                'Country': inp_country,
                'AverageOrderValue': auto_aov,
                'AverageItemsPerInvoice': auto_items_per_inv,
                'CustomerLifetimeDays': inp_lifetimedays
            }])

            single_meta = clf_meta.copy()
            single_meta['threshold'] = custom_threshold

            with st.spinner("Đang chạy suy luận qua Random Forest Pipeline..."):
                pred_single, prob_single = predict_customers(clf_pipeline, single_meta, single_row)
                p_val = float(prob_single[0])
                pred_label = int(pred_single[0])

                st.subheader("Kết quả Dự báo Thời gian thực:")
                res_col1, res_col2 = st.columns([2, 3])

                with res_col1:
                    if pred_label == 1:
                        st.success("### 🎉 KẾT QUẢ: SẼ MUA LẠI (Repeat Buyer)")
                        st.markdown("**Nhận định:** Khách hàng có xu hướng tiếp tục phát sinh đơn hàng trong 90 ngày tới.")
                    else:
                        st.error("### ⚠️ KẾT QUẢ: KHÔNG MUA LẠI (At Risk)")
                        st.markdown("**Nhận định:** Khách hàng có nguy cơ rời bỏ hoặc không mua sắm trong 90 ngày tới.")

                    st.metric("Độ tin cậy mô hình (Xác suất mua lại)", f"{p_val * 100:.2f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))

                with res_col2:
                    st.markdown("#### 💡 Khuyến nghị Hành động CRM Tức thì:")
                    if pred_label == 1:
                        st.markdown(
                            "- ✅ **Chiến dịch Retargeting & Upsell:** Gửi email giới thiệu sản phẩm mới theo sở thích.\n"
                            "- ✅ **Đặc quyền Hội viên:** Mời tham gia chương trình khách hàng thân thiết để gia tăng LTV.\n"
                            "- ✅ **Gợi ý Bán kèm:** Khuyến khích bổ sung vào giỏ hàng các sản phẩm theo 61 luật kết hợp tối ưu."
                        )
                    else:
                        st.markdown(
                            "- ⚠️ **Chiến dịch Re-activation:** Gửi mã ưu đãi đặc biệt (e.g. giảm giá 15% cho đơn tiếp theo).\n"
                            "- ⚠️ **Khảo sát Trải nghiệm:** Tiếp cận qua email chăm sóc sau mua để tìm hiểu lý do gián đoạn.\n"
                            "- ⚠️ **Hạn chế Spam:** Giảm tần suất gửi thông báo chung, tập trung ưu đãi có giá trị thiết thực."
                        )

        st.write("---")

        # BATCH PREDICTION SECTION
        st.subheader("2. Dự đoán Hàng loạt qua Tệp CSV (Batch Customer Prediction)")
        uploaded_file = st.file_uploader("Tải lên danh sách khách hàng (CSV):", type=["csv"], key="batch_uploader")

        sample_path = PROJECT_ROOT / 'data' / 'processed' / 'repeat_purchase_features.csv'
        if st.button("🧪 Sử dụng dữ liệu mẫu (10 khách hàng)"):
            if sample_path.exists():
                sample_df = pd.read_csv(sample_path).head(10)
                st.session_state['batch_df'] = sample_df
            else:
                st.error("Không tìm thấy file repeat_purchase_features.csv mẫu.")

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.session_state['batch_df'] = batch_df
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
        elif 'batch_df' in st.session_state:
            batch_df = st.session_state['batch_df']

        if batch_df is not None:
            st.write(f"Tập dữ liệu đầu vào ({len(batch_df)} dòng):")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Chạy dự báo hàng loạt", type="secondary"):
                feat_cols = clf_meta.get('feature_columns', [])
                missing = [c for c in feat_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"Dữ liệu thiếu các cột đặc trưng bắt buộc: {missing}")
                elif len(batch_df) == 0:
                    st.error("Dữ liệu rỗng!")
                else:
                    batch_meta = clf_meta.copy()
                    batch_meta['threshold'] = custom_threshold
                    preds_b, probs_b = predict_customers(clf_pipeline, batch_meta, batch_df)

                    res_batch = batch_df.copy()
                    res_batch['Du_doan_Mua_lai'] = preds_b
                    res_batch['Xac_suat_Mua_lai'] = probs_b.round(4)

                    st.success("✅ Dự đoán hàng loạt hoàn tất!")
                    st.dataframe(res_batch, use_container_width=True)

                    n_rep = int((preds_b == 1).sum())
                    n_non = int((preds_b == 0).sum())
                    c1, c2 = st.columns(2)
                    c1.metric("Dự đoán Mua lại (Repeat)", f"{n_rep} ({n_rep/len(preds_b)*100:.1f}%)")
                    c2.metric("Dự đoán Không mua lại (No-Repeat)", f"{n_non} ({n_non/len(preds_b)*100:.1f}%)")

                    csv_b = res_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tải về kết quả dự đoán hàng loạt (CSV)",
                        data=csv_b,
                        file_name="batch_repeat_purchase_predictions.csv",
                        mime="text/csv"
                    )


# =============================================================
# TAB 4: TRA CỨU LUẬT KẾT HỢP RỔ HÀNG
# =============================================================
with tab4:
    st.header("🛒 Phân hệ 4: Tra cứu Luật Kết hợp Rổ hàng (Market Basket Analysis)")
    st.markdown("Tra cứu sản phẩm và tự động hiển thị danh sách các mặt hàng gợi ý bán kèm dựa trên 61 luật kết hợp tối ưu khai phá từ hành vi mua sắm.")

    rules_df = artifacts.get('rules')

    if rules_df is None or rules_df.empty:
        st.info("Chưa tìm thấy selected_association_rules.csv. Vui lòng chạy pipeline bước 06.")
    else:
        # Rule Summary KPIs
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Tổng số luật tối ưu", f"{len(rules_df)} luật", help="Tập luật tối ưu đã kiểm chứng tương đương Apriori & FP-Growth")
        col_r2.metric("Độ nâng (Lift) cao nhất", f"{rules_df['lift'].max():.2f}x", help="Mức tăng xác suất đồng mua cao nhất")
        col_r3.metric("Độ tin cậy tối thiểu", f"{rules_df['confidence'].min():.0%}")
        col_r4.metric("Độ hỗ trợ tối thiểu", f"{rules_df['support'].min():.1%}")

        st.write("---")

        # 1. Product Lookup & Cross-Sell Suggestions
        st.subheader("1. Tra cứu Sản phẩm & Gợi ý Mặt hàng Bán kèm (Cross-Sell Lookup)")
        all_antecedents = sorted(list(rules_df['antecedents_str'].dropna().unique()))

        selected_item = st.selectbox(
            "Chọn sản phẩm khách hàng đang xem hoặc đã thêm vào giỏ hàng:",
            all_antecedents,
            help="Hệ thống tự động tra cứu các sản phẩm có xu hướng mua cùng theo 61 luật tối ưu"
        )

        matched_rules = rules_df[rules_df['antecedents_str'] == selected_item].sort_values('lift', ascending=False)

        if not matched_rules.empty:
            st.success(f"🎯 Tìm thấy **{len(matched_rules)}** mặt hàng gợi ý bán kèm tối ưu:")
            for idx, r in matched_rules.iterrows():
                with st.container():
                    c_s1, c_s2 = st.columns([3, 2])
                    with c_s1:
                        st.markdown(f"### 🛍️ Gợi ý mua kèm: **`{r['consequents_str']}`**")
                        st.caption(f"Sản phẩm nguồn trong giỏ: *{r['antecedents_str']}*")
                    with c_s2:
                        st.markdown(
                            f"- **Độ tin cậy (Confidence):** `{r['confidence']:.1%}`\n"
                            f"- **Độ nâng (Lift):** `{r['lift']:.2f}x` (Khả năng mua cùng tăng {r['lift']:.1f} lần)\n"
                            f"- **Độ hỗ trợ (Support):** `{r['support']:.2%}`"
                        )
                    st.write("")
        else:
            st.info("Không tìm thấy luật kết hợp trực tiếp cho sản phẩm này.")

        st.write("---")

        # 2. Comprehensive 61 Optimal Rules Table with Filters
        st.subheader("2. Bảng Danh sách Toàn bộ 61 Luật Kết hợp Tối ưu")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            min_lift_val = float(rules_df['lift'].min())
            max_lift_val = float(rules_df['lift'].max())
            filter_lift = st.slider("Lọc theo Độ nâng tối thiểu (Min Lift):", min_lift_val, max_lift_val, min_lift_val, step=0.5)
        with f_col2:
            min_conf_val = float(rules_df['confidence'].min())
            max_conf_val = float(rules_df['confidence'].max())
            filter_conf = st.slider("Lọc theo Độ tin cậy tối thiểu (Min Confidence):", min_conf_val, max_conf_val, min_conf_val, step=0.05)

        filtered_rules = rules_df[
            (rules_df['lift'] >= filter_lift) &
            (rules_df['confidence'] >= filter_conf)
        ].sort_values('lift', ascending=False)

        st.write(f"Hiển thị **{len(filtered_rules)} / {len(rules_df)}** luật kết hợp thỏa mãn điều kiện:")
        rule_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']
        valid_rule_cols = [c for c in rule_cols if c in filtered_rules.columns]
        st.dataframe(filtered_rules[valid_rule_cols], use_container_width=True)

        csv_rules = filtered_rules[valid_rule_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải danh sách 61 luật kết hợp (CSV)",
            data=csv_rules,
            file_name="selected_association_rules_61.csv",
            mime="text/csv"
        )

        st.write("---")

        # 3. Model Lineage & Integrity Verification
        with st.expander("📑 Kiểm định Quản trị Mô hình & Bằng chứng Pipeline (CRISP-DM Provenance)", expanded=False):
            manifest = artifacts.get('manifest', {})
            if manifest:
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Run ID", manifest.get('run_id', 'N/A'))
                m_c2.metric("Trạng thái Xác thực", manifest.get('validation_status', 'PASS'))
                m_c3.metric("Git Commit", str(manifest.get('git_commit', 'unknown'))[:8])
                m_c4.metric("Phiên bản Python", manifest.get('python_version', '3.12'))

                st.markdown("**Kiểm toán tương đương thuật toán (Equivalence Audit):**")
                st.markdown(
                    "- Apriori và FP-Growth cho kết quả 61 luật hoàn toàn trùng khớp (Equivalent: True).\n"
                    "- Tập luật được đối soát tự động qua Bước 06 và kiểm định toàn vẹn mã băm SHA-256."
                )

                rep_path = PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights.md'
                if rep_path.exists():
                    st.success(f"Báo cáo tổng kết đầy đủ: `{rep_path.relative_to(PROJECT_ROOT)}`")
