import numpy as np
import pandas as pd
from src.model_comparison import load_input


# ====================================================================
# 1. CUSTOMER SEGMENT INSIGHTS
# ====================================================================
def build_customer_segment_insights():
    """
    Merge clusters + RFM + repeat purchase data to create segment profiles.
    All numbers come from actual data — nothing is hard-coded.
    Reports both total cluster size and eligible cohort sample size for repeat rate.
    """
    clusters_df = load_input('customer_clusters')
    rfm_df = load_input('rfm_features')
    repeat_df = load_input('repeat_features')

    # Merge full RFM
    merged = clusters_df[['CustomerID', 'Cluster', 'BusinessSegment']].merge(
        rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary',
                'TotalItems', 'UniqueProducts', 'ActiveDays',
                'AverageOrderValue', 'CustomerLifetimeDays']],
        on='CustomerID', how='left'
    )

    # Merge repeat purchase label (at prediction cutoff)
    repeat_sub = repeat_df[['CustomerID', 'repeat_purchase_90d']].copy()
    merged = merged.merge(repeat_sub, on='CustomerID', how='left')

    total_customers = len(merged)

    insights = []
    for cluster_id in sorted(merged['Cluster'].unique()):
        seg = merged[merged['Cluster'] == cluster_id]
        n = len(seg)
        pct = n / total_customers * 100

        # RFM statistics
        rec_mean = seg['Recency'].mean()
        rec_med = seg['Recency'].median()
        freq_mean = seg['Frequency'].mean()
        freq_med = seg['Frequency'].median()
        mon_mean = seg['Monetary'].mean()
        mon_med = seg['Monetary'].median()
        aov = seg['AverageOrderValue'].mean() if 'AverageOrderValue' in seg.columns else np.nan

        # Cohort repeat purchase rate calculation
        labeled_mask = seg['repeat_purchase_90d'].notna()
        n_eligible = int(labeled_mask.sum())
        n_repeat = int((seg['repeat_purchase_90d'] == 1).sum())
        n_unlabeled = n - n_eligible
        coverage_pct = round(n_eligible / n * 100, 2) if n > 0 else 0.0

        if n_eligible > 0:
            repeat_rate = round(n_repeat / n_eligible, 4)
        else:
            repeat_rate = np.nan

        # Business segment name from data
        if 'BusinessSegment' in seg.columns and not seg['BusinessSegment'].mode().empty:
            biz_name = seg['BusinessSegment'].mode().iloc[0]
        else:
            biz_name = f'Cluster {cluster_id}'

        # Data-driven interpretation
        interp = _interpret_segment(rec_mean, freq_mean, mon_mean, repeat_rate, n, total_customers)

        insights.append({
            'cluster_id': cluster_id,
            'business_segment_name': biz_name,
            'customer_count': n,
            'customer_percentage': round(pct, 2),
            'eligible_labeled_customers': n_eligible,
            'repeat_customers': n_repeat,
            'unlabeled_customers': n_unlabeled,
            'cohort_coverage_pct': coverage_pct,
            'recency_mean': round(rec_mean, 1),
            'recency_median': round(rec_med, 1),
            'frequency_mean': round(freq_mean, 1),
            'frequency_median': round(freq_med, 1),
            'monetary_mean': round(mon_mean, 2),
            'monetary_median': round(mon_med, 2),
            'average_order_value': round(aov, 2) if not np.isnan(aov) else np.nan,
            'repeat_purchase_rate': repeat_rate,
            'business_interpretation': interp,
            'cohort_note': 'Repeat rate is evaluated retrospectively on customers active before cutoff date.',
            'evidence_source': 'data/processed/customer_clusters.csv + rfm_customer_features.csv + repeat_purchase_features.csv',
        })

    return pd.DataFrame(insights)


def _interpret_segment(rec, freq, mon, repeat_rate, n, total):
    """Generate dynamic segment interpretation without hardcoded labels."""
    pct = n / total * 100
    parts = []

    if rec < 50:
        parts.append(f"Khách hàng mua hàng rất gần đây (Recency trung bình {rec:.0f} ngày)")
    elif rec < 150:
        parts.append(f"Khách hàng giao dịch mức độ vừa phải (Recency trung bình {rec:.0f} ngày)")
    else:
        parts.append(f"Khách hàng có nguy cơ rời bỏ hoặc không hoạt động (Recency trung bình {rec:.0f} ngày)")

    if freq >= 5:
        parts.append(f"tần suất đặt hàng cao ({freq:.1f} đơn)")
    elif freq >= 2:
        parts.append(f"tần suất đặt hàng trung bình ({freq:.1f} đơn)")
    else:
        parts.append(f"tần suất đặt hàng thấp ({freq:.1f} đơn)")

    if mon >= 2000:
        parts.append(f"giá trị chi tiêu rất lớn (doanh thu trung bình £{mon:,.0f})")
    elif mon >= 500:
        parts.append(f"giá trị chi tiêu mức trung bình (doanh thu trung bình £{mon:,.0f})")
    else:
        parts.append(f"giá trị chi tiêu thấp (doanh thu trung bình £{mon:,.0f})")

    if not np.isnan(repeat_rate):
        parts.append(f"Tỷ lệ mua lại 90 ngày sau cutoff đạt {repeat_rate:.1%}")

    parts.append(f"Chiếm {pct:.1f}% tổng số khách hàng ({n:,} khách hàng)")
    return ". ".join(parts) + "."


# ====================================================================
# 2. ACTION PLAN
# ====================================================================
def build_action_plan(segment_insights_df):
    """
    Generate structured action plan based on segment characteristics.
    Links: Finding -> Evidence -> Eligible Segment -> Proposed Action -> KPI -> Verification Method -> Limitations.
    """
    if segment_insights_df is None or segment_insights_df.empty:
        raise ValueError("Cannot build action plan: segment_insights_df is empty or None")

    actions = []
    for _, row in segment_insights_df.iterrows():
        rec = row['recency_mean']
        freq = row['frequency_mean']
        mon = row['monetary_mean']
        repeat = row.get('repeat_purchase_rate', np.nan)
        cluster = row['cluster_id']
        name = row['business_segment_name']
        n_cust = row['customer_count']
        pct_cust = row['customer_percentage']

        action_list = []
        if rec > 150:
            action_list.append((
                'Reactivation / Win-back',
                f"Gửi chiến dịch tái kích hoạt qua email cá nhân hóa kèm mã ưu đãi độc quyền",
                f"Recency trung bình = {rec:.0f} ngày (> 150 ngày không phát sinh đơn)",
                "Tỷ lệ phản hồi mở email ≥ 15%; Tỷ lệ đặt lại đơn ≥ 5% trong 30 ngày",
                "A/B Testing 50/50: Nhóm thử nghiệm nhận email ưu đãi vs Nhóm đối chứng nhận bản tin chuẩn",
                "Chi phí chiết khấu có thể làm giảm biên lợi nhuận; chỉ áp dụng cho khách từng có Monetary khá"
            ))
        if rec <= 60 and freq >= 3:
            action_list.append((
                'Retention & VIP Loyalty',
                f"Triển khai chương trình khách hàng thân thiết VIP (ưu tiên giao hàng, quyền mua sớm bộ sưu tập mới)",
                f"Recency trung bình = {rec:.0f} ngày, Tần suất = {freq:.1f} đơn, Doanh thu = £{mon:,.0f}",
                "Tỷ lệ duy trì đơn hàng (Retention Rate) ≥ 80% trong 6 tháng kế tiếp",
                "Theo dõi tỷ lệ quay lại tự nhiên so với các quý trước (Cohort analysis)",
                "Chi phí vận hành chương trình tích điểm/ưu đãi thành viên"
            ))
        if mon > 1000:
            action_list.append((
                'Premium Upsell & Cross-sell',
                f"Giới thiệu các dòng sản phẩm cao cấp hoặc combo quà tặng giá trị cao",
                f"Chi tiêu trung bình £{mon:,.0f} (> £1,000)",
                "Tăng giá trị đơn hàng trung bình (AOV) thêm 10%",
                "Đo lường AOV của nhóm khách nhận gợi ý so với nhóm không nhận gợi ý",
                "Cần kiểm soát tồn kho các mặt hàng cao cấp trước khi đẩy mạnh gợi ý"
            ))
        if not np.isnan(repeat) and repeat < 0.4:
            action_list.append((
                'First-to-Second Purchase Nurturing',
                f"Chiến dịch nuôi dưỡng sau đơn hàng đầu tiên (hướng dẫn sử dụng sản phẩm, quà tặng đơn thứ hai)",
                f"Tỷ lệ mua lại chỉ đạt {repeat:.1%} (< 40%)",
                "Nâng tỷ lệ khách hàng mua lại lần 2 từ dưới 40% lên 45%",
                "Thử nghiệm email tự động gửi vào ngày thứ 14 sau đơn hàng đầu",
                "Rủi ro gây phiền toái nếu tần suất gửi email quá dày đặc"
            ))
        if not np.isnan(repeat) and repeat >= 0.6:
            action_list.append((
                'Advocacy & Referral Campaign',
                f"Kêu gọi đánh giá sản phẩm và giới thiệu bạn bè nhận thưởng hai chiều (Referral Program)",
                f"Tỷ lệ mua lại rất cao: {repeat:.1%} (≥ 60%)",
                "Tỷ lệ khách hàng giới thiệu thêm người dùng mới ≥ 8%",
                "Gắn mã giới thiệu cá nhân và đối chiếu số lượt đăng ký mới qua mã",
                "Cần chống gian lận tự tạo tài khoản phụ để nhận thưởng"
            ))

        if not action_list:
            action_list.append((
                'General Engagement',
                "Duy trì tương tác định kỳ qua bản tin cập nhật sản phẩm theo mùa",
                f"Recency={rec:.0f}, Freq={freq:.1f}, Monetary=£{mon:,.0f}",
                "Duy trì tỷ lệ mở email bản tin ≥ 20%",
                "Theo dõi định kỳ hàng tháng",
                "Không có ưu đãi tài chính trực tiếp"
            ))

        for strat, act, evid, kpi, verif, limit in action_list:
            actions.append({
                'cluster_id': cluster,
                'business_segment_name': name,
                'customer_count': n_cust,
                'customer_percentage': pct_cust,
                'strategy': strat,
                'recommended_action': act,
                'evidence': evid,
                'target_kpi': kpi,
                'verification_method': verif,
                'limitations': limit
            })

    return pd.DataFrame(actions)


# ====================================================================
# 3. PRODUCT ASSOCIATION INSIGHTS
# ====================================================================
def build_product_association_insights():
    """Generate business interpretations for top association rules."""
    rules = load_input('selected_rules')

    if rules is None or rules.empty:
        return pd.DataFrame()

    # Sort by lift descending, take top 20
    top = rules.nlargest(20, 'lift').copy()

    insights = []
    for _, row in top.iterrows():
        ant = row.get('antecedents_str', '')
        con = row.get('consequents_str', '')
        sup = row['support']
        conf = row['confidence']
        lift_val = row['lift']
        conv = row.get('conviction', np.nan)

        # Evidence quality assessment
        if sup >= 0.03 and conf >= 0.7 and lift_val >= 3:
            quality = 'Strong'
        elif sup >= 0.015 and conf >= 0.5 and lift_val >= 2:
            quality = 'Moderate'
        else:
            quality = 'Exploratory'

        # Limitations
        limitations = []
        if sup < 0.02:
            limitations.append(f"Độ hỗ trợ nhỏ ({sup:.3f}) — chỉ xuất hiện trong tập nhỏ giao dịch")
        if conf < 0.6:
            limitations.append(f"Độ tin cậy vừa phải ({conf:.2f}) — xác suất đồng mua chưa tuyệt đối")
        limitations.append("Quan hệ đồng xuất hiện tương quan, không chứng minh quan hệ nhân quả (causality)")

        interp = (
            f"Khách hàng mua '{ant}' có khả năng cao cũng mua '{con}' "
            f"(xác suất cao gấp {lift_val:.1f} lần so với khi hai sản phẩm xuất hiện độc lập). "
            f"Mô hình này xuất hiện trong {sup:.1%} tổng số giỏ hàng."
        )

        action = f"Đề xuất gợi ý chéo '{con}' khi khách hàng đưa '{ant}' vào giỏ hàng."
        if lift_val > 10:
            action += " Mức độ liên kết rất mạnh — có thể tạo combo đóng gói sẵn (product bundle)."
        elif lift_val > 3:
            action += " Mức độ liên kết tốt — hiển thị tại trang thanh toán (checkout recommendation)."

        insights.append({
            'antecedents': ant,
            'consequents': con,
            'support': round(sup, 4),
            'confidence': round(conf, 4),
            'lift': round(lift_val, 2),
            'conviction': round(conv, 4) if (not np.isnan(conv) and not np.isinf(conv)) else ('+Infinity' if np.isinf(conv) else np.nan),
            'evidence_quality': quality,
            'business_interpretation': interp,
            'recommended_action': action,
            'limitations': '; '.join(limitations),
        })

    return pd.DataFrame(insights)


# ====================================================================
# 4. FEATURE INSIGHTS
# ====================================================================
def build_feature_insights():
    """Interpret feature importance from classification model."""
    fi = load_input('feature_importance')

    if fi is None or fi.empty:
        return pd.DataFrame()

    # Load metadata for model name
    try:
        meta = load_input('class_metadata')
        model_name = meta.get('selected_model', 'Unknown')
    except Exception:
        model_name = 'Unknown'

    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    fi['rank'] = fi.index + 1
    fi['source_model'] = model_name

    # Categorize features
    rfm_cols = {'Recency', 'Frequency', 'Monetary'}
    behavioral_cols = {'TotalItems', 'UniqueProducts', 'ActiveDays',
                       'AverageOrderValue', 'AverageItemsPerInvoice', 'CustomerLifetimeDays'}

    def categorize(feat):
        base = feat.split('_')[0] if '_' in feat else feat
        if base in rfm_cols or feat in rfm_cols:
            return 'RFM'
        elif base in behavioral_cols or feat in behavioral_cols:
            return 'Behavioral'
        elif feat.startswith('Country'):
            return 'Geographic'
        return 'Other'

    fi['feature_group'] = fi['feature'].apply(categorize)

    # Interpretation
    interpretations = []
    for _, row in fi.iterrows():
        feat = row['feature']
        imp = row['importance']
        rank = row['rank']

        if rank <= 3:
            strength = "Top-3 biến quan trọng nhất"
            desc = f"Biến này có đóng góp phân tách mạnh nhất đối với dự đoán mua lại trong mô hình {model_name}."
        elif rank <= 10:
            strength = "Biến quan trọng mức trung bình"
            desc = f"Biến này có mức đóng góp vừa phải vào các nút phân nhánh của mô hình {model_name}."
        else:
            strength = "Biến có đóng góp thấp"
            desc = f"Biến này đóng góp không đáng kể vào quyết định dự báo của mô hình {model_name}."

        interp = (
            f"{strength} (tầm quan trọng = {imp:.4f}). {desc} "
            f"Lưu ý: Tầm quan trọng phụ thuộc vào mô hình và không chứng minh quan hệ nhân quả."
        )
        interpretations.append(interp)

    fi['interpretation'] = interpretations
    fi['limitations'] = 'Feature importance phản ánh tỷ lệ giảm độ vẩn đục (Gini/Impurity) trong mô hình cây, không phải tác động biên nhân quả.'

    return fi


# ====================================================================
# 5. GENERATE ALL INSIGHTS (ENFORCED COMPLETION)
# ====================================================================
def generate_all_insights():
    """
    Generate all business insights from model outputs.
    Guarantees that mandatory outputs (such as action plan) are generated or raises an error.
    """
    results = {}

    # 1. Segment insights (mandatory)
    seg_insights = build_customer_segment_insights()
    if seg_insights is None or seg_insights.empty:
        raise RuntimeError("CRITICAL: build_customer_segment_insights() returned empty result.")
    results['customer_segment_insights'] = seg_insights

    # 2. Action plan (mandatory, requires seg_insights)
    action_plan = build_action_plan(seg_insights)
    if action_plan is None or action_plan.empty:
        raise RuntimeError("CRITICAL: build_action_plan() failed to create action plan.")
    results['customer_segment_action_plan'] = action_plan

    # 3. Product association insights
    results['product_association_insights'] = build_product_association_insights()

    # 4. Feature importance insights
    results['classification_feature_insights'] = build_feature_insights()

    return results
