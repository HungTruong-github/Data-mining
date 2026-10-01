"""
Step 07 — Business Insights: generate customer segment profiles, action plans,
product association insights, and feature interpretation from real model outputs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.model_comparison import load_input


# ──────────────────────────────────────────────
# CUSTOMER SEGMENT INSIGHTS
# ──────────────────────────────────────────────
def build_customer_segment_insights():
    """
    Merge clusters + RFM + repeat purchase data to create segment profiles.
    All numbers come from actual data — nothing is hard-coded.
    """
    clusters_df = load_input('customer_clusters')
    rfm_df = load_input('rfm_features')
    repeat_df = load_input('repeat_features')

    # Merge
    merged = clusters_df[['CustomerID', 'Cluster', 'BusinessSegment']].merge(
        rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary',
                 'TotalItems', 'UniqueProducts', 'ActiveDays',
                 'AverageOrderValue', 'CustomerLifetimeDays']],
        on='CustomerID', how='left'
    )

    # Merge repeat purchase label
    repeat_sub = repeat_df[['CustomerID', 'repeat_purchase_90d']].copy()
    merged = merged.merge(repeat_sub, on='CustomerID', how='left')

    total_customers = len(merged)

    insights = []
    for cluster_id in sorted(merged['Cluster'].unique()):
        seg = merged[merged['Cluster'] == cluster_id]
        n = len(seg)
        pct = n / total_customers * 100

        # RFM stats
        rec_mean = seg['Recency'].mean()
        rec_med = seg['Recency'].median()
        freq_mean = seg['Frequency'].mean()
        freq_med = seg['Frequency'].median()
        mon_mean = seg['Monetary'].mean()
        mon_med = seg['Monetary'].median()
        aov = seg['AverageOrderValue'].mean() if 'AverageOrderValue' in seg.columns else np.nan

        # Repeat purchase rate
        if 'repeat_purchase_90d' in seg.columns:
            repeat_rate = seg['repeat_purchase_90d'].mean()
        else:
            repeat_rate = np.nan

        # Business segment name from data
        biz_name = seg['BusinessSegment'].mode().iloc[0] if 'BusinessSegment' in seg.columns and not seg['BusinessSegment'].mode().empty else f'Cluster {cluster_id}'

        # Data-driven interpretation
        interp = _interpret_segment(rec_mean, freq_mean, mon_mean, repeat_rate, n, total_customers)

        insights.append({
            'cluster_id': cluster_id,
            'business_segment_name': biz_name,
            'customer_count': n,
            'customer_percentage': round(pct, 2),
            'recency_mean': round(rec_mean, 1),
            'recency_median': round(rec_med, 1),
            'frequency_mean': round(freq_mean, 1),
            'frequency_median': round(freq_med, 1),
            'monetary_mean': round(mon_mean, 2),
            'monetary_median': round(mon_med, 2),
            'average_order_value': round(aov, 2) if not np.isnan(aov) else np.nan,
            'repeat_purchase_rate': round(repeat_rate, 4) if not np.isnan(repeat_rate) else np.nan,
            'business_interpretation': interp,
            'evidence_source': 'data/processed/customer_clusters.csv + rfm_customer_features.csv + repeat_purchase_features.csv',
        })

    return pd.DataFrame(insights)


def _interpret_segment(rec, freq, mon, repeat_rate, n, total):
    """Generate data-driven interpretation based on actual metrics."""
    parts = []
    if rec < 60:
        parts.append("Recently active customers")
    elif rec > 200:
        parts.append("Inactive/dormant customers")
    else:
        parts.append("Moderately active customers")

    if freq >= 5:
        parts.append(f"with high purchase frequency (avg {freq:.0f} orders)")
    elif freq <= 2:
        parts.append(f"with low purchase frequency (avg {freq:.0f} orders)")

    if mon > 1000:
        parts.append(f"and high monetary value (avg {mon:,.0f})")
    elif mon < 300:
        parts.append(f"and low monetary value (avg {mon:,.0f})")

    if not np.isnan(repeat_rate):
        parts.append(f"Repeat purchase rate: {repeat_rate:.1%}")

    return '. '.join(parts) + '.'


# ──────────────────────────────────────────────
# ACTION PLAN
# ──────────────────────────────────────────────
def build_action_plan(segment_insights_df):
    """Generate action plan based on segment characteristics."""
    actions = []
    for _, row in segment_insights_df.iterrows():
        rec = row['recency_mean']
        freq = row['frequency_mean']
        mon = row['monetary_mean']
        repeat = row.get('repeat_purchase_rate', np.nan)
        cluster = row['cluster_id']
        name = row['business_segment_name']

        action_list = []
        if rec > 200:
            action_list.append(('win-back', f"Send reactivation campaign — {rec:.0f} days since last purchase"))
        if rec < 60 and freq >= 3:
            action_list.append(('retention', f"Loyalty program — active customers with {freq:.0f} avg orders"))
        if mon > 1000:
            action_list.append(('upsell', f"Premium product recommendations — avg spend {mon:,.0f}"))
        if not np.isnan(repeat) and repeat < 0.4:
            action_list.append(('reactivation', f"Targeted discount — only {repeat:.0%} repeat rate"))
        if not np.isnan(repeat) and repeat > 0.6:
            action_list.append(('cross-sell', f"Cross-sell bundles — {repeat:.0%} already repeat"))
        if freq <= 2 and rec < 100:
            action_list.append(('onboarding', f"Onboarding email series — new/infrequent buyers"))

        if not action_list:
            action_list.append(('monitoring', "Continue monitoring — no immediate action required"))

        for strategy, detail in action_list:
            actions.append({
                'cluster_id': cluster,
                'business_segment_name': name,
                'strategy': strategy,
                'recommended_action': detail,
                'customer_count': row['customer_count'],
                'customer_percentage': row['customer_percentage'],
                'evidence_metric': f"Recency={rec:.0f}, Freq={freq:.0f}, Monetary={mon:,.0f}",
            })

    return pd.DataFrame(actions)


# ──────────────────────────────────────────────
# PRODUCT ASSOCIATION INSIGHTS
# ──────────────────────────────────────────────
def build_product_association_insights():
    """Generate business interpretations for top association rules."""
    rules = load_input('selected_rules')

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
        if sup >= 0.05 and conf >= 0.7 and lift_val >= 3:
            quality = 'Strong'
        elif sup >= 0.03 and conf >= 0.5 and lift_val >= 2:
            quality = 'Moderate'
        else:
            quality = 'Weak'

        # Limitations
        limitations = []
        if sup < 0.03:
            limitations.append(f"Low support ({sup:.3f}) — affects only a small fraction of transactions")
        if conf < 0.5:
            limitations.append(f"Low confidence ({conf:.2f}) — association is not strong")
        limitations.append("Association does not imply causation")

        interp = (
            f"Customers who buy {ant} also tend to buy {con} "
            f"({lift_val:.1f}x more likely than random). "
            f"This pattern appears in {sup:.1%} of all transactions."
        )

        action = f"Consider cross-selling {con} to customers buying {ant}."
        if lift_val > 10:
            action += " Strong co-purchase pattern — display as bundle."
        elif lift_val > 3:
            action += " Moderate co-purchase — recommend in checkout."

        insights.append({
            'antecedents': ant,
            'consequents': con,
            'support': round(sup, 4),
            'confidence': round(conf, 4),
            'lift': round(lift_val, 2),
            'conviction': round(conv, 4) if not np.isnan(conv) else np.nan,
            'evidence_quality': quality,
            'business_interpretation': interp,
            'recommended_action': action,
            'limitation': '; '.join(limitations),
        })

    return pd.DataFrame(insights)


# ──────────────────────────────────────────────
# FEATURE INSIGHTS
# ──────────────────────────────────────────────
def build_feature_insights():
    """Interpret feature importance from classification model."""
    fi = load_input('feature_importance')

    # Load metadata for model name
    try:
        meta = load_input('class_metadata')
        model_name = meta.get('selected_model', 'Unknown')
    except:
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
            strength = "Top-3 most important feature"
        elif rank <= 10:
            strength = "Moderately important feature"
        else:
            strength = "Low importance feature"

        interp = (
            f"{strength} (importance={imp:.4f}). "
            f"This feature has a strong association with repeat purchase prediction "
            f"but does NOT imply causation. Result is model-dependent ({model_name})."
        )
        interpretations.append(interp)

    fi['interpretation'] = interpretations
    fi['limitation'] = 'Importance is model-dependent; does not imply causal relationship'

    return fi
