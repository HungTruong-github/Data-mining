"""
Figure generation for CRISP-DM Step 07 (model comparison & insights).

Split out of ``src.model_comparison`` so that comparison/validation/report logic
and plotting logic live in separate modules. ``generate_all_step_07_figures`` is
still re-exported from ``src.model_comparison`` for backward compatibility.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'figures'


def generate_all_step_07_figures(clust_comp, class_comp, assoc_comp, seg_insights, prod_insights, feat_insights, output_dir=None):
    """
    Generate and save all 10 standard CRISP-DM Step 07 figures synchronously.
    Shared by runner script and Jupyter notebook.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / 'model_comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        fig.savefig(output_dir / name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 1. clustering_model_comparison.png
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2b5c8f' if r else '#a0c4ff' for r in clust_comp.get('is_selected', [False]*len(clust_comp))]
    labels = [f"{r['algorithm']}\n(K={r['n_clusters']})" for _, r in clust_comp.iterrows()]
    scores = clust_comp['silhouette_score'].values
    bars = ax.bar(range(len(clust_comp)), scores, color=colors, width=0.5)
    ax.set_xticks(range(len(clust_comp)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Algorithm Comparison (Silhouette Score)', fontsize=12, fontweight='bold')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(scores) * 1.2 if len(scores) else 1)
    _save(fig, 'clustering_model_comparison.png')

    # 2. classification_model_comparison.png
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(class_comp))
    width = 0.35
    cv_f1s = class_comp.get('cv_f1_mean', class_comp.get('cv_f1', pd.Series([0]*len(class_comp)))).values
    test_f1s = class_comp.get('test_f1', pd.Series([0]*len(class_comp))).values
    ax.bar(x - width/2, cv_f1s, width, label='CV F1-Score', color='#3a86ff')
    ax.bar(x + width/2, test_f1s, width, label='Test F1-Score', color='#8338ec')
    ax.set_xticks(x)
    ax.set_xticklabels(class_comp['model'].values, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('F1-Score')
    ax.set_title('Classification Model Comparison (CV vs Test F1)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    for i in range(len(class_comp)):
        ax.text(x[i] - width/2, cv_f1s[i] + 0.02, f"{cv_f1s[i]:.3f}", ha='center', fontsize=8)
        ax.text(x[i] + width/2, test_f1s[i] + 0.02, f"{test_f1s[i]:.3f}", ha='center', fontsize=8)
    _save(fig, 'classification_model_comparison.png')

    # 3. clustering_quality_metrics.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kmeans_df = clust_comp[clust_comp['algorithm'].isin(['K-Means', 'KMeans'])].sort_values('n_clusters')
    if not kmeans_df.empty:
        ax1.plot(kmeans_df['n_clusters'], kmeans_df['silhouette_score'], 'o-', color='#2b5c8f', linewidth=2)
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('K-Means Silhouette Score vs K', fontweight='bold')
        ax2.plot(kmeans_df['n_clusters'], kmeans_df['davies_bouldin_score'], 's-', color='#e63946', linewidth=2)
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Davies-Bouldin Index (Lower is Better)')
        ax2.set_title('K-Means Davies-Bouldin vs K', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'clustering_quality_metrics.png')

    # 4. classification_cv_vs_test.png
    fig, ax = plt.subplots(figsize=(8, 4))
    non_dummy = class_comp[class_comp['model'] != 'DummyClassifier']
    if not non_dummy.empty:
        models = non_dummy['model'].values
        cv = non_dummy.get('cv_f1_mean', non_dummy.get('cv_f1', pd.Series([0]*len(non_dummy)))).values
        test = non_dummy.get('test_f1', pd.Series([0]*len(non_dummy))).values
        x = np.arange(len(models))
        ax.plot(x, cv, 'o--', label='CV F1', color='#3a86ff', markersize=8)
        ax.plot(x, test, 's-', label='Test F1', color='#ff006e', markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title('Generalization Check: CV F1 vs Test F1 (Non-Baseline Models)', fontweight='bold')
        ax.legend()
        ax.set_ylim(0.4, 0.9)
    _save(fig, 'classification_cv_vs_test.png')

    # 5. cluster_size_distribution.png
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2b5c8f', '#d94e34']
    labels = [f"Cluster {c}\n{n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
    bars = ax.bar(labels, seg_insights['customer_count'], color=colors[:len(seg_insights)], width=0.4)
    ax.set_ylabel('Customer Count')
    ax.set_title('Customer Distribution by Segment', fontweight='bold')
    for bar, pct in zip(bars, seg_insights['customer_percentage']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')
    _save(fig, 'cluster_size_distribution.png')

    # 6. segment_rfm_profile.png
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [('recency_mean', 'Recency (Days, Lower = Better)', '#2a9d8f'),
               ('frequency_mean', 'Frequency (Orders, Higher = Better)', '#e76f51'),
               ('monetary_mean', 'Monetary (Spend £, Higher = Better)', '#2b5c8f')]
    x_labels = [f"C{c}: {n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(x_labels, seg_insights[col], color=color, width=0.4)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for bar in bars:
            val = bar.get_height()
            fmt = f"£{val:,.0f}" if 'Spend' in title else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.02, fmt, ha='center', fontsize=9)
    fig.tight_layout()
    _save(fig, 'segment_rfm_profile.png')

    # 7. segment_repeat_purchase_rate.png
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'repeat_purchase_rate' in seg_insights.columns:
        rates = seg_insights['repeat_purchase_rate'].values
        labels = [f"Cluster {c}\n({n})" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
        bars = ax.bar(labels, rates, color=['#2b5c8f', '#d94e34'], width=0.4)
        ax.set_ylabel('Repeat Purchase Rate (90 Days)')
        ax.set_title('Repeat Purchase Rate by Segment', fontweight='bold')
        ax.set_ylim(0, 1.15)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", ha='center', fontweight='bold', fontsize=11)
    _save(fig, 'segment_repeat_purchase_rate.png')

    # 8. top_association_rules.png
    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = prod_insights.head(10).iloc[::-1]
    labels = [f"{r['antecedents']} -> {r['consequents']}" for _, r in top10.iterrows()]
    bars = ax.barh(range(len(top10)), top10['lift'], color='#4361ee', height=0.6)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Lift Value')
    ax.set_title('Top 10 Product Association Rules by Lift', fontweight='bold')
    for bar, lift in zip(bars, top10['lift']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f"{lift:.1f}x", va='center', fontsize=8)
    _save(fig, 'top_association_rules.png')

    # 9. feature_importance_top20.png
    fig, ax = plt.subplots(figsize=(9, 5))
    top15 = feat_insights.head(15).iloc[::-1]
    colors = {'RFM': '#e76f51', 'Behavioral': '#2a9d8f', 'Geographic': '#2b5c8f', 'Other': '#8d99ae'}
    bar_colors = [colors.get(g, '#8d99ae') for g in top15['feature_group']]
    bars = ax.barh(range(len(top15)), top15['importance'], color=bar_colors, height=0.6)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['feature'], fontsize=9)
    ax.set_xlabel('Feature Importance (Gini / Mean Decrease Impurity)')
    ax.set_title('Top 15 Predictive Features for 90-Day Repeat Purchase', fontweight='bold')
    for bar, imp in zip(bars, top15['importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{imp:.4f}", va='center', fontsize=8)
    _save(fig, 'feature_importance_top20.png')

    # 10. insight_summary_dashboard.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].pie(seg_insights['customer_count'], labels=[f"C{c}: {n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])],
                   autopct='%1.1f%%', colors=['#2b5c8f', '#d94e34'], startangle=90)
    axes[0, 0].set_title('A. Customer Segment Distribution', fontweight='bold', fontsize=10)

    mc = class_comp.sort_values('test_f1', ascending=True)
    axes[0, 1].barh(mc['model'], mc['test_f1'], color='#3a86ff', height=0.5)
    axes[0, 1].set_xlabel('Test F1-Score')
    axes[0, 1].set_title('B. Classification Models Performance', fontweight='bold', fontsize=10)
    for i, v in enumerate(mc['test_f1']):
        axes[0, 1].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)

    axes[1, 0].bar([f"C{c}" for c in seg_insights['cluster_id']], seg_insights['repeat_purchase_rate'], color=['#2b5c8f', '#d94e34'], width=0.4)
    axes[1, 0].set_ylabel('Repeat Purchase Rate')
    axes[1, 0].set_title('C. Segment Repeat Purchase Behavior', fontweight='bold', fontsize=10)
    for i, v in enumerate(seg_insights['repeat_purchase_rate']):
        axes[1, 0].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9, fontweight='bold')

    axes[1, 1].axis('off')
    seg_by_spend = seg_insights.sort_values('monetary_mean', ascending=False)
    best_c = seg_by_spend.iloc[0]
    lost_c = seg_by_spend.iloc[-1]
    best_name = best_c['business_segment_name']
    best_pct = best_c['customer_percentage']
    best_rep = best_c.get('repeat_purchase_rate', 0)
    best_spend = best_c['monetary_mean']
    lost_name = lost_c['business_segment_name']
    lost_pct = lost_c['customer_percentage']
    lost_rep = lost_c.get('repeat_purchase_rate', 0)
    lost_rec = lost_c['recency_mean']

    feat_text = ""
    if feat_insights is not None and not feat_insights.empty:
        top1 = feat_insights.iloc[0]
        top2 = feat_insights.iloc[1] if len(feat_insights) > 1 else None
        feat_text = f"• Top Predictors: {top1['feature']} ({top1['importance']:.1%})"
        if top2 is not None:
            feat_text += f",\n  {top2['feature']} ({top2['importance']:.1%})."
    else:
        feat_text = "• Predictor Importance: Recency and\n  Monetary are key purchase predictors."

    rules_text = ""
    if assoc_comp is not None and not assoc_comp.empty:
        sel_assoc = assoc_comp[assoc_comp['is_selected'].astype(bool)]
        row_assoc = sel_assoc.iloc[0] if not sel_assoc.empty else assoc_comp.iloc[0]
        n_rules = row_assoc['valid_rule_count']
        max_lift = row_assoc.get('max_rule_lift', row_assoc.get('max_lift', 0))
        algo_name = row_assoc['algorithm']
        rules_text = f"• Co-Purchase Rules ({algo_name}): {n_rules} rules (Lift > 1.0).\n  Top rule lift = {max_lift:.1f}x."
    else:
        rules_text = "• Co-Purchase Rules: Validated Lift > 1.0."

    summary_text = (
        "D. Key Strategic Findings\n"
        "----------------------------------------\n"
        f"• {best_name} ({best_pct:.1f}%): {best_rep:.1%} repeat rate,\n"
        f"  Avg spend £{best_spend:,.0f}. Action: VIP retention.\n\n"
        f"• {lost_name} ({lost_pct:.1f}%): {lost_rep:.1%} repeat rate,\n"
        f"  Recency {lost_rec:.0f} days. Action: Win-back campaign.\n\n"
        f"{feat_text}\n\n"
        f"{rules_text}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    fig.suptitle('CRISP-DM Step 07: Data Mining Insights Dashboard', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'insight_summary_dashboard.png')
