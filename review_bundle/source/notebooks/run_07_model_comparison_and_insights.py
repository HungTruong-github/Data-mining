"""
Runner for Step 07: Model Comparison and Insights.
Executes all evaluation, comparison, visualization, report generation, and manifest updates.
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_comparison import (
    load_input, validate_inputs, build_clustering_comparison,
    build_classification_comparison, build_association_comparison,
    validate_selected_rules, verify_model_artifacts, file_sha256,
    generate_report, generate_summary_json, generate_manifest,
    INPUT_FILES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, EVIDENCE_DIR,
    MODELS_CLUSTERING_DIR, MODELS_CLASSIFICATION_DIR, DATA_PROCESSED_DIR
)
from src.insights import (
    build_customer_segment_insights, build_action_plan,
    build_product_association_insights, build_feature_insights
)

TABLES_MC = TABLES_DIR / 'model_comparison'
TABLES_INS = TABLES_DIR / 'insights'
FIGURES_MC = FIGURES_DIR / 'model_comparison'
REPORTS = REPORTS_DIR
EVIDENCE = EVIDENCE_DIR

for d in [TABLES_MC, TABLES_INS, FIGURES_MC, REPORTS, EVIDENCE]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def main():
    total_start = time.time()
    print("=" * 60)
    print("  07. MODEL COMPARISON AND INSIGHTS")
    print("=" * 60)

    # -- STEP 1: Validate Inputs --
    print("\n--- STEP 1: Validate Inputs ---")
    val_checks = validate_inputs()
    for c in val_checks:
        print(f"  {c}")

    # Verify model artifacts loadable
    art_checks = verify_model_artifacts()
    for c in art_checks:
        print(f"  {c}")

    # -- STEP 2: Clustering Comparison --
    print("\n--- STEP 2: Clustering Model Comparison ---")
    clust_comp = build_clustering_comparison()
    clust_comp.to_csv(TABLES_MC / 'clustering_model_comparison.csv', index=False)
    selected_clust = clust_comp[clust_comp['is_selected'].astype(bool)]
    if not selected_clust.empty:
        sc_row = selected_clust.iloc[0]
        print(f"  Selected: {sc_row['algorithm']} K={sc_row['n_clusters']}")
        print(f"  Silhouette={sc_row['silhouette_score']:.4f}  DB={sc_row['davies_bouldin_score']:.4f}")
    print(f"  Total configs evaluated: {len(clust_comp)}")

    # -- STEP 3: Classification Comparison --
    print("\n--- STEP 3: Classification Model Comparison ---")
    class_comp = build_classification_comparison()
    class_comp.to_csv(TABLES_MC / 'classification_model_comparison.csv', index=False)
    selected_class = class_comp[class_comp['is_selected'].astype(bool)]
    if not selected_class.empty:
        mc = selected_class.iloc[0]
        cv_f1 = mc.get('cv_f1_mean', mc.get('cv_f1', 0))
        print(f"  Selected: {mc['model']} (CV F1={cv_f1:.4f})")
        print(f"  Test: F1={mc.get('test_f1', 0):.4f}  AUC={mc.get('test_roc_auc', 0):.4f}")
    print(f"  Total models: {len(class_comp)}")

    # -- STEP 4: Association Rules Comparison --
    print("\n--- STEP 4: Association Rules Comparison ---")
    assoc_comp = build_association_comparison()
    assoc_comp.to_csv(TABLES_MC / 'association_rules_comparison.csv', index=False)
    selected_assoc = assoc_comp[assoc_comp['is_selected'].astype(bool)]
    if not selected_assoc.empty:
        sa = selected_assoc.iloc[0]
        print(f"  Selected: {sa['algorithm']} ({sa['valid_rule_count']} rules, {sa['runtime_seconds']:.2f}s)")

    # Validate rules quality
    rule_checks, rules_df = validate_selected_rules()
    for c in rule_checks:
        print(f"  {c}")

    # -- STEP 5: Customer Segment Insights --
    print("\n--- STEP 5: Customer Segment Insights ---")
    seg_insights = build_customer_segment_insights()
    seg_insights.to_csv(TABLES_INS / 'customer_segment_insights.csv', index=False)
    for _, row in seg_insights.iterrows():
        print(f"  Cluster {row['cluster_id']}: {row['business_segment_name']} "
              f"({row['customer_count']:,} customers, {row['customer_percentage']:.1f}%)")
        print(f"    R={row['recency_mean']:.0f} F={row['frequency_mean']:.0f} M={row['monetary_mean']:,.0f} "
              f"Repeat={row.get('repeat_purchase_rate', 'N/A')}")

    # Action plan
    action_plan = build_action_plan(seg_insights)
    action_plan.to_csv(TABLES_INS / 'customer_segment_action_plan.csv', index=False)
    print(f"  Action plan: {len(action_plan)} strategies")

    # -- STEP 6: Product Association Insights --
    print("\n--- STEP 6: Product Association Insights ---")
    prod_insights = build_product_association_insights()
    prod_insights.to_csv(TABLES_INS / 'product_association_insights.csv', index=False)
    print(f"  Top {len(prod_insights)} association rules with business interpretation")
    for _, row in prod_insights.head(3).iterrows():
        print(f"    {row['antecedents']} -> {row['consequents']} (Lift={row['lift']:.1f}, {row['evidence_quality']})")

    # -- STEP 7: Feature Insights --
    print("\n--- STEP 7: Classification Feature Insights ---")
    feat_insights = build_feature_insights()
    feat_insights.to_csv(TABLES_INS / 'classification_feature_insights.csv', index=False)
    for _, row in feat_insights.head(5).iterrows():
        print(f"  #{row['rank']}: {row['feature']} ({row['importance']:.4f}) [{row['feature_group']}]")

    # -- STEP 8: Visualizations --
    print("\n--- STEP 8: Visualizations ---")
    _plot_clustering_comparison(clust_comp)
    _plot_classification_comparison(class_comp)
    _plot_clustering_quality(clust_comp)
    _plot_cv_vs_test(class_comp)
    _plot_cluster_size(seg_insights)
    _plot_segment_rfm(seg_insights)
    _plot_segment_repeat_rate(seg_insights)
    _plot_top_rules(prod_insights)
    _plot_feature_importance(feat_insights)
    _plot_insight_summary(seg_insights, class_comp, assoc_comp, feat=feat_insights)

    # -- STEP 9: Generate Report --
    print("\n--- STEP 9: Generate Report ---")
    _generate_report(clust_comp, class_comp, assoc_comp, seg_insights,
                     action_plan, prod_insights, feat_insights)

    # -- STEP 10: Generate Summary JSON --
    print("\n--- STEP 10: Generate Summary JSON ---")
    _generate_summary_json(clust_comp, class_comp, assoc_comp, seg_insights)

    # -- STEP 11: Pipeline Manifest --
    print("\n--- STEP 11: Pipeline Manifest ---")
    _generate_manifest()

    # -- STEP 12: Final Validation --
    print("\n--- STEP 12: Final Validation ---")
    required_outputs = [
        TABLES_MC / 'clustering_model_comparison.csv',
        TABLES_MC / 'classification_model_comparison.csv',
        TABLES_MC / 'association_rules_comparison.csv',
        TABLES_INS / 'customer_segment_insights.csv',
        TABLES_INS / 'customer_segment_action_plan.csv',
        TABLES_INS / 'product_association_insights.csv',
        TABLES_INS / 'classification_feature_insights.csv',
        FIGURES_MC / 'clustering_model_comparison.png',
        FIGURES_MC / 'classification_model_comparison.png',
        FIGURES_MC / 'clustering_quality_metrics.png',
        FIGURES_MC / 'classification_cv_vs_test.png',
        FIGURES_MC / 'cluster_size_distribution.png',
        FIGURES_MC / 'segment_rfm_profile.png',
        FIGURES_MC / 'segment_repeat_purchase_rate.png',
        FIGURES_MC / 'top_association_rules.png',
        FIGURES_MC / 'feature_importance_top20.png',
        FIGURES_MC / 'insight_summary_dashboard.png',
        REPORTS / '07_model_comparison_and_insights.md',
        REPORTS / '07_model_comparison_and_insights_summary.json',
        EVIDENCE / 'pipeline_manifest.json',
    ]
    all_exist = True
    for f in required_outputs:
        if f.exists() and f.stat().st_size > 0:
            print(f"  [OK] {f.name}")
        else:
            print(f"  [FAIL] {f.name}")
            all_exist = False

    assert all_exist, "Some required outputs missing!"

    # Strict check: pipeline_manifest must have validation_status == PASS
    manifest_data = json.loads((EVIDENCE / 'pipeline_manifest.json').read_text(encoding='utf-8'))
    if manifest_data.get('validation_status') != 'PASS':
        raise RuntimeError(f"Step 07 FAILED: pipeline_manifest.json has validation_status='FAIL'. Missing: {manifest_data.get('missing_files')}")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  STEP 07 COMPLETE in {elapsed:.2f}s")
    print(f"  Tables: {TABLES_MC}, {TABLES_INS}")
    print(f"  Figures: {FIGURES_MC}")
    print(f"  Reports: {REPORTS}")
    print(f"{'=' * 60}")

# -- PLOTTING HELPERS --
def _save(fig, name):
    path = FIGURES_MC / name
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {name}")

def _plot_clustering_comparison(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2b5c8f' if r else '#a0c4ff' for r in df.get('is_selected', [False]*len(df))]
    labels = [f"{r['algorithm']}\n(K={r['n_clusters']})" for _, r in df.iterrows()]
    scores = df['silhouette_score'].values
    bars = ax.bar(range(len(df)), scores, color=colors, width=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Algorithm Comparison (Silhouette Score)', fontsize=12, fontweight='bold')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(scores) * 1.2 if len(scores) else 1)
    _save(fig, 'clustering_model_comparison.png')

def _plot_classification_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    width = 0.35
    cv_f1s = df.get('cv_f1_mean', df.get('cv_f1', pd.Series([0]*len(df)))).values
    test_f1s = df.get('test_f1', pd.Series([0]*len(df))).values
    ax.bar(x - width/2, cv_f1s, width, label='CV F1-Score', color='#3a86ff')
    ax.bar(x + width/2, test_f1s, width, label='Test F1-Score', color='#8338ec')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'].values, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('F1-Score')
    ax.set_title('Classification Model Comparison (CV vs Test F1)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    for i in range(len(df)):
        ax.text(x[i] - width/2, cv_f1s[i] + 0.02, f"{cv_f1s[i]:.3f}", ha='center', fontsize=8)
        ax.text(x[i] + width/2, test_f1s[i] + 0.02, f"{test_f1s[i]:.3f}", ha='center', fontsize=8)
    _save(fig, 'classification_model_comparison.png')

def _plot_clustering_quality(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kmeans_df = df[df['algorithm'].isin(['K-Means', 'KMeans'])].sort_values('n_clusters')
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

def _plot_cv_vs_test(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    non_dummy = df[df['model'] != 'DummyClassifier']
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

def _plot_cluster_size(seg):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2b5c8f', '#d94e34']
    labels = [f"Cluster {c}\n{n}" for c, n in zip(seg['cluster_id'], seg['business_segment_name'])]
    bars = ax.bar(labels, seg['customer_count'], color=colors[:len(seg)], width=0.4)
    ax.set_ylabel('Customer Count')
    ax.set_title('Customer Distribution by Segment', fontweight='bold')
    for bar, pct in zip(bars, seg['customer_percentage']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')
    _save(fig, 'cluster_size_distribution.png')

def _plot_segment_rfm(seg):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [('recency_mean', 'Recency (Days, Lower = Better)', '#2a9d8f'),
               ('frequency_mean', 'Frequency (Orders, Higher = Better)', '#e76f51'),
               ('monetary_mean', 'Monetary (Spend £, Higher = Better)', '#2b5c8f')]
    x_labels = [f"C{c}: {n}" for c, n in zip(seg['cluster_id'], seg['business_segment_name'])]
    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(x_labels, seg[col], color=color, width=0.4)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for bar in bars:
            val = bar.get_height()
            fmt = f"£{val:,.0f}" if 'Spend' in title else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.02, fmt, ha='center', fontsize=9)
    fig.tight_layout()
    _save(fig, 'segment_rfm_profile.png')

def _plot_segment_repeat_rate(seg):
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'repeat_purchase_rate' in seg.columns:
        rates = seg['repeat_purchase_rate'].values
        labels = [f"Cluster {c}\n({n})" for c, n in zip(seg['cluster_id'], seg['business_segment_name'])]
        bars = ax.bar(labels, rates, color=['#2b5c8f', '#d94e34'], width=0.4)
        ax.set_ylabel('Repeat Purchase Rate (90 Days)')
        ax.set_title('Repeat Purchase Rate by Segment', fontweight='bold')
        ax.set_ylim(0, 1.15)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", ha='center', fontweight='bold', fontsize=11)
    _save(fig, 'segment_repeat_purchase_rate.png')

def _plot_top_rules(prod):
    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = prod.head(10).iloc[::-1]
    labels = [f"{r['antecedents']} -> {r['consequents']}" for _, r in top10.iterrows()]
    bars = ax.barh(range(len(top10)), top10['lift'], color='#4361ee', height=0.6)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Lift Value')
    ax.set_title('Top 10 Product Association Rules by Lift', fontweight='bold')
    for bar, lift in zip(bars, top10['lift']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f"{lift:.1f}x", va='center', fontsize=8)
    _save(fig, 'top_association_rules.png')

def _plot_feature_importance(feat):
    fig, ax = plt.subplots(figsize=(9, 5))
    top15 = feat.head(15).iloc[::-1]
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

def _plot_insight_summary(seg, class_comp, assoc_comp, feat=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Panel 1: Cluster sizes
    axes[0, 0].pie(seg['customer_count'], labels=[f"C{c}: {n}" for c, n in zip(seg['cluster_id'], seg['business_segment_name'])],
                   autopct='%1.1f%%', colors=['#2b5c8f', '#d94e34'], startangle=90)
    axes[0, 0].set_title('A. Customer Segment Distribution', fontweight='bold', fontsize=10)

    # Panel 2: Model comparison
    mc = class_comp.sort_values('test_f1', ascending=True)
    axes[0, 1].barh(mc['model'], mc['test_f1'], color='#3a86ff', height=0.5)
    axes[0, 1].set_xlabel('Test F1-Score')
    axes[0, 1].set_title('B. Classification Models Performance', fontweight='bold', fontsize=10)
    for i, v in enumerate(mc['test_f1']):
        axes[0, 1].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)

    # Panel 3: Repeat rates
    axes[1, 0].bar([f"C{c}" for c in seg['cluster_id']], seg['repeat_purchase_rate'], color=['#2b5c8f', '#d94e34'], width=0.4)
    axes[1, 0].set_ylabel('Repeat Purchase Rate')
    axes[1, 0].set_title('C. Segment Repeat Purchase Behavior', fontweight='bold', fontsize=10)
    for i, v in enumerate(seg['repeat_purchase_rate']):
        axes[1, 0].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9, fontweight='bold')

    # Panel 4: Association rules & Key Strategic Findings
    axes[1, 1].axis('off')

    # Dynamically extract values from artifacts
    seg_by_spend = seg.sort_values('monetary_mean', ascending=False)
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
    if feat is not None and not feat.empty:
        top1 = feat.iloc[0]
        top2 = feat.iloc[1] if len(feat) > 1 else None
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

def _generate_report(clust_comp, class_comp, assoc_comp, seg, actions, prod, feat):
    generate_report(clust_comp, class_comp, assoc_comp, seg=seg, actions=actions,
                    prod=prod, feat=feat, output_path=REPORTS / '07_model_comparison_and_insights.md')
    print("  [OK] 07_model_comparison_and_insights.md")


def _generate_summary_json(clust_comp, class_comp, assoc_comp, seg):
    generate_summary_json(clust_comp, class_comp, assoc_comp, seg=seg,
                          output_path=REPORTS / '07_model_comparison_and_insights_summary.json')
    print("  [OK] 07_model_comparison_and_insights_summary.json")


def _generate_manifest():
    generate_manifest(output_path=EVIDENCE / 'pipeline_manifest.json')
    print("  [OK] pipeline_manifest.json")


if __name__ == '__main__':
    main()
