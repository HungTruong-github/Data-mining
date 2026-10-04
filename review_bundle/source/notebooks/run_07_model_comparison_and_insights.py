"""
Runner for Step 07: Model Comparison and Insights.
Executes all evaluation, comparison, visualization, report generation, and manifest updates.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_comparison import (
    validate_inputs,
    verify_model_artifacts,
    build_clustering_comparison,
    build_classification_comparison,
    build_association_comparison,
    validate_selected_rules,
    export_all_step_07_outputs,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    EVIDENCE_DIR,
)
from src.insights import (
    build_customer_segment_insights,
    build_action_plan,
    build_product_association_insights,
    build_feature_insights,
)

TABLES_MC = TABLES_DIR / 'model_comparison'
TABLES_INS = TABLES_DIR / 'insights'
FIGURES_MC = FIGURES_DIR / 'model_comparison'
REPORTS = REPORTS_DIR
EVIDENCE = EVIDENCE_DIR

for d in [TABLES_MC, TABLES_INS, FIGURES_MC, REPORTS, EVIDENCE]:
    d.mkdir(parents=True, exist_ok=True)

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
    selected_clust = clust_comp[clust_comp['is_selected'].astype(bool)]
    if not selected_clust.empty:
        sc_row = selected_clust.iloc[0]
        print(f"  Selected: {sc_row['algorithm']} K={sc_row['n_clusters']}")
        print(f"  Silhouette={sc_row['silhouette_score']:.4f}  DB={sc_row['davies_bouldin_score']:.4f}")
    print(f"  Total configs evaluated: {len(clust_comp)}")

    # -- STEP 3: Classification Comparison --
    print("\n--- STEP 3: Classification Model Comparison ---")
    class_comp = build_classification_comparison()
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
    for _, row in seg_insights.iterrows():
        print(f"  Cluster {row['cluster_id']}: {row['business_segment_name']} "
              f"({row['customer_count']:,} customers, {row['customer_percentage']:.1f}%)")
        print(f"    R={row['recency_mean']:.0f} F={row['frequency_mean']:.0f} M={row['monetary_mean']:,.0f} "
              f"Repeat={row.get('repeat_purchase_rate', 'N/A')}")

    # Action plan
    action_plan = build_action_plan(seg_insights)
    print(f"  Action plan: {len(action_plan)} strategies")

    # -- STEP 6: Product Association Insights --
    print("\n--- STEP 6: Product Association Insights ---")
    prod_insights = build_product_association_insights()
    print(f"  Top {len(prod_insights)} association rules with business interpretation")
    for _, row in prod_insights.head(3).iterrows():
        print(f"    {row['antecedents']} -> {row['consequents']} (Lift={row['lift']:.1f}, {row['evidence_quality']})")

    # -- STEP 7: Feature Insights --
    print("\n--- STEP 7: Classification Feature Insights ---")
    feat_insights = build_feature_insights()
    for _, row in feat_insights.head(5).iterrows():
        print(f"  #{row['rank']}: {row['feature']} ({row['importance']:.4f}) [{row['feature_group']}]")

    # -- STEP 8, 9, 10, 11: Export All Synchronized Step 07 Outputs --
    print("\n--- Exporting All Synchronized Step 07 Outputs (Tables, Figures, Report, Summary, Manifest) ---")
    res = export_all_step_07_outputs(
        clust_comp=clust_comp,
        class_comp=class_comp,
        assoc_comp=assoc_comp,
        seg_insights=seg_insights,
        action_plan=action_plan,
        prod_insights=prod_insights,
        feat_insights=feat_insights,
        save_figures=True
    )
    print("  [OK] Exported all tables, 10 figures, report markdown, summary json, and manifest.")

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
if __name__ == '__main__':
    main()
