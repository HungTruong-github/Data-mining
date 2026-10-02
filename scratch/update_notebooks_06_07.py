import nbformat

def update_notebooks():
    # --- Update Notebook 06 ---
    nb6 = nbformat.read('notebooks/06_association_rules.ipynb', as_version=4)

    # Cell 1: imports
    nb6.cells[1].source = """import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import PROCESSED_DIR, TABLES_ASSOCIATION, FIGURES_ASSOCIATION
from src.association_rules import (
    load_basket_matrix, load_basket_from_long, map_stockcode_to_description,
    run_apriori, run_fpgrowth, generate_rules, validate_rules, build_algorithm_comparison,
    frozenset_to_names, compare_rule_sets, calculate_min_baskets, benchmark_algorithms,
    run_sensitivity_analysis
)
from src.visualization import (
    plot_top_products, plot_algorithm_comparison, plot_scatter_support_confidence_lift, plot_top_rules_by_lift
)

RULES_TBL_DIR = TABLES_ASSOCIATION / 'association_rules'
RULES_TBL_DIR.mkdir(parents=True, exist_ok=True)

%matplotlib inline"""

    # Cell 5: benchmark & execution
    nb6.cells[5].source = """min_support = 0.02
min_confidence = 0.5
min_baskets = calculate_min_baskets(min_support, stats['n_invoices'])

print(f"Tham số: min_support={min_support} (yêu cầu tối thiểu ceil({min_support} x {stats['n_invoices']:,}) = {min_baskets} hóa đơn), min_confidence={min_confidence}")

print("\\n--- Thực hiện benchmark 3 lần lặp Apriori & FP-Growth ---")
bench = benchmark_algorithms(basket, min_support=min_support, n_runs=3)
apriori_time = bench['apriori_median']
fpgrowth_time = bench['fpgrowth_median']
print(f"Apriori median: {apriori_time:.2f}s (IQR: {bench['apriori_iqr']:.2f}s, runs: {bench['apriori_times']})")
print(f"FP-Growth median: {fpgrowth_time:.2f}s (IQR: {bench['fpgrowth_iqr']:.2f}s, runs: {bench['fpgrowth_times']})")

apriori_itemsets, _ = run_apriori(basket, min_support=min_support)
fpgrowth_itemsets, _ = run_fpgrowth(basket, min_support=min_support)

plot_algorithm_comparison(apriori_itemsets, fpgrowth_itemsets, apriori_time, fpgrowth_time)"""

    # Cell 7: validation, equivalence audit, exception raising
    nb6.cells[7].source = """apriori_rules = generate_rules(apriori_itemsets, min_confidence=min_confidence)
fpgrowth_rules = generate_rules(fpgrowth_itemsets, min_confidence=min_confidence)

apriori_valid, apriori_checks = validate_rules(apriori_rules, min_support, min_confidence)
fpgrowth_valid, fpgrowth_checks = validate_rules(fpgrowth_rules, min_support, min_confidence)

print("Apriori Validations:")
for c in apriori_checks: print(f"  {c}")
    
print("\\nFP-Growth Validations:")
for c in fpgrowth_checks: print(f"  {c}")

apriori_lift = apriori_valid[apriori_valid['lift'] > 1].copy()
fpgrowth_lift = fpgrowth_valid[fpgrowth_valid['lift'] > 1].copy()
print(f"\\nRules hợp lệ (Lift > 1) -> Apriori: {len(apriori_lift)}, FP-Growth: {len(fpgrowth_lift)}")

# Xác thực tương đương nội dung chuẩn hóa bằng canonical StockCodes
print("\\n--- Xác thực tương đương nội dung (Canonical StockCode Matching) ---")
equiv_res = compare_rule_sets(apriori_lift, fpgrowth_lift, 'Apriori', 'FP-Growth')
print(f"  Tương đương (is_equivalent): {equiv_res['is_equivalent']}")
print(f"  Khớp {equiv_res['matched_count']}/{equiv_res['count_Apriori']} luật")
print(f"  Độ lệch max: support={equiv_res['max_support_diff']:.6e}, conf={equiv_res['max_confidence_diff']:.6e}, lift={equiv_res['max_lift_diff']:.6e}")

if not equiv_res['is_equivalent']:
    raise ValueError(f"CRITICAL: Apriori và FP-Growth không tương đương nội dung! Chi tiết: {equiv_res}")

pd.DataFrame([equiv_res]).to_csv(TABLES_ASSOCIATION / 'association_rules_equivalence_audit.csv', index=False)
print("  [OK] Đã xuất association_rules_equivalence_audit.csv")"""

    # Cell 8: markdown
    nb6.cells[8].source = """## 4. Lựa chọn thuật toán & Lưu kết quả
Lựa chọn thuật toán dựa trên kết quả benchmark thực nghiệm (thời gian chạy median qua 3 lần lặp, độ biến thiên IQR). Thuật toán có median runtime thấp hơn sẽ được chọn động làm thuật toán chính cho downstream pipeline."""

    # Cell 9: sensitivity & dynamic selection
    nb6.cells[9].source = """# Phân tích độ nhạy (Sensitivity Analysis)
print("\\n--- Phân tích độ nhạy tham số (Support & Confidence) ---")
sensitivity_df = run_sensitivity_analysis(basket)
sensitivity_df.to_csv(TABLES_ASSOCIATION / 'rule_sensitivity_analysis.csv', index=False)
print(f"  [OK] Đã lưu rule_sensitivity_analysis.csv ({len(sensitivity_df)} điểm lưới)")

def max_itemset_size(df): return int(df['itemsets'].apply(len).max()) if not df.empty else 0

apriori_info = {'algorithm': 'Apriori', 'min_support': min_support, 'min_confidence': min_confidence, 'runtime': apriori_time, 'n_itemsets': len(apriori_itemsets), 'n_rules': len(apriori_rules), 'n_valid_rules': len(apriori_lift), 'max_itemset_size': max_itemset_size(apriori_itemsets), 'max_lift': apriori_lift['lift'].max() if len(apriori_lift)>0 else 0}
fpgrowth_info = {'algorithm': 'FP-Growth', 'min_support': min_support, 'min_confidence': min_confidence, 'runtime': fpgrowth_time, 'n_itemsets': len(fpgrowth_itemsets), 'n_rules': len(fpgrowth_rules), 'n_valid_rules': len(fpgrowth_lift), 'max_itemset_size': max_itemset_size(fpgrowth_itemsets), 'max_lift': fpgrowth_lift['lift'].max() if len(fpgrowth_lift)>0 else 0}

algo_comp = build_algorithm_comparison(apriori_info, fpgrowth_info)
display(algo_comp)
algo_comp.to_csv(TABLES_ASSOCIATION / 'association_algorithm_comparison.csv', index=False)

def save_rules_csv(rules, algorithm, filepath):
    if rules.empty:
        pd.DataFrame().to_csv(filepath, index=False)
        return
    df = rules.copy()
    df['antecedents_str'] = df['antecedents'].apply(lambda x: frozenset_to_names(x, sc_to_desc))
    df['consequents_str'] = df['consequents'].apply(lambda x: frozenset_to_names(x, sc_to_desc))
    df['antecedents_codes'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    df['consequents_codes'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
    df['algorithm'] = algorithm
    cols = ['antecedents_str', 'consequents_str', 'antecedents_codes', 'consequents_codes', 'antecedent_length', 'consequent_length', 'support', 'confidence', 'lift', 'algorithm']
    if 'leverage' in df.columns: cols.append('leverage')
    if 'conviction' in df.columns: cols.append('conviction')
    df[cols].to_csv(filepath, index=False)

# Lựa chọn động thuật toán dựa trên thực nghiệm median runtime
if apriori_time <= fpgrowth_time:
    selected_rules = apriori_lift
    selected_algo = 'Apriori'
else:
    selected_rules = fpgrowth_lift
    selected_algo = 'FP-Growth'

print(f"Thuật toán được chọn thực nghiệm: {selected_algo} ({len(selected_rules)} rules, median runtime={min(apriori_time, fpgrowth_time):.2f}s)")

save_rules_csv(apriori_lift, 'Apriori', TABLES_ASSOCIATION / 'apriori_rules.csv')
save_rules_csv(apriori_lift, 'Apriori', RULES_TBL_DIR / 'apriori_rules.csv')
save_rules_csv(fpgrowth_lift, 'FP-Growth', TABLES_ASSOCIATION / 'fpgrowth_rules.csv')
save_rules_csv(fpgrowth_lift, 'FP-Growth', RULES_TBL_DIR / 'fpgrowth_rules.csv')
save_rules_csv(selected_rules, selected_algo, TABLES_ASSOCIATION / 'selected_association_rules.csv')
save_rules_csv(selected_rules, selected_algo, RULES_TBL_DIR / 'selected_association_rules.csv')"""

    # Cell 11 & 12: use selected_rules
    nb6.cells[11].source = """plot_scatter_support_confidence_lift(selected_rules)"""
    nb6.cells[12].source = """plot_top_rules_by_lift(selected_rules, sc_to_desc, frozenset_to_names)"""

    # Cell 14: use selected_rules
    nb6.cells[14].source = """top_rules = selected_rules.copy()
top_rules['quality'] = top_rules['lift'] * top_rules['confidence']
top_rules = top_rules.nlargest(min(10, len(top_rules)), 'quality')

insights = []
for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
    ant_names = frozenset_to_names(rule['antecedents'], sc_to_desc)
    con_names = frozenset_to_names(rule['consequents'], sc_to_desc)
    insights.append({
        'rule_id': i,
        'antecedents': ant_names,
        'consequents': con_names,
        'support': rule['support'],
        'confidence': rule['confidence'],
        'lift': rule['lift'],
        'evidence_quality': 'High' if rule['lift'] > 3.0 and rule['confidence'] > 0.6 else 'Moderate',
        'business_action': f"Gợi ý kèm '{con_names}' khi khách hàng thêm '{ant_names}' vào giỏ hàng (Lift={rule['lift']:.1f}x)"
    })

insights_df = pd.DataFrame(insights)
display(insights_df)
insights_df.to_csv(TABLES_ASSOCIATION / 'association_business_insights.csv', index=False)"""

    nbformat.write(nb6, 'notebooks/06_association_rules.ipynb')
    print("Updated notebooks/06_association_rules.ipynb")

    # --- Update Notebook 07 ---
    nb7 = nbformat.read('notebooks/07_model_comparison_and_insights.ipynb', as_version=4)

    # Cell 16: artifact check without swallowing exceptions
    nb7.cells[16].source = """artifact_checks = verify_model_artifacts()
for c in artifact_checks:
    print(c)
if any('[FAIL]' in c for c in artifact_checks):
    raise RuntimeError(f"Artifact verification failed: {artifact_checks}")
print("\\n[OK] All model artifacts verified.")"""

    # Cell 18: insights generation without swallowing exceptions
    nb7.cells[18].source = """insights = generate_all_insights()

# Save insights tables
for name, df_insight in insights.items():
    if isinstance(df_insight, pd.DataFrame) and not df_insight.empty:
        path = OUTPUTS_DIR / 'tables' / 'insights' / f'{name}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df_insight.to_csv(path, index=False)
        print(f"[OK] Saved {name}.csv ({len(df_insight)} rows)")
    elif isinstance(df_insight, dict):
        print(f"[INFO] {name}: {len(df_insight)} entries")"""

    # Cell 20: report & manifest without swallowing exceptions, and asserting validation_status == PASS
    nb7.cells[20].source = """# Generate report
report_path = generate_report(
    clustering_comp, classification_comp, assoc_comp,
    output_path=REPORTS_DIR / '07_model_comparison_and_insights.md'
)
print(f"[OK] Report saved to: {report_path}")

# Generate manifest
manifest = generate_manifest()
manifest_path = EVIDENCE_DIR / 'pipeline_manifest.json'
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)
print(f"[OK] Manifest saved to: {manifest_path}")

if manifest.get('validation_status') != 'PASS':
    missing = manifest.get('missing_files', [])
    raise RuntimeError(f"Step 07 FAILED: pipeline_manifest.json validation_status is 'FAIL'. Missing files: {missing}")"""

    # Cell 22: final validation assert
    nb7.cells[22].source = """manifest_data = json.loads((EVIDENCE_DIR / 'pipeline_manifest.json').read_text(encoding='utf-8'))
if manifest_data.get('validation_status') != 'PASS':
    raise RuntimeError(f"Step 07 FAILED: pipeline_manifest.json has validation_status='FAIL'!")

print("=" * 60)
print("  07. MODEL COMPARISON AND INSIGHTS — COMPLETE")
print("=" * 60)
print(f"\\nOutputs:")
print(f"  Tables: {TABLES_MC}")
print(f"  Figures: {FIGURES_MC}")
print(f"  Report: {REPORTS_DIR / '07_model_comparison_and_insights.md'}")
print(f"  Manifest: {EVIDENCE_DIR / 'pipeline_manifest.json'}")"""

    nbformat.write(nb7, 'notebooks/07_model_comparison_and_insights.ipynb')
    print("Updated notebooks/07_model_comparison_and_insights.ipynb")

if __name__ == '__main__':
    update_notebooks()
