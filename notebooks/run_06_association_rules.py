"""
Runner script for 06_association_rules.
Run from project root: python notebooks/run_06_association_rules.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import PROCESSED_DIR, FIGURES_ASSOCIATION, TABLES_ASSOCIATION
from src.association_rules import (
    load_basket_matrix, load_basket_from_long, map_stockcode_to_description,
    run_apriori, run_fpgrowth, generate_rules, validate_rules, build_algorithm_comparison, frozenset_to_names
)
from src.visualization import (
    plot_top_products, plot_algorithm_comparison, plot_scatter_support_confidence_lift, plot_top_rules_by_lift
)

RULES_TBL_DIR = TABLES_ASSOCIATION / 'association_rules'
RULES_TBL_DIR.mkdir(parents=True, exist_ok=True)

print("=== 06. ASSOCIATION RULES ===")
basket_matrix_path = PROCESSED_DIR / 'association_basket_matrix.csv'
basket_long_path = PROCESSED_DIR / 'association_basket_long.csv'

if basket_matrix_path.exists():
    basket, stats = load_basket_matrix(basket_matrix_path)
else:
    basket, stats = load_basket_from_long(basket_long_path)

sc_to_desc = map_stockcode_to_description(basket_long_path)
product_freq = basket.sum(axis=0).sort_values(ascending=False)

min_support = 0.02
min_confidence = 0.5

print("\n--- Running Apriori & FP-Growth ---")
try:
    apriori_itemsets, apriori_time = run_apriori(basket, min_support=min_support)
except:
    min_support = 0.03
    apriori_itemsets, apriori_time = run_apriori(basket, min_support=min_support)

fpgrowth_itemsets, fpgrowth_time = run_fpgrowth(basket, min_support=min_support)

apriori_rules = generate_rules(apriori_itemsets, min_confidence=min_confidence)
fpgrowth_rules = generate_rules(fpgrowth_itemsets, min_confidence=min_confidence)

apriori_valid, _ = validate_rules(apriori_rules, min_support, min_confidence)
fpgrowth_valid, _ = validate_rules(fpgrowth_rules, min_support, min_confidence)

apriori_lift = apriori_valid[apriori_valid['lift'] > 1].copy()
fpgrowth_lift = fpgrowth_valid[fpgrowth_valid['lift'] > 1].copy()

def max_itemset_size(df): return int(df['itemsets'].apply(len).max()) if not df.empty else 0

apriori_info = {'algorithm': 'Apriori', 'min_support': min_support, 'min_confidence': min_confidence, 'runtime': apriori_time, 'n_itemsets': len(apriori_itemsets), 'n_rules': len(apriori_rules), 'n_valid_rules': len(apriori_lift), 'max_itemset_size': max_itemset_size(apriori_itemsets), 'max_lift': apriori_lift['lift'].max() if len(apriori_lift)>0 else 0}
fpgrowth_info = {'algorithm': 'FP-Growth', 'min_support': min_support, 'min_confidence': min_confidence, 'runtime': fpgrowth_time, 'n_itemsets': len(fpgrowth_itemsets), 'n_rules': len(fpgrowth_rules), 'n_valid_rules': len(fpgrowth_lift), 'max_itemset_size': max_itemset_size(fpgrowth_itemsets), 'max_lift': fpgrowth_lift['lift'].max() if len(fpgrowth_lift)>0 else 0}

algo_comp = build_algorithm_comparison(apriori_info, fpgrowth_info)
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

save_rules_csv(apriori_lift, 'Apriori', RULES_TBL_DIR / 'apriori_rules.csv')
save_rules_csv(fpgrowth_lift, 'FP-Growth', RULES_TBL_DIR / 'fpgrowth_rules.csv')
save_rules_csv(fpgrowth_lift, 'FP-Growth', RULES_TBL_DIR / 'selected_association_rules.csv')

top_rules = fpgrowth_lift.copy()
top_rules['quality'] = top_rules['lift'] * top_rules['confidence']
top_rules = top_rules.nlargest(min(10, len(top_rules)), 'quality')
insights = []
for i, (_, rule) in enumerate(top_rules.head(5).iterrows()):
    insights.append({
        'rank': i + 1,
        'antecedent': frozenset_to_names(rule['antecedents'], sc_to_desc),
        'consequent': frozenset_to_names(rule['consequents'], sc_to_desc),
        'support': round(rule['support'], 4),
        'confidence': round(rule['confidence'], 4),
        'lift': round(rule['lift'], 4),
    })
pd.DataFrame(insights).to_csv(TABLES_ASSOCIATION / 'association_business_insights.csv', index=False)

# Visualizations
print("\n--- Generating Visualizations ---")
import matplotlib
matplotlib.use('Agg')
plot_top_products(product_freq, sc_to_desc, stats['n_invoices'], save_path=FIGURES_ASSOCIATION / 'top_products_frequency.png')
plot_algorithm_comparison(apriori_itemsets, fpgrowth_itemsets, apriori_time, fpgrowth_time, save_path=FIGURES_ASSOCIATION / 'algorithm_comparison.png')
plot_scatter_support_confidence_lift(fpgrowth_lift, save_path=FIGURES_ASSOCIATION / 'scatter_support_confidence_lift.png')
plot_top_rules_by_lift(fpgrowth_lift, sc_to_desc, frozenset_to_names, save_path=FIGURES_ASSOCIATION / 'top_rules_lift.png')

print("=== ASSOCIATION RULES RUNNER COMPLETE ===")
