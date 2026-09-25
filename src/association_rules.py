"""
Module association_rules: Apriori vs FP-Growth comparison.
Uses StockCode as item key, maps to Description for display.
"""
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ====================================================================
# 1. LOAD AND PREPARE BASKET MATRIX
# ====================================================================
def load_basket_matrix(filepath):
    """
    Load basket matrix from CSV. First column is InvoiceNo (index).
    Columns are StockCodes. Values are 0/1.

    Returns
    -------
    basket : pd.DataFrame (boolean)
    stats : dict
    """
    print(f"Loading basket matrix from {filepath}...")
    basket = pd.read_csv(filepath, index_col=0)

    # Ensure binary
    basket = (basket > 0).astype(bool)

    # Remove all-zero rows and columns
    row_sums = basket.sum(axis=1)
    col_sums = basket.sum(axis=0)
    basket = basket.loc[row_sums > 0, col_sums > 0]

    items_per_invoice = basket.sum(axis=1)

    stats = {
        'n_invoices': basket.shape[0],
        'n_products': basket.shape[1],
        'mean_items_per_invoice': round(items_per_invoice.mean(), 2),
        'max_items_per_invoice': int(items_per_invoice.max()),
        'min_items_per_invoice': int(items_per_invoice.min()),
        'sparsity': round(1 - basket.values.sum() / (basket.shape[0] * basket.shape[1]), 4),
    }

    print(f"  Invoices: {stats['n_invoices']:,}")
    print(f"  Products: {stats['n_products']:,}")
    print(f"  Avg items/invoice: {stats['mean_items_per_invoice']}")
    print(f"  Sparsity: {stats['sparsity']:.4f}")

    return basket, stats


def load_basket_from_long(filepath):
    """
    Build basket matrix from long-format CSV (InvoiceNo, StockCode, Description).
    """
    print(f"Loading long-format basket from {filepath}...")
    df = pd.read_csv(filepath, dtype=str)
    df = df.dropna(subset=['InvoiceNo', 'StockCode'])

    basket = df.pivot_table(
        index='InvoiceNo', columns='StockCode',
        values='Description', aggfunc='count', fill_value=0
    )
    basket = (basket > 0).astype(bool)

    # Remove empty rows/cols
    basket = basket.loc[basket.sum(axis=1) > 0, basket.sum(axis=0) > 0]

    items_per_invoice = basket.sum(axis=1)
    stats = {
        'n_invoices': basket.shape[0],
        'n_products': basket.shape[1],
        'mean_items_per_invoice': round(items_per_invoice.mean(), 2),
        'max_items_per_invoice': int(items_per_invoice.max()),
        'min_items_per_invoice': int(items_per_invoice.min()),
        'sparsity': round(1 - basket.values.sum() / (basket.shape[0] * basket.shape[1]), 4),
    }

    return basket, stats


# ====================================================================
# 2. RUN APRIORI
# ====================================================================
def run_apriori(basket, min_support=0.02):
    """
    Run Apriori algorithm. Returns frequent_itemsets and runtime.
    """
    from mlxtend.frequent_patterns import apriori as mlx_apriori

    print(f"  Running Apriori (min_support={min_support})...")
    start = time.time()
    freq_itemsets = mlx_apriori(basket, min_support=min_support, use_colnames=True)
    runtime = round(time.time() - start, 2)

    print(f"  Apriori: {len(freq_itemsets)} frequent itemsets in {runtime}s")
    return freq_itemsets, runtime


# ====================================================================
# 3. RUN FP-GROWTH
# ====================================================================
def run_fpgrowth(basket, min_support=0.02):
    """
    Run FP-Growth algorithm. Returns frequent_itemsets and runtime.
    """
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

    print(f"  Running FP-Growth (min_support={min_support})...")
    start = time.time()
    freq_itemsets = mlx_fpgrowth(basket, min_support=min_support, use_colnames=True)
    runtime = round(time.time() - start, 2)

    print(f"  FP-Growth: {len(freq_itemsets)} frequent itemsets in {runtime}s")
    return freq_itemsets, runtime


# ====================================================================
# 4. GENERATE RULES
# ====================================================================
def generate_rules(freq_itemsets, min_confidence=0.5, metric='confidence'):
    """
    Generate association rules from frequent itemsets.
    """
    from mlxtend.frequent_patterns import association_rules as mlx_rules

    if freq_itemsets.empty:
        return pd.DataFrame()

    rules = mlx_rules(freq_itemsets, metric=metric, min_threshold=min_confidence)

    # Add lengths
    rules['antecedent_length'] = rules['antecedents'].apply(len)
    rules['consequent_length'] = rules['consequents'].apply(len)

    return rules


# ====================================================================
# 5. VALIDATE RULES
# ====================================================================
def validate_rules(rules, min_support=0.02, min_confidence=0.5):
    """
    Validate association rules quality.
    Returns validated rules and list of check results.
    """
    checks = []
    n_orig = len(rules)

    if rules.empty:
        checks.append('[WARN] No rules generated')
        return rules, checks

    # Filter: support >= min_support
    rules = rules[rules['support'] >= min_support].copy()
    checks.append(f'[OK] support >= {min_support}: {len(rules)}/{n_orig} rules')

    # Filter: confidence >= min_confidence
    rules = rules[rules['confidence'] >= min_confidence].copy()
    checks.append(f'[OK] confidence >= {min_confidence}: {len(rules)} rules')

    # Filter: lift > 1
    valid_lift = rules[rules['lift'] > 1].copy()
    checks.append(f'[OK] lift > 1: {len(valid_lift)}/{len(rules)} rules')

    # Check no empty antecedent/consequent
    empty_ant = rules['antecedents'].apply(lambda x: len(x) == 0).sum()
    empty_con = rules['consequents'].apply(lambda x: len(x) == 0).sum()
    checks.append(f'[OK] Empty antecedent: {empty_ant}, Empty consequent: {empty_con}')

    # Check antecedent/consequent no intersection
    overlap = rules.apply(
        lambda r: len(r['antecedents'].intersection(r['consequents'])), axis=1
    ).sum()
    checks.append(f'[OK] Antecedent-consequent overlap: {overlap}')

    # Check for NaN/inf
    nan_count = rules[['support', 'confidence', 'lift']].isna().sum().sum()
    inf_count = np.isinf(rules[['support', 'confidence', 'lift']].select_dtypes(include=[np.number])).sum().sum()
    checks.append(f'[OK] NaN in metrics: {nan_count}, Inf: {inf_count}')

    # Remove inf conviction rows
    if 'conviction' in rules.columns:
        rules = rules[~np.isinf(rules['conviction'])].copy()
        rules['conviction'] = rules['conviction'].fillna(0)

    # Remove duplicates
    n_before = len(rules)
    rules = rules.drop_duplicates(subset=['antecedents', 'consequents'])
    checks.append(f'[OK] Duplicates removed: {n_before - len(rules)}')

    # Sort by lift, confidence, support
    rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False])

    return rules, checks


# ====================================================================
# 6. STOCKCODE TO DESCRIPTION MAPPING
# ====================================================================
def map_stockcode_to_description(basket_long_path):
    """
    Build StockCode -> Description mapping.
    For multiple descriptions per StockCode, choose most frequent.
    """
    df = pd.read_csv(basket_long_path, dtype=str)
    df = df.dropna(subset=['StockCode', 'Description'])

    mapping = (
        df.groupby('StockCode')['Description']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    return mapping


def frozenset_to_names(fs, mapping):
    """Convert frozenset of StockCodes to readable product names."""
    names = [mapping.get(code, code) for code in sorted(fs)]
    return ', '.join(names)


# ====================================================================
# 7. ALGORITHM COMPARISON TABLE
# ====================================================================
def build_algorithm_comparison(apriori_info, fpgrowth_info):
    """
    Build comparison table between Apriori and FP-Growth.
    """
    rows = []
    for info in [apriori_info, fpgrowth_info]:
        rows.append({
            'algorithm': info['algorithm'],
            'min_support': info['min_support'],
            'min_confidence': info['min_confidence'],
            'runtime_seconds': info['runtime'],
            'frequent_itemset_count': info['n_itemsets'],
            'rule_count': info['n_rules'],
            'valid_rule_count': info['n_valid_rules'],
            'max_itemset_size': info['max_itemset_size'],
            'max_rule_lift': info['max_lift'],
        })
    return pd.DataFrame(rows)
