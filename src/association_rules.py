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
    Validate association rules quality and enforce mathematical constraints.
    Returns validated rules and detailed check results.

    Enforced criteria:
    - support >= min_support
    - confidence >= min_confidence
    - lift > 1.0 (positive association only; lift <= 1 means independent or negative)
    - Non-empty antecedent and consequent
    - Disjoint antecedent and consequent (no overlap)
    - Valid numeric values in core metrics (no NaN/Inf in support, confidence, lift)
    - Mathematical preservation of valid +inf conviction (when confidence == 1.0)
    - Unique rules (drop duplicates)
    """
    checks = []
    n_orig = len(rules)

    if rules is None or rules.empty:
        checks.append('[WARN] Empty rules input')
        return pd.DataFrame(), checks

    # 1. Enforce support threshold
    rules = rules[rules['support'] >= min_support].copy()
    checks.append(f'[OK] support >= {min_support}: {len(rules)}/{n_orig} rules')

    # 2. Enforce confidence threshold
    rules = rules[rules['confidence'] >= min_confidence].copy()
    checks.append(f'[OK] confidence >= {min_confidence}: {len(rules)} rules')

    # 3. Enforce lift > 1.0 (actually filter the returned rules!)
    rules = rules[rules['lift'] > 1.0].copy()
    checks.append(f'[OK] lift > 1.0 enforced: {len(rules)} rules')

    # 4. Filter empty antecedent / consequent
    valid_ant = rules['antecedents'].apply(lambda x: len(x) > 0)
    valid_con = rules['consequents'].apply(lambda x: len(x) > 0)
    rules = rules[valid_ant & valid_con].copy()
    checks.append(f'[OK] Non-empty antecedent and consequent: {len(rules)} rules')

    # 5. Filter antecedent-consequent overlap (must be disjoint sets)
    no_overlap = rules.apply(
        lambda r: len(r['antecedents'].intersection(r['consequents'])) == 0, axis=1
    )
    rules = rules[no_overlap].copy()
    checks.append(f'[OK] Disjoint antecedent and consequent (no overlap): {len(rules)} rules')

    # 6. Check core metrics for NaN / Inf
    core_metrics = ['support', 'confidence', 'lift']
    nan_count = rules[core_metrics].isna().sum().sum()
    inf_count = np.isinf(rules[core_metrics].select_dtypes(include=[np.number])).sum().sum()
    if nan_count > 0 or inf_count > 0:
        rules = rules[~rules[core_metrics].isna().any(axis=1)].copy()
        rules = rules[~np.isinf(rules[core_metrics]).any(axis=1)].copy()
        checks.append(f'[WARN] Filtered {nan_count} NaNs and {inf_count} Infs from core metrics')
    else:
        checks.append('[OK] Core metrics are finite and non-null')

    # 7. Preserve valid +inf conviction mathematically without mechanical removal
    # Conviction = (1 - sup(consequent)) / (1 - confidence). When confidence == 1, conviction is +inf.
    # We do NOT drop these rules because confidence=1 is the strongest logical implication!

    # 8. Remove duplicate rules
    n_before = len(rules)
    # Ensure sets are hashable frozensets for deduplication
    rules['ant_canonical'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['con_canonical'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
    rules = rules.drop_duplicates(subset=['ant_canonical', 'con_canonical']).copy()
    rules = rules.drop(columns=['ant_canonical', 'con_canonical'])
    checks.append(f'[OK] Duplicates removed: {n_before - len(rules)} (Remaining: {len(rules)})')

    # 9. Sort by lift, confidence, support descending
    rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False]).reset_index(drop=True)

    return rules, checks


def canonical_rule_key(antecedents, consequents):
    """Generate unambiguous canonical string key for a rule based on sorted StockCodes."""
    ant_str = ', '.join(sorted(antecedents))
    con_str = ', '.join(sorted(consequents))
    return f"{{{ant_str}}} -> {{{con_str}}}"


def compare_rule_sets(rules_a, rules_b, algo_a_name="Apriori", algo_b_name="FP-Growth", tolerance=1e-5):
    """
    Compare two sets of association rules by canonical StockCode representation.
    Verifies actual content equivalence beyond simple count matching.

    Returns
    -------
    dict with comparison metrics:
      - is_equivalent: bool
      - count_a: int
      - count_b: int
      - matched_count: int
      - only_in_a: list
      - only_in_b: list
      - max_support_diff: float
      - max_confidence_diff: float
      - max_lift_diff: float
    """
    df_a = rules_a.copy()
    df_b = rules_b.copy()

    df_a['canonical_key'] = df_a.apply(lambda r: canonical_rule_key(r['antecedents'], r['consequents']), axis=1)
    df_b['canonical_key'] = df_b.apply(lambda r: canonical_rule_key(r['antecedents'], r['consequents']), axis=1)

    keys_a = set(df_a['canonical_key'])
    keys_b = set(df_b['canonical_key'])

    only_in_a = sorted(list(keys_a - keys_b))
    only_in_b = sorted(list(keys_b - keys_a))
    common_keys = sorted(list(keys_a & keys_b))

    merged = df_a.merge(df_b, on='canonical_key', suffixes=('_a', '_b'))

    if not merged.empty:
        sup_diff = (merged['support_a'] - merged['support_b']).abs().max()
        conf_diff = (merged['confidence_a'] - merged['confidence_b']).abs().max()
        lift_diff = (merged['lift_a'] - merged['lift_b']).abs().max()
    else:
        sup_diff, conf_diff, lift_diff = 0.0, 0.0, 0.0

    is_equivalent = (
        len(only_in_a) == 0 and
        len(only_in_b) == 0 and
        sup_diff <= tolerance and
        conf_diff <= tolerance and
        lift_diff <= tolerance
    )

    return {
        'is_equivalent': is_equivalent,
        f'count_{algo_a_name}': len(df_a),
        f'count_{algo_b_name}': len(df_b),
        'matched_count': len(common_keys),
        f'only_in_{algo_a_name}': only_in_a,
        f'only_in_{algo_b_name}': only_in_b,
        'max_support_diff': float(sup_diff),
        'max_confidence_diff': float(conf_diff),
        'max_lift_diff': float(lift_diff),
    }


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
    Build comparison table between Apriori and FP-Growth and mark selected algorithm dynamically.
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
            'is_selected': False,
        })
    df = pd.DataFrame(rows)
    if not df.empty and 'runtime_seconds' in df.columns:
        fastest_idx = df['runtime_seconds'].idxmin()
        df.loc[fastest_idx, 'is_selected'] = True
    return df
