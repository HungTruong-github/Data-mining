# Review Notebook: 06_association_rules.ipynb
*Source Path: `D:/Project/Data-mininng/notebooks/06_association_rules.ipynb`*

---

# 06. Association Rules (Khai phá Luật kết hợp)

**Mục tiêu:**
- Sử dụng Basket Matrix (one-hot encoding) theo **StockCode**.
- Đánh giá và so sánh hai thuật toán: **Apriori** và **FP-Growth**.
- Sinh các luật kết hợp với min_support và min_confidence.
- Lọc các luật có ý nghĩa (Lift > 1) và đề xuất **Business Insights** (Cross-selling).

```python
# [Cell 1 - Execution Count: 1]
import sys, os
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

%matplotlib inline
```

## 1. Tải và Xác thực Basket Matrix

```python
# [Cell 3 - Execution Count: 2]
basket_matrix_path = PROCESSED_DIR / 'association_basket_matrix.csv'
basket_long_path = PROCESSED_DIR / 'association_basket_long.csv'

if basket_matrix_path.exists():
    basket, stats = load_basket_matrix(basket_matrix_path)
else:
    basket, stats = load_basket_from_long(basket_long_path)

sc_to_desc = map_stockcode_to_description(basket_long_path)
product_freq = basket.sum(axis=0).sort_values(ascending=False)

plot_top_products(product_freq, sc_to_desc, stats['n_invoices'])
```

**Output (stdout):**
```text
Loading basket matrix from D:\Project\Data-mininng\data\processed\association_basket_matrix.csv...
```

**Output (stdout):**
```text
Invoices: 19,773
  Products: 3,908
  Avg items/invoice: 26.16
  Sparsity: 0.9933
```

**Result Display:**
```text
<Figure size 1400x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 2. So sánh Apriori và FP-Growth

```python
# [Cell 5 - Execution Count: 3]
min_support = 0.02
min_confidence = 0.5
min_baskets = calculate_min_baskets(min_support, stats['n_invoices'])

print(f"Tham số: min_support={min_support} (yêu cầu tối thiểu ceil({min_support} x {stats['n_invoices']:,}) = {min_baskets} hóa đơn), min_confidence={min_confidence}")

print("\n--- Thực hiện benchmark 3 lần lặp Apriori & FP-Growth ---")
bench = benchmark_algorithms(basket, min_support=min_support, n_runs=3)
apriori_time = bench['apriori_median']
fpgrowth_time = bench['fpgrowth_median']
print(f"Apriori median: {apriori_time:.2f}s (IQR: {bench['apriori_iqr']:.2f}s, runs: {bench['apriori_times']})")
print(f"FP-Growth median: {fpgrowth_time:.2f}s (IQR: {bench['fpgrowth_iqr']:.2f}s, runs: {bench['fpgrowth_times']})")

apriori_itemsets, _ = run_apriori(basket, min_support=min_support)
fpgrowth_itemsets, _ = run_fpgrowth(basket, min_support=min_support)

plot_algorithm_comparison(apriori_itemsets, fpgrowth_itemsets, apriori_time, fpgrowth_time)
```

**Output (stdout):**
```text
Tham số: min_support=0.02 (yêu cầu tối thiểu ceil(0.02 x 19,773) = 396 hóa đơn), min_confidence=0.5

--- Thực hiện benchmark 3 lần lặp Apriori & FP-Growth ---
```

**Output (stdout):**
```text
Apriori median: 7.64s (IQR: 2.23s, runs: [4.6299, 7.6363, 9.08])
FP-Growth median: 7.13s (IQR: 2.14s, runs: [8.5085, 7.1299, 4.227])
  Running Apriori (min_support=0.02)...
```

**Output (stdout):**
```text
Apriori: 389 frequent itemsets in 8.96s
  Running FP-Growth (min_support=0.02)...
```

**Output (stdout):**
```text
FP-Growth: 389 frequent itemsets in 6.38s
```

**Result Display:**
```text
<Figure size 1400x600 with 2 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 3. Sinh Luật (Generate Rules) và Xác thực (Validate)
Xác thực cả hai thuật toán (kiểm tra Lift > 1, chống giao thoa, chống empty, loại NaN).

```python
# [Cell 7 - Execution Count: 4]
apriori_rules = generate_rules(apriori_itemsets, min_confidence=min_confidence)
fpgrowth_rules = generate_rules(fpgrowth_itemsets, min_confidence=min_confidence)

apriori_valid, apriori_checks = validate_rules(apriori_rules, min_support, min_confidence)
fpgrowth_valid, fpgrowth_checks = validate_rules(fpgrowth_rules, min_support, min_confidence)

print("Apriori Validations:")
for c in apriori_checks: print(f"  {c}")
    
print("\nFP-Growth Validations:")
for c in fpgrowth_checks: print(f"  {c}")

apriori_lift = apriori_valid[apriori_valid['lift'] > 1].copy()
fpgrowth_lift = fpgrowth_valid[fpgrowth_valid['lift'] > 1].copy()
print(f"\nRules hợp lệ (Lift > 1) -> Apriori: {len(apriori_lift)}, FP-Growth: {len(fpgrowth_lift)}")

# Xác thực tương đương nội dung chuẩn hóa bằng canonical StockCodes
print("\n--- Xác thực tương đương nội dung (Canonical StockCode Matching) ---")
equiv_res = compare_rule_sets(apriori_lift, fpgrowth_lift, 'Apriori', 'FP-Growth')
print(f"  Tương đương (is_equivalent): {equiv_res['is_equivalent']}")
print(f"  Khớp {equiv_res['matched_count']}/{equiv_res['count_Apriori']} luật")
print(f"  Độ lệch max: support={equiv_res['max_support_diff']:.6e}, conf={equiv_res['max_confidence_diff']:.6e}, lift={equiv_res['max_lift_diff']:.6e}")

if not equiv_res['is_equivalent']:
    raise ValueError(f"CRITICAL: Apriori và FP-Growth không tương đương nội dung! Chi tiết: {equiv_res}")

pd.DataFrame([equiv_res]).to_csv(TABLES_ASSOCIATION / 'association_rules_equivalence_audit.csv', index=False)
print("  [OK] Đã xuất association_rules_equivalence_audit.csv")
```

**Output (stdout):**
```text
Apriori Validations:
  [OK] support >= 0.02: 61/61 rules
  [OK] confidence >= 0.5: 61 rules
  [OK] lift > 1.0 enforced: 61 rules
  [OK] Non-empty antecedent and consequent: 61 rules
  [OK] Disjoint antecedent and consequent (no overlap): 61 rules
  [OK] Core metrics are finite and non-null
  [OK] Duplicates removed: 0 (Remaining: 61)

FP-Growth Validations:
  [OK] support >= 0.02: 61/61 rules
  [OK] confidence >= 0.5: 61 rules
  [OK] lift > 1.0 enforced: 61 rules
  [OK] Non-empty antecedent and consequent: 61 rules
  [OK] Disjoint antecedent and consequent (no overlap): 61 rules
  [OK] Core metrics are finite and non-null
  [OK] Duplicates removed: 0 (Remaining: 61)

Rules hợp lệ (Lift > 1) -> Apriori: 61, FP-Growth: 61

--- Xác thực tương đương nội dung (Canonical StockCode Matching) ---
  Tương đương (is_equivalent): True
  Khớp 61/61 luật
  Độ lệch max: support=0.000000e+00, conf=0.000000e+00, lift=0.000000e+00
  [OK] Đã xuất association_rules_equivalence_audit.csv
```

## 4. Lựa chọn thuật toán & Lưu kết quả
Lựa chọn thuật toán dựa trên kết quả benchmark thực nghiệm (thời gian chạy median qua 3 lần lặp, độ biến thiên IQR). Thuật toán có median runtime thấp hơn sẽ được chọn động làm thuật toán chính cho downstream pipeline.

```python
# [Cell 9 - Execution Count: 5]
# Phân tích độ nhạy (Sensitivity Analysis)
print("\n--- Phân tích độ nhạy tham số (Support & Confidence) ---")
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
save_rules_csv(selected_rules, selected_algo, RULES_TBL_DIR / 'selected_association_rules.csv')
```

**Output (stdout):**
```text
--- Phân tích độ nhạy tham số (Support & Confidence) ---
```

**Output (stdout):**
```text
[OK] Đã lưu rule_sensitivity_analysis.csv (12 điểm lưới)
```

**Result Display:**
```text
algorithm  min_support  min_confidence  runtime_seconds  \
0    Apriori         0.02             0.5           7.6363   
1  FP-Growth         0.02             0.5           7.1299   

   frequent_itemset_count  rule_count  valid_rule_count  max_itemset_size  \
0                     389          61                61                 3   
1                     389          61                61                 3   

   max_rule_lift  is_selected  
0      18.231107        False  
1      18.231107         True
```

**Output (stdout):**
```text
Thuật toán được chọn thực nghiệm: FP-Growth (61 rules, median runtime=7.13s)
```

## 5. Trực quan hóa Rules

```python
# [Cell 11 - Execution Count: 6]
plot_scatter_support_confidence_lift(selected_rules)
```

**Result Display:**
```text
<Figure size 1200x800 with 2 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

```python
# [Cell 12 - Execution Count: 7]
plot_top_rules_by_lift(selected_rules, sc_to_desc, frozenset_to_names)
```

**Result Display:**
```text
<Figure size 1400x800 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

## 6. Business Insights (Gợi ý kinh doanh)
Lấy top 5 luật tốt nhất dựa trên Tích `Lift * Confidence` để tư vấn Cross-selling.
*Lưu ý: Khai phá luật chỉ chứng minh sự đồng xuất hiện (co-occurrence) chứ không mang tính nhân quả (causality).*

```python
# [Cell 14 - Execution Count: 8]
top_rules = selected_rules.copy()
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
insights_df.to_csv(TABLES_ASSOCIATION / 'association_business_insights.csv', index=False)
```

**Result Display:**
```text
rule_id                                        antecedents  \
0        1  PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY ...   
1        2  GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY ...   
2        3                     PINK REGENCY TEACUP AND SAUCER   
3        4                     PINK REGENCY TEACUP AND SAUCER   
4        5  GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY...   
5        6  REGENCY CAKESTAND 3 TIER, GREEN REGENCY TEACUP...   
6        7  REGENCY CAKESTAND 3 TIER, ROSES REGENCY TEACUP...   
7        8                     PINK REGENCY TEACUP AND SAUCER   
8        9                 GARDENERS KNEELING PAD CUP OF TEA    
9       10                    GREEN REGENCY TEACUP AND SAUCER   

                                         consequents   support  confidence  \
0                    GREEN REGENCY TEACUP AND SAUCER  0.027361    0.904682   
1                   ROSES REGENCY TEACUP AND SAUCER   0.027361    0.856013   
2                    GREEN REGENCY TEACUP AND SAUCER  0.031963    0.826144   
3  GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY...  0.027361    0.707190   
4                     PINK REGENCY TEACUP AND SAUCER  0.027361    0.705346   
5                   ROSES REGENCY TEACUP AND SAUCER   0.020634    0.801572   
6                    GREEN REGENCY TEACUP AND SAUCER  0.020634    0.777143   
7                   ROSES REGENCY TEACUP AND SAUCER   0.030243    0.781699   
8                  GARDENERS KNEELING PAD KEEP CALM   0.027613    0.720317   
9                   ROSES REGENCY TEACUP AND SAUCER   0.038790    0.757157   

        lift evidence_quality  \
0  17.658719             High   
1  15.892900             High   
2  16.125707             High   
3  18.231107             High   
4  18.231107             High   
5  14.882138             High   
6  15.169246             High   
7  14.513184             High   
8  15.600023             High   
9  14.057525             High   

                                     business_action  
0  Gợi ý kèm 'GREEN REGENCY TEACUP AND SAUCER' kh...  
1  Gợi ý kèm 'ROSES REGENCY TEACUP AND SAUCER ' k...  
2  Gợi ý kèm 'GREEN REGENCY TEACUP AND SAUCER' kh...  
3  Gợi ý kèm 'GREEN REGENCY TEACUP AND SAUCER, RO...  
4  Gợi ý kèm 'PINK REGENCY TEACUP AND SAUCER' khi...  
5  Gợi ý kèm 'ROSES REGENCY TEACUP AND SAUCER ' k...  
6  Gợi ý kèm 'GREEN REGENCY TEACUP AND SAUCER' kh...  
7  Gợi ý kèm 'ROSES REGENCY TEACUP AND SAUCER ' k...  
8  Gợi ý kèm 'GARDENERS KNEELING PAD KEEP CALM ' ...  
9  Gợi ý kèm 'ROSES REGENCY TEACUP AND SAUCER ' k...
```

