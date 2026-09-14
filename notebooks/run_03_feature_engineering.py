"""
Script chay Feature Engineering & RFM - tuong duong notebook 03.
Pipeline: Customer Data -> RFM Features -> RFM Scores -> Repeat Purchase Label
           -> Basket Data -> Visualizations -> Data Quality Checks

Chay: .venv\\Scripts\\python.exe notebooks\\run_03_feature_engineering.py
"""
import sys
import os
import warnings

# Fix encoding cho Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    INTERIM_DIR, PROCESSED_DIR,
    FIGURES_FEATURE_ENGINEERING, TABLES_FEATURE_ENGINEERING,
    REPEAT_PURCHASE_WINDOW_DAYS, RAW_DATA_CSV,
)
from src.feature_engineering import (
    build_rfm_features,
    create_rfm_scores,
    assign_rfm_segment,
    summarize_rfm_segments,
    build_repeat_purchase_features,
    build_basket_data,
    summarize_features,
    validate_feature_data,
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style('whitegrid')
sns.set_palette('husl')

FIG_DIR = FIGURES_FEATURE_ENGINEERING
TBL_DIR = TABLES_FEATURE_ENGINEERING

print('=' * 70)
print('    03. FEATURE ENGINEERING & RFM ANALYSIS')
print('=' * 70)

# ====================================================================
# 1. DOC VA CHUAN HOA CUSTOMER DATA
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 1: DOC VA CHUAN HOA CUSTOMER DATA')
print('-' * 70)

df_cust = pd.read_csv(INTERIM_DIR / 'customer_transactions.csv')
df_cust['InvoiceDate'] = pd.to_datetime(df_cust['InvoiceDate'])
df_cust['CustomerID'] = df_cust['CustomerID'].astype(int)
df_cust['InvoiceNo'] = df_cust['InvoiceNo'].astype(str)
df_cust['StockCode'] = df_cust['StockCode'].astype(str)

# Kiem tra cac cot bat buoc
required_cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country', 'TotalAmount']
missing_cols = [c for c in required_cols if c not in df_cust.columns]
assert len(missing_cols) == 0, f'Thieu cac cot: {missing_cols}'

# Kiem tra du lieu
assert df_cust['CustomerID'].isnull().sum() == 0, 'CustomerID co gia tri thieu!'
assert df_cust['InvoiceNo'].isnull().sum() == 0, 'InvoiceNo co gia tri thieu!'
assert (df_cust['Quantity'] > 0).all(), 'Co Quantity <= 0!'
assert (df_cust['UnitPrice'] > 0).all(), 'Co UnitPrice <= 0!'
assert (df_cust['TotalAmount'] > 0).all(), 'Co TotalAmount <= 0!'

n_rows = len(df_cust)
n_customers = df_cust['CustomerID'].nunique()
n_invoices = df_cust['InvoiceNo'].nunique()
n_products = df_cust['StockCode'].nunique()
date_min = df_cust['InvoiceDate'].min()
date_max = df_cust['InvoiceDate'].max()
total_revenue = df_cust['TotalAmount'].sum()
total_items = df_cust['Quantity'].sum()

print(f'  So dong                : {n_rows:,}')
print(f'  So khach hang          : {n_customers:,}')
print(f'  So hoa don             : {n_invoices:,}')
print(f'  So san pham            : {n_products:,}')
print(f'  Ngay nho nhat          : {date_min}')
print(f'  Ngay lon nhat          : {date_max}')
print(f'  Tong doanh thu         : {total_revenue:,.2f} GBP')
print(f'  Tong so luong san pham : {total_items:,}')

# ====================================================================
# 2. TAO RFM CUSTOMER FEATURES
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 2: TAO RFM CUSTOMER FEATURES')
print('-' * 70)

rfm, reference_date = build_rfm_features(df_cust)

print(f'  reference_date         : {reference_date}')
print(f'  So khach hang RFM      : {len(rfm):,}')
print(f'  Recency   - min: {rfm["Recency"].min()}, max: {rfm["Recency"].max()}, mean: {rfm["Recency"].mean():.1f}')
print(f'  Frequency - min: {rfm["Frequency"].min()}, max: {rfm["Frequency"].max()}, mean: {rfm["Frequency"].mean():.1f}')
print(f'  Monetary  - min: {rfm["Monetary"].min():.2f}, max: {rfm["Monetary"].max():.2f}, mean: {rfm["Monetary"].mean():.2f}')

# ====================================================================
# 3. TAO RFM SCORE VA SEGMENT
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 3: TAO RFM SCORE VA SEGMENT')
print('-' * 70)

rfm = create_rfm_scores(rfm)
rfm['RFM_Segment'] = rfm.apply(assign_rfm_segment, axis=1)

print('  Phan bo R_Score:')
print(rfm['R_Score'].value_counts().sort_index().to_string())
print('  Phan bo F_Score:')
print(rfm['F_Score'].value_counts().sort_index().to_string())
print('  Phan bo M_Score:')
print(rfm['M_Score'].value_counts().sort_index().to_string())

# Luu RFM customer features
rfm.to_csv(PROCESSED_DIR / 'rfm_customer_features.csv', index=False)
print(f'\n  [OK] Luu: {PROCESSED_DIR / "rfm_customer_features.csv"}')
print(f'       {len(rfm):,} dong x {len(rfm.columns)} cot')

# Tong hop segment
segment_summary = summarize_rfm_segments(rfm)
segment_summary.to_csv(TBL_DIR / 'rfm_segment_summary.csv', index=False)
print(f'  [OK] Luu: {TBL_DIR / "rfm_segment_summary.csv"}')

print('\n  Phan khuc RFM:')
for _, row in segment_summary.iterrows():
    print(f'    {row["RFM_Segment"]:20s}: {int(row["CustomerCount"]):>5,} KH ({row["CustomerPercent"]:>5.1f}%) '
          f'| R={row["AverageRecency"]:.0f} F={row["AverageFrequency"]:.1f} M={row["AverageMonetary"]:,.0f}')

# ====================================================================
# 4. CHUAN BI CLUSTERING FEATURES
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 4: CHUAN BI CLUSTERING FEATURES')
print('-' * 70)

clustering_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary',
                   'TotalItems', 'UniqueProducts', 'AverageOrderValue',
                   'CustomerLifetimeDays']
rfm_cluster = rfm[clustering_cols].copy()

# Log transform
rfm_cluster['Log_Recency'] = np.log1p(rfm_cluster['Recency'])
rfm_cluster['Log_Frequency'] = np.log1p(rfm_cluster['Frequency'])
rfm_cluster['Log_Monetary'] = np.log1p(rfm_cluster['Monetary'])

rfm_cluster.to_csv(PROCESSED_DIR / 'rfm_clustering_features.csv', index=False)
print(f'  [OK] Luu: {PROCESSED_DIR / "rfm_clustering_features.csv"}')
print(f'       {len(rfm_cluster):,} dong x {len(rfm_cluster.columns)} cot')
print(f'  Cac cot: {list(rfm_cluster.columns)}')

# ====================================================================
# 5. TAO NHAN REPEAT PURCHASE 90 NGAY
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 5: TAO NHAN REPEAT PURCHASE 90 NGAY')
print('-' * 70)

repeat_features, audit_df, repeat_info = build_repeat_purchase_features(df_cust)

print(f'  max_date               : {repeat_info["max_date"]}')
print(f'  cutoff_date            : {repeat_info["cutoff_date"]}')
print(f'  feature_ref_date       : {repeat_info["feature_ref_date"]}')
print(f'  window_days            : {repeat_info["window_days"]}')
print(f'  Khach co lich su       : {repeat_info["customers_with_history"]:,}')
print(f'  Khach tao label        : {repeat_info["customers_in_features"]:,}')
print(f'  Label 1 (mua lai)      : {repeat_info["label_1_count"]:,}')
print(f'  Label 0 (khong mua)    : {repeat_info["label_0_count"]:,}')
print(f'  Ty le mua lai          : {repeat_info["repeat_rate"]:.2f}%')

# Luu
repeat_features.to_csv(PROCESSED_DIR / 'repeat_purchase_features.csv', index=False)
print(f'\n  [OK] Luu: {PROCESSED_DIR / "repeat_purchase_features.csv"}')
print(f'       {len(repeat_features):,} dong x {len(repeat_features.columns)} cot')

# Kiem tra khong chua cot cam
assert 'future_invoice_count' not in repeat_features.columns, 'DATA LEAKAGE: future_invoice_count!'
assert 'future_revenue' not in repeat_features.columns, 'DATA LEAKAGE: future_revenue!'
print('  [OK] Khong co data leakage (khong chua future info)')

audit_df.to_csv(TBL_DIR / 'repeat_purchase_label_audit.csv', index=False)
print(f'  [OK] Luu audit: {TBL_DIR / "repeat_purchase_label_audit.csv"}')

# Class balance
class_balance = repeat_features['repeat_purchase_90d'].value_counts().reset_index()
class_balance.columns = ['repeat_purchase_90d', 'Count']
class_balance['Percentage'] = round(class_balance['Count'] / class_balance['Count'].sum() * 100, 2)
class_balance.to_csv(TBL_DIR / 'repeat_purchase_class_balance.csv', index=False)
print(f'  [OK] Luu: {TBL_DIR / "repeat_purchase_class_balance.csv"}')

print('\n  Class Balance:')
for _, row in class_balance.iterrows():
    label_name = 'Mua lai' if row['repeat_purchase_90d'] == 1 else 'Khong mua'
    print(f'    Label {int(row["repeat_purchase_90d"])} ({label_name:10s}): {int(row["Count"]):>5,} ({row["Percentage"]:.2f}%)')

if class_balance['Percentage'].min() < 30:
    print('\n  [CANH BAO] Du lieu mat can bang. Buoc modeling can:')
    print('    - Stratified train/test split')
    print('    - Class weight')
    print('    - Danh gia bang Precision, Recall, F1, ROC-AUC, PR-AUC')

# ====================================================================
# 6. CHUAN BI BASKET DATA
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 6: CHUAN BI BASKET DATA CHO ASSOCIATION RULES')
print('-' * 70)

df_prod = pd.read_csv(INTERIM_DIR / 'product_transactions.csv')
basket_long, basket_stats = build_basket_data(df_prod)

print(f'  Tong so dong basket    : {basket_stats["total_rows"]:,}')
print(f'  So hoa don duy nhat    : {basket_stats["unique_invoices"]:,}')
print(f'  So san pham duy nhat   : {basket_stats["unique_products"]:,}')
print(f'  TB san pham/hoa don    : {basket_stats["avg_items_per_invoice"]:.2f}')

basket_long.to_csv(PROCESSED_DIR / 'association_basket_long.csv', index=False)
print(f'\n  [OK] Luu: {PROCESSED_DIR / "association_basket_long.csv"}')

# Kiem tra kich thuoc one-hot matrix
n_inv = basket_stats['unique_invoices']
n_prod = basket_stats['unique_products']
matrix_size_gb = n_inv * n_prod * 1 / (1024**3)  # boolean = 1 byte
print(f'  Kich thuoc one-hot uoc tinh: {n_inv:,} x {n_prod:,} = {matrix_size_gb:.2f} GB')

if matrix_size_gb < 1.0:
    print('  -> Kich thuoc hop ly, tao ma tran one-hot.')
    basket_matrix = basket_long.pivot_table(
        index='InvoiceNo', columns='StockCode',
        values='Description', aggfunc='count', fill_value=0
    )
    basket_matrix = (basket_matrix > 0).astype(int)
    basket_matrix.to_csv(PROCESSED_DIR / 'association_basket_matrix.csv')
    print(f'  [OK] Luu: {PROCESSED_DIR / "association_basket_matrix.csv"}')
    print(f'       {basket_matrix.shape[0]:,} hoa don x {basket_matrix.shape[1]:,} san pham')
else:
    print(f'  -> Ma tran qua lon ({matrix_size_gb:.2f} GB), chi luu dang dai.')

# ====================================================================
# 7. FEATURE SUMMARY
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 7: FEATURE SUMMARY')
print('-' * 70)

# Feature summary cho RFM
rfm_summary = summarize_features(rfm)
rfm_summary.to_csv(TBL_DIR / 'feature_summary.csv', index=False)
print(f'  [OK] Luu: {TBL_DIR / "feature_summary.csv"}')

# Feature engineering summary
output_files = [
    str(PROCESSED_DIR / 'rfm_customer_features.csv'),
    str(PROCESSED_DIR / 'rfm_clustering_features.csv'),
    str(PROCESSED_DIR / 'repeat_purchase_features.csv'),
    str(PROCESSED_DIR / 'association_basket_long.csv'),
    str(TBL_DIR / 'rfm_segment_summary.csv'),
    str(TBL_DIR / 'repeat_purchase_label_audit.csv'),
    str(TBL_DIR / 'repeat_purchase_class_balance.csv'),
    str(TBL_DIR / 'feature_summary.csv'),
]

fe_summary = pd.DataFrame([
    {'Metric': 'Customer transactions input rows', 'Value': str(n_rows)},
    {'Metric': 'Customer count (input)', 'Value': str(n_customers)},
    {'Metric': 'reference_date', 'Value': str(reference_date)},
    {'Metric': 'cutoff_date', 'Value': str(repeat_info['cutoff_date'])},
    {'Metric': 'Customers with RFM features', 'Value': str(len(rfm))},
    {'Metric': 'Customers eligible for label', 'Value': str(repeat_info['customers_in_features'])},
    {'Metric': 'Label 1 count', 'Value': str(repeat_info['label_1_count'])},
    {'Metric': 'Label 0 count', 'Value': str(repeat_info['label_0_count'])},
    {'Metric': 'Repeat purchase rate (%)', 'Value': str(repeat_info['repeat_rate'])},
    {'Metric': 'RFM feature count', 'Value': str(len(rfm.columns))},
    {'Metric': 'Basket invoices', 'Value': str(basket_stats['unique_invoices'])},
    {'Metric': 'Basket products', 'Value': str(basket_stats['unique_products'])},
    {'Metric': 'Output files', 'Value': '; '.join([os.path.basename(f) for f in output_files])},
])
fe_summary.to_csv(TBL_DIR / 'feature_engineering_summary.csv', index=False)
print(f'  [OK] Luu: {TBL_DIR / "feature_engineering_summary.csv"}')

# ====================================================================
# 8. TRUC QUAN HOA
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 8: TAO BIEU DO')
print('-' * 70)

# --- 8.1 Phan phoi RFM (Recency, Frequency, Monetary) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(rfm['Recency'], bins=50, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[0].set_title('Phan phoi Recency (ngay)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Recency (ngay)', fontsize=11)
axes[0].set_ylabel('So khach hang', fontsize=11)
axes[0].annotate(f'n={len(rfm):,}\nMedian={rfm["Recency"].median():.0f}\nMean={rfm["Recency"].mean():.0f}',
                 xy=(0.65, 0.75), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))

axes[1].hist(rfm['Frequency'], bins=50, color='#22c55e', edgecolor='white', alpha=0.85)
axes[1].set_title('Phan phoi Frequency (so hoa don)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Frequency', fontsize=11)
axes[1].set_ylabel('So khach hang', fontsize=11)
axes[1].annotate(f'n={len(rfm):,}\nMedian={rfm["Frequency"].median():.0f}\nMean={rfm["Frequency"].mean():.1f}',
                 xy=(0.65, 0.75), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='#f0fdf4', alpha=0.8))

axes[2].hist(rfm['Monetary'], bins=50, color='#f59e0b', edgecolor='white', alpha=0.85)
axes[2].set_title('Phan phoi Monetary (GBP)', fontsize=13, fontweight='bold', pad=10)
axes[2].set_xlabel('Monetary (GBP)', fontsize=11)
axes[2].set_ylabel('So khach hang', fontsize=11)
axes[2].annotate(f'n={len(rfm):,}\nMedian={rfm["Monetary"].median():,.0f}\nMean={rfm["Monetary"].mean():,.0f}',
                 xy=(0.65, 0.75), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='#fffbeb', alpha=0.8))

plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] rfm_distribution.png')

# --- 8.2 Phan phoi RFM_Score ---
fig, ax = plt.subplots(figsize=(12, 5))
score_counts = rfm['RFM_Score'].value_counts().sort_index()
ax.bar(score_counts.index, score_counts.values, color='#6366f1', edgecolor='white', alpha=0.9)
ax.set_title('Phan phoi RFM Score (R + F + M)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('RFM Score', fontsize=12)
ax.set_ylabel('So khach hang', fontsize=12)
ax.set_xticks(score_counts.index)
for i, (x, y) in enumerate(zip(score_counts.index, score_counts.values)):
    ax.text(x, y + max(score_counts.values) * 0.01, f'{y:,}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] rfm_score_distribution.png')

# --- 8.3 Khach hang theo RFM Segment ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

seg_order = segment_summary.sort_values('CustomerCount', ascending=True)

# Count
colors_seg = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9']
axes[0].barh(seg_order['RFM_Segment'], seg_order['CustomerCount'],
             color=colors_seg[:len(seg_order)], edgecolor='white')
axes[0].set_title('So khach hang theo Phan khuc RFM', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('So khach hang', fontsize=11)
for j, (cnt, pct) in enumerate(zip(seg_order['CustomerCount'], seg_order['CustomerPercent'])):
    axes[0].text(cnt + max(seg_order['CustomerCount']) * 0.02, j,
                 f'{int(cnt):,} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')

# Revenue
axes[1].barh(seg_order['RFM_Segment'], seg_order['TotalMonetary'],
             color=colors_seg[:len(seg_order)], edgecolor='white')
axes[1].set_title('Tong doanh thu theo Phan khuc RFM (GBP)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Doanh thu (GBP)', fontsize=11)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:,.0f}K'))
for j, val in enumerate(seg_order['TotalMonetary']):
    axes[1].text(val + max(seg_order['TotalMonetary']) * 0.02, j,
                 f'{val:,.0f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_segment_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] rfm_segment_overview.png')

# --- 8.4 Ty le nhan repeat_purchase_90d ---
fig, ax = plt.subplots(figsize=(8, 6))
# Sort index de nhan 0 (khong mua lai) va 1 (mua lai) dung thu tu
label_counts = repeat_features['repeat_purchase_90d'].value_counts().sort_index()
labels = [
    f'Khong mua lai (0)\n{label_counts[0]:,} khach ({label_counts[0]/len(repeat_features)*100:.1f}%)',
    f'Mua lai (1)\n{label_counts[1]:,} khach ({label_counts[1]/len(repeat_features)*100:.1f}%)'
]
colors_label = ['#ef4444', '#22c55e']
wedges, _, autotexts = ax.pie(
    label_counts.values, labels=labels, colors=colors_label,
    autopct='%1.1f%%', startangle=90, explode=[0.03, 0.03],
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 12}
)
ax.set_title(f'Ty le nhan repeat_purchase_90d\n(cutoff: {repeat_info["cutoff_date"].strftime("%Y-%m-%d")})',
             fontsize=14, fontweight='bold', pad=12)
total_labeled = len(repeat_features)
ax.text(0, -1.3, f'Tong: {total_labeled:,} khach hang', ha='center', fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(FIG_DIR / 'repeat_purchase_label_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] repeat_purchase_label_balance.png')

# --- 8.5 Ma tran tuong quan ---
corr_cols = ['Recency', 'Frequency', 'Monetary', 'TotalItems', 'UniqueProducts',
             'AverageOrderValue', 'CustomerLifetimeDays', 'ActiveDays']
corr_data = rfm[corr_cols]
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'fontsize': 9})
ax.set_title('Ma tran tuong quan cac bien RFM', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] rfm_correlation_matrix.png')

# --- 8.6 Phan phoi truoc va sau log-transform ---
log_vars = [('Recency', 'Log_Recency'), ('Frequency', 'Log_Frequency'), ('Monetary', 'Log_Monetary')]
rfm_with_log = rfm.copy()
rfm_with_log['Log_Recency'] = np.log1p(rfm_with_log['Recency'])
rfm_with_log['Log_Frequency'] = np.log1p(rfm_with_log['Frequency'])
rfm_with_log['Log_Monetary'] = np.log1p(rfm_with_log['Monetary'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (orig, logged) in enumerate(log_vars):
    # Truoc log
    axes[0, i].hist(rfm_with_log[orig], bins=50, color='#93c5fd', edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f'{orig} (goc)', fontsize=12, fontweight='bold', pad=8)
    axes[0, i].set_ylabel('So khach hang', fontsize=10)
    axes[0, i].annotate(f'Skew={rfm_with_log[orig].skew():.2f}',
                        xy=(0.65, 0.85), xycoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))

    # Sau log
    axes[1, i].hist(rfm_with_log[logged], bins=50, color='#86efac', edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'{logged} (log1p)', fontsize=12, fontweight='bold', pad=8)
    axes[1, i].set_ylabel('So khach hang', fontsize=10)
    axes[1, i].annotate(f'Skew={rfm_with_log[logged].skew():.2f}',
                        xy=(0.65, 0.85), xycoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='#f0fdf4', alpha=0.8))

plt.suptitle('So sanh phan phoi truoc va sau Log-Transform', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_log_transform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] rfm_log_transform_comparison.png')

# ====================================================================
# 9. DATA QUALITY CHECKS
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 9: DATA QUALITY CHECKS')
print('-' * 70)

checks, all_pass = validate_feature_data(
    rfm_df=rfm,
    repeat_df=repeat_features,
    basket_df=basket_long,
    cutoff_date=repeat_info['cutoff_date'],
    df_transactions=df_cust,
    audit_df=audit_df,
)

for c in checks:
    print(f'  {c}')

# Kiem tra bo sung
# 11. Du lieu raw van giu nguyen
from src.data_loader import load_raw_data
df_raw_check = load_raw_data()
assert df_raw_check.shape == (541909, 8), f'LOI: Du lieu goc bi thay doi! {df_raw_check.shape}'
print(f'  [PASS] Du lieu raw van nguyen ven: {df_raw_check.shape[0]:,} x {df_raw_check.shape[1]}')

# 12. Kiem tra cac file output
expected_files = [
    PROCESSED_DIR / 'rfm_customer_features.csv',
    PROCESSED_DIR / 'rfm_clustering_features.csv',
    PROCESSED_DIR / 'repeat_purchase_features.csv',
    PROCESSED_DIR / 'association_basket_long.csv',
    TBL_DIR / 'rfm_segment_summary.csv',
    TBL_DIR / 'repeat_purchase_label_audit.csv',
    TBL_DIR / 'repeat_purchase_class_balance.csv',
    TBL_DIR / 'feature_summary.csv',
    TBL_DIR / 'feature_engineering_summary.csv',
    FIG_DIR / 'rfm_distribution.png',
    FIG_DIR / 'rfm_score_distribution.png',
    FIG_DIR / 'rfm_segment_overview.png',
    FIG_DIR / 'repeat_purchase_label_balance.png',
    FIG_DIR / 'rfm_correlation_matrix.png',
    FIG_DIR / 'rfm_log_transform_comparison.png',
]

print('\n  Kiem tra file output:')
for f in expected_files:
    exists = f.exists()
    status = '[OK]' if exists else '[MISSING]'
    print(f'  {status} {f.name}')
    if not exists:
        all_pass = False

# 13. Doc lai cac file CSV de dam bao
for csv_file in [f for f in expected_files if str(f).endswith('.csv')]:
    try:
        test_df = pd.read_csv(csv_file)
        assert len(test_df) > 0, f'{csv_file.name} rong!'
    except Exception as e:
        print(f'  [FAIL] Khong doc duoc {csv_file.name}: {e}')
        all_pass = False

assert all_pass, 'Co check FAIL! Xem lai cac loi o tren.'
print('\n  [OK] TAT CA CHECKS DAU PASS!')

# ====================================================================
# TONG KET
# ====================================================================
print('\n' + '=' * 70)
print('    TONG KET FEATURE ENGINEERING & RFM')
print('=' * 70)
print(f'\n  Input rows             : {n_rows:,}')
print(f'  Customers (input)      : {n_customers:,}')
print(f'  reference_date         : {reference_date}')
print(f'  cutoff_date            : {repeat_info["cutoff_date"]}')
print(f'  RFM features           : {len(rfm):,} khach hang x {len(rfm.columns)} cot')
print(f'  Clustering features    : {len(rfm_cluster):,} khach x {len(rfm_cluster.columns)} cot')
print(f'  Repeat purchase feat   : {len(repeat_features):,} khach x {len(repeat_features.columns)} cot')
print(f'  Label 1 (mua lai)      : {repeat_info["label_1_count"]:,}')
print(f'  Label 0 (khong mua)    : {repeat_info["label_0_count"]:,}')
print(f'  Ty le mua lai          : {repeat_info["repeat_rate"]:.2f}%')
print(f'  Basket invoices        : {basket_stats["unique_invoices"]:,}')
print(f'  Basket products        : {basket_stats["unique_products"]:,}')
print(f'\n  Figures: {FIG_DIR}')
print(f'  Tables : {TBL_DIR}')
print(f'  Data   : {PROCESSED_DIR}')
print('=' * 70)
