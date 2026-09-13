"""
Script chay Data Preparation & Cleaning - tuong duong notebook 02.
Pipeline: Raw -> Standardize -> Flag -> Clean Invoice -> Clean StockCode
           -> Fill Missing -> Remove Invalid -> Dedup -> Outlier -> Save

Chay: .venv\\Scripts\\python.exe notebooks\\run_02_eda_and_cleaning.py
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
    RAW_DATA_CSV, INTERIM_DIR,
    FIGURES_DATA_PREPARATION, TABLES_DATA_PREPARATION,
    NON_PRODUCT_STOCK_CODES
)
from src.data_loader import load_raw_data
from src.preprocessing import (
    standardize_dtypes,
    add_transaction_flags,
    classify_stock_codes,
    fill_missing_descriptions,
    report_missing_values,
    remove_invalid_transactions,
    handle_duplicates,
    detect_outliers,
    create_clean_datasets,
    save_cleaning_report
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style('whitegrid')
sns.set_palette('husl')

print('=' * 70)
print('    02. DATA PREPARATION & CLEANING')
print('=' * 70)

# ====================================================================
# 1. TAI DU LIEU GOC
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 1: TAI DU LIEU GOC')
print('-' * 70)

df_raw = load_raw_data()
raw_info = {
    'rows': df_raw.shape[0],
    'cols': df_raw.shape[1],
    'columns': list(df_raw.columns),
    'invoices': df_raw['InvoiceNo'].nunique(),
    'products': df_raw['StockCode'].nunique(),
    'customers': df_raw['CustomerID'].nunique(),
}

print(f'  Kich thuoc: {raw_info["rows"]:,} dong x {raw_info["cols"]} cot')
print(f'  Hoa don: {raw_info["invoices"]:,}')
print(f'  San pham: {raw_info["products"]:,}')
print(f'  Khach hang: {raw_info["customers"]:,}')

# Luu ban sao de xu ly (KHONG thay doi df_raw)
df = df_raw.copy()

# ====================================================================
# 2. CHUAN HOA KIEU DU LIEU
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 2: CHUAN HOA KIEU DU LIEU')
print('-' * 70)

df = standardize_dtypes(df)

print('  Kieu du lieu sau chuan hoa:')
print(df.dtypes.to_string())
print(f'\n  Cot TotalAmount da tao: min={df["TotalAmount"].min():,.2f}, max={df["TotalAmount"].max():,.2f}')

# Khoang thoi gian
date_min = df['InvoiceDate'].min()
date_max = df['InvoiceDate'].max()
date_range = (date_max - date_min).days
print(f'  Thoi gian: {date_min} -> {date_max} ({date_range} ngay)')

# ====================================================================
# 3. THEM CO GIAO DICH
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 3: THEM CO GIAO DICH')
print('-' * 70)

df = add_transaction_flags(df)

print(f'  IsCancelled (C)      : {df["IsCancelled"].sum():,} dong')
print(f'  IsAdjust (A)         : {df["IsAdjust"].sum():,} dong')
print(f'  IsDuplicate          : {df["IsDuplicate"].sum():,} dong')
print(f'  IsQuantityInvalid    : {df["IsQuantityInvalid"].sum():,} dong')
print(f'  IsPriceInvalid       : {df["IsPriceInvalid"].sum():,} dong')

# ====================================================================
# 4. XU LY INVOICENO (C & A) - THONG KE
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 4: PHAN TICH HOA DON HUY (C) VA ADJUST BAD DEBT (A)')
print('-' * 70)

# Thong ke Invoice C
cancelled_df = df[df['IsCancelled']]
print(f'  Hoa don huy (C):')
print(f'    So dong      : {len(cancelled_df):,}')
print(f'    So hoa don   : {cancelled_df["InvoiceNo"].nunique():,}')
print(f'    TotalAmount  : {cancelled_df["TotalAmount"].sum():,.2f}')

# Thong ke Invoice A
adjust_df = df[df['IsAdjust']]
print(f'  Adjust bad debt (A):')
print(f'    So dong      : {len(adjust_df):,}')
print(f'    So hoa don   : {adjust_df["InvoiceNo"].nunique():,}')
print(f'    TotalAmount  : {adjust_df["TotalAmount"].sum():,.2f}')

# Normal
normal_df = df[~df['IsCancelled'] & ~df['IsAdjust']]
print(f'  Giao dich binh thuong:')
print(f'    So dong      : {len(normal_df):,}')

# Luu bang invoice_cleaning_summary
invoice_summary = pd.DataFrame([
    {
        'InvoiceType': 'Normal (numeric)',
        'RowCount': len(normal_df),
        'UniqueInvoices': normal_df['InvoiceNo'].nunique(),
        'TotalAmount': round(normal_df['TotalAmount'].sum(), 2),
        'Action': 'Giu lai',
        'Reason': 'Giao dich mua hang binh thuong'
    },
    {
        'InvoiceType': 'Cancelled (C)',
        'RowCount': len(cancelled_df),
        'UniqueInvoices': cancelled_df['InvoiceNo'].nunique(),
        'TotalAmount': round(cancelled_df['TotalAmount'].sum(), 2),
        'Action': 'Loai bo',
        'Reason': 'Hoa don huy, Quantity am, khong phai giao dich mua'
    },
    {
        'InvoiceType': 'Adjust bad debt (A)',
        'RowCount': len(adjust_df),
        'UniqueInvoices': adjust_df['InvoiceNo'].nunique(),
        'TotalAmount': round(adjust_df['TotalAmount'].sum(), 2),
        'Action': 'Loai bo',
        'Reason': 'But toan ke toan dieu chinh no xau, UnitPrice cuc lon (+-11,062)'
    }
])
invoice_summary.to_csv(TABLES_DATA_PREPARATION / 'invoice_cleaning_summary.csv', index=False)
print(f'\n  [OK] Luu: {TABLES_DATA_PREPARATION / "invoice_cleaning_summary.csv"}')

# ====================================================================
# 5. PHAN LOAI STOCKCODE
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 5: PHAN LOAI STOCKCODE PHI SAN PHAM')
print('-' * 70)

sc_class = classify_stock_codes(
    df, save_path=TABLES_DATA_PREPARATION / 'stockcode_classification.csv'
)
print(f'  Tong StockCode bat dau bang chu cai: {len(sc_class)}')

for _, row in sc_class.iterrows():
    print(f'    {row["StockCode"]:15s} | {row["RowCount"]:>5,} dong | {row["Category"]:15s} | {row["Action"]}')

print(f'\n  [OK] Luu: {TABLES_DATA_PREPARATION / "stockcode_classification.csv"}')

# ====================================================================
# 6. XU LY MISSING VALUES
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 6: XU LY MISSING VALUES')
print('-' * 70)

# Missing truoc xu ly
missing_before = report_missing_values(df)
print('  Missing truoc xu ly:')
for _, row in missing_before[missing_before['MissingCount'] > 0].iterrows():
    print(f'    {row["Column"]:15s}: {int(row["MissingCount"]):,} ({row["MissingPercent"]:.2f}%)')

# Dien Description thieu
df, desc_filled = fill_missing_descriptions(df)
print(f'\n  Da dien {desc_filled:,} dong Description thieu')

# Missing sau xu ly
missing_after = report_missing_values(
    df, save_path=TABLES_DATA_PREPARATION / 'missing_values_after_cleaning.csv'
)
print('  Missing sau xu ly Description:')
for _, row in missing_after[missing_after['MissingCount'] > 0].iterrows():
    print(f'    {row["Column"]:15s}: {int(row["MissingCount"]):,} ({row["MissingPercent"]:.2f}%)')

print(f'\n  [OK] Luu: {TABLES_DATA_PREPARATION / "missing_values_after_cleaning.csv"}')

# ====================================================================
# 7. LOAI GIAO DICH KHONG HOP LE
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 7: LOAI GIAO DICH KHONG HOP LE')
print('-' * 70)

df_valid, removal_stats = remove_invalid_transactions(df)

print(f'  Raw rows                     : {removal_stats["raw_rows"]:,}')
print(f'  Loai Invoice C (huy)         : -{removal_stats["cancelled_rows"]:,} dong ({removal_stats["cancelled_invoices"]:,} hoa don)')
print(f'  Loai Invoice A (adjust)      : -{removal_stats["adjust_rows"]:,} dong ({removal_stats["adjust_invoices"]:,} hoa don)')
print(f'  Sau loai C & A               : {removal_stats["after_invoice_clean"]:,}')
print(f'  Loai Quantity <= 0           : -{removal_stats["qty_invalid_rows"]:,} dong')
print(f'  Sau loai Quantity invalid     : {removal_stats["after_qty_clean"]:,}')
print(f'  Loai UnitPrice <= 0          : -{removal_stats["price_invalid_rows"]:,} dong')
print(f'  Sau loai Price invalid        : {removal_stats["after_price_clean"]:,}')

# ====================================================================
# 8. XU LY DUPLICATE
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 8: XU LY DUPLICATE')
print('-' * 70)

df_dedup, dup_stats = handle_duplicates(df_valid)

print(f'  Truoc dedup  : {dup_stats["before_dedup"]:,}')
print(f'  Duplicate    : {dup_stats["duplicate_count"]:,} ({dup_stats["duplicate_pct"]:.2f}%)')
print(f'  Sau dedup    : {dup_stats["after_dedup"]:,}')

# ====================================================================
# 9. XU LY STOCKCODE PHI SAN PHAM (cho cleaned_transactions)
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 9: LOAI STOCKCODE PHI SAN PHAM KHOI DU LIEU SACH')
print('-' * 70)

sc_str = df_dedup['StockCode'].astype(str)
non_product_mask = sc_str.isin(NON_PRODUCT_STOCK_CODES)
non_product_count = non_product_mask.sum()

# Loai non-product SC khoi cleaned_transactions
df_cleaned = df_dedup[~non_product_mask].copy()
print(f'  StockCode phi san pham: {non_product_count:,} dong')
print(f'  Sau loai non-product SC: {len(df_cleaned):,}')

# ====================================================================
# 10. PHAT HIEN OUTLIER
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 10: PHAT HIEN OUTLIER (IQR)')
print('-' * 70)

df_cleaned, outlier_summary = detect_outliers(df_cleaned, ['Quantity', 'UnitPrice', 'TotalAmount'])

print('  Outlier theo IQR:')
for _, row in outlier_summary.iterrows():
    print(f'    {row["Column"]:15s}: {int(row["OutlierCount"]):,} dong ({row["OutlierPercent"]:.2f}%) '
          f'| Bounds: [{row["LowerBound"]:.2f}, {row["UpperBound"]:.2f}]')

outlier_summary.to_csv(TABLES_DATA_PREPARATION / 'outlier_summary.csv', index=False)
print(f'\n  [OK] Luu: {TABLES_DATA_PREPARATION / "outlier_summary.csv"}')

# ====================================================================
# 11. TAO CAC TAP DU LIEU SACH
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 11: TAO CAC TAP DU LIEU SACH')
print('-' * 70)

datasets = create_clean_datasets(df_cleaned, NON_PRODUCT_STOCK_CODES)

print(f'  cleaned_transactions   : {datasets["stats"]["cleaned_rows"]:,} dong')
print(f'  customer_transactions  : {datasets["stats"]["customer_rows"]:,} dong')
print(f'  product_transactions   : {datasets["stats"]["product_rows"]:,} dong')
print(f'  Missing CustomerID loai: {datasets["stats"]["missing_customerid_removed"]:,}')
print(f'  Non-product SC loai    : {datasets["stats"]["non_product_removed"]:,}')

# Luu files
datasets['cleaned'].to_csv(INTERIM_DIR / 'cleaned_transactions.csv', index=False)
print(f'\n  [OK] Luu: {INTERIM_DIR / "cleaned_transactions.csv"}')

datasets['customer'].to_csv(INTERIM_DIR / 'customer_transactions.csv', index=False)
print(f'  [OK] Luu: {INTERIM_DIR / "customer_transactions.csv"}')

datasets['product'].to_csv(INTERIM_DIR / 'product_transactions.csv', index=False)
print(f'  [OK] Luu: {INTERIM_DIR / "product_transactions.csv"}')

# ====================================================================
# 12. LUU BAO CAO CLEANING SUMMARY
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 12: LUU BAO CAO CLEANING SUMMARY')
print('-' * 70)

pipeline_stats = [
    {'Stage': '01_raw_data', 'Rows': removal_stats['raw_rows'],
     'RowsRemoved': 0, 'Reason': 'Du lieu goc'},
    {'Stage': '02_remove_cancelled_C', 'Rows': removal_stats['after_invoice_clean'] + removal_stats['adjust_rows'],
     'RowsRemoved': removal_stats['cancelled_rows'], 'Reason': 'Loai hoa don huy (InvoiceNo bat dau bang C)'},
    {'Stage': '03_remove_adjust_A', 'Rows': removal_stats['after_invoice_clean'],
     'RowsRemoved': removal_stats['adjust_rows'], 'Reason': 'Loai adjust bad debt (InvoiceNo bat dau bang A)'},
    {'Stage': '04_remove_qty_invalid', 'Rows': removal_stats['after_qty_clean'],
     'RowsRemoved': removal_stats['qty_invalid_rows'], 'Reason': 'Loai Quantity <= 0 (khong thuoc C/A)'},
    {'Stage': '05_remove_price_invalid', 'Rows': removal_stats['after_price_clean'],
     'RowsRemoved': removal_stats['price_invalid_rows'], 'Reason': 'Loai UnitPrice <= 0'},
    {'Stage': '06_remove_duplicates', 'Rows': dup_stats['after_dedup'],
     'RowsRemoved': dup_stats['duplicate_count'], 'Reason': 'Loai dong trung lap hoan toan'},
    {'Stage': '07_remove_non_product_sc', 'Rows': len(df_cleaned),
     'RowsRemoved': non_product_count, 'Reason': 'Loai StockCode phi san pham (POST, DOT, M...)'},
    {'Stage': '08_final_cleaned', 'Rows': datasets['stats']['cleaned_rows'],
     'RowsRemoved': 0, 'Reason': 'Du lieu sach cuoi cung'},
    {'Stage': '09_customer_transactions', 'Rows': datasets['stats']['customer_rows'],
     'RowsRemoved': datasets['stats']['missing_customerid_removed'],
     'Reason': 'Chi giu giao dich co CustomerID (cho RFM, K-Means)'},
    {'Stage': '10_product_transactions', 'Rows': datasets['stats']['product_rows'],
     'RowsRemoved': datasets['stats']['non_product_removed'],
     'Reason': 'Loai them gift voucher (cho association rules)'},
]

cleaning_report = save_cleaning_report(
    pipeline_stats, TABLES_DATA_PREPARATION / 'cleaning_summary.csv'
)
print(cleaning_report.to_string(index=False))
print(f'\n  [OK] Luu: {TABLES_DATA_PREPARATION / "cleaning_summary.csv"}')

# ====================================================================
# 13. BIEU DO TRUC QUAN HOA
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 13: TAO BIEU DO')
print('-' * 70)

FIG_DIR = FIGURES_DATA_PREPARATION

# --- 13.1 Cleaning Funnel (Waterfall) ---
fig, ax = plt.subplots(figsize=(14, 6))

funnel_data = cleaning_report[cleaning_report['Stage'].str.startswith(('01', '02', '03', '04', '05', '06', '07', '08'))]
stages = [s.split('_', 1)[1].replace('_', ' ').title() for s in funnel_data['Stage']]
rows = funnel_data['Rows'].values.astype(int)

colors = ['#3b82f6'] + ['#ef4444'] * (len(rows) - 2) + ['#22c55e']
bars = ax.bar(range(len(stages)), rows, color=colors, edgecolor='white', width=0.6, alpha=0.9)

ax.set_title('Pipeline Lam sach Du lieu: So dong qua tung buoc', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('So dong', fontsize=12)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, rotation=35, ha='right', fontsize=9.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:,.0f}K'))

for bar, row_count, removed in zip(bars, rows, funnel_data['RowsRemoved'].values.astype(int)):
    label = f'{row_count:,}'
    if removed > 0:
        label += f'\n(-{removed:,})'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rows) * 0.01,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'cleaning_funnel.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] cleaning_funnel.png')

# --- 13.2 Removal Reasons Donut ---
removal_labels = ['Huy (C)', 'Adjust (A)', 'Qty <= 0', 'Price <= 0', 'Duplicate', 'Non-product SC']
removal_values = [
    removal_stats['cancelled_rows'],
    removal_stats['adjust_rows'],
    removal_stats['qty_invalid_rows'],
    removal_stats['price_invalid_rows'],
    dup_stats['duplicate_count'],
    non_product_count,
]

fig, ax = plt.subplots(figsize=(10, 7))
removal_colors = ['#ef4444', '#f97316', '#eab308', '#a855f7', '#6366f1', '#0ea5e9']
wedges, _ = ax.pie(
    removal_values, colors=removal_colors, startangle=90,
    explode=[0.05] * len(removal_values),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)
total_removed = sum(removal_values)
ax.text(0, 0, f'Tong loai\n{total_removed:,}\ndong', ha='center', va='center', fontsize=13, fontweight='bold')
ax.set_title('Ty le dong bi loai theo nguyen nhan', fontsize=14, fontweight='bold', pad=12)

legend_labels = [f'{l}: {v:,} ({v/total_removed*100:.1f}%)' for l, v in zip(removal_labels, removal_values)]
ax.legend(wedges, legend_labels, title='Nguyen nhan', loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig(FIG_DIR / 'removal_reasons_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] removal_reasons_pie.png')

# --- 13.3 Boxplot Quantity Before/After ---
df_raw_std = standardize_dtypes(df_raw.copy())
# Chi lay giao dich binh thuong de so sanh
raw_normal = df_raw_std[~df_raw_std['InvoiceNo'].str.startswith('C') & ~df_raw_std['InvoiceNo'].str.startswith('A')]
raw_qty = raw_normal[(raw_normal['Quantity'] > 0) & (raw_normal['Quantity'] <= 100)]['Quantity']
clean_qty = datasets['cleaned'][(datasets['cleaned']['Quantity'] > 0) & (datasets['cleaned']['Quantity'] <= 100)]['Quantity']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].boxplot(raw_qty.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#93c5fd', edgecolor='#1e40af'),
                medianprops=dict(color='#dc2626', linewidth=2))
axes[0].set_title('Quantity TRUOC lam sach (1-100)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_ylabel('Quantity', fontsize=11)
axes[0].annotate(f'n = {len(raw_qty):,}\nMedian = {raw_qty.median():.0f}\nMean = {raw_qty.mean():.1f}',
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='aliceblue', edgecolor='#1e40af', alpha=0.8), fontsize=10)

axes[1].boxplot(clean_qty.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#86efac', edgecolor='#166534'),
                medianprops=dict(color='#dc2626', linewidth=2))
axes[1].set_title('Quantity SAU lam sach (1-100)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_ylabel('Quantity', fontsize=11)
axes[1].annotate(f'n = {len(clean_qty):,}\nMedian = {clean_qty.median():.0f}\nMean = {clean_qty.mean():.1f}',
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='#f0fdf4', edgecolor='#166534', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / 'boxplot_quantity_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] boxplot_quantity_before_after.png')

# --- 13.4 Boxplot UnitPrice Before/After ---
raw_price = raw_normal[(raw_normal['UnitPrice'] > 0) & (raw_normal['UnitPrice'] <= 20)]['UnitPrice']
clean_price = datasets['cleaned'][(datasets['cleaned']['UnitPrice'] > 0) & (datasets['cleaned']['UnitPrice'] <= 20)]['UnitPrice']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].boxplot(raw_price.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#fde68a', edgecolor='#92400e'),
                medianprops=dict(color='#dc2626', linewidth=2))
axes[0].set_title('UnitPrice TRUOC lam sach (0-20)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_ylabel('UnitPrice (GBP)', fontsize=11)
axes[0].annotate(f'n = {len(raw_price):,}\nMedian = {raw_price.median():.2f}\nMean = {raw_price.mean():.2f}',
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='#fffbeb', edgecolor='#92400e', alpha=0.8), fontsize=10)

axes[1].boxplot(clean_price.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#c4b5fd', edgecolor='#5b21b6'),
                medianprops=dict(color='#dc2626', linewidth=2))
axes[1].set_title('UnitPrice SAU lam sach (0-20)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_ylabel('UnitPrice (GBP)', fontsize=11)
axes[1].annotate(f'n = {len(clean_price):,}\nMedian = {clean_price.median():.2f}\nMean = {clean_price.mean():.2f}',
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='#f5f3ff', edgecolor='#5b21b6', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / 'boxplot_unitprice_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] boxplot_unitprice_before_after.png')

# --- 13.5 Distribution TotalAmount Before/After (log-scale) ---
raw_ta = raw_normal[raw_normal['TotalAmount'] > 0]['TotalAmount']
clean_ta = datasets['cleaned'][datasets['cleaned']['TotalAmount'] > 0]['TotalAmount']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(np.log10(raw_ta), bins=50, color='#93c5fd', edgecolor='white', alpha=0.85)
axes[0].set_title('TotalAmount TRUOC lam sach (Log10)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('log10(TotalAmount)', fontsize=11)
axes[0].set_ylabel('So dong', fontsize=11)
axes[0].annotate(f'n = {len(raw_ta):,}\nMedian = {raw_ta.median():.2f}\nMean = {raw_ta.mean():.2f}',
                 xy=(0.05, 0.78), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='aliceblue', edgecolor='#1e40af', alpha=0.8), fontsize=10)

axes[1].hist(np.log10(clean_ta), bins=50, color='#86efac', edgecolor='white', alpha=0.85)
axes[1].set_title('TotalAmount SAU lam sach (Log10)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('log10(TotalAmount)', fontsize=11)
axes[1].set_ylabel('So dong', fontsize=11)
axes[1].annotate(f'n = {len(clean_ta):,}\nMedian = {clean_ta.median():.2f}\nMean = {clean_ta.mean():.2f}',
                 xy=(0.05, 0.78), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='#f0fdf4', edgecolor='#166534', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / 'dist_totalamount_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] dist_totalamount_before_after.png')

# --- 13.6 Doanh thu theo thang TRUOC/SAU ---
df_raw_std['InvoiceMonth'] = df_raw_std['InvoiceDate'].dt.to_period('M').astype(str)
raw_monthly = (
    raw_normal.assign(InvoiceMonth=raw_normal['InvoiceDate'].dt.to_period('M').astype(str))
    .groupby('InvoiceMonth')['TotalAmount'].sum()
)

clean_data = datasets['cleaned'].copy()
clean_data['InvoiceMonth'] = pd.to_datetime(clean_data['InvoiceDate']).dt.to_period('M').astype(str)
clean_monthly = clean_data.groupby('InvoiceMonth')['TotalAmount'].sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(raw_monthly.index, raw_monthly.values, color='#93c5fd', edgecolor='white', width=0.65)
axes[0].set_title('Doanh thu theo thang TRUOC lam sach', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Thang', fontsize=11)
axes[0].set_ylabel('Doanh thu (GBP)', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:,.0f}K'))

axes[1].bar(clean_monthly.index, clean_monthly.values, color='#86efac', edgecolor='white', width=0.65)
axes[1].set_title('Doanh thu theo thang SAU lam sach', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Thang', fontsize=11)
axes[1].set_ylabel('Doanh thu (GBP)', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:,.0f}K'))

plt.tight_layout()
plt.savefig(FIG_DIR / 'revenue_by_month_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] revenue_by_month_before_after.png')

# --- 13.7 Outlier Detection Boxplot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for i, col in enumerate(['Quantity', 'UnitPrice', 'TotalAmount']):
    vals = datasets['cleaned'][col]
    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    # Cap cho de doc
    plot_vals = vals[vals <= vals.quantile(0.99)]

    axes[i].boxplot(plot_vals.values, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#c4b5fd', edgecolor='#5b21b6'),
                    medianprops=dict(color='#dc2626', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='#f97316', markersize=3, alpha=0.4))
    axes[i].set_title(f'{col}\n(P99 view)', fontsize=12, fontweight='bold', pad=10)
    axes[i].set_ylabel(col, fontsize=10)

    outlier_count = (vals > upper).sum() + (vals < Q1 - 1.5 * IQR).sum()
    axes[i].annotate(f'IQR Outlier: {outlier_count:,}\n({outlier_count/len(vals)*100:.1f}%)',
                     xy=(0.5, 0.92), xycoords='axes fraction', ha='center',
                     bbox=dict(boxstyle='round', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.9),
                     fontsize=10, fontweight='bold')

plt.suptitle('Phat hien Outlier bang IQR (Du lieu da lam sach)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'outlier_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] outlier_detection.png')

# --- 13.8 Missing Values Heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Truoc
raw_missing = df_raw.isnull().sum()
raw_missing = raw_missing[raw_missing > 0]
if len(raw_missing) > 0:
    axes[0].barh(raw_missing.index, raw_missing.values, color='#ef4444', edgecolor='white')
    axes[0].set_title('Missing Values TRUOC lam sach', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('So dong thieu', fontsize=11)
    for j, (col, val) in enumerate(raw_missing.items()):
        axes[0].text(val + raw_missing.max() * 0.02, j, f'{val:,} ({val/len(df_raw)*100:.1f}%)',
                     va='center', fontsize=10, fontweight='bold')
else:
    axes[0].text(0.5, 0.5, 'Khong co missing values', ha='center', va='center', transform=axes[0].transAxes)

# Sau
clean_missing = datasets['cleaned'].isnull().sum()
clean_missing = clean_missing[clean_missing > 0]
if len(clean_missing) > 0:
    axes[1].barh(clean_missing.index, clean_missing.values, color='#f59e0b', edgecolor='white')
    axes[1].set_title('Missing Values SAU lam sach', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('So dong thieu', fontsize=11)
    for j, (col, val) in enumerate(clean_missing.items()):
        axes[1].text(val + clean_missing.max() * 0.02, j, f'{val:,} ({val/len(datasets["cleaned"])*100:.1f}%)',
                     va='center', fontsize=10, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'Khong co missing values', ha='center', va='center', fontsize=14,
                 transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='#d1fae5'))
    axes[1].set_title('Missing Values SAU lam sach', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(FIG_DIR / 'missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] missing_values_heatmap.png')

# ====================================================================
# 14. KIEM TRA DU LIEU GOC KHONG BI THAY DOI
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 14: KIEM TRA DU LIEU GOC KHONG BI THAY DOI')
print('-' * 70)

df_verify = load_raw_data()
assert df_verify.shape[0] == raw_info['rows'], 'LOI: Du lieu goc bi thay doi so dong!'
assert df_verify.shape[1] == raw_info['cols'], 'LOI: Du lieu goc bi thay doi so cot!'
print(f'  [OK] Du lieu goc van nguyen ven: {df_verify.shape[0]:,} x {df_verify.shape[1]}')

# ====================================================================
# 15. KIEM TRA FILE OUTPUT TON TAI
# ====================================================================
print('\n' + '-' * 70)
print('  BUOC 15: KIEM TRA FILE OUTPUT')
print('-' * 70)

expected_files = [
    INTERIM_DIR / 'cleaned_transactions.csv',
    INTERIM_DIR / 'customer_transactions.csv',
    INTERIM_DIR / 'product_transactions.csv',
    TABLES_DATA_PREPARATION / 'invoice_cleaning_summary.csv',
    TABLES_DATA_PREPARATION / 'stockcode_classification.csv',
    TABLES_DATA_PREPARATION / 'missing_values_after_cleaning.csv',
    TABLES_DATA_PREPARATION / 'outlier_summary.csv',
    TABLES_DATA_PREPARATION / 'cleaning_summary.csv',
    FIG_DIR / 'cleaning_funnel.png',
    FIG_DIR / 'removal_reasons_pie.png',
    FIG_DIR / 'boxplot_quantity_before_after.png',
    FIG_DIR / 'boxplot_unitprice_before_after.png',
    FIG_DIR / 'dist_totalamount_before_after.png',
    FIG_DIR / 'revenue_by_month_before_after.png',
    FIG_DIR / 'outlier_detection.png',
    FIG_DIR / 'missing_values_heatmap.png',
]

for f in expected_files:
    exists = f.exists()
    status = '[OK]' if exists else '[MISSING]'
    print(f'  {status} {f.name}')

# ====================================================================
# TONG KET
# ====================================================================
print('\n' + '=' * 70)
print('    TONG KET DATA PREPARATION & CLEANING')
print('=' * 70)
print(f'\n  Raw rows               : {raw_info["rows"]:,}')
print(f'  Cleaned rows           : {datasets["stats"]["cleaned_rows"]:,}')
print(f'  Customer trans rows    : {datasets["stats"]["customer_rows"]:,}')
print(f'  Product trans rows     : {datasets["stats"]["product_rows"]:,}')
print(f'\n  Cancelled (C)          : {removal_stats["cancelled_rows"]:,}')
print(f'  Adjust (A)             : {removal_stats["adjust_rows"]:,}')
print(f'  Quantity <= 0          : {removal_stats["qty_invalid_rows"]:,}')
print(f'  UnitPrice <= 0         : {removal_stats["price_invalid_rows"]:,}')
print(f'  Duplicate              : {dup_stats["duplicate_count"]:,}')
print(f'  Non-product SC         : {non_product_count:,}')
print(f'  Missing CustomerID     : {datasets["stats"]["missing_customerid_removed"]:,}')
print(f'  Description filled     : {desc_filled:,}')
print(f'\n  Figures: {FIG_DIR}')
print(f'  Tables : {TABLES_DATA_PREPARATION}')
print(f'  Data   : {INTERIM_DIR}')
print('=' * 70)
