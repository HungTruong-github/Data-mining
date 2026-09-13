"""
Script chay Data Understanding - tuong duong notebook 01.
Kiem tra ky luong tat ca cac loai InvoiceNo, StockCode dac biet,
adjust bad debt, va cac van de du lieu.
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
    RAW_DATA_CSV, FIGURES_DIR, TABLES_DIR,
    FIGURES_DATA_UNDERSTANDING, TABLES_DATA_UNDERSTANDING
)
from src.data_loader import load_raw_data

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')
sns.set_palette('husl')

print('=' * 60)
print('    01. DATA UNDERSTANDING (REVISED)')
print('=' * 60)

# ====================================================================
# 1. TAI DU LIEU
# ====================================================================
df = load_raw_data()
print(f'\nKich thuoc dataset: {df.shape[0]:,} dong x {df.shape[1]} cot')
print(f'Cot: {list(df.columns)}')
print(f'\n-- 5 dong dau --')
print(df.head().to_string())

# Chuyen InvoiceDate sang datetime
if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print('\n[OK] Da chuyen InvoiceDate sang datetime')

# Dam bao InvoiceNo la string
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

print(f'\n-- Kieu du lieu --')
print(df.dtypes)

# ====================================================================
# 2. MISSING VALUES
# ====================================================================
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing Count', ascending=False)
print(f'\n-- Gia tri thieu --')
print(missing_df)
print(f'\nTong dong co it nhat 1 gia tri thieu: {df.isnull().any(axis=1).sum():,}')

# ====================================================================
# 3. DUPLICATES
# ====================================================================
dup_count = df.duplicated().sum()
dup_pct = dup_count / len(df) * 100
print(f'\n-- Dong trung lap --')
print(f'So dong trung: {dup_count:,} ({dup_pct:.2f}%)')

# ====================================================================
# 4. UNIQUE VALUES
# ====================================================================
print(f'\n-- So luong unique values --')
print(f'  InvoiceNo  : {df["InvoiceNo"].nunique():,}')
print(f'  StockCode  : {df["StockCode"].nunique():,}')
print(f'  Description: {df["Description"].nunique():,}')
print(f'  CustomerID : {df["CustomerID"].nunique():,}')
print(f'  Country    : {df["Country"].nunique():,}')

# ====================================================================
# 5. KHOANG THOI GIAN
# ====================================================================
date_min = df['InvoiceDate'].min()
date_max = df['InvoiceDate'].max()
date_range = (date_max - date_min).days
print(f'\n-- Khoang thoi gian --')
print(f'  Tu: {date_min}')
print(f'  Den: {date_max}')
print(f'  Tong: {date_range} ngay')

# ====================================================================
# 6. PHAN LOAI INVOICENO THEO KY TU DAU
# ====================================================================
df['InvoicePrefix'] = df['InvoiceNo'].str[0]
print(f'\n-- Phan bo ky tu dau cua InvoiceNo --')
prefix_counts = df['InvoicePrefix'].value_counts()
print(prefix_counts.to_string())

numeric_invoices = df[df['InvoiceNo'].str.match(r'^\d')]
cancelled_invoices_df = df[df['InvoiceNo'].str.startswith('C')]
adjust_invoices = df[df['InvoiceNo'].str.startswith('A')]

print(f'\nGiao dich binh thuong (so): {len(numeric_invoices):,} dong')
print(f'Hoa don huy (C)           : {len(cancelled_invoices_df):,} dong')
print(f'Adjust bad debt (A)       : {len(adjust_invoices):,} dong')

# -- 6.1 Hoa don huy (C) --
df['IsCancelled'] = df['InvoiceNo'].str.startswith('C')
cancelled_count = df['IsCancelled'].sum()
cancelled_unique = df.loc[df['IsCancelled'], 'InvoiceNo'].nunique()
total_invoices = df['InvoiceNo'].nunique()

print(f'\n-- Hoa don huy (C) --')
print(f'  Dong huy: {cancelled_count:,} ({cancelled_count/len(df)*100:.2f}%)')
print(f'  Hoa don huy unique: {cancelled_unique:,} / {total_invoices:,}')

# -- 6.2 Adjust bad debt (A) --
df['IsAdjust'] = df['InvoiceNo'].str.startswith('A')
adjust_count = df['IsAdjust'].sum()
adjust_unique = df.loc[df['IsAdjust'], 'InvoiceNo'].nunique()

print(f'\n-- Adjust bad debt (A) --')
print(f'  Dong adjust: {adjust_count:,}')
print(f'  InvoiceNo unique: {adjust_unique:,}')

if adjust_count > 0:
    print(f'\n  Chi tiet tat ca dong Adjust bad debt:')
    print(adjust_invoices[['InvoiceNo', 'StockCode', 'Description',
                           'Quantity', 'UnitPrice', 'CustomerID']].to_string())
    adj_total = (adjust_invoices['Quantity'] * adjust_invoices['UnitPrice']).sum()
    print(f'\n  Tong TotalAmount cua Adjust: {adj_total:,.2f}')
    print(f'  UnitPrice abs max: {adjust_invoices["UnitPrice"].abs().max():,.2f}')

# ====================================================================
# 7. STOCKCODE PHI SAN PHAM
# ====================================================================
sc_str = df['StockCode'].astype(str)
alpha_mask = sc_str.str.match(r'^[A-Za-z]')
special_sc = df[alpha_mask]

print(f'\n-- StockCode bat dau bang chu cai --')
print(f'Tong dong: {len(special_sc):,} ({len(special_sc)/len(df)*100:.2f}%)')

NON_PRODUCT_CODES = {
    'POST': 'Postage',
    'DOT': 'Dotcom Postage',
    'M': 'Manual adjustment',
    'C2': 'Carriage',
    'D': 'Discount',
    'S': 'Samples',
    'BANK CHARGES': 'Bank Charges',
    'AMAZONFEE': 'Amazon Fee',
    'CRUK': 'CRUK Commission',
    'B': 'Bad debt adjust',
    'PADS': 'Pads',
    'm': 'Manual (lowercase)',
}

print(f'\n  Phan loai StockCode phi san pham:')
for code, desc in NON_PRODUCT_CODES.items():
    count = len(df[df['StockCode'] == code])
    if count > 0:
        print(f'    {code:15s}: {count:>5,} dong - {desc}')

# ====================================================================
# 8. DESCRIPTION CHUA "ADJUST" HOAC "BAD DEBT"
# ====================================================================
desc_lower = df['Description'].fillna('').str.lower()
adjust_desc_mask = desc_lower.str.contains('adjust') | desc_lower.str.contains('bad debt')
adjust_desc_rows = df[adjust_desc_mask]

print(f'\n-- Dong co Description chua "adjust" hoac "bad debt" --')
print(f'Tong: {len(adjust_desc_rows):,} dong')

if len(adjust_desc_rows) > 0:
    zero_price = (adjust_desc_rows['UnitPrice'] == 0).sum()
    nonzero_price = (adjust_desc_rows['UnitPrice'] != 0).sum()
    print(f'  UnitPrice = 0 : {zero_price} dong (dieu chinh so luong)')
    print(f'  UnitPrice != 0: {nonzero_price} dong (anh huong doanh thu)')
    print(f'  Tat ca thieu CustomerID: {adjust_desc_rows["CustomerID"].isnull().all()}')

# ====================================================================
# 9. QUANTITY <= 0, UNITPRICE <= 0
# ====================================================================
qty_neg = (df['Quantity'] <= 0).sum()
price_neg = (df['UnitPrice'] <= 0).sum()

print(f'\n-- Gia tri bat thuong --')
print(f'  Quantity <= 0 : {qty_neg:,}')
print(f'  UnitPrice <= 0: {price_neg:,}')

qty_neg_cancelled = ((df['Quantity'] <= 0) & df['IsCancelled']).sum()
qty_neg_adjust = ((df['Quantity'] <= 0) & df['IsAdjust']).sum()
qty_neg_other = qty_neg - qty_neg_cancelled - qty_neg_adjust
print(f'  Quantity <= 0 VA C : {qty_neg_cancelled:,}')
print(f'  Quantity <= 0 VA A : {qty_neg_adjust:,}')
print(f'  Quantity <= 0 KHAC : {qty_neg_other:,}')

# ====================================================================
# 10. TOTALAMOUNT VA THONG KE MO TA
# ====================================================================
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

print(f'\n-- TotalAmount --')
print(f'  Min: {df["TotalAmount"].min():,.2f}')
print(f'  Max: {df["TotalAmount"].max():,.2f}')
print(f'  Mean: {df["TotalAmount"].mean():,.2f}')
print(f'  Tong: {df["TotalAmount"].sum():,.2f}')

desc_stats = df.describe()
print(f'\n-- Thong ke mo ta --')
print(desc_stats.to_string())

# ====================================================================
# 11. LUU TABLES
# ====================================================================
raw_summary = pd.DataFrame({
    'Metric': [
        'Total Rows', 'Total Columns',
        'Unique InvoiceNo', 'Unique StockCode', 'Unique CustomerID', 'Unique Country',
        'Date Range Start', 'Date Range End', 'Date Range Days',
        'Missing CustomerID', 'Missing CustomerID %',
        'Missing Description', 'Missing Description %',
        'Duplicate Rows', 'Duplicate Rows %',
        'Cancelled Lines (C)', 'Cancelled Invoices (C)',
        'Adjust Bad Debt Lines (A)', 'Adjust Bad Debt Invoices (A)',
        'Adjust/Bad Debt Description Lines',
        'Non-Product StockCode Lines',
        'Quantity <= 0 (total)', 'Quantity <= 0 (not C, not A)',
        'UnitPrice <= 0',
        'Total Revenue (all data)', 'Mean TotalAmount', 'Median TotalAmount'
    ],
    'Value': [
        df.shape[0], df.shape[1],
        df['InvoiceNo'].nunique(), df['StockCode'].nunique(),
        df['CustomerID'].nunique(), df['Country'].nunique(),
        str(date_min), str(date_max), date_range,
        df['CustomerID'].isnull().sum(),
        f"{df['CustomerID'].isnull().sum()/len(df)*100:.2f}%",
        df['Description'].isnull().sum(),
        f"{df['Description'].isnull().sum()/len(df)*100:.2f}%",
        dup_count, f"{dup_pct:.2f}%",
        cancelled_count, cancelled_unique,
        adjust_count, adjust_unique,
        len(adjust_desc_rows),
        len(special_sc),
        qty_neg, qty_neg_other,
        price_neg,
        f"{df['TotalAmount'].sum():,.2f}",
        f"{df['TotalAmount'].mean():,.2f}",
        f"{df['TotalAmount'].median():,.2f}"
    ]
})

# Luu bang vao thu muc con outputs/tables/data_understanding/
raw_summary.to_csv(TABLES_DATA_UNDERSTANDING / 'raw_summary.csv', index=False)
print(f'\n[OK] Luu: {TABLES_DATA_UNDERSTANDING / "raw_summary.csv"}')

desc_stats.to_csv(TABLES_DATA_UNDERSTANDING / 'descriptive_statistics.csv')
print(f'[OK] Luu: {TABLES_DATA_UNDERSTANDING / "descriptive_statistics.csv"}')

# ====================================================================
# 12. BIEU DO (RO RANG, KHONG BI EP, LUXURY STYLING)
# ====================================================================
print('\n-- Tao bieu do ro rang, khong bi ep vao outputs/figures/data_understanding/ --')

# 12.1 Phan phoi Quantity
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

qty_normal = df[(df['Quantity'] > 0) & (df['Quantity'] <= 50)]['Quantity']
pct_normal = len(qty_normal) / len(df) * 100
mean_qty = qty_normal.mean()
median_qty = qty_normal.median()

axes[0].hist(qty_normal, bins=50, color='#2b5c8f', edgecolor='white', alpha=0.85)
axes[0].axvline(median_qty, color='#d95f02', linestyle='--', linewidth=2, label=f'Median: {median_qty:.0f}')
axes[0].axvline(mean_qty, color='#e7298a', linestyle=':', linewidth=2, label=f'Mean: {mean_qty:.1f}')
axes[0].set_title('Phan phoi Quantity thong thuong (1 - 50 don vi)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Quantity (So luong san pham / dong)', fontsize=11)
axes[0].set_ylabel('So luong dong', fontsize=11)
axes[0].legend(loc='upper right', frameon=True)
axes[0].annotate(
    f'Chiem {pct_normal:.1f}% tong so dong\nQ1=1, Median=3, Q3=10',
    xy=(0.60, 0.65), xycoords='axes fraction',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', edgecolor='#2b5c8f', alpha=0.8),
    fontsize=9.5
)

# Toan bo Quantity tren thang Log10 (Outliers & Cancellation)
qty_pos = df[df['Quantity'] > 0]['Quantity']
qty_neg_vals = df[df['Quantity'] < 0]['Quantity'].abs()

axes[1].hist(
    np.log10(qty_pos), bins=35, color='#2ca02c', edgecolor='white', alpha=0.7,
    label=f'Giao dich mua (>0): {len(qty_pos):,} dong'
)
axes[1].hist(
    np.log10(qty_neg_vals), bins=35, color='#d62728', edgecolor='white', alpha=0.7,
    label=f'Huy/Am (|Quantity|<0): {len(qty_neg_vals):,} dong'
)
axes[1].set_title('Toan canh phan phoi Quantity (Thang Log10)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('log10(|Quantity|)  [10^0=1, 10^1=10, 10^2=100, 10^3=1k, 10^4=10k, 80k+]', fontsize=10.5)
axes[1].set_ylabel('So luong dong', fontsize=11)
axes[1].set_xticks([0, 1, 2, 3, 4, 5])
axes[1].set_xticklabels(['1', '10', '100', '1,000', '10,000', '80,000+'])
axes[1].legend(loc='upper right', frameon=True)
axes[1].annotate(
    f'Cuc dai: +{df["Quantity"].max():,}\nCuc tieu: {df["Quantity"].min():,}',
    xy=(0.05, 0.78), xycoords='axes fraction',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff2f2', edgecolor='#d62728', alpha=0.8),
    fontsize=9.5
)

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'dist_quantity.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] dist_quantity.png')

# 12.2 Phan phoi UnitPrice
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

price_normal = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] <= 15)]['UnitPrice']
pct_price_normal = len(price_normal) / len(df) * 100
mean_price = price_normal.mean()
median_price = price_normal.median()

axes[0].hist(price_normal, bins=50, color='#d95f02', edgecolor='white', alpha=0.85)
axes[0].axvline(median_price, color='#1f77b4', linestyle='--', linewidth=2, label=f'Median: £{median_price:.2f}')
axes[0].axvline(mean_price, color='#2ca02c', linestyle=':', linewidth=2, label=f'Mean: £{mean_price:.2f}')
axes[0].set_title('Phan phoi UnitPrice thong thuong (£0 - £15)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('UnitPrice (£)', fontsize=11)
axes[0].set_ylabel('So luong dong', fontsize=11)
axes[0].legend(loc='upper right', frameon=True)
axes[0].annotate(
    f'Chiem {pct_price_normal:.1f}% tong du lieu\n99% gia < £15\nUnitPrice=0: {(df["UnitPrice"] == 0).sum():,} dong',
    xy=(0.55, 0.60), xycoords='axes fraction',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff7ec', edgecolor='#d95f02', alpha=0.8),
    fontsize=9.5
)

valid_prices = df[df['UnitPrice'] > 0]['UnitPrice']
axes[1].hist(np.log10(valid_prices), bins=45, color='#7570b3', edgecolor='white', alpha=0.85)
axes[1].set_title('Toan canh phan phoi UnitPrice (Thang Log10)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('log10(UnitPrice)  [£0.01 den £38,970]', fontsize=11)
axes[1].set_ylabel('So luong dong', fontsize=11)
axes[1].set_xticks([-2, -1, 0, 1, 2, 3, 4])
axes[1].set_xticklabels(['£0.01', '£0.10', '£1', '£10', '£100', '£1,000', '£10,000+'])
axes[1].annotate(
    f'Max: £{df["UnitPrice"].max():,.2f} (Manual)\nMin: £{df["UnitPrice"].min():,.2f} (Adjust bad debt)\nUnitPrice <= 0: {(df["UnitPrice"] <= 0).sum():,} dong',
    xy=(0.05, 0.75), xycoords='axes fraction',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3eef8', edgecolor='#7570b3', alpha=0.8),
    fontsize=9.5
)

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'dist_unitprice.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] dist_unitprice.png')

# 12.3 Doanh thu theo thang (khong bi ep, x-axis categorical)
df_valid = df[
    (~df['IsCancelled']) &
    (~df['IsAdjust']) &
    (df['Quantity'] > 0) &
    (df['UnitPrice'] > 0)
].copy()
df_valid['InvoiceMonth'] = df_valid['InvoiceDate'].dt.to_period('M').astype(str)
monthly_revenue = df_valid.groupby('InvoiceMonth')['TotalAmount'].sum()

fig, ax = plt.subplots(figsize=(13, 6))
bar_colors = ['#3b82f6' if m != '2011-12' else '#f59e0b' for m in monthly_revenue.index]
bars = ax.bar(monthly_revenue.index, monthly_revenue.values, color=bar_colors, edgecolor='white', width=0.65, alpha=0.9)

ax.set_title('Doanh thu theo thang (Giao dich hop le, loai tru Huy & No xau)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Thang (YYYY-MM)', fontsize=12)
ax.set_ylabel('Doanh thu (£)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x*1e-3:,.0f}K' if x < 1e6 else f'£{x*1e-6:,.2f}M'))
ax.set_ylim(0, monthly_revenue.max() * 1.15)

for bar in bars:
    height = bar.get_height()
    label = f'£{height/1e3:.0f}K' if height < 1e6 else f'£{height/1e6:.2f}M'
    ax.text(
        bar.get_x() + bar.get_width() / 2, height + monthly_revenue.max() * 0.015,
        label, ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

ax.annotate(
    '* Thang 12/2011 chi co du lieu den ngay 09/12 (chua du ca thang)',
    xy=(0.02, 0.92), xycoords='axes fraction',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbeb', edgecolor='#f59e0b', alpha=0.9),
    fontsize=9.5, fontstyle='italic'
)

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'revenue_by_month.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] revenue_by_month.png')

# 12.4 Top 10 san pham
top_products = (
    df_valid.groupby(['StockCode', 'Description'])['TotalAmount']
    .sum().sort_values(ascending=False).head(10).reset_index()
)
top_products['Label'] = top_products.apply(
    lambda r: f"{r['StockCode']} - {str(r['Description'])[:32]}" if len(str(r['Description'])) > 32 else f"{r['StockCode']} - {r['Description']}",
    axis=1
)

fig, ax = plt.subplots(figsize=(13, 6.5))
colors = sns.color_palette('mako', 10)[::-1]
bars = ax.barh(top_products['Label'][::-1], top_products['TotalAmount'][::-1], color=colors, edgecolor='white', height=0.65)

ax.set_title('Top 10 San pham co Doanh thu Cao nhat', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Tong Doanh thu (£)', fontsize=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x*1e-3:,.0f}K'))
ax.set_xlim(0, top_products['TotalAmount'].max() * 1.15)

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + top_products['TotalAmount'].max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f'£{w:,.0f}', ha='left', va='center', fontsize=9.5, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'top10_products_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] top10_products_revenue.png')

# 12.5 Top quoc gia theo doanh thu (UK vs International)
country_revenue = df_valid.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
top_countries_all = country_revenue.head(10)
non_uk_countries = country_revenue.drop('United Kingdom', errors='ignore').head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

bar_c1 = ['#1f4e79' if c == 'United Kingdom' else '#5c82a6' for c in top_countries_all.index]
bars1 = axes[0].barh(top_countries_all.index[::-1], top_countries_all.values[::-1], color=bar_c1[::-1], edgecolor='white', height=0.65)
axes[0].set_title('Top 10 Quoc gia (Bao gom UK)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Doanh thu (£)', fontsize=11)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x*1e-6:,.1f}M' if x >= 1e6 else f'£{x*1e-3:,.0f}K'))
axes[0].set_xlim(0, top_countries_all.max() * 1.18)

total_rev = df_valid['TotalAmount'].sum()
for bar in bars1:
    w = bar.get_width()
    pct = w / total_rev * 100
    axes[0].text(
        w + top_countries_all.max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f'£{w/1e3:,.0f}K ({pct:.1f}%)' if w < 1e6 else f'£{w/1e6:,.2f}M ({pct:.1f}%)',
        ha='left', va='center', fontsize=9, fontweight='bold'
    )

bars2 = axes[1].barh(
    non_uk_countries.index[::-1], non_uk_countries.values[::-1],
    color=sns.color_palette('viridis', 10)[::-1], edgecolor='white', height=0.65
)
axes[1].set_title('Top 10 Quoc gia Quoc te (Ngoai tru UK)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Doanh thu (£)', fontsize=11)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x*1e-3:,.0f}K'))
axes[1].set_xlim(0, non_uk_countries.max() * 1.18)

non_uk_total = country_revenue.drop('United Kingdom', errors='ignore').sum()
for bar in bars2:
    w = bar.get_width()
    pct = w / non_uk_total * 100
    axes[1].text(
        w + non_uk_countries.max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f'£{w:,.0f} ({pct:.1f}%)',
        ha='left', va='center', fontsize=9, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'top_countries_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] top_countries_revenue.png')

# 12.6 Phan bo loai InvoiceNo (Log-scale Bar + Donut Chart voi Legend rieng)
invoice_type_counts = pd.Series({
    'Binh thuong (numeric)': len(numeric_invoices),
    'Huy giao dich (C)': len(cancelled_invoices_df),
    'Adjust bad debt (A)': len(adjust_invoices)
})

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
inv_colors = ['#2ca02c', '#d62728', '#ff7f0e']

# Subplot 1: Bar chart voi thang Log (hien thi ro ca 3 loai)
bars = axes[0].bar(invoice_type_counts.index, invoice_type_counts.values, color=inv_colors, edgecolor='white', width=0.55)
axes[0].set_yscale('log')
axes[0].set_title('So dong theo loai InvoiceNo (Thang Logarit)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_ylabel('So dong (log scale)', fontsize=11)
axes[0].set_ylim(1, len(df) * 3)

for bar, count in zip(bars, invoice_type_counts.values):
    pct = count / len(df) * 100
    lbl = f'{count:,}\n({pct:.2f}%)' if pct >= 0.01 else f'{count:,}\n({pct:.4f}%)'
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.35,
        lbl, ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

# Subplot 2: Donut chart voi Legend rieng biet (khong chong de text)
wedges, _ = axes[1].pie(
    invoice_type_counts.values,
    colors=inv_colors,
    startangle=90,
    explode=(0, 0.12, 0.28),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)

axes[1].text(0, 0, f'Tong cong\n{len(df):,}\ndong', ha='center', va='center', fontsize=12, fontweight='bold')
axes[1].set_title('Ty le thanh phan loai hoa don', fontsize=13, fontweight='bold', pad=10)

legend_labels = [
    f'{k}: {v:,} ({v/len(df)*100:.2f}%)' if v/len(df)*100 >= 0.01 else f'{k}: {v:,} ({v/len(df)*100:.4f}%)'
    for k, v in invoice_type_counts.items()
]
axes[1].legend(
    wedges, legend_labels,
    title='Loai hoa don',
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fontsize=10,
    title_fontsize=11,
    frameon=True
)

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'invoice_type_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] invoice_type_distribution.png')

# ====================================================================
# TONG KET
# ====================================================================
print('\n' + '=' * 60)
print('    [OK] DATA UNDERSTANDING (REVISED) HOAN TAT')
print(f'    Figures: {FIGURES_DATA_UNDERSTANDING}')
print(f'    Tables : {TABLES_DATA_UNDERSTANDING}')
print('=' * 60)
