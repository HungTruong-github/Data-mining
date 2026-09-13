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

from src.config import RAW_DATA_CSV, FIGURES_DIR, TABLES_DIR
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

raw_summary.to_csv(TABLES_DIR / 'raw_summary.csv', index=False)
print(f'\n[OK] Luu: {TABLES_DIR / "raw_summary.csv"}')

desc_stats.to_csv(TABLES_DIR / 'descriptive_statistics.csv')
print(f'[OK] Luu: {TABLES_DIR / "descriptive_statistics.csv"}')

# ====================================================================
# 12. BIEU DO
# ====================================================================
print('\n-- Tao bieu do --')

# 12.1 Phan phoi Quantity
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(df['Quantity'], bins=100, color='#4C72B0', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribution of Quantity (all)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Quantity')
axes[0].set_ylabel('Count')
qty_filtered = df[(df['Quantity'] > 0) & (df['Quantity'] <= 50)]['Quantity']
axes[1].hist(qty_filtered, bins=50, color='#55A868', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribution of Quantity (1-50)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Quantity')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dist_quantity.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] dist_quantity.png')

# 12.2 Phan phoi UnitPrice
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(df['UnitPrice'], bins=100, color='#C44E52', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribution of UnitPrice (all)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('UnitPrice')
axes[0].set_ylabel('Count')
price_filtered = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] <= 10)]['UnitPrice']
axes[1].hist(price_filtered, bins=50, color='#DD8452', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribution of UnitPrice (0-10)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('UnitPrice')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dist_unitprice.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] dist_unitprice.png')

# 12.3 Doanh thu theo thang (loai C va A)
df_valid = df[
    (~df['IsCancelled']) &
    (~df['IsAdjust']) &
    (df['Quantity'] > 0) &
    (df['UnitPrice'] > 0)
].copy()
df_valid['InvoiceMonth'] = df_valid['InvoiceDate'].dt.to_period('M')
monthly_revenue = df_valid.groupby('InvoiceMonth')['TotalAmount'].sum()

fig, ax = plt.subplots(figsize=(14, 5))
monthly_revenue.plot(kind='bar', ax=ax, color='#4C72B0', edgecolor='white', alpha=0.85)
ax.set_title('Monthly Revenue (excl. cancelled & adjust)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(monthly_revenue.values):
    ax.text(i, v + monthly_revenue.max() * 0.01, f'{v/1000:.0f}K',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'revenue_by_month.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] revenue_by_month.png')

# 12.4 Top 10 san pham
top_products = (
    df_valid.groupby(['StockCode', 'Description'])['TotalAmount']
    .sum().sort_values(ascending=False).head(10).reset_index()
)
top_products['Label'] = top_products['Description'].str[:30]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(top_products['Label'][::-1], top_products['TotalAmount'][::-1],
               color=sns.color_palette('viridis', 10)[::-1], edgecolor='white')
ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
ax.set_xlabel('Revenue')
for bar in bars:
    w = bar.get_width()
    ax.text(w + top_products['TotalAmount'].max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{w:,.0f}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'top10_products_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] top10_products_revenue.png')

# 12.5 Top quoc gia
country_revenue = df_valid.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
top_countries = country_revenue.head(15)

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1],
               color=sns.color_palette('coolwarm', 15)[::-1], edgecolor='white')
ax.set_title('Top 15 Countries by Revenue', fontsize=14, fontweight='bold')
ax.set_xlabel('Revenue')
for bar in bars:
    w = bar.get_width()
    ax.text(w + top_countries.max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{w:,.0f}', ha='left', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'top_countries_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] top_countries_revenue.png')

# 12.6 Phan bo loai InvoiceNo
invoice_type_counts = pd.Series({
    'Normal': len(numeric_invoices),
    'Cancelled (C)': len(cancelled_invoices_df),
    'Adjust (A)': len(adjust_invoices)
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#55A868', '#C44E52', '#DD8452']
invoice_type_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title('Rows by Invoice Type', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(invoice_type_counts.values):
    axes[0].text(i, v + len(df) * 0.005, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(invoice_type_counts.values,
            labels=[f'{k}\n({v:,})' for k, v in invoice_type_counts.items()],
            colors=colors, autopct='%1.2f%%', startangle=90)
axes[1].set_title('Invoice Type Ratio', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'invoice_type_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] invoice_type_distribution.png')

# ====================================================================
# TONG KET
# ====================================================================
print('\n' + '=' * 60)
print('    [OK] DATA UNDERSTANDING (REVISED) HOAN TAT')
print('=' * 60)
