# Review Notebook: 01_data_understanding.ipynb
*Source Path: `D:/Project/Data-mininng/notebooks/01_data_understanding.ipynb`*

---

# 01. Data Understanding — Online Retail Dataset

## Mục đích
Khám phá cấu trúc, chất lượng và đặc điểm chính của dataset gốc **trước khi** tiến hành tiền xử lý.

**Câu hỏi chính:**
- Dữ liệu có bao nhiêu giao dịch, sản phẩm, khách hàng?
- Tỷ lệ missing và duplicate như thế nào?
- Có những loại giao dịch đặc biệt nào (hủy, điều chỉnh, phi sản phẩm)?
- Phân phối Quantity, UnitPrice, TotalAmount ra sao?

**Nguồn dữ liệu:** UCI Online Retail Dataset — UK-based non-store online retail, 2010-12-01 → 2011-12-09.

```python
# [Cell 1 - Execution Count: 1]
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    get_raw_data_path, FIGURES_DIR, TABLES_DIR,
    FIGURES_DATA_UNDERSTANDING, TABLES_DATA_UNDERSTANDING
)
from src.data_loader import load_raw_data

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

%matplotlib inline
print("Setup complete.")
```

**Output (stdout):**
```text
Setup complete.
```

## 1. Load dữ liệu gốc

```python
# [Cell 3 - Execution Count: 2]
raw_path = get_raw_data_path()
df = load_raw_data(raw_path)
print(f"Dataset: {df.shape[0]:,} dòng × {df.shape[1]} cột")
print(f"File: {raw_path.name}")
print(f"\nCác cột: {list(df.columns)}")
df.head()
```

**Output (stdout):**
```text
[OK] Da tai du lieu: 541,909 dong x 8 cot
     File: Online Retail.xlsx
Dataset: 541,909 dòng × 8 cột
File: Online Retail.xlsx

Các cột: ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
```

**Result Display:**
```text
InvoiceNo StockCode                          Description  Quantity  \
0    536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6   
1    536365     71053                  WHITE METAL LANTERN         6   
2    536365    84406B       CREAM CUPID HEARTS COAT HANGER         8   
3    536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6   
4    536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6   

          InvoiceDate  UnitPrice  CustomerID         Country  
0 2010-12-01 08:26:00       2.55   17,850.00  United Kingdom  
1 2010-12-01 08:26:00       3.39   17,850.00  United Kingdom  
2 2010-12-01 08:26:00       2.75   17,850.00  United Kingdom  
3 2010-12-01 08:26:00       3.39   17,850.00  United Kingdom  
4 2010-12-01 08:26:00       3.39   17,850.00  United Kingdom
```

## 2. Kiểu dữ liệu và thống kê cơ bản

```python
# [Cell 5 - Execution Count: 3]
print("Kiểu dữ liệu:")
print(df.dtypes)
print(f"\nKhoảng thời gian: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"Tổng: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} ngày")

print(f"\nSố lượng unique:")
for col in ['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Country']:
    print(f"  {col:15s}: {df[col].nunique():,}")
```

**Output (stdout):**
```text
Kiểu dữ liệu:
InvoiceNo              object
StockCode              object
Description            object
Quantity                int64
InvoiceDate    datetime64[us]
UnitPrice             float64
CustomerID            float64
Country                   str
dtype: object

Khoảng thời gian: 2010-12-01 08:26:00 → 2011-12-09 12:50:00
Tổng: 373 ngày

Số lượng unique:
  InvoiceNo      : 25,900
```

**Output (stdout):**
```text
StockCode      : 4,070
  Description    : 4,223
  CustomerID     : 4,372
  Country        : 38
```

```python
# [Cell 6 - Execution Count: 4]
# Thống kê mô tả cho các cột số
desc_stats = df.describe()
print("Thống kê mô tả:")
desc_stats
```

**Output (stdout):**
```text
Thống kê mô tả:
```

**Result Display:**
```text
Quantity                 InvoiceDate  UnitPrice  CustomerID
count 541,909.00                      541909 541,909.00  406,829.00
mean        9.55  2011-07-04 13:34:57.156386       4.61   15,287.69
min   -80,995.00         2010-12-01 08:26:00 -11,062.06   12,346.00
25%         1.00         2011-03-28 11:34:00       1.25   13,953.00
50%         3.00         2011-07-19 17:17:00       2.08   15,152.00
75%        10.00         2011-10-19 11:27:00       4.13   16,791.00
max    80,995.00         2011-12-09 12:50:00  38,970.00   18,287.00
std       218.08                         NaN      96.76    1,713.60
```

## 3. Missing Values

Phân tích giá trị thiếu theo từng cột.

```python
# [Cell 8 - Execution Count: 5]
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing Count', ascending=False)
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(f"Tổng dòng có ít nhất 1 giá trị thiếu: {df.isnull().any(axis=1).sum():,}")
print(f"\nCustomerID thiếu: {missing['CustomerID']:,} ({missing_pct['CustomerID']:.2f}%)")
print(f"Description thiếu: {missing['Description']:,} ({missing_pct['Description']:.2f}%)")
missing_df
```

**Output (stdout):**
```text
Tổng dòng có ít nhất 1 giá trị thiếu: 135,080

CustomerID thiếu: 135,080 (24.93%)
Description thiếu: 1,454 (0.27%)
```

**Result Display:**
```text
Missing Count  Missing %
CustomerID          135080      24.93
Description           1454       0.27
```

## 4. Duplicate Analysis

```python
# [Cell 10 - Execution Count: 6]
n_dup = df.duplicated().sum()
print(f"Số dòng trùng lặp (trên tất cả 8 cột gốc): {n_dup:,} ({n_dup/len(df)*100:.2f}%)")
if n_dup > 0:
    print("\nVí dụ dòng trùng lặp:")
    dup_mask = df.duplicated(keep=False)
    display(df[dup_mask].head(10))
```

**Output (stdout):**
```text
Số dòng trùng lặp (trên tất cả 8 cột gốc): 5,268 (0.97%)

Ví dụ dòng trùng lặp:
```

**Result Display:**
```text
InvoiceNo StockCode                        Description  Quantity  \
485    536409     22111       SCOTTIE DOG HOT WATER BOTTLE         1   
489    536409     22866      HAND WARMER SCOTTY DOG DESIGN         1   
494    536409     21866        UNION JACK FLAG LUGGAGE TAG         1   
517    536409     21866        UNION JACK FLAG LUGGAGE TAG         1   
521    536409     22900    SET 2 TEA TOWELS I LOVE LONDON          1   
527    536409     22866      HAND WARMER SCOTTY DOG DESIGN         1   
537    536409     22900    SET 2 TEA TOWELS I LOVE LONDON          1   
539    536409     22111       SCOTTIE DOG HOT WATER BOTTLE         1   
548    536412     22327  ROUND SNACK BOXES SET OF 4 SKULLS         1   
555    536412     22327  ROUND SNACK BOXES SET OF 4 SKULLS         1   

            InvoiceDate  UnitPrice  CustomerID         Country  
485 2010-12-01 11:45:00       4.95   17,908.00  United Kingdom  
489 2010-12-01 11:45:00       2.10   17,908.00  United Kingdom  
494 2010-12-01 11:45:00       1.25   17,908.00  United Kingdom  
517 2010-12-01 11:45:00       1.25   17,908.00  United Kingdom  
521 2010-12-01 11:45:00       2.95   17,908.00  United Kingdom  
527 2010-12-01 11:45:00       2.10   17,908.00  United Kingdom  
537 2010-12-01 11:45:00       2.95   17,908.00  United Kingdom  
539 2010-12-01 11:45:00       4.95   17,908.00  United Kingdom  
548 2010-12-01 11:49:00       2.95   17,920.00  United Kingdom  
555 2010-12-01 11:49:00       2.95   17,920.00  United Kingdom
```

## 5. Invoice Types — Hủy đơn và Adjust

Phân tích prefix InvoiceNo: prefix "C" = hủy đơn, prefix "A" = adjust bad debt.

```python
# [Cell 12 - Execution Count: 7]
df['InvoicePrefix'] = df['InvoiceNo'].astype(str).str[0]
df.loc[df['InvoicePrefix'].str.isdigit(), 'InvoicePrefix'] = 'Normal'

prefix_counts = df['InvoicePrefix'].value_counts()
print("Phân bố InvoicePrefix:")
print(prefix_counts)

# Hóa đơn hủy
cancelled = df[df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"\nHóa đơn hủy (C): {len(cancelled):,} dòng ({len(cancelled)/len(df)*100:.2f}%)")
print(f"Số hóa đơn hủy unique: {cancelled['InvoiceNo'].nunique()}")

# Adjust bad debt
adjust = df[df['InvoiceNo'].astype(str).str.startswith('A')]
print(f"\nAdjust bad debt (A): {len(adjust):,} dòng")
if len(adjust) > 0:
    display(adjust)
```

**Output (stdout):**
```text
Phân bố InvoicePrefix:
InvoicePrefix
Normal    532618
C           9288
A              3
Name: count, dtype: int64

Hóa đơn hủy (C): 9,288 dòng (1.71%)
Số hóa đơn hủy unique: 3836
```

**Output (stdout):**
```text
Adjust bad debt (A): 3 dòng
```

**Result Display:**
```text
InvoiceNo StockCode      Description  Quantity         InvoiceDate  \
299982   A563185         B  Adjust bad debt         1 2011-08-12 14:50:00   
299983   A563186         B  Adjust bad debt         1 2011-08-12 14:51:00   
299984   A563187         B  Adjust bad debt         1 2011-08-12 14:52:00   

        UnitPrice  CustomerID         Country InvoicePrefix  
299982  11,062.06         NaN  United Kingdom             A  
299983 -11,062.06         NaN  United Kingdom             A  
299984 -11,062.06         NaN  United Kingdom             A
```

## 6. StockCode phi sản phẩm

Một số StockCode không phải sản phẩm vật lý (POST, DOT, M, C2, D, S, BANK CHARGES, AMAZONFEE...).

```python
# [Cell 14 - Execution Count: 8]
from src.config import NON_PRODUCT_STOCK_CODES

non_product = df[df['StockCode'].isin(NON_PRODUCT_STOCK_CODES)]
print(f"Dòng có StockCode phi sản phẩm: {len(non_product):,} ({len(non_product)/len(df)*100:.2f}%)")
print("\nPhân loại:")
for code in sorted(NON_PRODUCT_STOCK_CODES):
    count = len(df[df['StockCode'] == code])
    if count > 0:
        desc = df[df['StockCode'] == code]['Description'].mode().values[0] if count > 0 else ''
        print(f"  {code:15s}: {count:,} dòng — {desc}")
```

**Output (stdout):**
```text
Dòng có StockCode phi sản phẩm: 2,912 (0.54%)

Phân loại:
```

**Output (stdout):**
```text
AMAZONFEE      : 34 dòng — AMAZON FEE
  B              : 3 dòng — Adjust bad debt
```

**Output (stdout):**
```text
BANK CHARGES   : 37 dòng — Bank Charges
  C2             : 144 dòng — CARRIAGE
  CRUK           : 16 dòng — CRUK Commission
```

**Output (stdout):**
```text
D              : 77 dòng — Discount
  DOT            : 710 dòng — DOTCOM POSTAGE
```

**Output (stdout):**
```text
M              : 571 dòng — Manual
  POST           : 1,256 dòng — POSTAGE
```

**Output (stdout):**
```text
S              : 63 dòng — SAMPLES
  m              : 1 dòng — Manual
```

## 7. Giá trị bất thường

```python
# [Cell 16 - Execution Count: 9]
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

print("Giá trị bất thường:")
print(f"  Quantity <= 0 : {(df['Quantity'] <= 0).sum():,}")
print(f"  UnitPrice <= 0: {(df['UnitPrice'] <= 0).sum():,}")
print(f"  TotalAmount < 0: {(df['TotalAmount'] < 0).sum():,}")

print(f"\nTotalAmount:")
print(f"  Min: {df['TotalAmount'].min():,.2f}")
print(f"  Max: {df['TotalAmount'].max():,.2f}")
print(f"  Mean: {df['TotalAmount'].mean():,.2f}")
print(f"  Median: {df['TotalAmount'].median():,.2f}")
```

**Output (stdout):**
```text
Giá trị bất thường:
  Quantity <= 0 : 10,624
  UnitPrice <= 0: 2,517
  TotalAmount < 0: 9,290

TotalAmount:
  Min: -168,469.60
  Max: 168,469.60
  Mean: 17.99
  Median: 9.75
```

## 8. Phân phối chính

```python
# [Cell 18 - Execution Count: 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Quantity distribution (positive only, clipped at P99)
pos_qty = df[df['Quantity'] > 0]['Quantity']
q99 = pos_qty.quantile(0.99)
axes[0].hist(pos_qty[pos_qty <= q99], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title(f'Quantity (positive, ≤P99={q99:.0f})')
axes[0].set_xlabel('Quantity')

# UnitPrice distribution
pos_price = df[df['UnitPrice'] > 0]['UnitPrice']
p99 = pos_price.quantile(0.99)
axes[1].hist(pos_price[pos_price <= p99], bins=50, color='coral', edgecolor='white')
axes[1].set_title(f'UnitPrice (positive, ≤P99=£{p99:.2f})')
axes[1].set_xlabel('UnitPrice (£)')

# TotalAmount distribution
pos_total = df[df['TotalAmount'] > 0]['TotalAmount']
t99 = pos_total.quantile(0.99)
axes[2].hist(pos_total[pos_total <= t99], bins=50, color='seagreen', edgecolor='white')
axes[2].set_title(f'TotalAmount (positive, ≤P99=£{t99:.2f})')
axes[2].set_xlabel('TotalAmount (£)')

plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'distributions_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Lưu ý: Biểu đồ hiển thị dữ liệu dương, clipped tại percentile 99 để đọc rõ.")
print(f"Có {len(df[df['Quantity'] > q99]):,} dòng Quantity > P99={q99:.0f}")
```

**Result Display:**
```text
<Figure size 1500x400 with 3 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

**Output (stdout):**
```text
Lưu ý: Biểu đồ hiển thị dữ liệu dương, clipped tại percentile 99 để đọc rõ.
Có 4,950 dòng Quantity > P99=100
```

```python
# [Cell 19 - Execution Count: 11]
# Revenue by month
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly = df[df['TotalAmount'] > 0].groupby('Month')['TotalAmount'].sum()

fig, ax = plt.subplots(figsize=(12, 4))
monthly.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Monthly Revenue (£)')
ax.set_ylabel('Revenue (£)')
ax.set_xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DATA_UNDERSTANDING / 'revenue_by_month.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Lưu ý: Tháng 12/2011 chỉ có đến ngày 09/12, nên doanh thu thấp hơn các tháng đầy đủ.")
```

**Result Display:**
```text
<Figure size 1200x400 with 1 Axes>
```

*Note: Chart output generated. See corresponding PNG file in `figures/` directory.*

**Output (stdout):**
```text
Lưu ý: Tháng 12/2011 chỉ có đến ngày 09/12, nên doanh thu thấp hơn các tháng đầy đủ.
```

## 9. Top Products và Countries

```python
# [Cell 21 - Execution Count: 12]
# Top 10 products by revenue
product_rev = df[df['TotalAmount'] > 0].groupby('StockCode').agg(
    Revenue=('TotalAmount', 'sum'),
    Description=('Description', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'),
    Count=('Quantity', 'sum')
).sort_values('Revenue', ascending=False).head(10)

print("Top 10 sản phẩm theo doanh thu (£):")
display(product_rev)

# Top countries
country_rev = df[df['TotalAmount'] > 0].groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
print(f"\nTop 10 quốc gia theo doanh thu:")
print(f"United Kingdom chiếm {country_rev.iloc[0]/country_rev.sum()*100:.1f}% doanh thu trong top 10")
display(country_rev)
```

**Output (stdout):**
```text
Top 10 sản phẩm theo doanh thu (£):
```

**Result Display:**
```text
Revenue                         Description  Count
StockCode                                                      
DOT       206,248.77                      DOTCOM POSTAGE    706
22423     174,484.74            REGENCY CAKESTAND 3 TIER  13879
23843     168,469.60         PAPER CRAFT , LITTLE BIRDIE  80995
85123A    104,518.80  WHITE HANGING HEART T-LIGHT HOLDER  37660
47566      99,504.33                       PARTY BUNTING  18295
85099B     94,340.05             JUMBO BAG RED RETROSPOT  48474
23166      81,700.92      MEDIUM CERAMIC TOP STORAGE JAR  78033
M          78,110.27                              Manual   7224
POST       78,101.88                             POSTAGE   3150
23084      66,964.99                  RABBIT NIGHT LIGHT  30788
```

**Output (stdout):**
```text
Top 10 quốc gia theo doanh thu:
United Kingdom chiếm 87.0% doanh thu trong top 10
```

**Result Display:**
```text
Country
United Kingdom   9,025,222.08
Netherlands        285,446.34
EIRE               283,453.96
Germany            228,867.14
France             209,715.11
Australia          138,521.31
Spain               61,577.11
Switzerland         57,089.90
Belgium             41,196.34
Sweden              38,378.33
Name: TotalAmount, dtype: float64
```

## 10. Tổng kết Data Understanding

### Phát hiện chính:
1. **Missing CustomerID**: 24.93% (135,080 dòng) — không thể aggregate cấp khách hàng cho nhóm này
2. **Hủy đơn (C)**: 9,288 dòng (1.71%) — cần loại khỏi tập mua hàng hợp lệ
3. **Adjust bad debt (A)**: 3 dòng — loại bỏ
4. **StockCode phi sản phẩm**: ~2,995 dòng — POST, DOT, M, phí vận chuyển/dịch vụ
5. **Duplicate**: 5,268 dòng (0.97%) — giả định ghi nhận lặp
6. **Phân phối lệch**: Quantity, UnitPrice, TotalAmount đều lệch phải mạnh

### Vấn đề cần xử lý ở Notebook 02:
- Loại bỏ hóa đơn hủy, adjust, duplicate
- Loại StockCode phi sản phẩm
- Xử lý missing CustomerID (loại khỏi aggregate khách hàng)
- Xử lý missing Description (mapping theo StockCode)
- Xử lý outlier Quantity/UnitPrice

```python
# [Cell 23 - Execution Count: 13]
# Save summaries
raw_summary = pd.DataFrame({
    'Metric': ['Total Rows', 'Total Columns', 'Unique Invoices', 'Unique Products',
               'Unique Customers', 'Countries', 'Date Range Start', 'Date Range End',
               'Missing CustomerID', 'Missing Description', 'Duplicates',
               'Cancelled Invoices', 'Adjust Bad Debt'],
    'Value': [len(df), 8, df['InvoiceNo'].nunique(), df['StockCode'].nunique(),
              df['CustomerID'].nunique(), df['Country'].nunique(),
              str(df['InvoiceDate'].min()), str(df['InvoiceDate'].max()),
              df['CustomerID'].isnull().sum(), df['Description'].isnull().sum(),
              df.duplicated().sum(),
              len(df[df['InvoiceNo'].astype(str).str.startswith('C')]),
              len(df[df['InvoiceNo'].astype(str).str.startswith('A')])]
})
raw_summary.to_csv(TABLES_DATA_UNDERSTANDING / 'raw_summary.csv', index=False)
desc_stats.to_csv(TABLES_DATA_UNDERSTANDING / 'descriptive_statistics.csv')
print("[OK] Saved raw_summary.csv and descriptive_statistics.csv")
print(f"Figures: {FIGURES_DATA_UNDERSTANDING}")
print(f"Tables: {TABLES_DATA_UNDERSTANDING}")
```

**Output (stdout):**
```text
[OK] Saved raw_summary.csv and descriptive_statistics.csv
Figures: D:\Project\Data-mininng\outputs\figures\data_understanding
Tables: D:\Project\Data-mininng\outputs\tables\data_understanding
```

