"""
Module tien xu ly du lieu Online Retail.
Cung cap cac ham tai su dung cho pipeline Data Preparation (Phase 02 CRISP-DM).

Cac ham khong hard-code duong dan, su dung config.py de quan ly paths.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import NON_PRODUCT_STOCK_CODES


# ====================================================================
# 1. CHUAN HOA KIEU DU LIEU
# ====================================================================
def standardize_dtypes(df):
    """
    Chuan hoa kieu du lieu cac cot trong DataFrame.

    - InvoiceNo -> str
    - StockCode -> str
    - Description -> str (giu NaN)
    - InvoiceDate -> datetime
    - Quantity -> int64
    - UnitPrice -> float64
    - CustomerID -> float64 (giu NaN)
    - Tao cot TotalAmount = Quantity * UnitPrice

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame goc.

    Returns
    -------
    pd.DataFrame
        DataFrame da chuan hoa kieu du lieu.
    """
    df = df.copy()

    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip()

    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0)

    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    return df


# ====================================================================
# 2. THEM CO GIAO DICH
# ====================================================================
def add_transaction_flags(df):
    """
    Them cac co (flags) boolean de danh dau cac loai giao dich dac biet.

    Cac co duoc tao:
    - IsCancelled: InvoiceNo bat dau bang 'C'
    - IsAdjust: InvoiceNo bat dau bang 'A'
    - IsDuplicate: dong trung lap hoan toan (tru cot TotalAmount va cac cot flag)
    - IsQuantityInvalid: Quantity <= 0
    - IsPriceInvalid: UnitPrice <= 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame voi cac cot flag moi.
    """
    df = df.copy()

    df['IsCancelled'] = df['InvoiceNo'].str.startswith('C')
    df['IsAdjust'] = df['InvoiceNo'].str.startswith('A')

    # Danh dau duplicate tren cac cot goc (8 cot ban dau)
    original_cols = ['InvoiceNo', 'StockCode', 'Description',
                     'Quantity', 'InvoiceDate', 'UnitPrice',
                     'CustomerID', 'Country']
    cols_for_dup = [c for c in original_cols if c in df.columns]
    df['IsDuplicate'] = df.duplicated(subset=cols_for_dup, keep='first')

    df['IsQuantityInvalid'] = df['Quantity'] <= 0
    df['IsPriceInvalid'] = df['UnitPrice'] <= 0

    return df


# ====================================================================
# 3. PHAN LOAI STOCKCODE
# ====================================================================
def classify_stock_codes(df, save_path=None):
    """
    Phan loai StockCode thanh san pham that va phi san pham (phi, dieu chinh, dich vu).

    - non_product_codes tu config (POST, DOT, M, m, C2, D, S, BANK CHARGES, AMAZONFEE, CRUK, B)
    - gift_* : Gift voucher -> loai khoi product_transactions, giu trong cleaned_transactions
    - PADS, DCGS* : San pham that -> giu lai

    Parameters
    ----------
    df : pd.DataFrame
    save_path : Path or str, optional
        Duong dan luu file CSV phan loai.

    Returns
    -------
    pd.DataFrame
        Bang phan loai StockCode.
    """
    # Tim tat ca StockCode bat dau bang chu cai (khong phai so)
    sc_str = df['StockCode'].astype(str)
    alpha_mask = sc_str.str.match(r'^[A-Za-z]')
    special_df = df[alpha_mask].copy()

    # Thong ke theo StockCode
    sc_stats = special_df.groupby('StockCode').agg(
        Description=('Description', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'),
        RowCount=('StockCode', 'count'),
        TotalQuantity=('Quantity', 'sum'),
        TotalAmount=('TotalAmount', 'sum'),
    ).reset_index()

    # Phan loai
    def _classify(row):
        sc = row['StockCode']
        desc = str(row['Description']).lower()

        if sc in NON_PRODUCT_STOCK_CODES:
            return 'non_product', 'Loai bo', 'Phi dich vu / dieu chinh / khong phai san pham'
        elif sc.lower().startswith('gift') or 'gift' in desc and 'voucher' in desc:
            return 'gift_voucher', 'Giu (cleaned), Loai (product)', 'Gift voucher, khong phai san pham vat ly'
        elif sc.startswith('DCGS') or sc.startswith('dcgs'):
            return 'product', 'Giu lai', 'San pham dat biet co Description hop le'
        elif sc == 'PADS':
            return 'product', 'Giu lai', 'San pham phu kien (pads) co Description hop le'
        else:
            return 'other_alpha', 'Kiem tra', f'StockCode bat dau bang chu: {sc}'

    classifications = sc_stats.apply(_classify, axis=1, result_type='expand')
    classifications.columns = ['Category', 'Action', 'Reason']
    sc_classification = pd.concat([sc_stats, classifications], axis=1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sc_classification.to_csv(save_path, index=False)

    return sc_classification


# ====================================================================
# 4. DIEN DESCRIPTION THIEU
# ====================================================================
def fill_missing_descriptions(df):
    """
    Dien gia tri thieu cho cot Description.

    Chien luoc:
    1. Dien Description pho bien nhat (mode) theo StockCode.
    2. Neu khong tim thay -> dien 'Unknown Product'.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame da dien Description.
    int
        So dong da duoc dien.
    """
    df = df.copy()
    missing_before = df['Description'].isnull().sum()

    if missing_before == 0:
        return df, 0

    # Tao bang mode Description theo StockCode
    desc_mode = (
        df.dropna(subset=['Description'])
        .groupby('StockCode')['Description']
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    )

    # Dien theo StockCode
    mask = df['Description'].isnull()
    df.loc[mask, 'Description'] = df.loc[mask, 'StockCode'].map(desc_mode)

    # Neu van con thieu -> 'Unknown Product'
    df['Description'] = df['Description'].fillna('Unknown Product')

    filled_count = missing_before - df['Description'].isnull().sum()
    return df, filled_count


# ====================================================================
# 5. BAO CAO MISSING VALUES
# ====================================================================
def report_missing_values(df, save_path=None):
    """
    Tao bao cao missing values.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : Path or str, optional

    Returns
    -------
    pd.DataFrame
        Bang thong ke missing values.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        'Column': missing.index,
        'MissingCount': missing.values,
        'MissingPercent': missing_pct.values,
        'TotalRows': len(df)
    })
    report = report.sort_values('MissingCount', ascending=False).reset_index(drop=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(save_path, index=False)

    return report


# ====================================================================
# 6. LOAI GIAO DICH KHONG HOP LE
# ====================================================================
def remove_invalid_transactions(df):
    """
    Loai cac giao dich khong hop le khoi du lieu phan tich chinh.

    Loai bo:
    - InvoiceNo bat dau bang 'C' (hoa don huy)
    - InvoiceNo bat dau bang 'A' (adjust bad debt)
    - Quantity <= 0
    - UnitPrice <= 0

    Khong thay doi DataFrame goc, tra ve ban sao sach.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame co cac cot flag (IsCancelled, IsAdjust, IsQuantityInvalid, IsPriceInvalid).

    Returns
    -------
    pd.DataFrame
        DataFrame sach.
    dict
        Thong ke so dong loai bo theo tung loai.
    """
    stats = {}
    stats['raw_rows'] = len(df)

    # Loai Invoice C
    mask_c = df['IsCancelled']
    stats['cancelled_rows'] = mask_c.sum()
    stats['cancelled_invoices'] = df.loc[mask_c, 'InvoiceNo'].nunique()

    # Loai Invoice A
    mask_a = df['IsAdjust']
    stats['adjust_rows'] = mask_a.sum()
    stats['adjust_invoices'] = df.loc[mask_a, 'InvoiceNo'].nunique()

    # Sau khi loai C va A
    df_clean = df[~mask_c & ~mask_a].copy()
    stats['after_invoice_clean'] = len(df_clean)

    # Loai Quantity <= 0 (nhung dong con lai)
    mask_qty = df_clean['Quantity'] <= 0
    stats['qty_invalid_rows'] = mask_qty.sum()
    df_clean = df_clean[~mask_qty]
    stats['after_qty_clean'] = len(df_clean)

    # Loai UnitPrice <= 0
    mask_price = df_clean['UnitPrice'] <= 0
    stats['price_invalid_rows'] = mask_price.sum()
    df_clean = df_clean[~mask_price]
    stats['after_price_clean'] = len(df_clean)

    return df_clean, stats


# ====================================================================
# 7. XU LY DUPLICATE
# ====================================================================
def handle_duplicates(df):
    """
    Loai bo cac dong trung lap hoan toan.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame khong trung lap.
    dict
        Thong ke duplicate.
    """
    original_cols = ['InvoiceNo', 'StockCode', 'Description',
                     'Quantity', 'InvoiceDate', 'UnitPrice',
                     'CustomerID', 'Country']
    cols_for_dup = [c for c in original_cols if c in df.columns]

    before = len(df)
    dup_mask = df.duplicated(subset=cols_for_dup, keep='first')
    dup_count = dup_mask.sum()

    df_clean = df[~dup_mask].copy()

    stats = {
        'before_dedup': before,
        'duplicate_count': dup_count,
        'duplicate_pct': round(dup_count / before * 100, 2) if before > 0 else 0,
        'after_dedup': len(df_clean)
    }

    return df_clean, stats


# ====================================================================
# 8. PHAT HIEN OUTLIER
# ====================================================================
def detect_outliers(df, columns=None):
    """
    Phat hien outlier bang phuong phap IQR tren cac cot so.

    Them cac cot co:
    - IsQuantityOutlier
    - IsPriceOutlier
    - IsAmountOutlier

    KHONG tu dong xoa outlier. Chi danh dau de bao cao.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list, optional
        Danh sach cot kiem tra. Mac dinh: ['Quantity', 'UnitPrice', 'TotalAmount']

    Returns
    -------
    pd.DataFrame
        DataFrame voi cac cot co outlier.
    pd.DataFrame
        Bang thong ke outlier.
    """
    df = df.copy()
    if columns is None:
        columns = ['Quantity', 'UnitPrice', 'TotalAmount']

    flag_map = {
        'Quantity': 'IsQuantityOutlier',
        'UnitPrice': 'IsPriceOutlier',
        'TotalAmount': 'IsAmountOutlier'
    }

    outlier_stats = []

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)

        outlier_mask = (df[col] < lower) | (df[col] > upper)

        flag_name = flag_map.get(col, f'Is{col}Outlier')
        df[flag_name] = outlier_mask

        outlier_stats.append({
            'Column': col,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'LowerBound': round(lower, 2),
            'UpperBound': round(upper, 2),
            'P1': round(p1, 2),
            'P99': round(p99, 2),
            'OutlierCount': outlier_mask.sum(),
            'OutlierPercent': round(outlier_mask.sum() / len(df) * 100, 2),
            'Min': round(df[col].min(), 2),
            'Max': round(df[col].max(), 2),
            'Mean': round(df[col].mean(), 2),
            'Median': round(df[col].median(), 2),
        })

    outlier_summary = pd.DataFrame(outlier_stats)
    return df, outlier_summary


# ====================================================================
# 9. TAO CAC TAP DU LIEU SACH
# ====================================================================
def create_clean_datasets(df, non_product_codes=None):
    """
    Tao 3 tap du lieu sach tu DataFrame da xu ly.

    1. cleaned_transactions: Tat ca giao dich mua hop le.
    2. customer_transactions: Chi giao dich co CustomerID (cho RFM, K-Means).
    3. product_transactions: Loai non-product StockCode va gift voucher (cho association rules).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame da loai C/A, Qty<=0, Price<=0, duplicate.
    non_product_codes : set, optional
        Tap hop StockCode phi san pham. Mac dinh su dung tu config.

    Returns
    -------
    dict
        {'cleaned': df1, 'customer': df2, 'product': df3, 'stats': dict}
    """
    if non_product_codes is None:
        non_product_codes = NON_PRODUCT_STOCK_CODES

    # 1. Cleaned transactions (giu tat ca, ke ca khong co CustomerID)
    cleaned = df.copy()

    # 2. Customer transactions (chi co CustomerID)
    customer = cleaned.dropna(subset=['CustomerID']).copy()
    customer['CustomerID'] = customer['CustomerID'].astype(int)

    # 3. Product transactions (loai non-product SC va gift voucher)
    sc_str = cleaned['StockCode'].astype(str)
    gift_mask = sc_str.str.lower().str.startswith('gift')
    non_product_mask = sc_str.isin(non_product_codes) | gift_mask
    product = cleaned[~non_product_mask].copy()

    stats = {
        'cleaned_rows': len(cleaned),
        'customer_rows': len(customer),
        'product_rows': len(product),
        'missing_customerid_removed': len(cleaned) - len(customer),
        'non_product_removed': non_product_mask.sum(),
    }

    return {
        'cleaned': cleaned,
        'customer': customer,
        'product': product,
        'stats': stats
    }


# ====================================================================
# 10. LUU BAO CAO CLEANING
# ====================================================================
def save_cleaning_report(pipeline_stats, save_path):
    """
    Luu bao cao cleaning summary voi so dong o tung giai doan.

    Parameters
    ----------
    pipeline_stats : list of dict
        Moi dict co {'Stage': str, 'Rows': int, 'RowsRemoved': int, 'Reason': str}
    save_path : Path or str
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(pipeline_stats)
    report_df.to_csv(save_path, index=False)
    return report_df
