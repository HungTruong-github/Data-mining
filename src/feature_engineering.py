"""
Module Feature Engineering cho project Online Retail.
Cung cap cac ham tai su dung cho pipeline Feature Engineering (Phase 03 CRISP-DM).

Cac ham khong hard-code duong dan, su dung config.py de quan ly paths.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import REPEAT_PURCHASE_WINDOW_DAYS


# ====================================================================
# 1. XAY DUNG BANG RFM CUSTOMER FEATURES
# ====================================================================
def build_rfm_features(df, reference_date=None):
    """
    Tao bang dac trung cap khach hang (RFM va cac bien mo rong).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame giao dich khach hang (customer_transactions).
        Yeu cau cac cot: CustomerID, InvoiceNo, InvoiceDate, Quantity,
                         UnitPrice, TotalAmount, StockCode, Country.
    reference_date : pd.Timestamp, optional
        Ngay tham chieu de tinh Recency.
        Mac dinh: max(InvoiceDate) + 1 ngay.

    Returns
    -------
    pd.DataFrame
        Bang RFM moi dong la mot khach hang.
    pd.Timestamp
        reference_date da su dung.
    """
    df = df.copy()

    # Dam bao kieu du lieu
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)

    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Tinh InvoiceDateOnly de dem ActiveDays
    df['InvoiceDateOnly'] = df['InvoiceDate'].dt.date

    # Gom theo CustomerID
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum'),
        TotalItems=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
        UniqueInvoices=('InvoiceNo', 'nunique'),
        ActiveDays=('InvoiceDateOnly', 'nunique'),
        FirstPurchaseDate=('InvoiceDate', 'min'),
        LastPurchaseDate=('InvoiceDate', 'max'),
        Country=('Country', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'),
    ).reset_index()

    # Tinh cac bien dan xuat
    rfm['AverageOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['AverageItemsPerInvoice'] = rfm['TotalItems'] / rfm['Frequency']
    rfm['CustomerLifetimeDays'] = (
        rfm['LastPurchaseDate'] - rfm['FirstPurchaseDate']
    ).dt.days

    return rfm, reference_date


# ====================================================================
# 2. TAO RFM SCORE (1-5)
# ====================================================================
def _safe_qcut(series, q, labels, ascending=True):
    """
    Chia series thanh q nhom bang qcut, xu ly truong hop quantile trung.

    Parameters
    ----------
    series : pd.Series
    q : int
        So nhom can chia.
    labels : list
        Nhan cho tung nhom.
    ascending : bool
        True: gia tri nho -> nhan lon (dung cho Recency).
        False: gia tri lon -> nhan lon (dung cho Frequency, Monetary).
    """
    try:
        if ascending:
            # Recency: thap -> score cao
            result = pd.qcut(series, q=q, labels=labels, duplicates='drop')
        else:
            result = pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        # Fallback: dung rank roi chia deu
        ranks = series.rank(method='first', ascending=(not ascending))
        n = len(ranks)
        bin_size = n / q
        result = pd.Series(
            [labels[min(int((r - 1) / bin_size), q - 1)] for r in ranks],
            index=series.index
        )
    return result.astype(int)


def create_rfm_scores(rfm_df, n_bins=5):
    """
    Tao diem RFM tu 1 den n_bins cho tung khach hang.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Bang RFM voi cac cot Recency, Frequency, Monetary.
    n_bins : int
        So nhom diem (mac dinh: 5).

    Returns
    -------
    pd.DataFrame
        Bang RFM da co them R_Score, F_Score, M_Score, RFM_Score, RFM_Code.
    """
    df = rfm_df.copy()
    labels = list(range(1, n_bins + 1))

    # Recency: thap -> score cao (khach moi mua gan day tot hon)
    df['R_Score'] = _safe_qcut(df['Recency'], q=n_bins,
                               labels=list(reversed(labels)), ascending=True)

    # Frequency: cao -> score cao
    df['F_Score'] = _safe_qcut(df['Frequency'], q=n_bins,
                               labels=labels, ascending=False)

    # Monetary: cao -> score cao
    df['M_Score'] = _safe_qcut(df['Monetary'], q=n_bins,
                               labels=labels, ascending=False)

    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
    df['RFM_Code'] = (
        df['R_Score'].astype(str) +
        df['F_Score'].astype(str) +
        df['M_Score'].astype(str)
    )

    return df


# ====================================================================
# 3. GAN PHAN KHUC RFM
# ====================================================================
def assign_rfm_segment(row):
    """
    Phan loai khach hang vao nhom dua tren R_Score, F_Score, M_Score.

    Logic phan loai:
    - Champions:        R >= 4 AND F >= 4 AND M >= 4
    - Loyal Customers:  F >= 4 AND M >= 3
    - Big Spenders:     M >= 4
    - Recent Customers: R >= 4 AND F <= 2
    - At Risk:          R <= 2 AND F >= 2
    - Regular:          Tat ca truong hop con lai

    Luu y: Day la phan khuc mo ta dua tren diem RFM, can kiem chung
    them bang K-Means clustering.

    Parameters
    ----------
    row : pd.Series
        Dong du lieu chua R_Score, F_Score, M_Score.

    Returns
    -------
    str
        Ten phan khuc khach hang.
    """
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4 and m >= 3:
        return 'Loyal Customers'
    elif m >= 4:
        return 'Big Spenders'
    elif r >= 4 and f <= 2:
        return 'Recent Customers'
    elif r <= 2 and f >= 2:
        return 'At Risk'
    else:
        return 'Regular Customers'


# ====================================================================
# 4. TONG HOP THONG KE PHAN KHUC RFM
# ====================================================================
def summarize_rfm_segments(rfm_df):
    """
    Tao bang thong ke tong hop theo phan khuc RFM.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Bang RFM da co cot RFM_Segment.

    Returns
    -------
    pd.DataFrame
        Bang thong ke: so khach, ty le, RFM trung binh, tong doanh thu.
    """
    total_customers = len(rfm_df)

    summary = rfm_df.groupby('RFM_Segment').agg(
        CustomerCount=('CustomerID', 'count'),
        AverageRecency=('Recency', 'mean'),
        AverageFrequency=('Frequency', 'mean'),
        AverageMonetary=('Monetary', 'mean'),
        TotalMonetary=('Monetary', 'sum'),
        AverageOrderValue=('AverageOrderValue', 'mean'),
    ).reset_index()

    summary['CustomerPercent'] = round(
        summary['CustomerCount'] / total_customers * 100, 2
    )

    # Sap xep theo so luong khach giam dan
    summary = summary.sort_values('CustomerCount', ascending=False).reset_index(drop=True)

    # Lam tron
    for col in ['AverageRecency', 'AverageFrequency', 'AverageMonetary',
                'TotalMonetary', 'AverageOrderValue']:
        summary[col] = summary[col].round(2)

    return summary


# ====================================================================
# 5. TAO NHAN REPEAT PURCHASE 90 NGAY (TRANH DATA LEAKAGE)
# ====================================================================
def build_repeat_purchase_features(df, window_days=None):
    """
    Tao feature va label cho bai toan du doan mua lai trong N ngay.

    CHIEN LUOC TRANH DATA LEAKAGE:
    - cutoff_date = max_date - window_days
    - Feature chi duoc tinh tu giao dich truoc hoac bang cutoff_date
    - Label duoc tao tu giao dich SAU cutoff_date

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame giao dich khach hang (customer_transactions).
    window_days : int, optional
        So ngay cua cua so du doan. Mac dinh: 90 (tu config).

    Returns
    -------
    pd.DataFrame
        Bang feature + label (repeat_purchase_90d).
    pd.DataFrame
        Bang audit de kiem tra nhan.
    dict
        Thong tin ve cutoff, reference dates va thong ke.
    """
    if window_days is None:
        window_days = REPEAT_PURCHASE_WINDOW_DAYS

    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)

    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=window_days)
    feature_ref_date = cutoff_date + pd.Timedelta(days=1)

    # ---- TAO FEATURE: chi tu du lieu <= cutoff_date ----
    df_before = df[df['InvoiceDate'] <= cutoff_date].copy()
    df_before['InvoiceDateOnly'] = df_before['InvoiceDate'].dt.date

    # Chi giu khach hang co it nhat 1 giao dich truoc cutoff
    customers_before = df_before['CustomerID'].unique()

    features = df_before.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (feature_ref_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum'),
        TotalItems=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
        UniqueInvoices=('InvoiceNo', 'nunique'),
        ActiveDays=('InvoiceDateOnly', 'nunique'),
        FirstPurchaseDate=('InvoiceDate', 'min'),
        LastPurchaseDate=('InvoiceDate', 'max'),
        Country=('Country', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'),
    ).reset_index()

    features['AverageOrderValue'] = features['Monetary'] / features['Frequency']
    features['AverageItemsPerInvoice'] = features['TotalItems'] / features['Frequency']
    features['CustomerLifetimeDays'] = (
        features['LastPurchaseDate'] - features['FirstPurchaseDate']
    ).dt.days

    # ---- TAO LABEL: tu du lieu trong (cutoff_date, cutoff_date + window_days] ----
    df_future = df[
        (df['InvoiceDate'] > cutoff_date) &
        (df['InvoiceDate'] <= cutoff_date + pd.Timedelta(days=window_days))
    ].copy()

    future_agg = df_future.groupby('CustomerID').agg(
        future_invoice_count=('InvoiceNo', 'nunique'),
        future_revenue=('TotalAmount', 'sum'),
    ).reset_index()

    # Merge va tao label
    features = features.merge(future_agg, on='CustomerID', how='left')
    features['future_invoice_count'] = features['future_invoice_count'].fillna(0).astype(int)
    features['future_revenue'] = features['future_revenue'].fillna(0.0)
    features['repeat_purchase_90d'] = (features['future_invoice_count'] >= 1).astype(int)

    # ---- TAO BANG AUDIT ----
    audit = features[['CustomerID', 'future_invoice_count', 'future_revenue',
                       'repeat_purchase_90d']].copy()
    audit['cutoff_date'] = cutoff_date

    # ---- TAO BANG MODELING (khong chua future info) ----
    modeling_features = features.drop(
        columns=['future_invoice_count', 'future_revenue',
                 'FirstPurchaseDate', 'LastPurchaseDate'],
        errors='ignore'
    )

    info = {
        'max_date': max_date,
        'cutoff_date': cutoff_date,
        'feature_ref_date': feature_ref_date,
        'window_days': window_days,
        'customers_with_history': len(customers_before),
        'customers_in_features': len(features),
        'label_1_count': int(features['repeat_purchase_90d'].sum()),
        'label_0_count': int((features['repeat_purchase_90d'] == 0).sum()),
        'repeat_rate': round(features['repeat_purchase_90d'].mean() * 100, 2),
    }

    return modeling_features, audit, info


# ====================================================================
# 6. CHUAN BI BASKET DATA CHO ASSOCIATION RULES
# ====================================================================
def build_basket_data(df):
    """
    Chuan bi du lieu dang gio hang cho khai pha luat ket hop.

    Moi hoa don la mot giao dich.
    Moi san pham (StockCode) chi xuat hien 1 lan trong moi hoa don.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame giao dich san pham (product_transactions).

    Returns
    -------
    pd.DataFrame
        Bang dang dai (InvoiceNo, StockCode, Description).
    dict
        Thong ke basket data.
    """
    df = df.copy()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)

    # Loai dong thieu InvoiceNo hoac StockCode
    df = df.dropna(subset=['InvoiceNo', 'StockCode'])

    # Chon cac cot can thiet
    basket = df[['InvoiceNo', 'StockCode', 'Description']].copy()

    # Moi san pham chi xuat hien 1 lan trong moi hoa don
    basket = basket.drop_duplicates(subset=['InvoiceNo', 'StockCode'], keep='first')

    stats = {
        'total_rows': len(basket),
        'unique_invoices': basket['InvoiceNo'].nunique(),
        'unique_products': basket['StockCode'].nunique(),
        'avg_items_per_invoice': round(
            basket.groupby('InvoiceNo')['StockCode'].count().mean(), 2
        ),
    }

    return basket, stats


# ====================================================================
# 7. TONG HOP FEATURE SUMMARY
# ====================================================================
def summarize_features(df, numeric_only=True):
    """
    Tao bang thong ke mo ta cho cac feature.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_only : bool
        Neu True, chi thong ke cac cot so.

    Returns
    -------
    pd.DataFrame
        Bang thong ke moi dong la mot feature.
    """
    if numeric_only:
        cols = df.select_dtypes(include=[np.number]).columns
    else:
        cols = df.columns

    rows = []
    for col in cols:
        s = df[col]
        row = {
            'Feature': col,
            'Dtype': str(s.dtype),
            'MissingCount': int(s.isnull().sum()),
            'MissingPercent': round(s.isnull().mean() * 100, 2),
            'UniqueCount': int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            row.update({
                'Min': round(desc.get('min', np.nan), 2),
                'Q1': round(s.quantile(0.25), 2),
                'Median': round(desc.get('50%', np.nan), 2),
                'Mean': round(desc.get('mean', np.nan), 2),
                'Q3': round(s.quantile(0.75), 2),
                'Max': round(desc.get('max', np.nan), 2),
            })
        else:
            for k in ['Min', 'Q1', 'Median', 'Mean', 'Q3', 'Max']:
                row[k] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ====================================================================
# 8. KIEM TRA CHAT LUONG DU LIEU (DATA QUALITY)
# ====================================================================
def validate_feature_data(rfm_df, repeat_df, basket_df, cutoff_date=None,
                          df_transactions=None, audit_df=None):
    """
    Kiem tra chat luong du lieu feature engineering va chong data leakage.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Bang RFM customer features.
    repeat_df : pd.DataFrame
        Bang repeat purchase features (modeling).
    basket_df : pd.DataFrame
        Bang basket data dang dai.
    cutoff_date : pd.Timestamp, optional
        Ngay cat de kiem tra thoi gian va data leakage.
    df_transactions : pd.DataFrame, optional
        Du lieu giao dich khach hang goc de kiem tra temporal leakage.
    audit_df : pd.DataFrame, optional
        Bang audit chua future info truoc khi loai bo.

    Returns
    -------
    list of str
        Danh sach ket qua kiem tra (PASS / FAIL).
    bool
        True neu tat ca checks deu pass.
    """
    checks = []
    all_pass = True

    # 1. CustomerID khong thieu trong RFM
    missing_id = rfm_df['CustomerID'].isnull().sum()
    status = 'PASS' if missing_id == 0 else 'FAIL'
    checks.append(f'[{status}] RFM: CustomerID khong thieu (missing={missing_id})')
    if status == 'FAIL':
        all_pass = False

    # 2. Moi CustomerID chi xuat hien 1 lan
    dup_count = rfm_df['CustomerID'].duplicated().sum()
    status = 'PASS' if dup_count == 0 else 'FAIL'
    checks.append(f'[{status}] RFM: CustomerID duy nhat (duplicate={dup_count})')
    if status == 'FAIL':
        all_pass = False

    # 3. Recency >= 0
    neg_recency = (rfm_df['Recency'] < 0).sum()
    status = 'PASS' if neg_recency == 0 else 'FAIL'
    checks.append(f'[{status}] RFM: Recency >= 0 (negative={neg_recency})')
    if status == 'FAIL':
        all_pass = False

    # 4. Frequency >= 1
    low_freq = (rfm_df['Frequency'] < 1).sum()
    status = 'PASS' if low_freq == 0 else 'FAIL'
    checks.append(f'[{status}] RFM: Frequency >= 1 (below={low_freq})')
    if status == 'FAIL':
        all_pass = False

    # 5. Monetary > 0
    neg_monetary = (rfm_df['Monetary'] <= 0).sum()
    status = 'PASS' if neg_monetary == 0 else 'FAIL'
    checks.append(f'[{status}] RFM: Monetary > 0 (non_positive={neg_monetary})')
    if status == 'FAIL':
        all_pass = False

    # 6. repeat_purchase_90d chi nhan 0 hoac 1
    unique_labels = set(repeat_df['repeat_purchase_90d'].unique())
    valid_labels = unique_labels.issubset({0, 1})
    status = 'PASS' if valid_labels else 'FAIL'
    checks.append(f'[{status}] Repeat: label chi nhan 0/1 (unique={unique_labels})')
    if status == 'FAIL':
        all_pass = False

    # 7. Khong co missing bat ngo trong feature chinh
    key_cols = ['Recency', 'Frequency', 'Monetary']
    for col in key_cols:
        if col in repeat_df.columns:
            miss = repeat_df[col].isnull().sum()
            status = 'PASS' if miss == 0 else 'FAIL'
            checks.append(f'[{status}] Repeat: {col} khong thieu (missing={miss})')
            if status == 'FAIL':
                all_pass = False

    # 8. Khong co future_invoice_count hoac future_revenue trong modeling
    forbidden = ['future_invoice_count', 'future_revenue']
    for col in forbidden:
        present = col in repeat_df.columns
        status = 'PASS' if not present else 'FAIL'
        checks.append(f'[{status}] Repeat: khong chua cot cam "{col}"')
        if status == 'FAIL':
            all_pass = False

    # 9. Khong co duplicate trong repeat features
    dup_repeat = repeat_df['CustomerID'].duplicated().sum()
    status = 'PASS' if dup_repeat == 0 else 'FAIL'
    checks.append(f'[{status}] Repeat: CustomerID duy nhat (duplicate={dup_repeat})')
    if status == 'FAIL':
        all_pass = False

    # 10. Khong co duplicate InvoiceNo + StockCode trong basket
    dup_basket = basket_df.duplicated(subset=['InvoiceNo', 'StockCode']).sum()
    status = 'PASS' if dup_basket == 0 else 'FAIL'
    checks.append(f'[{status}] Basket: khong trung InvoiceNo+StockCode (dup={dup_basket})')
    if status == 'FAIL':
        all_pass = False

    # 11. Kiem tra cutoff_date va Recency khong co giao dich sau cutoff
    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        # feature_ref_date = cutoff_ts + 1 day
        # LastPurchaseDate <= cutoff_ts => Recency >= 1
        if 'Recency' in repeat_df.columns:
            invalid_rec = (repeat_df['Recency'] < 1).sum()
            status = 'PASS' if invalid_rec == 0 else 'FAIL'
            checks.append(
                f'[{status}] Cutoff: Recency trong repeat_features >= 1 ngay '
                f'(khong co giao dich tu/sau feature_ref_date, invalid={invalid_rec})'
            )
            if status == 'FAIL':
                all_pass = False

    # 12. Kiem tra temporal leakage bang giao dich khach hang goc
    if df_transactions is not None and cutoff_date is not None:
        df_tx = df_transactions.copy()
        df_tx['InvoiceDate'] = pd.to_datetime(df_tx['InvoiceDate'])
        df_tx['CustomerID'] = df_tx['CustomerID'].astype(int).astype(str)
        repeat_cust_set = set(repeat_df['CustomerID'].astype(str).unique())

        # 12a. 100% khach hang trong repeat_df phai co giao dich <= cutoff_date
        tx_before = df_tx[df_tx['InvoiceDate'] <= cutoff_ts]
        cust_before = set(tx_before['CustomerID'].unique())
        not_before = repeat_cust_set - cust_before
        status = 'PASS' if len(not_before) == 0 else 'FAIL'
        checks.append(
            f'[{status}] Cutoff: 100% khach hang trong repeat_features '
            f'({len(repeat_df):,}) co giao dich truoc/tai cutoff_date '
            f'({cutoff_ts.strftime("%Y-%m-%d")})'
        )
        if status == 'FAIL':
            all_pass = False

        # 12b. Khong co khach hang moi sau cutoff bi ro ri vao repeat_features
        first_purchases = df_tx.groupby('CustomerID')['InvoiceDate'].min()
        cust_new_after = set(first_purchases[first_purchases > cutoff_ts].index)
        leaked_cust = repeat_cust_set.intersection(cust_new_after)
        status = 'PASS' if len(leaked_cust) == 0 else 'FAIL'
        checks.append(
            f'[{status}] Cutoff: Khong co khach hang moi sau cutoff bi ro ri vao feature '
            f'(khach moi={len(cust_new_after):,}, ro ri={len(leaked_cust)})'
        )
        if status == 'FAIL':
            all_pass = False

        # 12c. Kiem tra Frequency khop voi giao dich <= cutoff_date (khong cong don tuong lai)
        if 'Frequency' in repeat_df.columns:
            freq_before = tx_before.groupby('CustomerID')['InvoiceNo'].nunique()
            repeat_indexed = repeat_df.set_index(repeat_df['CustomerID'].astype(str))
            freq_diff = (repeat_indexed['Frequency'] != freq_before.reindex(repeat_indexed.index)).sum()
            status = 'PASS' if freq_diff == 0 else 'FAIL'
            checks.append(
                f'[{status}] Cutoff: Frequency tinh hoan toan tu giao dich <= cutoff_date '
                f'(mismatches={freq_diff})'
            )
            if status == 'FAIL':
                all_pass = False

        # 12d. Kiem tra nhan repeat_purchase_90d khop voi giao dich tuong lai
        tx_future = df_tx[
            (df_tx['InvoiceDate'] > cutoff_ts) &
            (df_tx['InvoiceDate'] <= cutoff_ts + pd.Timedelta(days=90))
        ]
        cust_with_future_tx = set(tx_future['CustomerID'].unique())
        repeat_label_1 = set(repeat_df[repeat_df['repeat_purchase_90d'] == 1]['CustomerID'].astype(str))
        repeat_label_0 = set(repeat_df[repeat_df['repeat_purchase_90d'] == 0]['CustomerID'].astype(str))

        invalid_1 = repeat_label_1 - cust_with_future_tx
        invalid_0 = repeat_label_0.intersection(cust_with_future_tx)
        status = 'PASS' if (len(invalid_1) == 0 and len(invalid_0) == 0) else 'FAIL'
        checks.append(
            f'[{status}] Repeat: Nhan repeat_purchase_90d xac thuc 100% voi hanh vi tuong lai '
            f'(loi nhan 1={len(invalid_1)}, loi nhan 0={len(invalid_0)})'
        )
        if status == 'FAIL':
            all_pass = False

    # 13. Kiem tra audit_df neu co
    if audit_df is not None:
        if 'repeat_purchase_90d' in audit_df.columns and 'future_invoice_count' in audit_df.columns:
            correct_label = (
                audit_df['repeat_purchase_90d'] == (audit_df['future_invoice_count'] >= 1).astype(int)
            ).all()
            status = 'PASS' if correct_label else 'FAIL'
            checks.append(
                f'[{status}] Audit: Logic gan nhan repeat_purchase_90d = '
                f'(future_invoice_count >= 1) hop le 100%'
            )
            if status == 'FAIL':
                all_pass = False

    return checks, all_pass
