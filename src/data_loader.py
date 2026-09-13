"""
Module tải dữ liệu gốc Online Retail.
Hỗ trợ cả CSV và XLSX tùy file có sẵn.
"""

import pandas as pd
from src.config import RAW_DATA_CSV


def load_raw_data(filepath=None):
    """
    Đọc file dữ liệu gốc Online Retail.

    Parameters
    ----------
    filepath : str or Path, optional
        Đường dẫn tùy chỉnh. Nếu None, sử dụng đường dẫn mặc định từ config.

    Returns
    -------
    pd.DataFrame
        DataFrame chứa dữ liệu gốc.
    """
    if filepath is None:
        filepath = RAW_DATA_CSV

    filepath = str(filepath)

    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, engine='openpyxl')
    elif filepath.endswith('.csv'):
        # Thử encoding phổ biến cho dataset Online Retail
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(f"Không thể đọc file CSV với các encoding đã thử: {filepath}")
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {filepath}")

    print(f"[OK] Da tai du lieu: {df.shape[0]:,} dong x {df.shape[1]} cot")
    return df
