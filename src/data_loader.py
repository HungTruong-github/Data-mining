"""
Module tải dữ liệu Online Retail.
Hỗ trợ đọc dữ liệu thô (raw), dữ liệu trung gian (interim) và dữ liệu đã xử lý (processed).
"""

import pandas as pd
from pathlib import Path
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


def load_processed_data(filename):
    """
    Tải file CSV từ thư mục data/processed/ một cách an toàn.
    """
    project_dir = Path(__file__).resolve().parents[1]
    file_path = project_dir / 'data' / 'processed' / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")
        
    return pd.read_csv(file_path)


def load_interim_data(filename):
    """
    Tải file CSV từ thư mục data/interim/.
    """
    project_dir = Path(__file__).resolve().parents[1]
    file_path = project_dir / 'data' / 'interim' / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")
        
    return pd.read_csv(file_path)

def save_rules(rules_df, filename):
    """
    Lưu danh sách luật kết hợp vào thư mục outputs/rules/
    """
    project_dir = Path(__file__).resolve().parents[1]
    save_path = project_dir / 'outputs' / 'rules' / filename
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rules_df.empty:
        csv_rules = rules_df.copy()
        # Chuyển đổi định dạng frozenset của mlxtend thành chuỗi string ngăn cách bởi dấu phẩy
        csv_rules['antecedents'] = csv_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        csv_rules['consequents'] = csv_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        csv_rules.to_csv(save_path, index=False)
        print(f"[OK] Đã lưu thành công danh sách luật kết hợp tại: {save_path}")
    else:
        print("Cảnh báo: DataFrame luật kết hợp trống, không có dữ liệu để lưu.")