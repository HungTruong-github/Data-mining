"""
Module tai du lieu Online Retail.
Ho tro doc du lieu tho (raw), du lieu trung gian (interim) va du lieu da xu ly (processed).
"""

import hashlib
import pandas as pd
from pathlib import Path
from src.config import INTERIM_DIR, PROCESSED_DIR, get_raw_data_path, REQUIRED_COLUMNS


def load_raw_data(filepath=None):
    """
    Doc file du lieu goc Online Retail.
    
    Neu filepath la None:
        1. Uu tien CSV
        2. Fallback sang XLSX
        3. Raise FileNotFoundError neu khong co ca hai

    Returns
    -------
    pd.DataFrame
        DataFrame chua du lieu goc.
    """
    if filepath is None:
        filepath = get_raw_data_path()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Khong tim thay file: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == '.xlsx':
        df = pd.read_excel(filepath, engine='openpyxl')
    elif suffix == '.csv':
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(f"Khong the doc file CSV voi cac encoding da thu: {filepath}")
    else:
        raise ValueError(f"Dinh dang file khong duoc ho tro: {suffix}. Chi ho tro .csv va .xlsx")

    print(f"[OK] Da tai du lieu: {df.shape[0]:,} dong x {df.shape[1]} cot")
    print(f"     File: {filepath.name}")
    return df


def validate_raw_schema(df):
    """
    Kiem tra cac cot bat buoc co ton tai trong DataFrame.
    Raise AssertionError neu thieu cot.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert len(missing) == 0, f"Thieu cac cot bat buoc: {missing}. Co: {list(df.columns)}"
    return True


def get_raw_metadata(filepath=None):
    """
    Tra ve metadata cua file raw: path, size, sha256, row/col count.
    """
    if filepath is None:
        filepath = get_raw_data_path()
    filepath = Path(filepath)
    
    file_size = filepath.stat().st_size
    
    # SHA256
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    
    return {
        'raw_path': str(filepath),
        'file_size_bytes': file_size,
        'file_size_mb': round(file_size / (1024*1024), 2),
        'sha256': sha.hexdigest(),
    }


def load_processed_data(filename):
    """Tai file CSV tu thu muc data/processed/."""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {file_path}")
    return pd.read_csv(file_path)


def load_interim_data(filename):
    """Tai file CSV tu thu muc data/interim/."""
    file_path = INTERIM_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {file_path}")
    return pd.read_csv(file_path)


def save_rules(rules_df, filename):
    """Luu danh sach luat ket hop vao thu muc outputs/rules/."""
    from src.config import RULES_DIR
    save_path = RULES_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rules_df.empty:
        csv_rules = rules_df.copy()
        csv_rules['antecedents'] = csv_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        csv_rules['consequents'] = csv_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        csv_rules.to_csv(save_path, index=False)
        print(f"[OK] Da luu luat ket hop tai: {save_path}")
    else:
        print("Canh bao: DataFrame luat ket hop trong.")
