import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    standardize_dtypes, 
    add_transaction_flags, 
    remove_invalid_transactions,
    handle_duplicates
)

@pytest.fixture
def sample_raw_data():
    """Tạo bộ dữ liệu thô giả lập chứa các lỗi nghiệp vụ phổ biến."""
    return pd.DataFrame({
        'InvoiceNo': ['536365', 'C536379', 'A563185', '536366', '536366', '536367'],
        'StockCode': ['85123A', 'D', 'B', '22632', '22632', 'POST'],
        'Description': ['WHITE HANGING HEART', 'Discount', 'Adjust bad debt', 'HAND WARMER', 'HAND WARMER', 'POSTAGE'],
        'Quantity': ['6', '-1', '1', '10', '10', '0'], # Có số lượng âm và = 0
        'UnitPrice': ['2.55', '27.50', '11062.06', '2.10', '2.10', '-5.0'], # Có giá trị âm
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 9:41', '8/12/2011 14:50', '12/1/2010 8:28', '12/1/2010 8:28', '12/1/2010 8:30'],
        'CustomerID': ['17850', '14527', np.nan, '17850', '17850', '12345'],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    })

def test_standardize_dtypes(sample_raw_data):
    """Kiểm tra việc ép kiểu dữ liệu và tính toán TotalAmount."""
    df_std = standardize_dtypes(sample_raw_data)
    
    assert pd.api.types.is_datetime64_any_dtype(df_std['InvoiceDate']), "InvoiceDate phải là datetime"
    assert pd.api.types.is_integer_dtype(df_std['Quantity']), "Quantity phải là số nguyên"
    assert df_std['TotalAmount'].iloc[0] == 6 * 2.55, "TotalAmount tính toán sai"

def test_add_transaction_flags(sample_raw_data):
    """Kiểm tra việc gắn cờ (flag) chính xác cho các giao dịch dị thường."""
    df_std = standardize_dtypes(sample_raw_data)
    df_flagged = add_transaction_flags(df_std)
    
    assert df_flagged.loc[1, 'IsCancelled'] == True, "Lỗi không nhận diện được hóa đơn C"
    assert df_flagged.loc[2, 'IsAdjust'] == True, "Lỗi không nhận diện được hóa đơn A"
    assert df_flagged.loc[4, 'IsDuplicate'] == True, "Lỗi không nhận diện được dòng trùng lặp"

def test_remove_invalid_transactions(sample_raw_data):
    """Kiểm tra việc loại bỏ các dòng không hợp lệ."""
    df_std = standardize_dtypes(sample_raw_data)
    df_flagged = add_transaction_flags(df_std)
    df_clean, stats = remove_invalid_transactions(df_flagged)
    
    assert len(df_clean) == 3, "Hệ thống lọc sai số lượng giao dịch hợp lệ"
    assert stats['cancelled_rows'] == 1, "Thống kê số dòng C sai"
    assert stats['adjust_rows'] == 1, "Thống kê số dòng A sai"

def test_handle_duplicates(sample_raw_data):
    """Kiểm tra thuật toán xóa dòng trùng lặp."""
    df_std = standardize_dtypes(sample_raw_data)
    df_flagged = add_transaction_flags(df_std)
    df_dedup, stats = handle_duplicates(df_flagged)
    
    assert stats['duplicate_count'] == 1, "Không đếm đúng số dòng trùng lặp"
    assert len(df_dedup) == len(sample_raw_data) - 1, "Không xóa đúng số dòng trùng lặp"