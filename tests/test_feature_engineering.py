import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    build_rfm_features,
    create_rfm_scores,
    assign_rfm_segment
)

@pytest.fixture
def sample_customer_transactions():
    """Tạo dữ liệu giao dịch sạch của khách hàng để test Feature Engineering."""
    return pd.DataFrame({
        'InvoiceNo': ['1001', '1002', '1003', '1004'],
        'StockCode': ['A1', 'A2', 'B1', 'B2'],
        'Quantity': [2, 1, 5, 2],
        'UnitPrice': [10.0, 20.0, 50.0, 15.0],
        'TotalAmount': [20.0, 20.0, 250.0, 30.0],
        'InvoiceDate': [
            '2011-10-01 10:00:00', 
            '2011-10-15 10:00:00', # Khách hàng 1 mua 2 lần
            '2011-12-01 10:00:00', 
            '2011-12-05 10:00:00'  # Khách hàng 2 mua 2 lần, mua gần đây hơn
        ],
        'CustomerID': ['1', '1', '2', '2'],
        'Country': ['UK', 'UK', 'UK', 'UK']
    })

def test_build_rfm_features(sample_customer_transactions):
    """Kiểm tra việc tính toán đúng các giá trị RFM."""
    # Set ngày tham chiếu là 2011-12-06
    ref_date = pd.to_datetime('2011-12-06 10:00:00')
    rfm, used_date = build_rfm_features(sample_customer_transactions, reference_date=ref_date)
    
    assert len(rfm) == 2, "Phải gom nhóm thành đúng 2 khách hàng"
    
    # Kiem tra Khach hang 1
    cust_1 = rfm[rfm['CustomerID'] == '1'].iloc[0]
    assert cust_1['Frequency'] == 2, "Tính sai Frequency"
    assert cust_1['Monetary'] == 40.0, "Tính sai Monetary"
    assert cust_1['Recency'] == 52, "Tính sai Recency (2011-12-06 trừ 2011-10-15 = 52 ngày)"

def test_create_rfm_scores():
    """Kiểm tra hàm qcut chia điểm từ 1-5."""
    # Tạo một dataframe RFM giả với phân phối rõ ràng
    dummy_rfm = pd.DataFrame({
        'CustomerID': ['1', '2', '3', '4', '5'],
        'Recency': [10, 20, 30, 40, 50],   # KH 1 mua gần nhất -> Điểm R phải cao nhất
        'Frequency': [50, 40, 30, 20, 10], # KH 1 mua nhiều nhất -> Điểm F phải cao nhất
        'Monetary': [500, 400, 300, 200, 100]
    })
    
    df_scored = create_rfm_scores(dummy_rfm, n_bins=5)
    
    cust_1 = df_scored[df_scored['CustomerID'] == '1'].iloc[0]
    cust_5 = df_scored[df_scored['CustomerID'] == '5'].iloc[0]
    
    assert cust_1['R_Score'] == 5, "Khách mua gần nhất phải có R_Score lớn nhất (5)"
    assert cust_1['F_Score'] == 5, "Khách mua nhiều nhất phải có F_Score lớn nhất (5)"
    assert cust_5['M_Score'] == 1, "Khách chi tiêu ít nhất phải có M_Score thấp nhất (1)"
    assert cust_1['RFM_Code'] == '555', "Lỗi nối chuỗi RFM_Code"

def test_assign_rfm_segment():
    """Kiểm tra logic phân nhóm khách hàng VIP."""
    # Giả lập 1 dòng dữ liệu của khách Champions (5-5-5)
    row_champion = pd.Series({'R_Score': 5, 'F_Score': 5, 'M_Score': 5})
    # Giả lập khách At Risk (R thấp, F cao)
    row_at_risk = pd.Series({'R_Score': 1, 'F_Score': 4, 'M_Score': 2})
    
    assert assign_rfm_segment(row_champion) == 'Champions', "Lỗi phân nhóm Champions"
    assert assign_rfm_segment(row_at_risk) == 'At Risk', "Lỗi phân nhóm At Risk"