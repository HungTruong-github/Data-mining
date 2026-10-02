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

def test_rfm_fallback_and_monotonicity():
    """Kiểm tra tính đơn điệu và hướng điểm khi có nhiều giá trị trùng (ties) gây kích hoạt fallback."""
    # Tập dữ liệu với 60% Frequency = 1 (kích hoạt fallback qcut)
    df_ties = pd.DataFrame({
        'CustomerID': [str(i) for i in range(10)],
        'Recency': [5, 5, 10, 20, 50, 100, 150, 200, 300, 300],
        'Frequency': [1, 1, 1, 1, 1, 1, 2, 4, 8, 15],
        'Monetary': [20, 20, 30, 50, 100, 200, 400, 800, 1500, 3000]
    })
    
    scored = create_rfm_scores(df_ties, n_bins=5)
    
    # Recency: giá trị nhỏ hơn phải có R_Score lớn hơn hoặc bằng
    rec_small = scored[scored['Recency'] == 5]['R_Score'].values
    rec_large = scored[scored['Recency'] == 300]['R_Score'].values
    assert rec_small.min() >= rec_large.max(), f"R_Score hướng sai: {rec_small} vs {rec_large}"
    assert rec_small.min() == rec_small.max(), "Các giá trị Recency giống nhau phải cùng điểm"
    
    # Frequency: giá trị lớn hơn phải có F_Score lớn hơn hoặc bằng
    freq_small = scored[scored['Frequency'] == 1]['F_Score'].values
    freq_large = scored[scored['Frequency'] == 15]['F_Score'].values
    assert freq_large.min() >= freq_small.max(), f"F_Score hướng sai: {freq_large} vs {freq_small}"
    assert freq_small.min() == freq_small.max(), "Các giá trị Frequency giống nhau phải cùng điểm"


def test_rfm_row_order_invariance():
    """Kiểm tra tính bất biến theo thứ tự dòng: xáo trộn thứ tự không được làm đổi điểm số."""
    df_base = pd.DataFrame({
        'CustomerID': [f"C_{i}" for i in range(20)],
        'Recency': [1, 2, 3, 5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200],
        'Frequency': [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
        'Monetary': [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
    })
    
    scored_original = create_rfm_scores(df_base, n_bins=5).set_index('CustomerID')
    
    # Xáo trộn dòng ngẫu nhiên
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df_base))
    df_shuffled = df_base.iloc[shuffled_indices].copy()
    
    scored_shuffled = create_rfm_scores(df_shuffled, n_bins=5).set_index('CustomerID')
    
    # Đối chiếu từng khách hàng
    for col in ['R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'RFM_Code']:
        diffs = (scored_original[col] != scored_shuffled.loc[scored_original.index, col]).sum()
        assert diffs == 0, f"Xáo trộn thứ tự dòng làm thay đổi kết quả cột {col} ở {diffs} khách hàng!"


def test_rfm_all_equal_values():
    """Kiểm tra trường hợp đặc biệt khi tất cả khách hàng có cùng giá trị (độ biến thiên = 0)."""
    df_equal = pd.DataFrame({
        'CustomerID': ['A', 'B', 'C', 'D'],
        'Recency': [30, 30, 30, 30],
        'Frequency': [2, 2, 2, 2],
        'Monetary': [100.0, 100.0, 100.0, 100.0]
    })
    scored = create_rfm_scores(df_equal, n_bins=5)
    # Tất cả phải nhận cùng một mức điểm hợp lệ
    assert len(scored['R_Score'].unique()) == 1
    assert len(scored['F_Score'].unique()) == 1
    assert len(scored['M_Score'].unique()) == 1
    assert scored['R_Score'].iloc[0] in [1, 2, 3, 4, 5]
