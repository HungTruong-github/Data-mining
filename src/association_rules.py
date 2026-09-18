import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def prepare_basket_matrix(df):
    """
    Chuyển đổi dữ liệu giao dịch sạch thành ma trận One-Hot (Basket Matrix).
    Sử dụng Description thay vì StockCode để dễ đọc kết quả luật kết hợp.
    """
    print("Đang chuẩn bị ma trận giỏ hàng (quá trình này có thể mất chút thời gian)...")
    
    # Nhóm theo hóa đơn và tên sản phẩm, đếm số lượng, sau đó pivot thành ma trận
    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    
    # Hàm chuyển đổi: Nếu số lượng <= 0 thì là 0, >= 1 thì là 1
    def encode_units(x):
        if x <= 0: return 0
        if x >= 1: return 1
    
    # Áp dụng hàm encode và ép kiểu sang boolean để mlxtend chạy nhanh hơn
    basket_sets = basket.map(encode_units).astype(bool)
    
    print(f"[OK] Đã tạo xong ma trận giỏ hàng với kích thước: {basket_sets.shape}")
    return basket_sets

def extract_association_rules(basket_matrix, min_support=0.02, min_threshold=0.5):
    """
    Khai phá luật kết hợp sử dụng thuật toán FP-Growth.
    """
    print(f"Đang chạy FP-Growth với min_support={min_support}...")
    
    # 1. Tìm tập phổ biến
    frequent_itemsets = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("Cảnh báo: Không tìm thấy tập sản phẩm phổ biến. Hãy thử giảm min_support.")
        return pd.DataFrame()
        
    print(f"Tìm thấy {len(frequent_itemsets)} tập phổ biến. Đang sinh luật...")
    
    # 2. Sinh luật kết hợp dựa trên min_threshold (mặc định cho độ tin cậy - confidence)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    print(f"[OK] Đã sinh thành công {len(rules)} luật kết hợp.")
    return rules