import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Cấu hình trang
st.set_page_config(page_title="Dự Đoán Khách Hàng Mua Lại", layout="wide")
st.title("🛍️ Dashboard Dự Đoán Khách Hàng Mua Lại (Repeat Purchase)")

# Load mô hình
@st.cache_resource
def load_model():
    model_path = Path('models/classification/best_classifier.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success(" Đã tải mô hình Gradient Boosting thành công!")
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}")
    st.stop()

st.write("---")
st.subheader("1. Tải lên tập dữ liệu khách hàng (CSV)")
uploaded_file = st.file_uploader("Vui lòng tải lên file đặc trưng khách hàng (ví dụ: tập test hoặc dữ liệu mới)", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    st.write("Dữ liệu đầu vào:")
    st.dataframe(df.head())
    
    if st.button("Chạy Dự Đoán"):
        with st.spinner('Đang xử lý dữ liệu và dự đoán...'):
            # Loại bỏ cột định danh và nhãn thực tế nếu có
            X = df.drop(columns=['CustomerID', 'Is_Repeat_Buyer', 'repeat_purchase_90d'], errors='ignore')
            
            # Xử lý One-hot encoding để khớp với mô hình đã train
            X = pd.get_dummies(X, drop_first=True)
            X = X.fillna(0)
            
            # Đảm bảo các cột khớp hoàn toàn với mô hình
            model_features = model.feature_names_in_
            
            # Thêm các cột thiếu với giá trị 0
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            
            # Xóa các cột thừa và sắp xếp đúng thứ tự
            X = X[model_features]
            
            # Dự đoán
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            
            # Thêm kết quả vào DataFrame gốc
            df['Dự_Đoán_Mua_Lại'] = predictions
            df['Xác_Suất_Mua_Lại'] = probabilities.round(4)
            
            st.success("Dự đoán hoàn tất!")
            
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tổng số khách hàng", len(df))
            with col2:
                repeat_count = sum(predictions)
                st.metric("Số khách có khả năng mua lại", f"{repeat_count} ({(repeat_count/len(df)*100):.1f}%)")
                
            st.write("---")
            st.subheader("2. Chi tiết Dự Đoán")
            # Highlight khách hàng có xác suất mua lại > 50%
            st.dataframe(
                df[['CustomerID', 'Dự_Đoán_Mua_Lại', 'Xác_Suất_Mua_Lại']].style.map(
                    lambda x: 'background-color: lightgreen' if x == 1 else '', subset=['Dự_Đoán_Mua_Lại']
                )
            )