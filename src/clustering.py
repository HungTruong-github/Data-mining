import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

def prepare_clustering_features(df, features_to_scale=['Recency', 'Frequency', 'Monetary']):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler để tránh biến Monetary áp đảo.
    """
    X = df[features_to_scale].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Trả về dữ liệu đã scale và đối tượng scaler để dùng lại cho dữ liệu mới
    return X_scaled, scaler

def find_optimal_k(X_scaled, max_k=10):
    """
    Tính toán chỉ số WCSS (Elbow) và Silhouette Score để tìm K tối ưu.
    """
    wcss = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
    return {
        'k_values': list(k_values), 
        'wcss': wcss, 
        'silhouette_scores': silhouette_scores
    }

def train_kmeans(X_scaled, n_clusters, scaler=None, model_save_path=None):
    """
    Huấn luyện K-Means với số cụm K đã chốt và lưu mô hình.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Lưu mô hình nếu có đường dẫn
    if model_save_path:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        # Đóng gói cả model và scaler vào 1 file .pkl
        joblib.dump({'model': kmeans, 'scaler': scaler}, model_save_path)
        
    return kmeans, clusters