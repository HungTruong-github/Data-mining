import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_clusters(df, output_dir=None):
    """
    Trực quan hóa kết quả phân cụm K-Means bằng Boxplots và Scatter Plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe chứa dữ liệu RFM và cột 'Cluster'.
    output_dir : str hoặc Path, optional
        Đường dẫn thư mục để lưu biểu đồ.
    """
    # Tạo thư mục nếu có yêu cầu lưu ảnh
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    sns.set_theme(style="whitegrid")
    
    # --- 1. Vẽ Boxplots phân phối R, F, M ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = "Set2"
    
    sns.boxplot(x='Cluster', y='Recency', data=df, ax=axes[0], palette=palette)
    axes[0].set_title('Phân phối Recency theo Cụm', fontsize=12)
    axes[0].set_ylabel('Số ngày (Recency)')
    
    sns.boxplot(x='Cluster', y='Frequency', data=df, ax=axes[1], palette=palette)
    axes[1].set_title('Phân phối Frequency theo Cụm', fontsize=12)
    axes[1].set_ylabel('Số lần mua (Frequency)')
    
    sns.boxplot(x='Cluster', y='Monetary', data=df, ax=axes[2], palette=palette)
    axes[2].set_title('Phân phối Monetary theo Cụm', fontsize=12)
    axes[2].set_ylabel('Tổng chi tiêu (Monetary)')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'cluster_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 2. Vẽ Scatter Plot (Recency vs Monetary) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Recency', 
        y='Monetary', 
        hue='Cluster', 
        palette=palette, 
        s=80, 
        alpha=0.7,
        edgecolor=None
    )
    plt.title('Mức độ phân tách cụm: Recency vs Monetary', fontsize=14, pad=15)
    
    if output_dir:
        plt.savefig(output_dir / 'cluster_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()