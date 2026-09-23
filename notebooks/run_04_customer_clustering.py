"""
Runner script for 04_customer_clustering.
Run from project root: python notebooks/run_04_customer_clustering.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, FIGURES_CLUSTERING, TABLES_CLUSTERING,
    MODELS_CLUSTERING_DIR, RANDOM_STATE
)
from src.clustering import (
    prepare_clustering_features, run_kmeans, run_gmm, run_agglomerative, run_dbscan,
    build_comparison_table, select_best_model, create_cluster_profiles, assign_business_names,
    save_clustering_results, pca_2d
)
from src.visualization import (
    plot_elbow_curve, plot_silhouette_comparison, plot_metrics_comparison,
    plot_cluster_distribution, plot_cluster_rfm_boxplots, plot_pca_2d, plot_rfm_heatmap
)

print("=== 04. CUSTOMER CLUSTERING ===")
df = pd.read_csv(PROCESSED_DIR / 'rfm_clustering_features.csv')
print(f"Loaded {len(df):,} customers.")

# Validate
assert df['CustomerID'].isnull().sum() == 0, 'Missing CustomerID'
assert df['CustomerID'].duplicated().sum() == 0, 'Duplicate CustomerID'
assert (df['Recency'] >= 0).all() and (df['Frequency'] >= 1).all() and (df['Monetary'] > 0).all(), 'Invalid RFM values'

rfm_cols = ['Recency', 'Frequency', 'Monetary']
X_raw, X_log, X_scaled, scaler, feature_names, customer_ids = prepare_clustering_features(df, rfm_cols)

print("\n--- Running Algorithms ---")
k_range = range(2, 9)
all_results = []
kmeans_results = []

for k in k_range:
    labels, model, metrics = run_kmeans(X_scaled, k)
    kmeans_results.append(('K-Means', k, labels, model, metrics))
    all_results.append(('K-Means', k, labels, model, metrics))

for k in k_range:
    labels, model, metrics = run_gmm(X_scaled, k)
    all_results.append(('GMM', k, labels, model, metrics))

for k in k_range:
    labels, model, metrics = run_agglomerative(X_scaled, k, linkage='ward')
    all_results.append(('Agglomerative(ward)', k, labels, model, metrics))

dbscan_configs = [(0.3, 5), (0.5, 5), (0.7, 5), (1.0, 5), (0.5, 10)]
for eps, min_s in dbscan_configs:
    labels, model, metrics = run_dbscan(X_scaled, eps=eps, min_samples=min_s)
    all_results.append(('DBSCAN', metrics['n_clusters'], labels, model, metrics))

comparison_df = build_comparison_table(all_results)
best_labels, best_model, best_metrics = select_best_model(all_results, comparison_df)

best_algo = best_metrics['algorithm']
best_k = best_metrics['n_clusters']
print(f"\nSelected Model: {best_algo} K={best_k}")
print(f"Silhouette: {best_metrics['silhouette_score']:.4f}")
print(f"Davies-Bouldin: {best_metrics['davies_bouldin_score']:.4f}")

# Profile
profiles = create_cluster_profiles(df, best_labels, rfm_cols)
profiles = assign_business_names(profiles)
name_map = dict(zip(profiles['Cluster'].astype(int), profiles['BusinessName']))

# Save files
comparison_df.to_csv(TABLES_CLUSTERING / 'clustering_algorithm_comparison.csv', index=False)
profiles.to_csv(TABLES_CLUSTERING / 'cluster_profiles.csv', index=False)

df_result = df[['CustomerID'] + rfm_cols].copy()
df_result['Cluster'] = best_labels
df_result['Algorithm'] = best_algo
df_result['FeatureVersion'] = 'log1p_scaled'
df_result['BusinessSegment'] = df_result['Cluster'].map(name_map)
df_result.to_csv(PROCESSED_DIR / 'customer_clusters.csv', index=False)

config = {
    'algorithm': best_algo, 'n_clusters': best_k, 'feature_names': feature_names,
    'feature_version': 'log1p_scaled', 'metrics': best_metrics, 'n_customers': len(df)
}
save_clustering_results(best_model, scaler, best_labels, config, MODELS_CLUSTERING_DIR)

# Generate visualizations
print("\n--- Generating Visualizations ---")
import matplotlib
matplotlib.use('Agg')
plot_elbow_curve(kmeans_results, best_k, best_algo, save_path=FIGURES_CLUSTERING / 'elbow_curve.png')
plot_silhouette_comparison(comparison_df, k_range, save_path=FIGURES_CLUSTERING / 'silhouette_comparison.png')
plot_metrics_comparison(comparison_df, k_range, save_path=FIGURES_CLUSTERING / 'metrics_comparison.png')
plot_cluster_distribution(best_labels, len(df), name_map, best_algo, best_k, save_path=FIGURES_CLUSTERING / 'cluster_distribution.png')
plot_cluster_rfm_boxplots(df, best_labels, name_map, rfm_cols, best_algo, best_k, save_path=FIGURES_CLUSTERING / 'cluster_rfm_boxplots.png')
pca_df, pca_model = pca_2d(X_scaled, best_labels)
plot_pca_2d(pca_df, pca_model, name_map, best_algo, best_k, save_path=FIGURES_CLUSTERING / 'pca_2d_clusters.png')
plot_rfm_heatmap(profiles, best_k, save_path=FIGURES_CLUSTERING / 'rfm_heatmap.png')

print("=== CLUSTERING RUNNER COMPLETE ===")
