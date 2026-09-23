"""
Module visualization: Provides plotting functions for both runner scripts and notebooks.
Ensures identical and live-generated charts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ====================================================================
# CLUSTERING VISUALIZATIONS
# ====================================================================

def plot_elbow_curve(kmeans_results, best_k=None, best_algo=None, save_path=None):
    """Plot Elbow Curve for K-Means."""
    k_vals = [m['n_clusters'] for _, _, _, _, m in kmeans_results]
    wcss_vals = [m['wcss'] for _, _, _, _, m in kmeans_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_vals, wcss_vals, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('WCSS', fontsize=12)
    ax.set_title('Elbow Method (K-Means)', fontsize=14, fontweight='bold')
    ax.set_xticks(k_vals)
    if best_algo and best_algo.startswith('K-Means') and best_k:
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={best_k}')
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_silhouette_comparison(comparison_df, k_range, save_path=None):
    """Plot Silhouette scores across algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name in ['K-Means', 'GMM', 'Agglomerative(ward)']:
        algo_rows = comparison_df[comparison_df['algorithm'] == algo_name].sort_values('n_clusters')
        if not algo_rows.empty:
            ax.plot(algo_rows['n_clusters'], algo_rows['silhouette_score'],
                    'o-', linewidth=2, markersize=8, label=algo_name)

    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by K — Algorithm Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(comparison_df, k_range, save_path=None):
    """Plot Sil, DB, CH, and Dunn across algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics_to_plot = [
        ('silhouette_score', 'Silhouette Score (higher=better)', True),
        ('davies_bouldin_score', 'Davies-Bouldin Index (lower=better)', False),
        ('calinski_harabasz_score', 'Calinski-Harabasz Index (higher=better)', True),
        ('dunn_approximation', 'Dunn Approximation (higher=better)', True),
    ]

    for ax, (metric_name, title, higher_better) in zip(axes.flat, metrics_to_plot):
        for algo_name in ['K-Means', 'GMM', 'Agglomerative(ward)']:
            algo_rows = comparison_df[comparison_df['algorithm'] == algo_name].sort_values('n_clusters')
            if not algo_rows.empty and algo_rows[metric_name].notna().any():
                ax.plot(algo_rows['n_clusters'], algo_rows[metric_name],
                        'o-', linewidth=2, markersize=6, label=algo_name)
        ax.set_xlabel('K')
        ax.set_ylabel(metric_name)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xticks(list(k_range))

    plt.suptitle('Clustering Metrics Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_cluster_distribution(labels, n_customers, name_map, best_algo, best_k, save_path=None):
    """Plot customer counts per cluster."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    colors = sns.color_palette("Set2", n_colors=len(cluster_counts))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, count in zip(bars, cluster_counts.values):
        pct = count / n_customers * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    cluster_names = [name_map.get(i, f'Cluster {i}') for i in cluster_counts.index]
    ax.set_xticks(cluster_counts.index)
    ax.set_xticklabels(cluster_names, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title(f'Customer Distribution by Cluster ({best_algo}, K={best_k})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_cluster_rfm_boxplots(df, labels, name_map, rfm_cols, best_algo, best_k, save_path=None):
    """Plot RFM boxplots by cluster."""
    df_plot = df[rfm_cols].copy()
    df_plot['Cluster'] = labels
    df_plot['BusinessSegment'] = df_plot['Cluster'].map(name_map)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for i, col in enumerate(rfm_cols):
        sns.boxplot(x='BusinessSegment', y=col, data=df_plot, ax=axes[i],
                    palette="Set2", showfliers=False)
        axes[i].set_title(f'{col} by Cluster', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=20)

    plt.suptitle(f'RFM Distribution by Cluster ({best_algo}, K={best_k})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_pca_2d(pca_df, pca_model, name_map, best_algo, best_k, save_path=None):
    """Plot 2D PCA representation of clusters."""
    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster_id in sorted(pca_df['Cluster'].unique()):
        mask = pca_df['Cluster'] == cluster_id
        cluster_name = name_map.get(cluster_id, f'Cluster {cluster_id}')
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                   alpha=0.6, s=30, label=cluster_name, edgecolors='none')

    ev1 = round(pca_model.explained_variance_ratio_[0] * 100, 1)
    ev2 = round(pca_model.explained_variance_ratio_[1] * 100, 1)
    ax.set_xlabel(f'PC1 ({ev1}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev2}% variance)', fontsize=12)
    ax.set_title(f'PCA 2D Visualization ({best_algo}, K={best_k})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_rfm_heatmap(profiles, best_k, save_path=None):
    """Plot normalized RFM heatmap."""
    fig, ax = plt.subplots(figsize=(10, max(6, best_k + 1)))
    heatmap_data = profiles[['BusinessName', 'MeanRecency', 'MeanFrequency', 'MeanMonetary']].set_index('BusinessName')
    
    hm_scaler = MinMaxScaler()
    hm_normalized = pd.DataFrame(
        hm_scaler.fit_transform(heatmap_data),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )
    sns.heatmap(hm_normalized, annot=heatmap_data.values, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, annot_kws={'fontsize': 11})
    ax.set_title('RFM Profile Heatmap by Segment', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ====================================================================
# ASSOCIATION RULES VISUALIZATIONS
# ====================================================================

def plot_top_products(product_freq, sc_to_desc, n_invoices, save_path=None):
    """Plot Top 20 Most Frequent Products."""
    fig, ax = plt.subplots(figsize=(14, 8))
    top20 = product_freq.head(20)
    top20_names = [sc_to_desc.get(str(sc), str(sc))[:40] for sc in top20.index]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    bars = ax.barh(range(len(top20)), top20.values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20_names, fontsize=10)
    ax.set_xlabel('Number of Invoices', fontsize=12)
    ax.set_title('Top 20 Most Frequent Products', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, count in zip(bars, top20.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_algorithm_comparison(apriori_itemsets, fpgrowth_itemsets, apriori_runtime, fpgrowth_runtime, save_path=None):
    """Plot runtime and itemset counts comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    algos = ['Apriori', 'FP-Growth']
    counts = [len(apriori_itemsets), len(fpgrowth_itemsets)]
    bars = ax.bar(algos, counts, color=['#3b82f6', '#22c55e'], edgecolor='white', linewidth=2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frequent Itemsets', fontsize=12)
    ax.set_title('Frequent Itemsets Count', fontsize=13, fontweight='bold')

    ax = axes[1]
    runtimes = [apriori_runtime, fpgrowth_runtime]
    bars = ax.bar(algos, runtimes, color=['#3b82f6', '#22c55e'], edgecolor='white', linewidth=2)
    for bar, rt in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rt:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison', fontsize=13, fontweight='bold')

    plt.suptitle('Apriori vs FP-Growth Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_scatter_support_confidence_lift(rules, save_path=None):
    """Scatter plot: X=Support, Y=Confidence, Color=Lift"""
    if len(rules) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        rules['support'], rules['confidence'],
        c=rules['lift'], cmap='RdYlGn', alpha=0.7, s=40, edgecolors='gray', linewidths=0.3
    )
    cbar = plt.colorbar(scatter, ax=ax, label='Lift')
    ax.set_xlabel('Support', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Association Rules: Support vs Confidence (colored by Lift)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_top_rules_by_lift(rules, sc_to_desc, frozenset_to_names, save_path=None):
    """Plot top 20 rules by lift."""
    if len(rules) == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    top_lift = rules.nlargest(20, 'lift').copy()
    top_lift['rule_name'] = top_lift.apply(
        lambda r: frozenset_to_names(r['antecedents'], sc_to_desc)[:30] +
                  ' -> ' + frozenset_to_names(r['consequents'], sc_to_desc)[:30], axis=1)
    
    ax.barh(range(len(top_lift)), top_lift['lift'].values,
            color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_lift))))
    ax.set_yticks(range(len(top_lift)))
    ax.set_yticklabels(top_lift['rule_name'].values, fontsize=9)
    ax.set_xlabel('Lift', fontsize=12)
    ax.set_title('Top 20 Association Rules by Lift', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
