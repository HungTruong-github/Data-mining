"""
Module clustering: multi-algorithm customer segmentation.
Supports K-Means, GMM, Agglomerative, DBSCAN.
Computes Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn Index.
"""
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42


# ====================================================================
# 1. FEATURE PREPARATION
# ====================================================================
def prepare_clustering_features(df, rfm_cols=None):
    """
    Prepare RFM features: raw -> log1p -> StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain CustomerID and RFM columns.
    rfm_cols : list, optional
        Columns to use. Default: ['Recency', 'Frequency', 'Monetary'].

    Returns
    -------
    X_raw : np.ndarray
    X_log : np.ndarray
    X_scaled : np.ndarray
    scaler : StandardScaler
    feature_names : list
    customer_ids : pd.Series
    """
    if rfm_cols is None:
        rfm_cols = ['Recency', 'Frequency', 'Monetary']

    customer_ids = df['CustomerID'].copy()
    X_raw = df[rfm_cols].values.copy()
    X_log = np.log1p(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    return X_raw, X_log, X_scaled, scaler, rfm_cols, customer_ids


# ====================================================================
# 2. DUNN INDEX
# ====================================================================
def dunn_approximation(X, labels):
    """
    Compute Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter).
    Uses centroid-based distances for efficiency.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    # Compute centroids
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])

    # Min inter-cluster distance (between centroids)
    inter_dists = cdist(centroids, centroids)
    np.fill_diagonal(inter_dists, np.inf)
    min_inter = inter_dists.min()

    # Max intra-cluster diameter (max pairwise distance within any cluster)
    max_intra = 0
    for k in unique_labels:
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            # Use max distance from centroid * 2 as diameter approximation
            dists_to_centroid = np.sqrt(((cluster_points - centroids[k == unique_labels][0]) ** 2).sum(axis=1))
            diameter = 2 * dists_to_centroid.max()
            max_intra = max(max_intra, diameter)

    if max_intra == 0:
        return np.nan

    return min_inter / max_intra


# ====================================================================
# 3. COMPUTE ALL METRICS
# ====================================================================
def compute_clustering_metrics(X, labels):
    """
    Compute all clustering evaluation metrics.

    Returns dict with: silhouette, davies_bouldin, calinski_harabasz, dunn_index.
    """
    unique_labels = np.unique(labels[labels >= 0])  # exclude noise (-1)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return {
            'silhouette_score': np.nan,
            'davies_bouldin_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'dunn_approximation': np.nan,
        }

    # For metrics, only use non-noise points
    mask = labels >= 0
    X_valid = X[mask]
    labels_valid = labels[mask]

    metrics = {}
    try:
        metrics['silhouette_score'] = round(silhouette_score(X_valid, labels_valid), 4)
    except Exception:
        metrics['silhouette_score'] = np.nan

    try:
        metrics['davies_bouldin_score'] = round(davies_bouldin_score(X_valid, labels_valid), 4)
    except Exception:
        metrics['davies_bouldin_score'] = np.nan

    try:
        metrics['calinski_harabasz_score'] = round(calinski_harabasz_score(X_valid, labels_valid), 4)
    except Exception:
        metrics['calinski_harabasz_score'] = np.nan

    try:
        metrics['dunn_approximation'] = round(dunn_approximation(X_valid, labels_valid), 4)
    except Exception:
        metrics['dunn_approximation'] = np.nan

    return metrics


# ====================================================================
# 4. RUN INDIVIDUAL ALGORITHMS
# ====================================================================
def run_kmeans(X, n_clusters, random_state=RANDOM_STATE):
    """Run K-Means and return labels, model, metrics, runtime."""
    start = time.time()
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = model.fit_predict(X)
    runtime = round(time.time() - start, 4)

    metrics = compute_clustering_metrics(X, labels)
    metrics.update({
        'algorithm': 'K-Means',
        'n_clusters': n_clusters,
        'wcss': round(model.inertia_, 2),
        'noise_ratio': 0.0,
        'min_cluster_size': int(pd.Series(labels).value_counts().min()),
        'max_cluster_pct': round(pd.Series(labels).value_counts().max() / len(labels) * 100, 2),
        'runtime_seconds': runtime,
    })
    return labels, model, metrics


def run_gmm(X, n_clusters, random_state=RANDOM_STATE):
    """Run Gaussian Mixture Model."""
    start = time.time()
    model = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=3, max_iter=200)
    labels = model.fit_predict(X)
    runtime = round(time.time() - start, 4)

    metrics = compute_clustering_metrics(X, labels)
    metrics.update({
        'algorithm': 'GMM',
        'n_clusters': n_clusters,
        'bic': round(model.bic(X), 2),
        'aic': round(model.aic(X), 2),
        'noise_ratio': 0.0,
        'min_cluster_size': int(pd.Series(labels).value_counts().min()),
        'max_cluster_pct': round(pd.Series(labels).value_counts().max() / len(labels) * 100, 2),
        'runtime_seconds': runtime,
    })
    return labels, model, metrics


def run_agglomerative(X, n_clusters, linkage='ward'):
    """Run Agglomerative Clustering."""
    start = time.time()
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    runtime = round(time.time() - start, 4)

    metrics = compute_clustering_metrics(X, labels)
    metrics.update({
        'algorithm': f'Agglomerative({linkage})',
        'n_clusters': n_clusters,
        'noise_ratio': 0.0,
        'min_cluster_size': int(pd.Series(labels).value_counts().min()),
        'max_cluster_pct': round(pd.Series(labels).value_counts().max() / len(labels) * 100, 2),
        'runtime_seconds': runtime,
    })
    return labels, model, metrics


def run_dbscan(X, eps=0.5, min_samples=5):
    """Run DBSCAN."""
    start = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    runtime = round(time.time() - start, 4)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    noise_ratio = round(n_noise / len(labels) * 100, 2)

    metrics = compute_clustering_metrics(X, labels)

    # Cluster size stats (excluding noise)
    if n_clusters > 0:
        cluster_counts = pd.Series(labels[labels >= 0]).value_counts()
        min_cluster_size = int(cluster_counts.min())
        max_cluster_pct = round(cluster_counts.max() / len(labels) * 100, 2)
    else:
        min_cluster_size = 0
        max_cluster_pct = 0.0

    metrics.update({
        'algorithm': f'DBSCAN(eps={eps},min={min_samples})',
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'noise_count': n_noise,
        'min_cluster_size': min_cluster_size,
        'max_cluster_pct': max_cluster_pct,
        'runtime_seconds': runtime,
    })
    return labels, model, metrics


# ====================================================================
# 5. MULTI-ALGORITHM COMPARISON
# ====================================================================
def run_all_algorithms(X, k_range=range(2, 9)):
    """
    Run K-Means, GMM, Agglomerative for k in k_range, plus DBSCAN.
    Returns list of (algorithm_name, k, labels, model, metrics).
    """
    results = []

    # K-Means
    for k in k_range:
        labels, model, metrics = run_kmeans(X, k)
        results.append(('K-Means', k, labels, model, metrics))
        print(f"  K-Means K={k}: Silhouette={metrics['silhouette_score']:.4f}, "
              f"DB={metrics['davies_bouldin_score']:.4f}")

    # GMM
    for k in k_range:
        labels, model, metrics = run_gmm(X, k)
        results.append(('GMM', k, labels, model, metrics))
        print(f"  GMM K={k}: Silhouette={metrics['silhouette_score']:.4f}, "
              f"DB={metrics['davies_bouldin_score']:.4f}")

    # Agglomerative (Ward)
    for k in k_range:
        labels, model, metrics = run_agglomerative(X, k)
        results.append(('Agglomerative', k, labels, model, metrics))
        print(f"  Agglomerative K={k}: Silhouette={metrics['silhouette_score']:.4f}, "
              f"DB={metrics['davies_bouldin_score']:.4f}")

    # DBSCAN with several eps values
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_s in [5, 10]:
            labels, model, metrics = run_dbscan(X, eps=eps, min_samples=min_s)
            k = metrics['n_clusters']
            results.append(('DBSCAN', k, labels, model, metrics))
            print(f"  DBSCAN eps={eps} min={min_s}: clusters={k}, "
                  f"noise={metrics['noise_ratio']:.1f}%, "
                  f"Silhouette={metrics.get('silhouette_score', 'N/A')}")

    return results


def build_comparison_table(results):
    """Build DataFrame comparing all algorithm runs."""
    rows = []
    for algo_name, k, labels, model, metrics in results:
        row = {
            'algorithm': metrics.get('algorithm', algo_name),
            'n_clusters': metrics['n_clusters'],
            'silhouette_score': metrics.get('silhouette_score', np.nan),
            'davies_bouldin_score': metrics.get('davies_bouldin_score', np.nan),
            'calinski_harabasz_score': metrics.get('calinski_harabasz_score', np.nan),
            'dunn_approximation': metrics.get('dunn_approximation', np.nan),
            'noise_ratio': metrics.get('noise_ratio', 0.0),
            'min_cluster_size': metrics.get('min_cluster_size', 0),
            'max_cluster_pct': metrics.get('max_cluster_pct', 0.0),
            'runtime_seconds': metrics.get('runtime_seconds', 0.0),
            'is_selected': False,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ====================================================================
# 6. MODEL SELECTION
# ====================================================================
def select_best_model(results, comparison_df):
    """
    Select best model using multi-criteria ranking.
    Filters: n_clusters >= 2, max_cluster_pct < 90%, min_cluster_size >= 10.
    Ranks by: silhouette (desc), davies_bouldin (asc), calinski_harabasz (desc).
    """
    df = comparison_df.copy()

    # Filter valid configurations
    valid = df[
        (df['n_clusters'] >= 2) &
        (df['max_cluster_pct'] < 90) &
        (df['min_cluster_size'] >= 10) &
        (df['silhouette_score'].notna())
    ].copy()

    if valid.empty:
        # Fallback: relax constraints
        valid = df[
            (df['n_clusters'] >= 2) &
            (df['silhouette_score'].notna())
        ].copy()

    if valid.empty:
        print("WARNING: No valid configurations found!")
        return None, None, None

    # Rank each metric
    valid['rank_sil'] = valid['silhouette_score'].rank(ascending=False)
    valid['rank_db'] = valid['davies_bouldin_score'].rank(ascending=True)
    valid['rank_ch'] = valid['calinski_harabasz_score'].rank(ascending=False)
    valid['rank_dunn'] = valid['dunn_approximation'].rank(ascending=False)

    # Combined rank (lower is better)
    valid['combined_rank'] = valid['rank_sil'] + valid['rank_db'] + valid['rank_ch'] + valid['rank_dunn']

    # Sort by combined rank
    valid = valid.sort_values('combined_rank')
    best_idx = valid.index[0]

    # Find the corresponding result
    best_row = comparison_df.loc[best_idx]
    best_algo = best_row['algorithm']
    best_k = int(best_row['n_clusters'])

    # Find matching result tuple
    for algo_name, k, labels, model, metrics in results:
        if metrics.get('algorithm', algo_name) == best_algo and metrics['n_clusters'] == best_k:
            return labels, model, metrics

    return None, None, None


# ====================================================================
# 7. CLUSTER PROFILES
# ====================================================================
def create_cluster_profiles(df_rfm, labels, rfm_cols=None):
    """
    Create cluster profiles with RFM statistics.

    Parameters
    ----------
    df_rfm : pd.DataFrame with RFM columns
    labels : array-like cluster labels
    rfm_cols : list

    Returns
    -------
    pd.DataFrame with cluster profiles
    """
    if rfm_cols is None:
        rfm_cols = ['Recency', 'Frequency', 'Monetary']

    df = df_rfm.copy()
    df['Cluster'] = labels

    profiles = []
    for cluster_id in sorted(df['Cluster'].unique()):
        if cluster_id < 0:
            continue  # skip noise
        cluster_data = df[df['Cluster'] == cluster_id]
        profile = {
            'Cluster': cluster_id,
            'CustomerCount': len(cluster_data),
            'CustomerPercentage': round(len(cluster_data) / len(df) * 100, 2),
        }
        for col in rfm_cols:
            profile[f'Mean{col}'] = round(cluster_data[col].mean(), 2)
            profile[f'Median{col}'] = round(cluster_data[col].median(), 2)
        if 'Monetary' in rfm_cols:
            profile['TotalMonetary'] = round(cluster_data['Monetary'].sum(), 2)
        profiles.append(profile)

    return pd.DataFrame(profiles)


def assign_business_names(profiles):
    """
    Assign business segment names based on RFM profile characteristics.
    Uses relative ranking within profiles, not hard-coded thresholds.
    """
    df = profiles.copy()
    n = len(df)

    if n == 0:
        df['BusinessName'] = []
        return df

    # Rank clusters (lower rank = better for that metric direction)
    df['RecencyRank'] = df['MeanRecency'].rank(ascending=True)       # low recency = good
    df['FrequencyRank'] = df['MeanFrequency'].rank(ascending=False)  # high freq = good
    df['MonetaryRank'] = df['MeanMonetary'].rank(ascending=False)    # high monetary = good
    df['OverallScore'] = df['RecencyRank'] + df['FrequencyRank'] + df['MonetaryRank']

    # Sort by overall score (lower = better)
    df = df.sort_values('OverallScore').reset_index(drop=True)

    # Name assignment based on relative position
    name_pool = ['Best Customers', 'Loyal Customers', 'Potential Customers',
                 'At Risk', 'Lost Customers', 'Hibernating', 'New Customers',
                 'Occasional Buyers']

    names = {}
    for i, row in df.iterrows():
        cluster_id = row['Cluster']
        if i == 0:
            # Best overall RFM
            names[cluster_id] = 'Best Customers'
        elif row['MeanRecency'] == df['MeanRecency'].min() and i > 0:
            names[cluster_id] = 'Loyal Customers'
        elif row['MeanRecency'] == df['MeanRecency'].max():
            names[cluster_id] = 'Lost Customers'
        elif row['MeanFrequency'] <= df['MeanFrequency'].median() and row['MeanRecency'] > df['MeanRecency'].median():
            names[cluster_id] = 'At Risk'
        elif row['MeanMonetary'] > df['MeanMonetary'].median() and row['MeanRecency'] <= df['MeanRecency'].median():
            names[cluster_id] = 'Potential Customers'
        else:
            # Fallback
            remaining = [n for n in name_pool if n not in names.values()]
            names[cluster_id] = remaining[0] if remaining else f'Segment {cluster_id}'

    df['BusinessName'] = df['Cluster'].map(names)

    # Drop ranking columns
    df = df.drop(columns=['RecencyRank', 'FrequencyRank', 'MonetaryRank', 'OverallScore'])

    return df


# ====================================================================
# 8. SAVE RESULTS
# ====================================================================
def save_clustering_results(model, scaler, labels, config, save_dir):
    """
    Save model, scaler, and configuration.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_dir / 'clustering_model.pkl')
    joblib.dump(scaler, save_dir / 'scaler.pkl')
    joblib.dump(config, save_dir / 'clustering_config.pkl')

    print(f"  [OK] Model saved to {save_dir}")


# ====================================================================
# 9. PCA FOR VISUALIZATION
# ====================================================================
def pca_2d(X, labels):
    """Reduce to 2D using PCA for visualization."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    df = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
    df['Cluster'] = labels
    df['ExplainedVariance'] = [
        round(pca.explained_variance_ratio_[0] * 100, 1),
        round(pca.explained_variance_ratio_[1] * 100, 1),
    ] + [np.nan] * (len(df) - 2)

    return df, pca
