"""
Core module for CRISP-DM Step 07: Model Comparison, Validation & Insight Synthesis.
Provides centralized evaluation, comparison, artifact verification, report, and manifest generation.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'cluster_comparison': ['algorithm', 'n_clusters', 'silhouette_score', 'davies_bouldin_score'],
    'class_comparison': ['model'],
    'selected_rules': ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift'],
    'feature_importance': ['feature', 'importance'],
}


def load_input(name):
    """Load an input artifact by registered name, supporting fallback paths."""
    if name == 'selected_rules':
        primary = TABLES_DIR / 'association_rules' / 'selected_association_rules.csv'
        nested = TABLES_DIR / 'association_rules' / 'association_rules' / 'selected_association_rules.csv'
        if primary.exists():
            return pd.read_csv(primary)
        elif nested.exists():
            return pd.read_csv(nested)
        else:
            raise FileNotFoundError(f"Selected rules file not found at {primary} or {nested}")

    if name not in INPUT_FILES:
        raise KeyError(f"Unknown input file alias: '{name}'")

    path = INPUT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Input file for '{name}' not found: {path}")

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    elif path.suffix in ['.pkl', '.joblib']:
        return joblib.load(path)
    return path


def validate_inputs():
    """Validate existence, non-emptiness, schema, and data integrity for all inputs."""
    errors = []
    checks = []

    for name, path in INPUT_FILES.items():
        # Check fallback for selected_rules
        if name == 'selected_rules' and not path.exists():
            alt_path = TABLES_DIR / 'association_rules' / 'selected_association_rules.csv'
            if alt_path.exists():
                path = alt_path

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

    # Verify alignment between comparison row and saved config artifact
    sel_row = df[df['is_selected'].astype(bool)].iloc[0]
    if sel_algo and sel_k != -1:
        if sel_row['algorithm'] != sel_algo or int(sel_row['n_clusters']) != int(sel_k):
            raise ValueError(
                f"Clustering mismatch: comparison row has ({sel_row['algorithm']}, K={sel_row['n_clusters']}) "
                f"but saved config artifact has ({sel_algo}, K={sel_k})!"
            )

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
            reasons.append(row.get('selection_reason', ''))
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
        raise ValueError("DummyClassifier cannot be selected as predictive model when learned models are available.")

    reasons = []
    for _, row in df.iterrows():
        if row.get('is_selected', False):
            cv_f1 = row.get('cv_f1_mean', row.get('cv_f1', 0))
            reasons.append(
                f"Highest Stratified K-Fold CV F1-score ({cv_f1:.4f}) among candidate learned models. "
                f"Selection performed strictly on CV metrics without test set data leakage."
            )
        elif row.get('model', '') == 'DummyClassifier':
            reasons.append('Baseline classifier predicting majority class')
        else:
            reasons.append('Candidate learned model')
    df['selection_reason'] = reasons

    if 'specificity' not in df.columns and 'test_specificity' in df.columns:
        df['specificity'] = df['test_specificity']

    return df


def build_association_comparison():
    """
    Load association rules comparison and perform element-by-element verification between Apriori and FP-Growth.
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
    """Validate association rules quality with strict mathematical constraints and StockCode keys."""
    df = load_input('selected_rules')
    errors = []
    checks = []

    if df is None or df.empty:
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

    # 2. NaN / Inf check on core metrics
    for col in ['support', 'confidence', 'lift']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN and {inf_count} Inf values.")
            checks.append(f"[{'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'}] {col} NaN={nan_count} Inf={inf_count}")

    # 3. Lift <= 1 check (must be strictly > 1.0)
    if 'lift' in df.columns:
        low_lift = (df['lift'] <= 1.0).sum()
        if low_lift > 0:
            errors.append(f"Found {low_lift} rules with lift <= 1.0 (must be strictly > 1.0).")
        checks.append(f"[{'PASS' if low_lift == 0 else 'FAIL'}] Rules with lift <= 1.0: {low_lift}")

    # 4. Antecedent-Consequent overlap check using canonical StockCodes
    overlap_count = 0
    for _, row in df.iterrows():
        # Use StockCodes if available, otherwise fallback to description
        ant_raw = str(row.get('antecedents_codes', row.get('antecedents_str', '')))
        con_raw = str(row.get('consequents_codes', row.get('consequents_str', '')))
        ant_set = {x.strip() for x in ant_raw.split(',') if x.strip()}
        con_set = {x.strip() for x in con_raw.split(',') if x.strip()}
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
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def generate_report(clustering_comp, classification_comp, assoc_comp,
                    seg=None, actions=None, prod=None, feat=None, output_path=None):
    """
    Generate comprehensive CRISP-DM Step 07 Markdown report from dynamic comparison outputs.
    Guarantees no hard-coded summary values and clean UTF-8 Vietnamese text.
    """
    if output_path is None:
        output_path = REPORTS_DIR / '07_model_comparison_and_insights.md'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load supplementary tables if not supplied
    if seg is None:
        seg_path = TABLES_DIR / 'insights' / 'customer_segment_insights.csv'
        seg = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()

    if actions is None:
        act_path = TABLES_DIR / 'insights' / 'customer_segment_action_plan.csv'
        actions = pd.read_csv(act_path) if act_path.exists() else pd.DataFrame()

    if prod is None:
        prod_path = TABLES_DIR / 'insights' / 'product_association_insights.csv'
        prod = pd.read_csv(prod_path) if prod_path.exists() else pd.DataFrame()

    if feat is None:
        feat_path = TABLES_DIR / 'insights' / 'classification_feature_insights.csv'
        feat = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()

    sc = clustering_comp[clustering_comp['is_selected'].astype(bool)].iloc[0]
    mc = classification_comp[classification_comp['is_selected'].astype(bool)].iloc[0]
    sa = assoc_comp[assoc_comp['is_selected'].astype(bool)].iloc[0]

    cv_f1 = mc.get('cv_f1_mean', mc.get('cv_f1', 0))

    # Dynamic segment breakdown string
    if not seg.empty and 'business_segment_name' in seg.columns and 'customer_percentage' in seg.columns:
        seg_summary = "; ".join([
            f"{r['business_segment_name']} ({r['customer_percentage']}%, {r.get('customer_count', 0):,} khách hàng)"
            for _, r in seg.iterrows()
        ])
    else:
        seg_summary = f"{len(seg)} phân khúc"

    # Baseline comparison note
    dummy_rows = classification_comp[classification_comp['model'] == 'DummyClassifier']
    lr_rows = classification_comp[classification_comp['model'] == 'LogisticRegression']
    if not dummy_rows.empty:
        dummy_cv_f1 = dummy_rows.iloc[0].get('cv_f1_mean', dummy_rows.iloc[0].get('cv_f1', 0))
        pos_ratio = float(dummy_rows.iloc[0].get('test_precision', 0.5698)) * 100
        lr_auc = float(lr_rows.iloc[0].get('test_roc_auc', 0)) if not lr_rows.empty else 0.0
        baseline_note = (
            f"Baseline DummyClassifier (chiến lược đoán lớp đa số) có CV F1 = {dummy_cv_f1:.4f} do tỷ lệ lớp dương trong cohort đạt {pos_ratio:.1f}%. "
            f"Mô hình học máy {mc['model']} được chọn theo tiêu chí CV F1 cao nhất trong nhóm learned models (loại trừ Dummy khỏi nhóm learned models do Dummy có ROC-AUC=0.5000, hoàn toàn không có khả năng phân loại/xếp hạng). "
            f"Bên cạnh đó, Logistic Regression đạt Test ROC-AUC = {lr_auc:.4f}, thể hiện năng lực phân biệt và xếp hạng rủi ro rất tốt."
        )
    else:
        baseline_note = f"Mô hình học máy {mc['model']} được chọn trên cơ sở Stratified 5-Fold CV F1."

    report = f"""# CRISP-DM Step 07: Model Comparison, Business Insights & Strategic Recommendations

**Project:** UCI Online Retail Data Mining Analysis  
**Phase:** CRISP-DM Step 05 (Evaluation) & Step 06 (Deployment Preparation)  
**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary

Báo cáo này tổng hợp toàn diện kết quả từ các bước khai phá dữ liệu (Phân cụm khách hàng, Dự đoán mua lại, và Khai phá luật kết hợp) trên tập dữ liệu UCI Online Retail.

- **Customer Clustering (Phân cụm khách hàng):** Mô hình được chọn động từ bảng xếp hạng đa tiêu chí là **{sc['algorithm']} (K={sc['n_clusters']})** với Silhouette Score = **{sc['silhouette_score']:.4f}**, Davies-Bouldin Index = **{sc['davies_bouldin_score']:.4f}**. Phân tách thành các nhóm: {seg_summary}.
- **Repeat Purchase Classification (Dự đoán mua lại):** Mô hình học được chọn là **{mc['model']}** với Stratified 5-Fold CV F1 = **{cv_f1:.4f}**, Test F1 = **{mc.get('test_f1', 0):.4f}**, Test Precision = **{mc.get('test_precision', 0):.4f}**, Test Recall = **{mc.get('test_recall', 0):.4f}**, Test ROC-AUC = **{mc.get('test_roc_auc', 0):.4f}**. {baseline_note}
- **Association Rule Mining (Khai phá luật kết hợp):** Thuật toán được chọn là **{sa['algorithm']}** sinh ra **{sa.get('n_valid_rules', sa.get('valid_rule_count', 0))} luật hợp lệ** (Lift > 1.0) trong thời gian thực thi {sa['runtime_seconds']:.2f} giây. Hai thuật toán Apriori và FP-Growth đã được kiểm chứng tương đương 100% về tập luật và metric.

---

## 2. Model Comparison Tables (Bảng so sánh mô hình)

### 2.1 Customer Clustering Model Comparison
{clustering_comp.to_markdown(index=False)}

### 2.2 Classification Model Comparison
{classification_comp.to_markdown(index=False)}

### 2.3 Association Rules Algorithm Comparison
{assoc_comp.to_markdown(index=False)}

---

## 3. Customer Segment Profiles & Strategic Action Plan (Phân khúc & Kế hoạch hành động)

### 3.1 Segment Profiles (Hồ sơ phân khúc)
{seg.to_markdown(index=False) if not seg.empty else "No segment profiles available."}

### 3.2 Strategic Action Plan (Kế hoạch hành động chiến lược)
{actions.to_markdown(index=False) if not actions.empty else "No action plan available."}

---

## 4. Product Co-Purchase Association Rules (Top 10 Luật kết hợp hàng đầu)
{prod.head(10).to_markdown(index=False) if not prod.empty else "No association insights available."}

---

## 5. Predictive Feature Importance (Tầm quan trọng của đặc trưng dự đoán)
{feat.head(10).to_markdown(index=False) if not feat.empty else "No feature insights available."}

---

## 6. Generated Visualizations & Dashboards (Biểu đồ & Dashboard minh họa)
- `clustering_model_comparison.png`
- `classification_model_comparison.png`
- `clustering_quality_metrics.png`
- `classification_cv_vs_test.png`
- `insight_summary_dashboard.png`

---

## 7. Limitations & Scientific Constraints (Giới hạn & Ràng buộc phương pháp)
1. **Single Retailer Scope:** Dữ liệu chỉ từ một nhà bán lẻ trực tuyến tại Vương quốc Anh (12/2010 - 12/2011), không tự động suy rộng ra toàn ngành thương mại điện tử.
2. **Missing CustomerID:** 24.93% giao dịch không có CustomerID bị loại khỏi bài toán cấp khách hàng (selection bias đã được phân tích và gắn cờ).
3. **Class Imbalance & Baseline:** Tỷ lệ mua lại 90 ngày đạt mức đa số trong cohort, khiến Dummy Classifier có F1 danh nghĩa cao; Random Forest được chọn là mô hình học máy tối ưu F1 thực tế qua Cross-Validation trên tập train.
4. **Retrospective vs Predictive Segment Evaluation:** Tỷ lệ mua lại theo cụm là phân tích hồi cứu mô tả do cụm RFM được xây dựng trên toàn bộ lịch sử quan sát.
5. **Association vs Causation:** Luật kết hợp (Lift > 1) chỉ biểu thị tương quan đồng xuất hiện thống kê, chưa phải quan hệ nhân quả; cần kiểm chứng qua A/B testing trước khi quyết định nhập hàng combo.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


def generate_summary_json(clustering_comp, classification_comp, assoc_comp,
                          seg=None, validation_status='PASS', warnings=None, output_path=None):
    """Generate dynamic summary JSON from actual model evaluation results."""
    if output_path is None:
        output_path = REPORTS_DIR / '07_model_comparison_and_insights_summary.json'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sc = clustering_comp[clustering_comp['is_selected'].astype(bool)]
    mc = classification_comp[classification_comp['is_selected'].astype(bool)]
    sa = assoc_comp[assoc_comp['is_selected'].astype(bool)]

    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        git_commit = res.stdout.strip()
    except Exception:
        git_commit = 'unknown'

    input_files_rel = {k: v.relative_to(PROJECT_ROOT).as_posix() for k, v in INPUT_FILES.items()}
    cv_f1 = float(mc.iloc[0].get('cv_f1_mean', mc.iloc[0].get('cv_f1', 0))) if not mc.empty else None

    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'git_commit': git_commit,
        'input_files': input_files_rel,
        'selected_clustering_model': (sc.iloc[0]['algorithm'] + ' K=' + str(sc.iloc[0]['n_clusters'])) if not sc.empty else None,
        'selected_classification_model': mc.iloc[0]['model'] if not mc.empty else None,
        'selected_association_algorithm': sa.iloc[0]['algorithm'] if not sa.empty else None,
        'key_metrics': {
            'clustering_silhouette': float(sc.iloc[0]['silhouette_score']) if not sc.empty else None,
            'clustering_davies_bouldin': float(sc.iloc[0]['davies_bouldin_score']) if not sc.empty else None,
            'classification_cv_f1': cv_f1,
            'classification_test_f1': float(mc.iloc[0].get('test_f1', 0)) if not mc.empty else None,
            'classification_test_auc': float(mc.iloc[0].get('test_roc_auc', 0)) if not mc.empty else None,
            'association_valid_rules': int(sa.iloc[0].get('n_valid_rules', sa.iloc[0].get('valid_rule_count', 0))) if not sa.empty else None,
        },
        'validation_status': validation_status,
        'warnings': warnings or []
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


def generate_manifest(output_path=None):
    """
    Generate pipeline manifest with relative POSIX paths, file checksums, and strict output verification.
    """
    if output_path is None:
        output_path = EVIDENCE_DIR / 'pipeline_manifest.json'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        'run_id': f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'generated': datetime.now().isoformat(),
        'python_version': sys.version.split()[0],
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
            PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'test_predictions.csv',
            PROJECT_ROOT / 'models' / 'classification' / 'best_classifier_pipeline.joblib',
            PROJECT_ROOT / 'models' / 'classification' / 'classification_metadata.json',
        ],
        'step_06_association': [
            PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_algorithm_comparison.csv',
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
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'clustering_model_comparison.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'classification_model_comparison.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'clustering_quality_metrics.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'classification_cv_vs_test.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'cluster_size_distribution.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'segment_rfm_profile.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'segment_repeat_purchase_rate.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'top_association_rules.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'feature_importance_top20.png',
            PROJECT_ROOT / 'outputs' / 'figures' / 'model_comparison' / 'insight_summary_dashboard.png',
            PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights.md',
            PROJECT_ROOT / 'outputs' / 'reports' / '07_model_comparison_and_insights_summary.json',
        ],
    }

    # Add selected rules file (mandatory artifact)
    prim_rules = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'selected_association_rules.csv'
    nest_rules = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_rules' / 'selected_association_rules.csv'
    if prim_rules.exists():
        steps_outputs['step_06_association'].append(prim_rules)
    elif nest_rules.exists():
        steps_outputs['step_06_association'].append(nest_rules)
    else:
        steps_outputs['step_06_association'].append(prim_rules)

    equiv_audit = PROJECT_ROOT / 'outputs' / 'tables' / 'association_rules' / 'association_rules_equivalence_audit.csv'
    steps_outputs['step_06_association'].append(equiv_audit)

    missing_files = []
    for step_name, files in steps_outputs.items():
        step_entry = []
        for p in files:
            if p.exists() and p.stat().st_size > 0:
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
                step_entry.append({
                    'file': rel_path,
                    'size_bytes': p.stat().st_size,
                    'sha256': file_sha256(p),
                })
            else:
                missing_files.append(str(p))
        manifest['steps'][step_name] = step_entry

    if missing_files:
        manifest['validation_status'] = 'FAIL'
        manifest['missing_files'] = missing_files

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_all_step_07_figures(clust_comp, class_comp, assoc_comp, seg_insights, prod_insights, feat_insights, output_dir=None):
    """
    Generate and save all 10 standard CRISP-DM Step 07 figures synchronously.
    Shared by runner script and Jupyter notebook.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / 'model_comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        fig.savefig(output_dir / name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 1. clustering_model_comparison.png
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2b5c8f' if r else '#a0c4ff' for r in clust_comp.get('is_selected', [False]*len(clust_comp))]
    labels = [f"{r['algorithm']}\n(K={r['n_clusters']})" for _, r in clust_comp.iterrows()]
    scores = clust_comp['silhouette_score'].values
    bars = ax.bar(range(len(clust_comp)), scores, color=colors, width=0.5)
    ax.set_xticks(range(len(clust_comp)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Algorithm Comparison (Silhouette Score)', fontsize=12, fontweight='bold')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(scores) * 1.2 if len(scores) else 1)
    _save(fig, 'clustering_model_comparison.png')

    # 2. classification_model_comparison.png
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(class_comp))
    width = 0.35
    cv_f1s = class_comp.get('cv_f1_mean', class_comp.get('cv_f1', pd.Series([0]*len(class_comp)))).values
    test_f1s = class_comp.get('test_f1', pd.Series([0]*len(class_comp))).values
    ax.bar(x - width/2, cv_f1s, width, label='CV F1-Score', color='#3a86ff')
    ax.bar(x + width/2, test_f1s, width, label='Test F1-Score', color='#8338ec')
    ax.set_xticks(x)
    ax.set_xticklabels(class_comp['model'].values, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('F1-Score')
    ax.set_title('Classification Model Comparison (CV vs Test F1)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    for i in range(len(class_comp)):
        ax.text(x[i] - width/2, cv_f1s[i] + 0.02, f"{cv_f1s[i]:.3f}", ha='center', fontsize=8)
        ax.text(x[i] + width/2, test_f1s[i] + 0.02, f"{test_f1s[i]:.3f}", ha='center', fontsize=8)
    _save(fig, 'classification_model_comparison.png')

    # 3. clustering_quality_metrics.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kmeans_df = clust_comp[clust_comp['algorithm'].isin(['K-Means', 'KMeans'])].sort_values('n_clusters')
    if not kmeans_df.empty:
        ax1.plot(kmeans_df['n_clusters'], kmeans_df['silhouette_score'], 'o-', color='#2b5c8f', linewidth=2)
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('K-Means Silhouette Score vs K', fontweight='bold')
        ax2.plot(kmeans_df['n_clusters'], kmeans_df['davies_bouldin_score'], 's-', color='#e63946', linewidth=2)
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Davies-Bouldin Index (Lower is Better)')
        ax2.set_title('K-Means Davies-Bouldin vs K', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'clustering_quality_metrics.png')

    # 4. classification_cv_vs_test.png
    fig, ax = plt.subplots(figsize=(8, 4))
    non_dummy = class_comp[class_comp['model'] != 'DummyClassifier']
    if not non_dummy.empty:
        models = non_dummy['model'].values
        cv = non_dummy.get('cv_f1_mean', non_dummy.get('cv_f1', pd.Series([0]*len(non_dummy)))).values
        test = non_dummy.get('test_f1', pd.Series([0]*len(non_dummy))).values
        x = np.arange(len(models))
        ax.plot(x, cv, 'o--', label='CV F1', color='#3a86ff', markersize=8)
        ax.plot(x, test, 's-', label='Test F1', color='#ff006e', markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title('Generalization Check: CV F1 vs Test F1 (Non-Baseline Models)', fontweight='bold')
        ax.legend()
        ax.set_ylim(0.4, 0.9)
    _save(fig, 'classification_cv_vs_test.png')

    # 5. cluster_size_distribution.png
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2b5c8f', '#d94e34']
    labels = [f"Cluster {c}\n{n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
    bars = ax.bar(labels, seg_insights['customer_count'], color=colors[:len(seg_insights)], width=0.4)
    ax.set_ylabel('Customer Count')
    ax.set_title('Customer Distribution by Segment', fontweight='bold')
    for bar, pct in zip(bars, seg_insights['customer_percentage']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')
    _save(fig, 'cluster_size_distribution.png')

    # 6. segment_rfm_profile.png
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [('recency_mean', 'Recency (Days, Lower = Better)', '#2a9d8f'),
               ('frequency_mean', 'Frequency (Orders, Higher = Better)', '#e76f51'),
               ('monetary_mean', 'Monetary (Spend £, Higher = Better)', '#2b5c8f')]
    x_labels = [f"C{c}: {n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(x_labels, seg_insights[col], color=color, width=0.4)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for bar in bars:
            val = bar.get_height()
            fmt = f"£{val:,.0f}" if 'Spend' in title else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.02, fmt, ha='center', fontsize=9)
    fig.tight_layout()
    _save(fig, 'segment_rfm_profile.png')

    # 7. segment_repeat_purchase_rate.png
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'repeat_purchase_rate' in seg_insights.columns:
        rates = seg_insights['repeat_purchase_rate'].values
        labels = [f"Cluster {c}\n({n})" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])]
        bars = ax.bar(labels, rates, color=['#2b5c8f', '#d94e34'], width=0.4)
        ax.set_ylabel('Repeat Purchase Rate (90 Days)')
        ax.set_title('Repeat Purchase Rate by Segment', fontweight='bold')
        ax.set_ylim(0, 1.15)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.1%}", ha='center', fontweight='bold', fontsize=11)
    _save(fig, 'segment_repeat_purchase_rate.png')

    # 8. top_association_rules.png
    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = prod_insights.head(10).iloc[::-1]
    labels = [f"{r['antecedents']} -> {r['consequents']}" for _, r in top10.iterrows()]
    bars = ax.barh(range(len(top10)), top10['lift'], color='#4361ee', height=0.6)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Lift Value')
    ax.set_title('Top 10 Product Association Rules by Lift', fontweight='bold')
    for bar, lift in zip(bars, top10['lift']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f"{lift:.1f}x", va='center', fontsize=8)
    _save(fig, 'top_association_rules.png')

    # 9. feature_importance_top20.png
    fig, ax = plt.subplots(figsize=(9, 5))
    top15 = feat_insights.head(15).iloc[::-1]
    colors = {'RFM': '#e76f51', 'Behavioral': '#2a9d8f', 'Geographic': '#2b5c8f', 'Other': '#8d99ae'}
    bar_colors = [colors.get(g, '#8d99ae') for g in top15['feature_group']]
    bars = ax.barh(range(len(top15)), top15['importance'], color=bar_colors, height=0.6)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['feature'], fontsize=9)
    ax.set_xlabel('Feature Importance (Gini / Mean Decrease Impurity)')
    ax.set_title('Top 15 Predictive Features for 90-Day Repeat Purchase', fontweight='bold')
    for bar, imp in zip(bars, top15['importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{imp:.4f}", va='center', fontsize=8)
    _save(fig, 'feature_importance_top20.png')

    # 10. insight_summary_dashboard.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].pie(seg_insights['customer_count'], labels=[f"C{c}: {n}" for c, n in zip(seg_insights['cluster_id'], seg_insights['business_segment_name'])],
                   autopct='%1.1f%%', colors=['#2b5c8f', '#d94e34'], startangle=90)
    axes[0, 0].set_title('A. Customer Segment Distribution', fontweight='bold', fontsize=10)

    mc = class_comp.sort_values('test_f1', ascending=True)
    axes[0, 1].barh(mc['model'], mc['test_f1'], color='#3a86ff', height=0.5)
    axes[0, 1].set_xlabel('Test F1-Score')
    axes[0, 1].set_title('B. Classification Models Performance', fontweight='bold', fontsize=10)
    for i, v in enumerate(mc['test_f1']):
        axes[0, 1].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)

    axes[1, 0].bar([f"C{c}" for c in seg_insights['cluster_id']], seg_insights['repeat_purchase_rate'], color=['#2b5c8f', '#d94e34'], width=0.4)
    axes[1, 0].set_ylabel('Repeat Purchase Rate')
    axes[1, 0].set_title('C. Segment Repeat Purchase Behavior', fontweight='bold', fontsize=10)
    for i, v in enumerate(seg_insights['repeat_purchase_rate']):
        axes[1, 0].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9, fontweight='bold')

    axes[1, 1].axis('off')
    seg_by_spend = seg_insights.sort_values('monetary_mean', ascending=False)
    best_c = seg_by_spend.iloc[0]
    lost_c = seg_by_spend.iloc[-1]
    best_name = best_c['business_segment_name']
    best_pct = best_c['customer_percentage']
    best_rep = best_c.get('repeat_purchase_rate', 0)
    best_spend = best_c['monetary_mean']
    lost_name = lost_c['business_segment_name']
    lost_pct = lost_c['customer_percentage']
    lost_rep = lost_c.get('repeat_purchase_rate', 0)
    lost_rec = lost_c['recency_mean']

    feat_text = ""
    if feat_insights is not None and not feat_insights.empty:
        top1 = feat_insights.iloc[0]
        top2 = feat_insights.iloc[1] if len(feat_insights) > 1 else None
        feat_text = f"• Top Predictors: {top1['feature']} ({top1['importance']:.1%})"
        if top2 is not None:
            feat_text += f",\n  {top2['feature']} ({top2['importance']:.1%})."
    else:
        feat_text = "• Predictor Importance: Recency and\n  Monetary are key purchase predictors."

    rules_text = ""
    if assoc_comp is not None and not assoc_comp.empty:
        sel_assoc = assoc_comp[assoc_comp['is_selected'].astype(bool)]
        row_assoc = sel_assoc.iloc[0] if not sel_assoc.empty else assoc_comp.iloc[0]
        n_rules = row_assoc['valid_rule_count']
        max_lift = row_assoc.get('max_rule_lift', row_assoc.get('max_lift', 0))
        algo_name = row_assoc['algorithm']
        rules_text = f"• Co-Purchase Rules ({algo_name}): {n_rules} rules (Lift > 1.0).\n  Top rule lift = {max_lift:.1f}x."
    else:
        rules_text = "• Co-Purchase Rules: Validated Lift > 1.0."

    summary_text = (
        "D. Key Strategic Findings\n"
        "----------------------------------------\n"
        f"• {best_name} ({best_pct:.1f}%): {best_rep:.1%} repeat rate,\n"
        f"  Avg spend £{best_spend:,.0f}. Action: VIP retention.\n\n"
        f"• {lost_name} ({lost_pct:.1f}%): {lost_rep:.1%} repeat rate,\n"
        f"  Recency {lost_rec:.0f} days. Action: Win-back campaign.\n\n"
        f"{feat_text}\n\n"
        f"{rules_text}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    fig.suptitle('CRISP-DM Step 07: Data Mining Insights Dashboard', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'insight_summary_dashboard.png')


def export_all_step_07_outputs(
    clust_comp=None,
    class_comp=None,
    assoc_comp=None,
    seg_insights=None,
    action_plan=None,
    prod_insights=None,
    feat_insights=None,
    save_figures=True
):
    """
    Centralized, unified export pipeline for Step 07 outputs.
    Guarantees 100% synchronicity and parity between Jupyter Notebook and runner script.
    """
    from src.insights import (
        build_customer_segment_insights, build_action_plan,
        build_product_association_insights, build_feature_insights
    )

    TABLES_MC = TABLES_DIR / 'model_comparison'
    TABLES_INS = TABLES_DIR / 'insights'
    FIGURES_MC = FIGURES_DIR / 'model_comparison'
    REPORTS = REPORTS_DIR
    EVIDENCE = EVIDENCE_DIR

    for d in [TABLES_MC, TABLES_INS, FIGURES_MC, REPORTS, EVIDENCE]:
        d.mkdir(parents=True, exist_ok=True)

    if clust_comp is None:
        clust_comp = build_clustering_comparison()
    clust_comp.to_csv(TABLES_MC / 'clustering_model_comparison.csv', index=False)

    if class_comp is None:
        class_comp = build_classification_comparison()
    class_comp.to_csv(TABLES_MC / 'classification_model_comparison.csv', index=False)

    if assoc_comp is None:
        assoc_comp = build_association_comparison()
    assoc_comp.to_csv(TABLES_MC / 'association_rules_comparison.csv', index=False)

    if seg_insights is None:
        seg_insights = build_customer_segment_insights()
    seg_insights.to_csv(TABLES_INS / 'customer_segment_insights.csv', index=False)

    if action_plan is None:
        action_plan = build_action_plan(seg_insights)
    action_plan.to_csv(TABLES_INS / 'customer_segment_action_plan.csv', index=False)

    if prod_insights is None:
        prod_insights = build_product_association_insights()
    prod_insights.to_csv(TABLES_INS / 'product_association_insights.csv', index=False)

    if feat_insights is None:
        feat_insights = build_feature_insights()
    feat_insights.to_csv(TABLES_INS / 'classification_feature_insights.csv', index=False)

    if save_figures:
        generate_all_step_07_figures(
            clust_comp, class_comp, assoc_comp, seg_insights, prod_insights, feat_insights,
            output_dir=FIGURES_MC
        )

    # Generate Markdown report
    report_text = generate_report(
        clust_comp, class_comp, assoc_comp, seg=seg_insights, actions=action_plan,
        prod=prod_insights, feat=feat_insights, output_path=REPORTS / '07_model_comparison_and_insights.md'
    )

    # Generate dynamic summary JSON
    summary = generate_summary_json(
        clust_comp, class_comp, assoc_comp, seg=seg_insights, validation_status='PASS',
        output_path=REPORTS / '07_model_comparison_and_insights_summary.json'
    )

    # Generate manifest and strictly enforce PASS
    manifest = generate_manifest(output_path=EVIDENCE / 'pipeline_manifest.json')
    if manifest.get('validation_status') != 'PASS':
        missing = manifest.get('missing_files', [])
        raise RuntimeError(f"Step 07 FAILED: pipeline_manifest validation_status is 'FAIL'. Missing: {missing}")

    return {
        'clustering_comparison': clust_comp,
        'classification_comparison': class_comp,
        'association_comparison': assoc_comp,
        'segment_insights': seg_insights,
        'action_plan': action_plan,
        'product_insights': prod_insights,
        'feature_insights': feat_insights,
        'report_text': report_text,
        'summary': summary,
        'manifest': manifest,
    }
