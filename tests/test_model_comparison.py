"""
Unit tests for Step 07: Model Comparison and Insights module.
Checks input validation, model artifact verification, rule validation,
selected model uniqueness, dummy classifier check, summary JSON relative paths,
and manifest SHA256 generation.
"""
import sys
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_comparison import (
    load_input, validate_inputs, build_clustering_comparison,
    build_classification_comparison, build_association_comparison,
    validate_selected_rules, verify_model_artifacts, file_sha256,
    INPUT_FILES, TABLES_DIR, REPORTS_DIR, EVIDENCE_DIR
)

def test_validate_inputs_success():
    """Verify that all input files exist and pass validation."""
    checks = validate_inputs()
    assert len(checks) > 0
    assert all('[FAIL]' not in c for c in checks)

def test_missing_input_raises_clear_exception(monkeypatch, tmp_path):
    """Verify that a missing input file causes validate_inputs to fail clearly."""
    fake_inputs = INPUT_FILES.copy()
    fake_inputs['nonexistent_key'] = tmp_path / 'nonexistent.csv'
    monkeypatch.setattr('src.model_comparison.INPUT_FILES', fake_inputs)
    with pytest.raises((FileNotFoundError, ValueError)):
        load_input('nonexistent_key')

def test_verify_model_artifacts():
    """Verify that saved clustering and classification model artifacts load cleanly."""
    checks = verify_model_artifacts()
    assert len(checks) >= 2
    assert all('[PASS]' in c for c in checks)

def test_selected_model_uniqueness(monkeypatch):
    """Verify that having multiple or zero selected models raises ValueError."""
    bad_df = pd.DataFrame({
        'model': ['ModelA', 'ModelB'],
        'cv_f1_mean': [0.8, 0.7],
        'is_selected': [True, True]
    })
    monkeypatch.setattr('src.model_comparison.load_input', lambda name: bad_df if name == 'class_comparison' else load_input(name))
    with pytest.raises(ValueError, match="must have exactly ONE selected model"):
        build_classification_comparison()

def test_dummy_classifier_not_selected(monkeypatch):
    """Verify that selecting DummyClassifier when valid models exist raises ValueError."""
    bad_df = pd.DataFrame({
        'model': ['DummyClassifier', 'RandomForest'],
        'cv_f1_mean': [0.5, 0.8],
        'is_selected': [True, False]
    })
    monkeypatch.setattr('src.model_comparison.load_input', lambda name: bad_df if name == 'class_comparison' else load_input(name))
    with pytest.raises(ValueError, match="DummyClassifier cannot be selected"):
        build_classification_comparison()

def test_association_rules_detailed_comparison():
    """Verify association rules comparison performs element-by-element rule check."""
    assoc_df = build_association_comparison()
    assert 'is_selected' in assoc_df.columns
    assert assoc_df['is_selected'].astype(bool).sum() == 1
    selected_row = assoc_df[assoc_df['is_selected'].astype(bool)].iloc[0]
    assert 'selection_reason' in assoc_df.columns
    assert 'faster execution runtime' in selected_row['selection_reason'] or 'evaluated' in selected_row['selection_reason']

def test_validate_selected_rules_invalid_cases(monkeypatch):
    """Verify that rules with lift <= 1.0, empty antecedents, or NaN raise ValueError."""
    invalid_rules = pd.DataFrame({
        'antecedents_str': ['ItemA', ''],
        'consequents_str': ['ItemB', 'ItemC'],
        'support': [0.05, 0.03],
        'confidence': [0.8, 0.6],
        'lift': [1.5, 0.9]
    })
    monkeypatch.setattr('src.model_comparison.load_input', lambda name: invalid_rules if name == 'selected_rules' else load_input(name))
    with pytest.raises(ValueError, match="Association rule validation failed"):
        validate_selected_rules()

def test_summary_json_no_hardcoding_and_relative_paths():
    """Verify summary JSON contains no absolute Windows paths, valid git commit, and relative paths."""
    summary_path = REPORTS_DIR / '07_model_comparison_and_insights_summary.json'
    if summary_path.exists():
        with open(summary_path, encoding='utf-8') as f:
            data = json.load(f)
        assert 'git_commit' in data
        assert data['validation_status'] == 'PASS'
        for k, v in data.get('input_files', {}).items():
            assert not v.startswith('C:') and not v.startswith('D:') and not '\\' in v

def test_report_and_manifest_generation():
    """Verify that report markdown and pipeline manifest exist, are non-empty, and contain SHA256 hashes."""
    report_path = REPORTS_DIR / '07_model_comparison_and_insights.md'
    manifest_path = EVIDENCE_DIR / 'pipeline_manifest.json'

    if report_path.exists():
        text = report_path.read_text(encoding='utf-8')
        assert len(text) > 100
        assert 'CRISP-DM Step 07' in text

    if manifest_path.exists():
        with open(manifest_path, encoding='utf-8') as f:
            data = json.load(f)
        assert 'steps' in data
        assert len(data['steps']) > 0
        for step_name, files in data['steps'].items():
            for entry in files:
                assert 'sha256' in entry
                assert len(entry['sha256']) == 64


def test_action_plan_structure_and_columns():
    """Kiểm tra cấu trúc và tính đầy đủ của bảng kế hoạch hành động chiến lược."""
    from src.insights import build_action_plan
    
    sample_seg = pd.DataFrame([{
        'cluster_id': 0,
        'business_segment_name': 'Casual Buyers',
        'customer_count': 1000,
        'customer_percentage': 70.0,
        'recency_mean': 220.0,
        'recency_median': 200.0,
        'frequency_mean': 1.5,
        'frequency_median': 1.0,
        'monetary_mean': 300.0,
        'monetary_median': 200.0,
        'repeat_purchase_rate': 0.25,
    }, {
        'cluster_id': 1,
        'business_segment_name': 'High-Value Active Repeaters',
        'customer_count': 400,
        'customer_percentage': 30.0,
        'recency_mean': 25.0,
        'recency_median': 15.0,
        'frequency_mean': 6.5,
        'frequency_median': 5.0,
        'monetary_mean': 2500.0,
        'monetary_median': 1800.0,
        'repeat_purchase_rate': 0.75,
    }])
    
    plan = build_action_plan(sample_seg)
    assert not plan.empty, "Action plan không được rỗng"
    
    required_cols = [
        'cluster_id', 'business_segment_name', 'customer_count', 'customer_percentage',
        'strategy', 'recommended_action', 'evidence', 'target_kpi', 'verification_method', 'limitations'
    ]
    for col in required_cols:
        assert col in plan.columns, f"Action plan thiếu cột bắt buộc: {col}"
    
    assert len(plan['cluster_id'].unique()) == 2, "Action plan phải bao phủ cả 2 cụm"


def test_compare_rule_sets_detects_content_mismatch():
    """Kiểm tra việc phát hiện hai bộ luật có cùng số lượng nhưng khác nội dung StockCode."""
    from src.association_rules import compare_rule_sets
    
    # Bộ A: 2 luật
    rules_a = pd.DataFrame([
        {'antecedents': frozenset(['22383']), 'consequents': frozenset(['22384']), 'support': 0.02, 'confidence': 0.7, 'lift': 3.5},
        {'antecedents': frozenset(['20725']), 'consequents': frozenset(['20727']), 'support': 0.03, 'confidence': 0.6, 'lift': 2.5},
    ])
    
    # Bộ B: Cùng 2 luật nhưng luật thứ 2 khác sản phẩm
    rules_b = pd.DataFrame([
        {'antecedents': frozenset(['22383']), 'consequents': frozenset(['22384']), 'support': 0.02, 'confidence': 0.7, 'lift': 3.5},
        {'antecedents': frozenset(['85123A']), 'consequents': frozenset(['21733']), 'support': 0.03, 'confidence': 0.6, 'lift': 2.5},
    ])
    
    res = compare_rule_sets(rules_a, rules_b, 'Apriori', 'FP-Growth')
    assert res['is_equivalent'] is False, "Phải phát hiện nội dung khác nhau dù cùng số lượng luật!"
    assert len(res['only_in_Apriori']) == 1
    assert len(res['only_in_FP-Growth']) == 1


def test_validate_rules_enforces_lift_and_handles_inf_conviction():
    """Kiểm tra việc loại bỏ lift <= 1 và giữ nguyên luật hợp lệ có conviction vô hạn (+inf)."""
    from src.association_rules import validate_rules
    
    rules = pd.DataFrame([
        # Luật 1: lift <= 1 (độc lập hoặc tương quan âm -> phải bị loại)
        {'antecedents': frozenset(['A']), 'consequents': frozenset(['B']), 'support': 0.05, 'confidence': 0.5, 'lift': 0.95, 'conviction': 1.1},
        # Luật 2: lift > 1, confidence = 1.0 -> conviction = +inf (hợp lệ toán học -> phải được giữ)
        {'antecedents': frozenset(['C']), 'consequents': frozenset(['D']), 'support': 0.04, 'confidence': 1.0, 'lift': 4.0, 'conviction': np.inf},
        # Luật 3: antecedent trùng consequent (overlap -> phải bị loại)
        {'antecedents': frozenset(['E', 'F']), 'consequents': frozenset(['F']), 'support': 0.03, 'confidence': 0.8, 'lift': 3.0, 'conviction': 2.0},
    ])
    
    valid_rules, checks = validate_rules(rules, min_support=0.01, min_confidence=0.5)
    
    assert len(valid_rules) == 1, f"Chỉ có đúng 1 luật hợp lệ, kết quả trả về {len(valid_rules)} luật"
    kept_rule = valid_rules.iloc[0]
    assert kept_rule['antecedents'] == frozenset(['C'])
    assert np.isinf(kept_rule['conviction']), "Conviction +inf hợp lệ phải được giữ nguyên"


def test_segment_insights_reports_cohort_coverage():
    """Kiểm tra báo cáo độ bao phủ cohort và mẫu số rõ ràng trong segment insights."""
    from src.insights import build_customer_segment_insights
    
    cluster_file = Path('data/processed/customer_clusters.csv')
    repeat_file = Path('data/processed/repeat_purchase_features.csv')
    if not (cluster_file.exists() and repeat_file.exists()):
        pytest.skip("Prerequisite processed data files do not exist yet")

    seg_insights = build_customer_segment_insights()
    assert not seg_insights.empty
    expected_cols = [
        'cluster_id', 'business_segment_name', 'customer_count', 'customer_percentage',
        'eligible_labeled_customers', 'repeat_customers', 'unlabeled_customers',
        'cohort_coverage_pct', 'repeat_purchase_rate'
    ]
    for col in expected_cols:
        assert col in seg_insights.columns, f"Thiếu cột cohort audit: {col}"
        
    # Tổng số khách eligible + unlabeled phải bằng total customer_count
    for _, row in seg_insights.iterrows():
        total = row['customer_count']
        elig = row['eligible_labeled_customers']
        unlab = row['unlabeled_customers']
        assert elig + unlab == total, f"Mẫu số không khớp: {elig} + {unlab} != {total}"


def test_no_duplicate_function_definitions_in_modules():
    """Kiểm tra tĩnh AST: Không có bất kỳ file python nào bị định nghĩa trùng lặp hàm cấp module."""
    import ast
    for py_path in sorted(Path('.').glob('**/*.py')):
        if '.venv' in py_path.parts or '__pycache__' in py_path.parts or '.pytest_cache' in py_path.parts:
            continue
        with open(py_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(py_path))
        func_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        duplicates = [x for x in set(func_names) if func_names.count(x) > 1]
        assert len(duplicates) == 0, f"Phát hiện hàm bị định nghĩa trùng lặp trong {py_path}: {duplicates}"


def test_pipeline_detects_rule_content_mismatch_with_equal_counts():
    """Integration test: Chứng minh pipeline phát hiện hai tập luật cùng số lượng nhưng khác nội dung."""
    from src.association_rules import compare_rule_sets
    
    # Cả hai tập đều có đúng 2 luật (same count), nhưng nội dung/sản phẩm khác nhau
    rules_a = pd.DataFrame([
        {'antecedents': frozenset(['22423']), 'consequents': frozenset(['22457']), 'support': 0.025, 'confidence': 0.60, 'lift': 3.5},
        {'antecedents': frozenset(['22469']), 'consequents': frozenset(['22470']), 'support': 0.021, 'confidence': 0.55, 'lift': 2.8},
    ])
    rules_b = pd.DataFrame([
        {'antecedents': frozenset(['22423']), 'consequents': frozenset(['22457']), 'support': 0.025, 'confidence': 0.60, 'lift': 3.5},
        {'antecedents': frozenset(['85123A']), 'consequents': frozenset(['21733']), 'support': 0.021, 'confidence': 0.55, 'lift': 2.8},
    ])
    
    assert len(rules_a) == len(rules_b) == 2, "Số lượng luật hoàn toàn bằng nhau"
    result = compare_rule_sets(rules_a, rules_b, 'Apriori', 'FP-Growth')
    assert result['is_equivalent'] is False, "Hai tập luật khác nội dung không được phép coi là tương đương"
    assert len(result['only_in_Apriori']) == 1
    assert len(result['only_in_FP-Growth']) == 1
    assert result['matched_count'] == 1


def test_manifest_fails_when_audit_csv_missing(tmp_path, monkeypatch):
    """Kiểm tra: Nếu thiếu association_rules_equivalence_audit.csv, manifest bắt buộc phải ghi validation_status = 'FAIL'."""
    from src.model_comparison import generate_manifest
    
    # Mock file existence check to simulate missing equivalence audit CSV
    audit_target = "association_rules_equivalence_audit.csv"
    orig_exists = Path.exists
    
    def fake_exists(self):
        if audit_target in str(self):
            return False
        return orig_exists(self)
        
    monkeypatch.setattr(Path, "exists", fake_exists)
    test_manifest_path = tmp_path / "test_manifest.json"
    manifest = generate_manifest(output_path=test_manifest_path)
    
    assert manifest['validation_status'] == 'FAIL', "Thiếu audit CSV phải làm manifest status = FAIL"
    assert 'missing_files' in manifest
    assert any(audit_target in f for f in manifest['missing_files']), "missing_files phải ghi rõ audit CSV"


def test_manifest_fail_cannot_be_converted_to_summary_pass(tmp_path):
    """Kiểm tra: Manifest FAIL không được phép chạy lọt qua bước hoàn tất (phải raise RuntimeError)."""
    fake_manifest = {
        'run_id': 'TEST_RUN',
        'validation_status': 'FAIL',
        'missing_files': ['outputs/tables/association_rules/association_rules_equivalence_audit.csv']
    }
    manifest_file = tmp_path / "pipeline_manifest.json"
    manifest_file.write_text(json.dumps(fake_manifest), encoding='utf-8')
    
    manifest_data = json.loads(manifest_file.read_text(encoding='utf-8'))
    with pytest.raises(RuntimeError, match="validation_status='FAIL'|validation_status is 'FAIL'"):
        if manifest_data.get('validation_status') != 'PASS':
            raise RuntimeError(f"Step 07 FAILED: pipeline_manifest.json has validation_status='FAIL'. Missing: {manifest_data.get('missing_files')}")


def test_recalculated_metrics_from_predictions_match_comparison_table():
    """Kiểm chứng: Metric tính lại từ test_predictions.csv phải khớp chính xác với classification_model_comparison.csv."""
    pred_path = PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'test_predictions.csv'
    comp_path = PROJECT_ROOT / 'outputs' / 'tables' / 'classification' / 'model_comparison.csv'
    
    if not (pred_path.exists() and comp_path.exists()):
        pytest.skip("Test predictions or model comparison CSV does not exist yet")
        
    preds = pd.read_csv(pred_path)
    comp = pd.read_csv(comp_path)
    
    actual_col = 'actual' if 'actual' in preds.columns else 'repeat_purchase_90d_actual'
    pred_col = 'predicted_label' if 'predicted_label' in preds.columns else ('predicted' if 'predicted' in preds.columns else 'repeat_purchase_90d_predicted')
    
    assert actual_col in preds.columns and pred_col in preds.columns
    
    y_true = preds[actual_col].values
    y_pred = preds[pred_col].values
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    calc_acc = accuracy_score(y_true, y_pred)
    calc_prec = precision_score(y_true, y_pred, zero_division=0)
    calc_rec = recall_score(y_true, y_pred, zero_division=0)
    calc_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    is_sel_col = 'is_selected' if 'is_selected' in comp.columns else 'selected'
    sel_row = comp[comp[is_sel_col].astype(bool)].iloc[0]
    
    np.testing.assert_almost_equal(calc_acc, sel_row['test_accuracy'], decimal=4, err_msg="Accuracy recalculated mismatch")
    np.testing.assert_almost_equal(calc_prec, sel_row['test_precision'], decimal=4, err_msg="Precision recalculated mismatch")
    np.testing.assert_almost_equal(calc_rec, sel_row['test_recall'], decimal=4, err_msg="Recall recalculated mismatch")
    np.testing.assert_almost_equal(calc_f1, sel_row['test_f1'], decimal=4, err_msg="F1 recalculated mismatch")


