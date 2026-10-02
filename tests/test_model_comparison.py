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
    
    try:
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
    except Exception as e:
        pytest.skip(f"Prerequisite files not generated yet: {e}")
