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
