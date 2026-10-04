import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Manifest check
with open('outputs/evidence/pipeline_manifest.json', encoding='utf-8') as f:
    man = json.load(f)
print('Pipeline manifest validation_status:', man.get('validation_status'))

# Summary check
with open('outputs/reports/07_model_comparison_and_insights_summary.json', encoding='utf-8') as f:
    summ = json.load(f)
print('Summary selected classification:', summ.get('selected_models', {}).get('classification'))
print('Summary selected clustering:', summ.get('selected_models', {}).get('clustering'))
print('Summary selected association:', summ.get('selected_models', {}).get('association'))

# Comparison table check
comp = pd.read_csv('outputs/tables/classification/model_comparison.csv')
print('\nModel Comparison:')
print(comp[['model', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall', 'selected']])

# Test predictions recalculation check
preds = pd.read_csv('outputs/tables/classification/test_predictions.csv')
print('\nPredictions rows:', len(preds), 'CustomerID in col 0:', preds.columns[0] == 'CustomerID')
rf_row = comp[comp['selected'] == True].iloc[0]
recalc_f1 = f1_score(preds['repeat_purchase_90d_actual'], preds['repeat_purchase_90d_predicted'])
recalc_auc = roc_auc_score(preds['repeat_purchase_90d_actual'], preds['repeat_purchase_90d_probability'])
tbl_f1 = float(rf_row['test_f1'])
tbl_auc = float(rf_row['test_roc_auc'])
print(f'Recalculated Test F1: {recalc_f1:.4f} (Table: {tbl_f1:.4f})')
print(f'Recalculated Test AUC: {recalc_auc:.4f} (Table: {tbl_auc:.4f})')
assert abs(recalc_f1 - tbl_f1) < 1e-4, 'F1 mismatch!'
assert abs(recalc_auc - tbl_auc) < 1e-4, 'AUC mismatch!'
print('[PASS] Exact mathematical parity verified!')
