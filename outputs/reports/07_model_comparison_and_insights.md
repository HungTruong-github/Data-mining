# CRISP-DM Step 07: Model Comparison and Insights Report

Generated: 2026-10-02 12:26:42

## 1. Clustering Model Comparison

**Selected Model**: K-Means (K=2)

- Silhouette Score: 0.4330
- Davies-Bouldin Index: 0.8917
- Total configurations evaluated: 26

## 2. Classification Model Comparison

**Selected Model**: RandomForest

- CV F1 (mean): 0.6726
- Test F1: 0.7086
- Total models evaluated: 4

- Baseline (Dummy) CV F1: 0.7259
- **Warning**: Selected model does NOT exceed baseline

## 3. Association Rules Comparison

- **Apriori**: 61 rules, runtime=3.74s
- **FP-Growth**: 61 rules, runtime=4.15s

## 4. Limitations

- Single UK retailer — results do not generalize automatically
- Historical data (2010-12-01 → 2011-12-09) — temporal drift not evaluated
- Missing CustomerID (24.93%) — selection bias in customer cohort
- No margin/campaign response data — cannot compute actual ROI
- Correlational, not causal — associations do not prove causation
