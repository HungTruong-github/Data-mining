# 07. Model Comparison and Insights Report

**Generated**: 2026-10-01 09:42:07
**Random State**: 42

## 1. Objective

Compare all models from clustering (step 04), classification (step 05), and association rules (step 06).
Generate data-driven business insights with evidence.

## 2. Input Data and Provenance

| Input | Path | Status |
|-------|------|--------|
| rfm_features | `rfm_customer_features.csv` | OK |
| repeat_features | `repeat_purchase_features.csv` | OK |
| customer_clusters | `customer_clusters.csv` | OK |
| cluster_comparison | `clustering_algorithm_comparison.csv` | OK |
| cluster_profiles | `cluster_profiles.csv` | OK |
| class_comparison | `model_comparison.csv` | OK |
| cv_results | `cv_results.csv` | OK |
| test_predictions | `test_predictions.csv` | OK |
| feature_importance | `feature_importance.csv` | OK |
| assoc_comparison | `association_algorithm_comparison.csv` | OK |
| selected_rules | `selected_association_rules.csv` | OK |
| business_insights | `association_business_insights.csv` | OK |
| class_metadata | `classification_metadata.json` | OK |

## 3. Data Validation

All input files validated. Model artifacts verified loadable.

## 4. Clustering Model Comparison

- **Total configurations evaluated**: 26
- **Selected**: K-Means (K=2)
- **Silhouette Score**: 0.4330
- **Davies-Bouldin**: 0.8917
- **Selection Reason**: Silhouette=0.4330; DB=0.8917; Best combined ranking across Silhouette, DB, CH, Dunn metrics

See: `outputs/tables/model_comparison/clustering_model_comparison.csv`

## 5. Classification Model Comparison

- **Models evaluated**: DummyClassifier, LogisticRegression, DecisionTree, RandomForest
- **Selected**: RandomForest (CV F1=0.6726)
- **Test F1**: 0.7086
- **Test AUC**: 0.7438
- **Selection Reason**: Highest CV F1-score (0.6726) among non-baseline models. Model selected via StratifiedKFold CV on training set only. Test set NOT used for selection.

> Model was selected using **StratifiedKFold Cross-Validation on training set only**.
> Test set was used for final evaluation only, NOT for model selection.

See: `outputs/tables/model_comparison/classification_model_comparison.csv`

## 6. Association Rule Comparison

- **Algorithms**: Apriori, FP-Growth
- **Selected**: Apriori
- **Valid Rules**: 61
- **Runtime**: 4.14s
- **Selection Reason**: Both algorithms produce identical rules (61 valid rules). Selected for faster runtime (4.14s).

> Association rules indicate co-occurrence patterns, NOT causal relationships.

See: `outputs/tables/model_comparison/association_rules_comparison.csv`

## 7. Customer Segment Profiles

### Cluster 0: Best Customers
- Customers: 1,662 (38.4%)
- Recency: 26 days | Frequency: 8 | Monetary: 4,464
- Repeat purchase rate: 95.7%
- Recently active customers. with high purchase frequency (avg 8 orders). and high monetary value (avg 4,464). Repeat purchase rate: 95.7%.

### Cluster 1: Lost Customers
- Customers: 2,672 (61.6%)
- Recency: 134 days | Frequency: 2 | Monetary: 493
- Repeat purchase rate: 26.4%
- Moderately active customers. with low purchase frequency (avg 2 orders). Repeat purchase rate: 26.4%.

## 8. Business Recommendations

4 recommended actions generated across 2 segments.
See: `outputs/tables/insights/customer_segment_action_plan.csv`

## 9. Product Association Insights

Top 5 rules by lift:

| Antecedent | Consequent | Lift | Evidence |
|-----------|-----------|------|----------|
| PINK REGENCY TEACUP AND SAUCER | GREEN REGENCY TEACUP AND SAUCER, ROSES R | 18.2 | Weak |
| GREEN REGENCY TEACUP AND SAUCER, ROSES R | PINK REGENCY TEACUP AND SAUCER | 18.2 | Weak |
| PINK REGENCY TEACUP AND SAUCER, ROSES RE | GREEN REGENCY TEACUP AND SAUCER | 17.7 | Weak |
| GREEN REGENCY TEACUP AND SAUCER | PINK REGENCY TEACUP AND SAUCER, ROSES RE | 17.7 | Weak |
| PINK REGENCY TEACUP AND SAUCER | GREEN REGENCY TEACUP AND SAUCER | 16.1 | Moderate |

> These are correlation patterns. They do NOT establish cause-and-effect.

## 10. Classification Feature Insights

Top 5 features for repeat purchase prediction (RandomForest):

| Rank | Feature | Importance | Group |
|------|---------|-----------|-------|
| 1 | Recency | 0.1393 | RFM |
| 2 | Monetary | 0.1360 | RFM |
| 3 | UniqueProducts | 0.1298 | Behavioral |
| 4 | TotalItems | 0.1279 | Behavioral |
| 5 | AverageOrderValue | 0.1235 | Behavioral |

> Feature importance is model-dependent and does NOT imply causation.

## 11. Limitations and Risks

1. **Clustering**: K=2 is a coarse segmentation; business may need finer segments.
2. **Classification**: Random customer-level split, not temporal split.
3. **Association**: Low support thresholds may produce spurious rules.
4. **General**: All insights are correlational, not causal.
5. **Data**: Single-year UK-dominated e-commerce dataset.

## 12. Reproducibility

- Random state: 42
- All outputs reproducible via: `python run_pipeline.py`
- Individual step: `python notebooks/run_07_model_comparison_and_insights.py`

## 13. Output Inventory

See: `outputs/evidence/pipeline_manifest.json`
