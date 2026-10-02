# References — Online Retail Data Mining Project

## Dataset

1. Daqing Chen, Sai Liang Sain, and Kun Guo. **Online Retail Dataset**. UCI Machine Learning Repository, 2015.
   - URL: https://archive.ics.uci.edu/dataset/352/online+retail
   - Mục dùng: Raw data source — 541,909 giao dịch từ UK-based retailer, 2010-12-01 → 2011-12-09.
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Methodology

2. Pete Chapman et al. **CRISP-DM 1.0: Step-by-step data mining guide**. SPSS Inc., 2000.
   - URL: https://www.datascience-pm.com/crisp-dm-2/
   - Mục dùng: Quy trình CRISP-DM cho cấu trúc project (Business Understanding → Deployment).

## Clustering

3. scikit-learn: **Clustering documentation**.
   - URL: https://scikit-learn.org/stable/modules/clustering.html
   - Mục dùng: K-Means, Agglomerative, DBSCAN implementation và metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).

4. Peter J. Rousseeuw. **Silhouettes: A graphical aid to the interpretation and validation of cluster analysis**. Journal of Computational and Applied Mathematics, 20:53–65, 1987.
   - DOI: https://doi.org/10.1016/0377-0427(87)90125-7
   - Mục dùng: Silhouette score definition.

5. David L. Davies and Donald W. Bouldin. **A Cluster Separation Measure**. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2):224–227, 1979.
   - DOI: https://doi.org/10.1109/TPAMI.1979.4766909
   - Mục dùng: Davies-Bouldin Index definition.

## Classification

6. scikit-learn: **Model evaluation and selection**.
   - URL: https://scikit-learn.org/stable/modules/model_evaluation.html
   - Mục dùng: F1-score, Precision, Recall, ROC-AUC, Average Precision definitions.

7. scikit-learn: **Cross-validation: evaluating estimator performance**.
   - URL: https://scikit-learn.org/stable/modules/cross_validation.html
   - Mục dùng: StratifiedKFold protocol.

8. scikit-learn: **Common pitfalls and recommended practices**.
   - URL: https://scikit-learn.org/stable/common_pitfalls.html
   - Mục dùng: Data leakage prevention, pipeline best practices.

## Association Rule Mining

9. Rakesh Agrawal and Ramakrishnan Srikant. **Fast Algorithms for Mining Association Rules**. VLDB Conference, 1994.
   - URL: https://rakesh.agrawal-family.com/papers/vldb94apriori.pdf
   - Mục dùng: Apriori algorithm definition.

10. Jiawei Han, Jian Pei, and Yiwen Yin. **Mining Frequent Patterns without Candidate Generation**. SIGMOD Conference, 2000.
    - DOI: https://doi.org/10.1145/342009.335372
    - Mục dùng: FP-Growth algorithm definition.

11. Sebastian Raschka. **mlxtend: Frequent pattern mining**.
    - URL: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/
    - Mục dùng: Apriori, FP-Growth, association_rules implementation.

## RFM Analysis

12. Arthur Hughes. **Strategic Database Marketing**. McGraw-Hill, 2000.
    - Mục dùng: RFM segmentation methodology.

## Python Libraries (versions ghi tại runtime)

- pandas, numpy, scikit-learn, matplotlib, seaborn, mlxtend, joblib, streamlit
- Xem `requirements.txt` cho version cụ thể.