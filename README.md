# Week 4 — Unsupervised Learning & Customer Segmentation

AI/ML Internship at ITSimplera Solutions — customer segmentation on credit card behavioural data using K-Means and Hierarchical Clustering.

## Overview

This week moves into unsupervised learning there is no target column to predict. Instead, the goal is to discover natural, hidden groupings within customer behaviour and turn them into segments a business can act on: retention offers for dormant customers, loyalty perks for high-value spenders, and risk review for revolvers.

Two clustering approaches were applied and compared on the same dataset:
- **K-Means** — the primary, production-scale method
- **Agglomerative Hierarchical Clustering** — used as an independent validation method on a sample

## Dataset

| | |
|---|---|
| **Source** | [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) (Kaggle) |
| **Size** | 8,950 active credit card holders, 18 original features |
| **Period** | 6 months of behavioural data |
| **Feature themes** | balance, purchase behaviour (one-off vs. installment), cash advance usage, credit limit, payment history, account tenure |

`CUST_ID` was dropped before modeling it's a unique string identifier with no behavioural signal, and would only distort a distance-based algorithm.

## Methodology

**Preprocessing**
- 314 missing cells (313 in `MINIMUM_PAYMENTS`, 1 in `CREDIT_LIMIT`) were filled with the column **median** rather than the mean, since both fields are right-skewed by a small number of very high-limit customers.
- All 17 remaining features were standardized with `StandardScaler` (mean 0, std 1) before clustering mandatory because K-Means and hierarchical clustering both rely on Euclidean distance, and an unscaled feature like `CASH_ADVANCE` would dominate the distance calculation over a 0–1 feature like `BALANCE_FREQUENCY` purely due to its numeric range.

**Choosing k**
- K-Means was fit for every k from 2 to 10, recording inertia at each step.
- Silhouette Score was computed alongside it to cross-check the choice of k.

![Elbow Curve](https://github.com/user-attachments/assets/82b0c951-f118-40ee-b1ee-fd9a5450b3fb)

The elbow bends at k = 4–5, while silhouette actually peaks at k = 3 (0.251) before dipping a common, realistic disagreement, since the elbow method measures overall compactness and silhouette measures cluster separation. **k = 4** was selected: it sits at the elbow bend, its silhouette score (0.198) stays reasonable and close to the k = 3 peak, and it gives marketing/risk teams enough resolution to act on (k = 3 was found to merge two behaviourally distinct groups).

**Validation**
- Agglomerative Hierarchical Clustering (Ward linkage) was run on a random 300-row sample of the scaled data.
- Cutting the dendrogram at distance 28.82 produces exactly 4 clusters, confirming the same k = 4 structure holds hierarchically.

## Quantitative Insights

**Inertia and Silhouette Score by k**

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Inertia | 127,784.5 | 111,975.0 | 99,061.9 | 91,490.5 | 84,826.6 | 79,856.2 | 74,484.9 | 69,828.7 | 66,466.4 |
| Silhouette | 0.210 | 0.251 | 0.198 | 0.193 | 0.203 | 0.208 | 0.222 | 0.226 | 0.220 |

**Cluster sizes (k = 4)**

![Cluster Distribution](https://github.com/user-attachments/assets/044f9774-34b7-414b-81a7-6c5b3df8f52d)

**Cluster profile — mean feature values**

| Cluster | Balance | Purchases | Cash Advance | Credit Limit | Full Payment % |
|---|---|---|---|---|---|
| 0 — Dormant (3,977 · 44.4%) | 1,012.7 | 270.0 | 596.5 | 3,278.6 | 0.10 |
| 1 — High-Value (409 · 4.6%) | 3,551.2 | 7,681.6 | 653.6 | 9,696.9 | 0.30 |
| 2 — Cash-Advance Risk (1,197 · 13.4%) | 4,602.4 | 501.9 | 4,521.5 | 7,546.2 | 0.00 |
| 3 — Frequent Payer (3,367 · 37.6%) | 894.9 | 1,236.2 | 210.6 | 4,213.2 | 0.30 |

**K-Means vs. Agglomerative cross-tabulation (n = 300 sample)**

| K-Means \ Agglomerative | Agg. 0 | Agg. 1 | Agg. 2 | Agg. 3 |
|---|---|---|---|---|
| K-Means 0 | 1 | 6 | 124 | 0 |
| K-Means 1 | 15 | 0 | 0 | 0 |
| K-Means 2 | 1 | 31 | 6 | 1 |
| K-Means 3 | 81 | 0 | 33 | 1 |

Most K-Means clusters have one dominant matching Agglomerative cluster (e.g. 124 of K-Means Cluster 0 land in Agglomerative Cluster 2; 81 of K-Means Cluster 3 land in Agglomerative Cluster 0), showing real structural agreement on the core of each segment, with expected cross-over at the boundaries.

## Repository Structure

```
├── notebooks/
│   └── week4_clustering.ipynb
├── requirements.txt
└── README.md
├── data/
│   └── CC GENERAL.csv
├── Images/
```
