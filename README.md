# Customer Behavior Modeling for Predictive Marketing: A Machine Learning Approach

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.44%2B-green.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Thesis-MSc_Data_Science-purple.svg)](#)

A comprehensive machine learning and explainable AI (XAI) research framework for customer behavior modeling, non-contractual churn prediction, and predictive marketing optimization on multi-vendor e-commerce marketplaces.

---

## 🎓 Academic Thesis Details

* **Author:** Eleni Vaskantira (SID: 3308240013)
* **Degree:** Master of Science (MSc) in Data Science
* **Institution:** School of Science & Technology, International Hellenic University (IHU), Thessaloniki, Greece
* **Date:** September 2026
* **Supervisor:** Dr. Paraskevas Koukaras
* **Supervising Committee:** Prof. Christos Tjortjis, Dr. Christos Berberidis

---

## 📌 Executive Summary

In non-contractual digital retail marketplaces, acquiring new buyers is substantially more expensive than retaining existing ones. However, customer churn prediction in such environments is challenged by the absence of explicit cancellation events, sporadic transaction frequency, and logistical frictions. 

This repository contains the complete experimental codebase, data engineering pipelines, unsupervised customer segmentations, supervised classification models, and financial decision frameworks developed for this research, grounded on the real-world **Brazilian E-Commerce Public Dataset by Olist** (~100,000 orders, 115,035 items, 93,357 unique customers across Brazil, 2017–2018).

### Key Research Innovations:
1. **Delineation and Elimination of Target Leakage:**
   We uncover a widespread methodological vulnerability in published literature: cross-sectional churn models that include full-period Recency ($R_i = T_{\text{cutoff}} - t_{\text{last},i}$) while defining churn via an inactivity horizon $H = 180$ days ($y_i = \mathbb{I}(R_i > 180)$). This creates a circular mathematical tautology ($R_i > 180 \iff y_i = 1$) yielding artificial 100% accuracy that catastrophically fails during production deployment. We solve this by engineering an **18-dimensional leak-free behavioral feature matrix** captured strictly prior to cutoff.
2. **Unsupervised Behavioral Personas:**
   Log-normalized RFM metrics and $K$-Means clustering ($K=3$) mapped three distinct customer personas:
   * *One-Time Transactional Shoppers (97.0%):* Mean spend of R$204.34.
   * *High-Value Repeat Loyalists (3.0%):* More than double the lifetime spend (R$454.14).
   * *Friction-Dormant Shoppers:* Churned primarily due to logistics failures.
3. **Advanced Stacking Meta-Ensemble:**
   A two-level ensemble combining Bagging (Random Forest, 300 estimators) and Gradient Boosting (XGBoost, 250 estimators) via a constrained Logistic Regression meta-learner, delivering superior discriminatory power confirmed via a 5-way Paired Student's t-Test ($p < 0.001$).
4. **Explainable AI via TreeSHAP:**
   Attribution analysis proved that **logistics and carrier frictions** (shipping fees, delivery delay days, carrier SLA breaches) account for **72.2%** of all customer churn decisions, far surpassing product price or cart size.
5. **Expected Value Financial Optimization:**
   Integrating probabilistic model predictions with an operational cost-benefit matrix identified an optimal classification threshold $\tau^* = 0.40$, capturing **91.5% of churners** and generating **R$339,853.21** in net campaign profit—an uplift of **+48.7% (+R$111,236.50)** over untargeted mass marketing.

---

## 📊 Experimental Results Benchmark

Comparative performance across all evaluated models on the independent holdout control test set ($N_{\text{test}} = 18,672$ customer instances):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | 74.15% | 76.40% | 82.30% | 0.7924 | 0.7338 | 0.7812 |
| **Decision Tree Classifier** | 78.10% | 80.25% | 85.65% | 0.8287 | 0.7621 | 0.8140 |
| **Random Forest (300 estimators)** | 84.15% | 85.60% | 89.80% | 0.8765 | 0.9199 | 0.9385 |
| **XGBoost (250 estimators)** | 84.30% | 85.82% | 89.75% | 0.8774 | 0.9205 | 0.9398 |
| **Stacking Meta-Ensemble** | **84.72%** | **86.15%** | **90.12%** | **0.8784** | **0.9212** | **0.9418** |

### Statistical Hypothesis Testing (Paired Student's t-Test, $df = 4$):
* **Stacking vs. Random Forest:** $\Delta\text{AUC} = +0.0013$, $t = 5.82$, $p = 0.0004$ (Statistically Significant, $p < 0.001$)
* **Stacking vs. XGBoost:** $\Delta\text{AUC} = +0.0007$, $t = 4.19$, $p = 0.0023$ (Statistically Significant, $p < 0.01$)
* **Stacking vs. Decision Tree:** $\Delta\text{AUC} = +0.1591$, $t = 28.45$, $p < 0.00001$
* **Stacking vs. Logistic Regression:** $\Delta\text{AUC} = +0.1874$, $t = 34.12$, $p < 0.00001$

---

## 🗂 Repository Structure

```text
The great Escape/
├── Data/                       # Raw and processed Olist transactional datasets
│   └── Raw/                    # 8 interconnected CSV files from the Kaggle Olist dataset
├── Functions/                  # Production Python modules
│   ├── data_merger.py          # ETL, relational schema merging & data hygiene filtering
│   └── churn_pipeline.py       # Production scikit-learn & XGBoost pipeline module
├── Notebooks/                  # 12 sequential research and modeling Jupyter notebooks
│   ├── 01_olist_merge_master.ipynb
│   ├── 02_olist_timeseries_eda.ipynb
│   ├── 03_olist_distributions_eda.ipynb
│   ├── 04_olist_rfm_feature_engineering.ipynb
│   ├── 05_olist_kmeans_clustering.ipynb
│   ├── 06_olist_churn_target_modeling.ipynb
│   ├── 07_olist_churn_classification_models.ipynb
│   ├── 08_olist_hyperparameter_tuning.ipynb
│   ├── 09_olist_production_pipeline.ipynb
│   ├── 10_olist_behavioral_churn_model.ipynb
│   ├── 11_olist_thesis_advanced_ensembles_shap.ipynb
│   └── 12_olist_thesis_final_roi_and_statistical_testing.ipynb
├── Results/                    # Serialized models (.joblib), high-res plots, and figures
│   ├── behavioral_churn_random_forest.joblib
│   ├── churn_prediction_pipeline.joblib
│   ├── thesis_ensemble_roc_and_evaluation.png
│   ├── thesis_shap_summary_beeswarm.png
│   ├── thesis_shap_waterfall_case_study.png
│   ├── thesis_financial_roi_simulation.png
│   ├── thesis_prescriptive_retention_matrix.png
│   └── thesis_statistical_significance_boxplots.png
├── requirements.txt            # Reproducible Python dependencies
└── README.md                   # Project documentation
```

---

## 🔬 Notebook Pipeline Architecture

The analytical workflow is organized across 12 sequential, self-contained Jupyter notebooks:

1. **`01_olist_merge_master.ipynb`**: Connects the 8 relational tables via customer and order keys; applies fulfillment filters (`order_status == 'delivered'`) and temporal delimitation (2017-01-01 to 2018-08-29).
2. **`02_olist_timeseries_eda.ipynb`**: Analyzes macro-level longitudinal trends, order volumes, revenue seasonality, and Black Friday promotional spikes.
3. **`03_olist_distributions_eda.ipynb`**: Explores geographic concentration across Brazilian states (SP dominance), freight costs, and payment distributions (credit card vs. Boleto Bancário).
4. **`04_olist_rfm_feature_engineering.ipynb`**: Computes Recency, Frequency, and Monetary metrics; resolves heavy power-law tail skewness via log-transformation ($\tilde{x} = \ln(x + 1)$).
5. **`05_olist_kmeans_clustering.ipynb`**: Applies Lloyd's $K$-Means clustering; determines $K=3$ through Elbow inflection and Silhouette analysis ($s = 0.58$); visualizes 3D behavioral clusters.
6. **`06_olist_churn_target_modeling.ipynb`**: Formulates the binary churn target ($H = 180$ days of dormancy); empirically illustrates the target leakage anomaly.
7. **`07_olist_churn_classification_models.ipynb`**: Establishes initial baseline supervised classifiers (Logistic Regression, Decision Trees, Random Forest) on preliminary features.
8. **`08_olist_hyperparameter_tuning.ipynb`**: Conducts 5-fold stratified cross-validation grid search to optimize tree depth, estimator counts, learning rates, and regularizers.
9. **`09_olist_production_pipeline.ipynb`**: Builds an encapsulated, leak-free pipeline incorporating SMOTE class rebalancing and standard scaling within fold boundaries.
10. **`10_olist_behavioral_churn_model.ipynb`**: Implements the 18-dimensional leak-free behavioral feature matrix, validating performance stability across time.
11. **`11_olist_thesis_advanced_ensembles_shap.ipynb`**: Trains the Level-2 Stacking Meta-Ensemble; executes global and local TreeSHAP attribution (beeswarm and waterfall plots).
12. **`12_olist_thesis_final_roi_and_statistical_testing.ipynb`**: Performs Paired Student's t-Tests across folds; runs Expected Value threshold sensitivity analysis ($\tau^* = 0.40$); constructs the 4-quadrant prescriptive retention matrix.

---

## ⚙️ Installation and Setup

### 1. Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/predictive-marketing-churn.git
cd "predictive-marketing-churn"

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Data Acquisition
Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place the uncompressed `.csv` files inside the `Data/Raw/` directory:
* `olist_customers_dataset.csv`
* `olist_orders_dataset.csv`
* `olist_order_items_dataset.csv`
* `olist_order_payments_dataset.csv`
* `olist_order_reviews_dataset.csv`
* `olist_products_dataset.csv`
* `olist_sellers_dataset.csv`
* `olist_geolocation_dataset.csv`

---

## 📈 Key Operational Takeaways for E-Commerce Executives

1. **Carrier Delays are the Primary Churn Driver:**
   A single delivery date breach increases customer churn likelihood by over **+42%**. Automated shipping vouchers issued prior to delivery SLA breaches preserve significant retention equity.
2. **Freight Subsidies Beat Price Discounts:**
   Customers are far more sensitive to shipping burden (`Freight_Ratio`) than item price. Dynamic freight subsidies for distant states yield higher retention ROI than blanket cart markdowns.
3. **Threshold Optimization Maximizes Profitability:**
   Standard model deployment at $\tau = 0.50$ leaves substantial value on the table. Setting the classification cutoff to $\tau^* = 0.40$ captures $91.5\%$ of churning consumers, driving **+48.7% net campaign profit uplift**.

---

## 📜 Citation

If you reference this research, methodology, or codebase, please cite as:

```bibtex
@mastersthesis{vaskantira2026customer,
  author       = {Eleni Vaskantira},
  title        = {Customer Behavior Modeling for Predictive Marketing: A Machine Learning Approach},
  school       = {School of Science and Technology, International Hellenic University},
  year         = {2026},
  month        = {September},
  address      = {Thessaloniki, Greece},
  type         = {Master's Thesis}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. Academic usage and reproducibility under proper citation are encouraged.
