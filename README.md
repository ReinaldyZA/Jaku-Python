# ISPU Air Quality Classification for DKI Jakarta

Machine learning classification of Jakarta's Air Pollution Standard Index (ISPU) categories using **Random Forest**, **XGBoost**, and **SVM**, following the **CRISP-DM** methodology.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lavUHdIOjS2_v7yF3cjCpL75Th-RqemV?usp=sharing)

## Background

Air quality in DKI Jakarta directly affects public health. ISPU categorizes daily air quality based on six pollutants: PM10, PM2.5, SO₂, CO, O₃, and NO₂. Manual categorization is time-consuming and prone to inconsistency, so this project builds a predictive model to classify ISPU categories automatically.

**Objectives**

1. Build ISPU classification models using Random Forest, XGBoost, and SVM.
2. Compare model performance using accuracy, precision, recall, and F1-score.
3. Determine the best model for air quality classification in DKI Jakarta.

## Dataset

`Data_ISPU.csv` (semicolon-separated), 3,350 rows and 13 columns from 5 monitoring stations in DKI Jakarta (DKI1 Bundaran HI, DKI2 Kelapa Gading, DKI3 Jagakarsa, DKI4 Lubang Buaya, DKI5 Kebon Jeruk).

| Role | Columns |
|---|---|
| Features (X) | `pm_sepuluh`, `pm_duakomalima`, `sulfur_dioksida`, `karbon_monoksida`, `ozon`, `nitrogen_dioksida` |
| Target (y) | `kategori` |

| Category | ISPU Range |
|---|---|
| BAIK (Good) | 0 – 50 |
| SEDANG (Moderate) | 51 – 100 |
| TIDAK SEHAT (Unhealthy) | 101 – 200 |

## Methodology (CRISP-DM)

| Phase | Steps |
|---|---|
| 1. Business Understanding | Problem definition, objectives, target variable |
| 2. Data Understanding | Data structure, descriptive statistics, class distribution, histograms, boxplots, missing values, correlation heatmap |
| 3. Data Preparation | Remove invalid categories, standardize station names, numeric conversion, median imputation, IQR outlier removal, label encoding, 80:20 stratified split, Z-score scaling (SVM only) |
| 4. Modeling | Baseline models, hyperparameter tuning with GridSearchCV + Stratified 5-Fold CV |
| 5. Evaluation | Confusion matrix, classification report, cross-validation, feature importance, model comparison, SMOTE comparison |
| 6. Deployment | Save models, encoder, and scaler with `joblib`, predict new data |

After cleaning and outlier removal, the final dataset contains **3,057 rows** (2,445 train / 612 test).

## Results

Performance after hyperparameter tuning:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | CV Mean Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.9706 | 0.9550 | 0.9506 | 0.9527 | 0.9808 ± 0.0054 |
| **XGBoost** | **0.9771** | 0.9623 | **0.9645** | **0.9634** | **0.9845 ± 0.0021** |
| SVM | 0.9690 | **0.9666** | 0.9315 | 0.9457 | 0.9538 ± 0.0044 |

**Best model: XGBoost** with 97.71% test accuracy and the most stable cross-validation score.

Best hyperparameters:

| Model | Parameters |
|---|---|
| Random Forest | `n_estimators=100`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features='sqrt'` |
| XGBoost | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=1.0` |
| SVM | `kernel='rbf'`, `C=10`, `gamma=0.1` |

The notebook also compares results before and after SMOTE to address class imbalance.

## Output Files

| File | Description |
|---|---|
| `model_random_forest.pkl` | Tuned Random Forest model |
| `model_xgboost.pkl` | Tuned XGBoost model |
| `model_svm.pkl` | Tuned SVM model |
| `label_encoder.pkl` | Target label encoder |
| `standard_scaler.pkl` | Scaler for SVM input |
| `fitur_polutan.pkl` | List of feature columns |

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1lavUHdIOjS2_v7yF3cjCpL75Th-RqemV?usp=sharing).
2. Run all cells in order.
3. Upload `Data_ISPU.csv` when prompted.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, SMOTE, joblib, Google Colab

## Author

**Rei** ([@ReinaldyZA](https://github.com/ReinaldyZA))
