# 🚀 Machine Learning & Deep Learning MLOps Portfolio

👤 **Author Profile**

* **Name:** Mohamad Faisal Bashir
* **Class:** TK-47-04
* **NIM:** 101032300036
* **Institution:** Computer Engineering, Telkom University

This repository showcases two end-to-end machine learning projects, ranging from high-stakes fraud detection to deep learning regression modeling, with a heavy emphasis on **MLOps practices**, **experiment tracking**, **hyperparameter tuning**, **model evaluation**, and **cloud-native development** using **Azure Machine Learning**.

---

## 📁 Projects Overview

### 1. 🔍 End-to-End Fraud Detection Pipeline — Deep Learning Version

**Platform:** ![Azure Machine Learning](https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white)  
**File:** `fraud_detection.ipynb`

An advanced deep learning MLOps pipeline designed to detect fraudulent financial transactions from a highly imbalanced tabular dataset.

* **Objective:** Predict the probability of fraudulent transactions (`isFraud`) using a TensorFlow/Keras-based tabular neural network.
* **Dataset:** Uses `train_transaction.csv` as the main transaction dataset.
* **Strategy:**
    * **Memory Optimization:** Downcasts numeric columns to reduce RAM usage, important for large transaction datasets.
    * **Feature Engineering:** Creates transaction-time, transaction-amount, and email-domain-derived features.
    * **Handling Imbalance:** Applies **SMOTE/SMOTENC** only on the training split to reduce class imbalance without leaking information into validation or test data.
    * **Deep Learning Model:** Uses a tabular neural network with numeric inputs, categorical embeddings, dropout, batch normalization, and L2 regularization.
    * **Auto-Tuning:** Utilizes **Optuna** for hyperparameter optimization over 30 trials.
* **Workflow:**
    * Data ingestion and memory optimization.
    * Exploratory data analysis and fraud-ratio inspection.
    * Feature engineering from transaction data.
    * Train / validation / test split using stratification.
    * Missing value imputation, categorical mapping, scaling, and class imbalance handling.
    * Optuna tuning with MLflow tracking.
    * Final model training with best hyperparameters.
    * Threshold tuning using validation **F2-score**.
    * Final evaluation using **ROC-AUC**, **PR-AUC**, precision, recall, F1-score, and confusion matrix.
    * Model, preprocessing artifacts, metadata, and test predictions saved to `outputs/`.
* **Performance Snapshot:**
    * Dataset shape after loading: **138,133 rows × 394 columns**
    * Fraud ratio: **2.59%**
    * Best Optuna validation ROC-AUC: **0.9138**
    * Final test ROC-AUC: **0.8966**
    * Final test PR-AUC: **0.5783**
    * Tuned decision threshold: **0.8588**
* **Saved Artifacts:**
    * `outputs/fraud_detection_tabular_dl.keras`
    * `outputs/preprocessing_artifacts.joblib`
    * `outputs/feature_metadata.json`
    * `outputs/test_predictions.csv`

---

### 2. 📈 End-to-End Deep Learning Regression Pipeline with MLOps

**Platform:** ![Azure Machine Learning](https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white)  
**File:** `regression_dl.ipynb`

A highly reproducible deep learning regression framework integrated with the Azure ML ecosystem for professional experiment tracking, model comparison, and result interpretation.

* **Objective:** Predict a continuous target variable using deep learning regression while maintaining strict preprocessing, evaluation, and experiment tracking.
* **Dataset:** Uses `midterm-regresi-dataset.csv`, a tabular regression dataset. The notebook can automatically detect whether the dataset has a header and generate generic feature names if needed.
* **Strategy:**
    * **Data Cleaning:** Handles duplicated rows, missing values, high-cardinality columns, and outliers.
    * **Preprocessing:** Uses KNN imputation, categorical encoding, robust scaling, and target scaling.
    * **Feature Engineering:** Creates squared features, interaction features, and log-transformed features.
    * **Feature Selection:** Combines Mutual Information and ExtraTreesRegressor importance ranking.
    * **Target Imbalance Handling:** Applies **SMOTE-style / SMOTER-lite oversampling** for rare continuous target ranges on the training data only.
    * **Deep Learning Models:** Compares multiple TensorFlow/Keras MLP configurations.
    * **Auto-Tuning:** Uses **Optuna** with 15 trials to optimize the MLP architecture and training parameters.
    * **Interpretability:** Uses **LIME** to explain local feature contributions for individual regression predictions.
* **Workflow:**
    * Dataset loading and smart CSV header detection.
    * Exploratory data analysis and missing value audit.
    * Data cleaning, imputation, outlier treatment, and encoding.
    * Feature engineering and feature selection.
    * Train / validation / test split.
    * Scaling and SMOTER-lite target-range oversampling.
    * Baseline MLP model training.
    * Optuna hyperparameter tuning.
    * MLflow tracking for parameters, metrics, and artifacts.
    * Leaderboard generation and best-model selection.
    * LIME-based model interpretation.
* **Model Candidates:**
    * `MLP_Small`
    * `MLP_BatchNorm`
    * `MLP_Deep_Huber`
    * `MLP_Tuned_Optuna`
* **Best Model Snapshot:**
    * Best model by R²: **MLP_Deep_Huber**
    * R²: **0.6916**
    * RMSE: **12.8727**
    * MAE: **8.5664**
    * MAPE: **305.3967%**
* **Optuna Best Trial:**
    * Best validation RMSE: **12.7111**
    * Trials completed: **15**
    * Best activation: `elu`
    * Best batch size: `256`
    * Best loss: `huber`

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Cloud Platform** | Microsoft Azure Machine Learning |
| **MLOps & Tracking** | MLflow, Azure ML Studio, `azureml-mlflow`, `azureml-core` |
| **Deep Learning** | TensorFlow, Keras |
| **Optimization** | Optuna |
| **Class / Target Imbalance Handling** | SMOTE, SMOTENC, SMOTER-lite |
| **Explainability** | LIME |
| **Machine Learning Utilities** | Scikit-Learn, Imbalanced-Learn |
| **Data Processing** | Pandas, NumPy, Joblib |
| **Visualization** | Matplotlib, Seaborn, Plotly, Missingno |
| **Evaluation Metrics** | ROC-AUC, PR-AUC, F2-score, MSE, RMSE, MAE, R², MAPE |

---

## 📂 Repository Structure

```text
/
├── fraud_detection.ipynb          # Deep learning fraud detection pipeline
├── regression_dl.ipynb            # Deep learning regression pipeline
├── README.md
├── folder-data/                   # Dataset directory
│   ├── train_transaction.csv
│   └── midterm-regresi-dataset.csv
└── outputs/                       # Generated model artifacts and predictions
```

> **Note:** Dataset files are not included in this repository. Place the required CSV files inside `folder-data/` before running the notebooks.

---

## 🚀 Environment & Setup

### For Azure ML Projects

1. **Workspace:** Ensure you have access to an Azure Machine Learning workspace.
2. **Compute Instance:** Create and start a compute instance from Azure ML Studio.
3. **Recommended Compute:**
    * Fraud Detection: `Standard_E4ds_v4` or higher because transaction data can be memory intensive.
    * Regression DL: `Standard_F8s_v2`, `Standard_F16s_v2`, or GPU-backed compute if available.
4. **Tracking URI:** The notebooks are configured to use the Azure ML MLflow tracking URI when running inside Azure ML.
5. **Dataset Placement:** Upload the datasets into `folder-data/`.

Expected dataset structure:

```text
folder-data/
├── train_transaction.csv
└── midterm-regresi-dataset.csv
```

---

## ▶️ Running the Notebooks

### 1. Fraud Detection

Open:

```text
fraud_detection.ipynb
```

Then run the notebook from top to bottom.

The notebook installs and uses:

```python
tensorflow
optuna
mlflow
azureml-mlflow
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
plotly
joblib
```

After execution, the final model and artifacts are saved inside:

```text
outputs/
```

---

### 2. Regression Deep Learning

Open:

```text
regression_dl.ipynb
```

Then run the notebook from top to bottom.

The notebook installs and uses:

```python
optuna
mlflow
azureml-mlflow
azureml-core
lime
scikit-learn
tensorflow
pandas
numpy
matplotlib
seaborn
missingno
```

After execution, the notebook will produce:

* preprocessing outputs,
* model leaderboard,
* Optuna optimization results,
* MLflow experiment logs,
* LIME interpretation plots.

---

## 📊 Results Summary

| Project | Main Metric | Best / Final Result |
| :--- | :--- | :--- |
| Fraud Detection DL | Test ROC-AUC | **0.8966** |
| Fraud Detection DL | Test PR-AUC | **0.5783** |
| Fraud Detection DL | Tuned Threshold | **0.8588** |
| Regression DL | Best R² | **0.6916** |
| Regression DL | Best RMSE | **12.8727** |
| Regression DL | Best MAE | **8.5664** |

---

## 📈 MLflow Experiment Tracking

Both notebooks use MLflow for experiment tracking.

### Fraud Detection Experiment

```text
Fraud_Detection_Tabular_DL_Optuna
```

Tracked components include:

* Optuna trial parameters,
* ROC-AUC and PR-AUC,
* model architecture parameters,
* class imbalance configuration,
* final threshold,
* final artifacts and prediction outputs.

### Regression Experiment

```text
Midterm_Regression_DL_Pipeline
```

Tracked components include:

* baseline model configurations,
* Optuna hyperparameters,
* MSE, RMSE, MAE, R², and MAPE,
* training duration,
* number of trained epochs,
* model comparison results,
* LIME interpretation outputs.

To view results in Azure ML Studio:

1. Open **Azure ML Studio**.
2. Go to **Jobs** or **Experiments**.
3. Select the corresponding experiment name.
4. Open any run to inspect parameters, metrics, logs, and artifacts.

---

## 🧠 Model Interpretability

The regression notebook includes **LIME** to interpret how selected features affect individual predictions.

LIME is used to:

* explain typical, high, and low prediction samples,
* identify features that locally increase or decrease predicted values,
* produce aggregate feature importance across multiple test samples.

The fraud detection notebook focuses on threshold-based fraud probability interpretation using ROC Curve, Precision-Recall Curve, confusion matrix, and classification reports.

---

## 📄 License

Submitted as academic coursework.
