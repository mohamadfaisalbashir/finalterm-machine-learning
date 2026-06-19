# 🔍 Fraud Detection — End-to-End Deep Learning Pipeline on Azure ML

> **Author:** Mohamad Faisal Bashir · Class: TK-47-04 · NIM: 101032300036

An end-to-end deep learning pipeline for detecting fraudulent financial transactions using the IEEE-CIS Fraud Detection transaction dataset. This project covers the full workflow required in the assignment: data loading, memory optimization, exploratory data analysis, data cleaning, feature engineering, missing value handling, class imbalance handling using **SMOTE/SMOTENC**, tabular deep learning model training with TensorFlow/Keras, automated hyperparameter tuning using Optuna, evaluation with fraud-relevant metrics, and experiment tracking with **MLflow on Azure Machine Learning**.

> **Important note:** The assignment description mentions both transaction and identity tables. However, this implementation follows the development constraint that only `train_transaction.csv` is used. Therefore, the notebook creates train, validation, and test splits from `train_transaction.csv`.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Workflow](#-pipeline-workflow)
- [Key Design Decisions](#-key-design-decisions)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Azure ML Setup Guide](#-azure-ml-setup-guide)
- [Running the Notebook](#-running-the-notebook)
- [Results & Metrics](#-results--metrics)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Output Artifacts](#-output-artifacts)
- [Cost Management](#-cost-management)
- [License](#-license)

---

## 📌 Project Overview

Fraud detection is a highly imbalanced binary classification problem. In this dataset, fraudulent transactions represent only a small portion of all records, which makes simple accuracy unreliable. The objective is not only to classify transactions as fraud or non-fraud, but also to predict the probability that a transaction is fraudulent.

This project builds a deep learning-based fraud detection pipeline that handles:

- **High-dimensional tabular transaction data** from `train_transaction.csv`
- **Missing values** in numerical and categorical features
- **Class imbalance** using SMOTE for numerical-only cases and SMOTENC when categorical features are present
- **Feature engineering** from transaction time, transaction amount, and email-domain information
- **Numerical scaling and categorical encoding** for neural network input
- **Categorical embeddings** inside the neural network architecture
- **Automated hyperparameter tuning** using Optuna
- **Experiment tracking** using MLflow in Azure Machine Learning
- **Final model evaluation** using ROC-AUC, PR-AUC, F2-score, classification report, confusion matrix, ROC curve, and precision-recall curve

**Dataset file:**

| File | Description |
|------|-------------|
| `train_transaction.csv` | Financial transaction features and target label used for training, validation, and testing |

**Target variable:** `isFraud`  
- `0` = legitimate transaction  
- `1` = fraudulent transaction  

---

## 🔄 Pipeline Workflow

```text
1. Environment Setup & Library Imports
2. Configuration
3. Data Ingestion & Memory Optimization
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Train / Validation / Test Split
7. Feature Selection & Preprocessing for Deep Learning
8. Handling Class Imbalance with SMOTE / SMOTENC
9. TensorFlow Dataset Builder
10. Deep Learning Model Architecture
11. Hyperparameter Tuning with Optuna + MLflow Tracking
12. Hyperparameter Tuning Results
13. Final Model Training
14. Training Curve Visualization
15. Threshold Tuning and Model Evaluation
16. ROC Curve and Precision-Recall Curve
17. Save Model, Preprocessing Artifacts, and Predictions
18. MLflow Metrics Tracking and Local Artifact Saving
19. Inference Helper for New Transaction Data
```

---

## 🧠 Key Design Decisions

| Step | Decision | Rationale |
|------|----------|-----------|
| Dataset scope | Uses only `train_transaction.csv` | Matches the development constraint while still enabling full train/validation/test evaluation |
| Memory optimization | Downcasts numeric dtypes | Reduces RAM usage on Azure ML Compute Instance |
| Split strategy | 70% train, 15% validation, 15% test with stratification | Preserves fraud ratio across splits and allows unbiased final testing |
| Missing value handling | Median imputation for numerical features and special token for categorical features | Makes data compatible with neural networks |
| Feature engineering | Adds transaction amount log, day/hour features, and email-domain provider/suffix features | Captures useful fraud patterns from raw transaction data |
| Feature selection | Drops columns with >90% missing values and keeps valid numerical/categorical columns | Reduces noise and memory usage |
| Categorical processing | Rare-category handling and integer encoding | Prepares categorical data for embedding layers |
| Class imbalance | Uses SMOTE/SMOTENC only on training data | Avoids leakage into validation/test sets while improving fraud-class representation |
| Model architecture | Numerical branch + categorical embedding branch + dense layers | Adapts deep learning to mixed tabular data |
| Tuning | Optuna with ROC-AUC as objective | Searches neural network depth, units, dropout, learning rate, L2 regularization, embedding dimension, and batch size |
| Threshold tuning | F2-score on validation set | Prioritizes recall, which is important in fraud detection |
| Tracking | MLflow on Azure ML | Logs parameters, metrics, and experiment runs for reproducibility |

---

## 🛠 Tech Stack

| Category | Library / Tool |
|----------|----------------|
| Deep Learning | `tensorflow`, `keras` |
| Hyperparameter Tuning | `optuna` |
| Experiment Tracking | `mlflow`, `azureml-mlflow` |
| Class Imbalance Handling | `imbalanced-learn`, `SMOTE`, `SMOTENC` |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Artifact Serialization | `joblib`, `json` |
| Infrastructure | Azure Machine Learning Compute Instance |

---

## 📁 Repository Structure

```text
/
├── fraud_detection.ipynb              # Main deep learning notebook
├── folder-data/                       # Dataset directory
│   └── train_transaction.csv          # IEEE-CIS transaction training data
└── README.md
```

> **Note:** Dataset files are not included in the repository due to size. Download the IEEE-CIS Fraud Detection dataset from Kaggle and place `train_transaction.csv` inside `folder-data/`.

---

## ☁️ Azure ML Setup Guide

### Step 1 — Create Azure ML Workspace

1. Go to [portal.azure.com](https://portal.azure.com).
2. Search for **Azure Machine Learning**.
3. Click **Create** and fill in the workspace details:
   - **Subscription:** Azure for Students or your active Azure subscription
   - **Resource group:** for example, `UAS`
   - **Workspace name:** for example, `fsl-uas`
   - **Region:** Southeast Asia
   - **Storage account:** create new or use the automatically generated one
   - **Key vault:** create new or use the automatically generated one
   - **Application insights:** create new or use the automatically generated one
   - **Container registry:** `None` is acceptable for notebook-based training
4. Click **Review + Create**.
5. After deployment finishes, click **Go to resource** and launch **Azure Machine Learning Studio**.

### Step 2 — Create a Compute Instance

1. In Azure ML Studio, open **Compute > Compute instances**.
2. Click **+ New**.
3. Select **CPU** compute.
4. Recommended size:
   - **Minimum safe:** `Standard_E4ds_v4` with 4 cores and 32 GB RAM
   - **Faster option:** `Standard_E8ds_v4` or `Standard_E8ds_v5` if quota and budget allow
5. Enable **auto-shutdown** to reduce cost.
6. Wait until the compute status becomes **Running**.

> GPU is not required for this project because most of the workload is tabular preprocessing, SMOTE/SMOTENC, Optuna tuning, and CPU/RAM-heavy data manipulation.

### Step 3 — Upload Files via JupyterLab

1. Open the running compute instance and click **JupyterLab**.
2. Create a folder named `folder-data`.
3. Upload `train_transaction.csv` into `folder-data/`.
4. Upload `fraud_detection.ipynb` to the working directory.
5. The expected file structure is:

```text
/
├── fraud_detection.ipynb
└── folder-data/
    └── train_transaction.csv
```

---

## ▶️ Running the Notebook

1. Open `fraud_detection.ipynb` in Azure ML JupyterLab.
2. Select the kernel:

```text
Python 3.10 - Pytorch and Tensorflow
```

3. Run the dependency installation cell:

```python
%pip install tensorflow optuna mlflow azureml-mlflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn plotly joblib --quiet
```

4. Restart the kernel:

```text
Kernel > Restart Kernel
```

5. Run all cells from top to bottom.

For the first trial run, reduce compute time by setting:

```python
N_TRIALS = 5
EPOCHS_PER_TRIAL = 3
FINAL_EPOCHS = 10
SMOTE_SAMPLING_STRATEGY = 0.10
```

For the final run, the notebook configuration uses:

```python
N_TRIALS = 30
EPOCHS_PER_TRIAL = 10
FINAL_EPOCHS = 30
SMOTE_SAMPLING_STRATEGY = 0.30
```

---

## 📊 Results & Metrics

The final model is evaluated using validation and test data. Since this is an imbalanced fraud detection task, ROC-AUC and PR-AUC are more informative than accuracy alone.

### Dataset Summary from Notebook Run

| Item | Value |
|------|-------|
| Raw dataset shape | `(44195, 394)` |
| Shape after feature engineering | `(44195, 401)` |
| Train split | `30935` rows |
| Validation split | `6630` rows |
| Test split | `6630` rows |
| Numerical features used | `375` |
| Categorical features used | `18` |
| Total features used | `393` |
| SMOTE/SMOTENC strategy | `0.30` |
| Original train fraud count | `875` |
| Resampled train fraud count | `9018` |
| Resampled train rows | `39078` |

### Best Optuna Configuration

```python
{
    "hidden_units": [128, 128],
    "dropout_rate": 0.3479069935892086,
    "learning_rate": 0.0018648483566534718,
    "l2_reg": 5.58036728964346e-06,
    "max_embedding_dim": 64
}
```

**Best batch size:** `512`

### Final Evaluation

| Evaluation | Threshold | ROC-AUC | PR-AUC | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|-----------|-----------|---------|--------|----------|-----------------|--------------|----------|
| Validation Default | `0.5000` | `0.9091` | `0.5232` | `0.9386` | `0.2689` | `0.6845` | `0.3861` |
| Validation Tuned | `0.5545` | `0.9091` | `0.5232` | `0.9468` | `0.3014` | `0.6738` | `0.4165` |
| Test Default | `0.5000` | `0.8745` | `0.5647` | `0.9305` | `0.2335` | `0.6417` | `0.3424` |
| Test Tuned | `0.5545` | `0.8745` | `0.5647` | `0.9382` | `0.2560` | `0.6257` | `0.3634` |

**Best validation threshold using F2-score:** `0.5545`  
**Best validation F2-score:** `0.5403`

---

## 📈 MLflow Experiment Tracking

The notebook uses the following MLflow experiment:

```text
Fraud_Detection_Tabular_DL_Optuna
```

MLflow tracks:

- Optuna trial parameters
- Neural network architecture parameters
- Class imbalance settings
- SMOTE/SMOTENC configuration
- Validation ROC-AUC
- Validation PR-AUC
- Final test ROC-AUC
- Final test PR-AUC
- Best validation threshold

To view the experiment in Azure ML Studio:

1. Open **Azure Machine Learning Studio**.
2. Go to **Jobs** or **Experiments**.
3. Select:

```text
Fraud_Detection_Tabular_DL_Optuna
```

4. Open runs such as:
   - `trial_0`, `trial_1`, ..., `trial_29`
   - `final_tabular_dl_model`
   - `final_metrics_logging_safe`

### Artifact Logging Note

In some Azure ML notebook environments, `mlflow.log_artifact()` may raise an AzureML artifact repository compatibility error similar to:

```text
azureml_artifacts_builder() got an unexpected keyword argument 'tracking_uri'
```

To keep the workflow stable, the notebook logs metrics and parameters to MLflow while saving model artifacts locally in the `outputs/` directory.

---

## 📦 Output Artifacts

After running the notebook, the following artifacts are created locally:

| File | Description |
|------|-------------|
| `outputs/fraud_detection_tabular_dl.keras` | Final trained Keras model |
| `outputs/preprocessing_artifacts.joblib` | Numerical imputer, scaler, categorical mappings, selected columns, and threshold metadata |
| `outputs/feature_metadata.json` | Human-readable feature metadata |
| `outputs/test_predictions.csv` | Test-set prediction probabilities |

To download all artifacts from Azure ML JupyterLab:

```python
!zip -r fraud_detection_outputs.zip outputs
```

Then right-click `fraud_detection_outputs.zip` in the JupyterLab file explorer and select **Download**.

---

## 🧪 Inference Helper

The notebook includes an inference helper function:

```python
preprocess_new_transactions(raw_df, artifacts)
```

This function transforms new transaction rows using the saved preprocessing artifacts and aligns new input data with the training-time feature schema. The trained Keras model can then output fraud probabilities for new transactions.

## 📄 License

Submitted as academic coursework. Dataset usage is subject to the IEEE-CIS Fraud Detection competition rules on Kaggle.
