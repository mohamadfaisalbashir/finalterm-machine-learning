# 🔬 Midterm Regression DL Pipeline on Azure ML

> **Author:** Mohamad Faisal Bashir · Class: TK-47-04 · NIM: 101032300036

## 🚀 How to View the Full Notebook

Due to the comprehensive outputs, extensive training logs, MLflow experiment links, Optuna tuning results, model comparison charts, and LIME interpretation plots, this Jupyter Notebook may be too large to render natively on GitHub.

👉 **[Click here to view the fully rendered notebook via NBViewer](https://nbviewer.org/github/mohamadfaisalbashir/midterm-machine-learning/blob/main/regression/regression_dl.ipynb)**

> If the repository path is different, replace the NBViewer URL with the actual GitHub path of `regression_dl.ipynb`.

An end-to-end **deep learning regression pipeline** built with MLOps best practices, featuring TensorFlow/Keras MLP models, SMOTE-style regression oversampling, automated hyperparameter tuning via Optuna, model interpretation using LIME, and experiment tracking with MLflow — deployed on **Azure Machine Learning**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Workflow](#-pipeline-workflow)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Azure ML Setup Guide](#️-azure-ml-setup-guide)
- [Running the Notebook](#️-running-the-notebook)
- [Results & Metrics](#-results--metrics)
- [Model Interpretation with LIME](#-model-interpretation-with-lime)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Troubleshooting Notes](#-troubleshooting-notes)
- [Cost Management](#-cost-management)
- [License](#-license)

---

## 📌 Project Overview

This project implements a supervised **deep learning regression pipeline** to predict a continuous target variable from the provided tabular dataset. The workflow follows MLOps principles to make the experiment reproducible, trackable, and easier to evaluate in Azure Machine Learning.

The notebook covers data loading, data cleaning, missing value handling, outlier treatment, feature engineering, feature selection, deep learning model training, hyperparameter tuning, evaluation, model interpretation, and MLflow tracking.

**Dataset file:**

| File | Description |
|------|-------------|
| `midterm-regresi-dataset.csv` | Tabular regression dataset containing input features and one continuous target variable |

> The notebook automatically detects whether the CSV has a header. If no header is detected, generic column names such as `Feature_0`, `Feature_1`, ..., `Target` are generated automatically.

---

## 🔄 Pipeline Workflow

```text
1. Environment Setup & Library Imports
2. Dataset Loading & Smart CSV Header Detection
3. Exploratory Data Analysis (EDA)
4. Data Cleaning & Deduplication
5. Missing Value Handling
   - Numeric features: KNNImputer
   - Categorical features: most frequent imputation
6. Outlier Treatment
   - IQR-based Winsorization
7. Categorical Encoding
   - Label Encoding
   - One-Hot Encoding
   - Frequency Encoding
8. Feature Engineering
   - Squared features
   - Interaction features
   - log1p transformation
9. Feature Selection
   - Mutual Information
   - ExtraTreesRegressor feature importance
10. Train / Validation / Test Split
11. Feature Scaling with RobustScaler
12. Regression Target Imbalance Handling
   - SMOTE-style / SMOTER-lite oversampling
13. Target Scaling with StandardScaler
14. Baseline Deep Learning Model Training
15. Hyperparameter Tuning with Optuna
16. MLflow Experiment Tracking
17. Model Evaluation
   - MSE
   - RMSE
   - MAE
   - R²
   - MAPE
18. Model Interpretation with LIME
19. Final Result Interpretation
```

---

## 🛠 Tech Stack

| Category | Library / Tool |
|----------|----------------|
| Deep Learning | `tensorflow`, `keras` |
| Model Architecture | Multi-Layer Perceptron (MLP), Batch Normalization, Dropout, L2 Regularization |
| Hyperparameter Tuning | `optuna` |
| Experiment Tracking | `mlflow`, `azureml-mlflow`, `azureml-core` |
| Model Interpretation | `lime` |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Imputation | `KNNImputer`, `SimpleImputer` |
| Feature Selection | `mutual_info_regression`, `ExtraTreesRegressor` |
| Scaling | `RobustScaler`, `StandardScaler` |
| Visualization | `matplotlib`, `seaborn`, `missingno` |
| Infrastructure | Azure Machine Learning Compute Instance |

---

## 📁 Repository Structure

```text
/
├── regression_dl.ipynb                # Main deep learning regression notebook
├── folder-data/                       # Dataset directory (not tracked in Git)
│   └── midterm-regresi-dataset.csv
└── README.md
```

> **Note:** The dataset file is not included in this repository. Place `midterm-regresi-dataset.csv` inside `folder-data/` before running the notebook.

---

## ☁️ Azure ML Setup Guide

### Step 1 — Create Azure ML Workspace

1. Go to [portal.azure.com](https://portal.azure.com) and search for **Azure Machine Learning**.
2. Click **+ Create > New Workspace** and fill in the required fields:
   - **Resource group:** for example, `rg-mlops-faisal`
   - **Workspace name:** for example, `ml-workspace-faisal`
   - **Region:** Southeast Asia / nearby available region
3. Click **Review + Create**.
4. After deployment finishes, click **Go to resource**.
5. Click **Launch studio** to open Azure ML Studio.

### Step 2 — Create a Compute Instance

1. In Azure ML Studio, open **Compute > Compute instances > + New**.
2. Create a compute instance.

Recommended options:

| Compute Type | Example Size | Notes |
|--------------|--------------|-------|
| CPU | `Standard_F8s_v2` | Good balance for general notebook execution |
| CPU Memory-Optimized | `Standard_E4ds_v4` | Useful when the dataset is larger |
| GPU | `Standard_NC4as_T4_v3` or similar | Recommended if available for faster TensorFlow training |

3. Wait until the compute instance status becomes **Running**.

> The notebook includes mixed precision support. It can run on CPU, but GPU is recommended when available to speed up TensorFlow/Keras training.

### Step 3 — Upload Files via JupyterLab

1. On the compute instance page, click **JupyterLab**.
2. In the root directory, create a folder named exactly:

```text
folder-data
```

3. Upload the dataset file into that folder:

```text
midterm-regresi-dataset.csv
```

4. Upload the notebook file into the root directory:

```text
regression_dl.ipynb
```

Expected structure on the Azure ML compute instance:

```text
/ (Root)
 ├── folder-data/
 │    └── midterm-regresi-dataset.csv
 └── regression_dl.ipynb
```

---

## ▶️ Running the Notebook

1. Open `regression_dl.ipynb` in JupyterLab.
2. Run the package installation cell:

```python
%pip install -q optuna mlflow azureml-mlflow azureml-core lime scikit-learn tensorflow
%pip install -q pandas numpy matplotlib seaborn missingno
```

3. Restart the kernel:

```text
Kernel > Restart Kernel... > Restart
```

4. Run all cells:

```text
Run > Run All Cells
```

> ⚠️ Restarting the kernel after package installation is important because newly installed libraries may not be available until the kernel reloads.

---

## 📊 Results & Metrics

The notebook evaluates each model using regression metrics:

| Metric | Description |
|--------|-------------|
| MSE | Average squared prediction error |
| RMSE | Root mean squared error; easier to interpret because it has the same unit as the target |
| MAE | Average absolute error |
| R² | Proportion of target variance explained by the model |
| MAPE | Relative percentage error, calculated safely to avoid division by zero |

### Model Leaderboard

| Rank | Model | Stage | MSE | RMSE | MAE | R² | MAPE |
|------|-------|-------|-----|------|-----|----|------|
| 1 | `MLP_Deep_Huber` | Baseline | 165.7053 | 12.8727 | 8.5664 | 0.6916 | 305.3967 |
| 2 | `MLP_Tuned_Optuna` | Tuned | 155.7697 | 12.4808 | 7.9878 | 0.6657 | 726.1500 |
| 3 | `MLP_BatchNorm` | Baseline | 181.1037 | 13.4575 | 8.9601 | 0.6629 | 281.9607 |
| 4 | `MLP_Small` | Baseline | 182.2845 | 13.5013 | 9.0823 | 0.6607 | 294.8583 |

### Best Model

The best model based on the highest R² score is:

```text
MLP_Deep_Huber
```

Final performance:

```text
R²   : 0.6916
RMSE : 12.8727
MAE  : 8.5664
MAPE : 305.3967%
```

### Optuna Best Trial

The Optuna tuning process used **15 trials** and found the following best configuration:

```json
{
  "n_layers": 3,
  "units_l1": 128,
  "units_l2": 256,
  "units_l3": 64,
  "dropout_rate": 0.09759585735163344,
  "learning_rate": 0.0025719654260941984,
  "activation": "elu",
  "l2_reg": 0.0001510933026167445,
  "batch_size": 256,
  "loss_name": "huber"
}
```

> Although the tuned model achieved the lowest RMSE in the recorded leaderboard, the final best model was selected using the highest R² score.

---

## 🔎 Model Interpretation with LIME

LIME is used to explain how individual features influence model predictions.

The notebook includes:

- Local explanations for selected test samples:
  - Typical prediction near the median
  - Highest predicted value
  - Lowest predicted value
- Aggregate LIME feature importance across multiple test samples
- Visualization of positive and negative feature contributions

Top influential features based on aggregate LIME importance:

```text
Feature_87
Feature_1
Feature_89
Feature_88
Feature_63_x_Feature_88
Feature_13
Feature_63
Feature_80
Feature_54
Feature_75
```

---

## 📈 MLflow Experiment Tracking

The notebook uses MLflow to track experiments in Azure ML.

**Experiment name:**

```text
Midterm_Regression_DL_Pipeline
```

Each run logs important information such as:

- Model name
- Hidden layer configuration
- Dropout rate
- Learning rate
- Activation function
- L2 regularization
- Batch size
- Loss function
- Training time
- Number of trained epochs
- Evaluation metrics

To view the experiment in Azure ML Studio:

1. Open **Azure ML Studio**.
2. Go to **Jobs** or **Experiments**.
3. Open the experiment named:

```text
Midterm_Regression_DL_Pipeline
```

4. Select a run to inspect:
   - Parameters
   - Metrics
   - Training logs
   - Model comparison results
   - Optuna tuning output
   - Visualization outputs

---

## 🧩 Troubleshooting Notes

### 1. Notebook runs too long

To reduce runtime:

- Keep `N_TRIALS = 15` for basic tuning.
- Use a GPU compute instance if available.
- Keep larger batch sizes such as `256` or `512`.
- Avoid increasing epochs too much because EarlyStopping already controls convergence.

### 2. MLflow model artifact logging warning

If Azure ML shows a warning similar to:

```text
log_model skipped: API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404
```

The MLflow run, parameters, and metrics can still be tracked. If model artifact logging is required, adjust the logging method based on the Azure ML / MLflow version being used.

### 3. GitHub cannot render the notebook

If GitHub cannot render the full `.ipynb`, use NBViewer:

```text
https://nbviewer.org/
```

Then paste the GitHub URL of the notebook.

---

## 💡 Cost Management

> ⚠️ **Always stop your Azure ML compute instance when it is not being used.**

Steps:

1. Go to **Azure ML Studio**.
2. Open **Compute**.
3. Select your running compute instance.
4. Click **Stop**.
5. Confirm that the status changes to **Stopped**.

Stopping the compute instance helps prevent unnecessary Azure compute charges.

---

## 📄 License

Submitted as academic coursework.
