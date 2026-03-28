# 🔍 Advanced Credit Card Fraud Detection

**An end-to-end machine learning pipeline for real-time credit card fraud detection using XGBoost, deployed with a Streamlit web application.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results & Visualizations](#-results--visualizations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Overview

Credit card fraud is a critical challenge in the financial industry, resulting in billions of dollars in losses annually. This project implements a **high-performance fraud detection system** built on the IEEE-CIS Fraud Detection dataset, achieving a **ROC-AUC of 0.9342** and a **recall of 89.17%** on a highly imbalanced dataset (96.5% legitimate vs. 3.5% fraudulent transactions).

The system features:
- **Advanced feature engineering** — temporal, aggregation, identity-based, and behavioral features extracted from raw transactional data
- **XGBoost classifier** — optimized with early stopping, class-weight balancing, and hyperparameter tuning for imbalanced data
- **Production-ready inference** — a Streamlit web app that accepts CSV uploads and returns fraud predictions in real time

---

## 🏗 Project Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Raw IEEE-CIS Data                    │
│           (Transaction + Identity Tables)               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline               │
│  • Temporal features (hour, day, weekday)               │
│  • Transaction velocity (time_diff)                     │
│  • Card-level aggregations (count, mean, std)           │
│  • User identity (UID) creation + aggregations          │
│  • Browser/resolution parsing                           │
│  • Categorical encoding (one-hot + frequency)           │
│  • Missing value imputation                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  XGBoost Classifier                     │
│  • 5,000 estimators (early stopped at ~1,745)           │
│  • max_depth=40, learning_rate=0.008                    │
│  • scale_pos_weight for class imbalance                 │
│  • Optimized classification threshold: 0.0028           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web App                      │
│  • CSV upload → preprocessing → prediction              │
│  • Fraud probability scores + binary predictions        │
│  • Interactive data exploration                         │
└─────────────────────────────────────────────────────────┘
````

-----

## 📊 Dataset

This project uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle.

| Property | Value |
| :--- | :--- |
| **Total Transactions** | 590,540 |
| **Features** | 434 (after merging transaction + identity tables) |
| **Fraudulent (isFraud=1)** | \~3.5% (20,663 cases) |
| **Legitimate (isFraud=0)** | \~96.5% (569,877 cases) |
| **Training Subset** | 25% stratified sample (\~147,635 rows) |

The dataset includes transaction-level features (amount, product code, card details, address, email domain), Vesta-engineered features (V1–V339), and identity-linked features (device type, browser, screen resolution, OS).

-----

## ⚙️ Feature Engineering

The pipeline (`pipeline.py`) applies the following transformations:

| Stage | Description |
| :--- | :--- |
| **Temporal** | Extract `hour`, `day`, `weekday` from `TransactionDT` |
| **Velocity** | Compute `time_diff` — seconds between consecutive transactions per card |
| **Card Aggregations** | Per-card `count`, `mean`, `std` of `TransactionAmt` for `card1` and `card2` |
| **User ID (UID)** | Construct `card1 + addr1 + P_emaildomain` composite key → aggregate stats |
| **Device/Browser**| Parse `id_33` (screen resolution) into width/height; normalize `id_31` to major browser families |
| **Missing Values**| Numeric → `-999`; Categorical → `"missing"` |
| **Encoding** | Low-cardinality (≤10 unique) → one-hot encoding; High-cardinality → frequency encoding |

-----

## 📈 Model Performance

The XGBoost model was trained with **class-weight balancing** (`scale_pos_weight`) and **early stopping** (50 rounds patience), evaluated on a time-based 80/20 split:

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.9342** |
| **Recall** | **0.8917** (89.17%) |
| **Classification Threshold** | 0.0028 (optimized for high recall) |

### Key Model Hyperparameters

```python
XGBClassifier(
    n_estimators=5000,       # early stopped at ~1,745
    learning_rate=0.008,
    max_depth=40,
    subsample=0.8,
    colsample_bytree=0.7,
    scale_pos_weight=27.58,  # class imbalance ratio
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42
)
```

> **Note:** The model uses a very low threshold (0.0028) instead of the default 0.5, prioritizing **recall** (catching fraud) over precision — a common strategy in fraud detection where missing a fraudulent transaction is far more costly than reviewing a false positive.

-----

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **ML Framework** | XGBoost 3.2 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Evaluation** | scikit-learn (ROC-AUC, F1, Confusion Matrix) |
| **Web App** | Streamlit |
| **Notebook Environment** | Google Colab (TPU-accelerated) |

-----

## 📁 Project Structure

```text
CreditCardFraudDetection/
├── README.md
└── HamidNomanLeghari_Capstone_Project/
    ├── AdvanceCreditCardFraudDetection_notebook_HLeghari.ipynb   # Full EDA + training notebook
    ├── AdvancedCredit_Card_Fraud_Detection Presentation.pptx     # Project presentation
    └── CreditCardFraudDetection_App/
        ├── app.py              # Streamlit web application
        ├── pipeline.py         # Feature engineering pipeline
        └── xgb_model.json      # Trained XGBoost model (~555 MB)
```

-----

## 🚀 Getting Started

### Prerequisites

  - Python 3.10 or higher
  - pip package manager

### Installation

1.  **Clone the repository:**

<!-- end list -->

```bash
git clone [https://github.com/HAMIDNOMANLEGHARI/CreditCardFraudDetection.git](https://github.com/HAMIDNOMANLEGHARI/CreditCardFraudDetection.git)
cd CreditCardFraudDetection
```

2.  **Install dependencies:**

<!-- end list -->

```bash
pip install streamlit pandas numpy xgboost scikit-learn matplotlib
```

3.  **Navigate to the app directory:**

<!-- end list -->

```bash
cd HamidNomanLeghari_Capstone_Project/CreditCardFraudDetection_App
```

4.  **Run the Streamlit app:**

<!-- end list -->

```bash
streamlit run app.py
```

5.  **Open your browser** at `http://localhost:8501`

> **Note:** The trained model file (`xgb_model.json`) is \~555 MB. If using Git LFS, make sure to pull LFS objects after cloning. Alternatively, retrain the model using the provided notebook.

-----

## 💡 Usage

1.  **Prepare your data** — ensure your CSV follows the IEEE-CIS transaction format with columns like `TransactionDT`, `TransactionAmt`, `card1`, `card2`, `ProductCD`, etc.
2.  **Upload via the web app** — drag and drop your CSV file into the Streamlit interface.
3.  **View results** — the app runs the full feature engineering pipeline and returns:
      - **Fraud\_Prediction** — binary label (0 = legitimate, 1 = fraud)
      - **Fraud\_Probability** — model confidence score (0.0 – 1.0)
4.  **Review flagged transactions** — the app highlights the total count of detected fraudulent transactions for quick triage.

-----

## 📊 Results & Visualizations

The model demonstrates strong discriminative ability on the validation set:

  - **Confusion Matrix** — shows the trade-off between catching fraud (high recall) and false positives.
  - **Normalized Confusion Matrix** — percentage-based view for class-imbalanced evaluation.
  - **AUC Training Curve** — validation AUC steadily improves from 0.84 → 0.93 over \~1,745 boosting rounds.

> Full visualizations and EDA are available in the [Jupyter notebook](https://www.google.com/search?q=HamidNomanLeghari_Capstone_Project/AdvanceCreditCardFraudDetection_notebook_HLeghari.ipynb).

-----

## 🔮 Future Improvements

  - [ ] **Threshold optimization** — systematic F1/F-beta threshold tuning via precision-recall curves.
  - [ ] **Feature selection** — SHAP-based feature importance analysis to reduce the 535-dimensional feature space.
  - [ ] **Ensemble methods** — combine XGBoost with LightGBM and CatBoost for a stacking ensemble.
  - [ ] **Real-time streaming** — integrate with Apache Kafka for live transaction scoring.
  - [ ] **Model monitoring** — add drift detection for production deployment.
  - [ ] **API deployment** — wrap the model in a FastAPI/Flask REST endpoint for microservice integration.

-----

## 👤 Author

**Hamid Noman Leghari**

> Capstone Project — Advanced Credit Card Fraud Detection

-----

\<div align="center"\>

⭐ **If you found this project useful, please consider giving it a star\!** ⭐

\</div\>

```

