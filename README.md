# AI-ML
Vityarthi Project Winter Semester
#  Diabetes Prediction with K-Nearest Neighbours

A supervised machine learning project that predicts whether a patient is diabetic based on clinical diagnostic measurements, using the K-Nearest Neighbours (KNN) algorithm with cross-validated hyperparameter tuning.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project builds a binary classifier that answers a single clinical question:

> **Can we predict whether a patient has diabetes based on standard diagnostic measurements?**

Given eight numerical health indicators (glucose level, BMI, age, etc.), the model outputs a prediction: **diabetic (1)** or **non-diabetic (0)**.

Key features of this implementation:
- **Automatic K optimisation** — cross-validation selects the best number of neighbours (K) from 1 to 39, rather than using an arbitrary default
- **Proper data scaling** — StandardScaler prevents features with large ranges from dominating distance calculations
- **Leak-free pipeline** — the scaler is fit only on training data and applied to the test set, not the other way around
- **Visual diagnostics** — an Error Rate vs. K plot lets you inspect the bias-variance trade-off

---

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

| Property | Detail |
|---|---|
| File | `diabetes.csv` |
| Rows | 768 patient records |
| Features | 8 numerical inputs |
| Target | `Outcome` — 0 (non-diabetic) or 1 (diabetic) |
| Class split | ~65% negative, ~35% positive |

**Feature descriptions:**

| # | Feature | Description |
|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant |
| 2 | `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| 3 | `BloodPressure` | Diastolic blood pressure (mm Hg) |
| 4 | `SkinThickness` | Triceps skinfold thickness (mm) |
| 5 | `Insulin` | 2-hour serum insulin (μU/mL) |
| 6 | `BMI` | Body mass index (kg/m²) |
| 7 | `DiabetesPedigreeFunction` | Genetic likelihood score based on family history |
| 8 | `Age` | Age in years |

> **Note:** Several features contain `0` values that are biologically implausible (e.g. a BMI of 0). These represent missing measurements, not true zeros. The current implementation does not impute them — see [Limitations](#limitations--future-work).

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project root as `diabetes.csv`.

---

## Project Structure

```
diabetes-knn/
│
├── diabetes.csv          # Dataset (download separately — see above)
├── diabetes_knn.py       # Main script
├── README.md             # This file

```

**Python packages:**

```
pandas
scikit-learn
matplotlib
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/diabetes-knn.git
cd diabetes-knn
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project root folder.

---

## Usage

**Run the full pipeline:**

```bash
python diabetes_knn.py
```

This will:
1. Load and split the dataset (80% train / 20% test)
2. Scale all features using StandardScaler
3. Search K values from 1 to 39 using 5-fold cross-validation
4. Print the optimal K to the console
5. Train the final model and evaluate it on the test set
6. Print a confusion matrix and full classification report
7. Display an Error Rate vs. K plot

**Example console output:**

```
Optimal K found: 19

--- Model Performance ---
[[88 11]
 [22 33]]

              precision    recall  f1-score   support

           0       0.80      0.89      0.84        99
           1       0.75      0.60      0.67        55

    accuracy                           0.79       154
   macro avg       0.78      0.74      0.76       154
weighted avg       0.78      0.79      0.78       154
```

> Your exact numbers may vary slightly depending on your environment's random state handling.

---

## How It Works

```
Raw CSV
   │
   ├─ Features (X): columns 0–7
   └─ Target   (y): column 8 (Outcome)
          │
          ▼
   Train / Test Split  (80% / 20%, random_state=42)
          │
          ▼
   StandardScaler  ──► fit on X_train only, then transform both splits
          │
          ▼
   K Search (K = 1 … 39)
     └─ 5-fold cross-validation on X_train_scaled
     └─ record mean error rate per K
          │
          ▼
   Best K = argmin(error_rate)
          │
          ▼
   KNeighborsClassifier(n_neighbors=best_k)
     └─ fit on X_train_scaled
     └─ predict on X_test_scaled
          │
          ▼
   Evaluation
     ├─ Confusion Matrix
     ├─ Classification Report (precision / recall / F1)
     └─ Error Rate vs. K plot
```

**Why K-Nearest Neighbours?**

KNN is non-parametric — it makes no assumptions about the distribution of the data, which suits a medical dataset that may not follow a simple distribution. It is also interpretable: a prediction for any patient can be explained by pointing to the K most similar patients in the training set.

**Why cross-validation for K selection?**

With only ~614 training samples, a single validation split would produce an unstable error estimate. Cross-validation uses all training data across multiple folds, yielding a more reliable guide to generalisation performance.

---

## Results

The model typically achieves around **77–80% accuracy** on the test set, with stronger performance on the non-diabetic class (class 0) than the diabetic class (class 1) — reflecting the class imbalance in the data.

In a medical screening context, **recall on the positive class (diabetic)** is the more important metric: a missed diabetic patient (false negative) carries higher clinical cost than a false alarm (false positive).

---

## Limitations & Future Work

| Limitation | Suggested Fix |
|---|---|
| Zero values in Glucose, BMI, etc. are biologically impossible but are not imputed | Replace zeros in those columns with the column median before scaling |
| Moderate class imbalance (~65/35) may suppress recall on the diabetic class | Apply SMOTE oversampling or use `class_weight='balanced'` in the classifier |
| Only KNN is evaluated | Benchmark against Logistic Regression, Random Forest, and SVM |
| No feature selection | Investigate removing correlated features (e.g. SkinThickness + BMI) |
| No prediction interface | Wrap the trained model in a Flask API or Streamlit app |


