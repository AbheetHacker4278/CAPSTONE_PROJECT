"""
train_model.py
--------------
Trains a scikit-learn Pipeline (StandardScaler + SVM) on the Wisconsin Breast
Cancer dataset and saves it as model.pkl.

Run this script once (or whenever you want to retrain):
    python train_model.py

Expected output:
    Cross-validation accuracy: ~97-98%
    model.pkl saved.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading Wisconsin Breast Cancer dataset...")
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
)
column_names = [
    "id",
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
]

try:
    df = pd.read_csv(url, names=column_names)
    print(f"Dataset loaded from UCI. Shape: {df.shape}")
except Exception as e:
    print(f"Could not download dataset ({e}). Trying local copy...")
    # Fallback: look for a local CSV in the project directory
    import os
    local_path = os.path.join("Breast cancer prediction", "data.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Loaded local dataset. Shape: {df.shape}")
    else:
        raise RuntimeError(
            "Dataset not found. Please ensure an internet connection or place "
            "data.csv in the 'Breast cancer prediction' folder."
        )

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
# Drop the ID column (not a feature)
df.drop(columns=["id"], inplace=True, errors="ignore")

# Replace '?' with NaN and impute with median (more robust than ffill)
df = df.replace("?", np.nan)
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")
median_bare = df["bare_nuclei"].median()
df["bare_nuclei"] = df["bare_nuclei"].fillna(median_bare)
df["bare_nuclei"] = df["bare_nuclei"].astype(int)

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# Features and target
feature_cols = [
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]
X = df[feature_cols].values
y = df["class"].values  # 2 = Benign, 4 = Malignant

print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 3. Build Pipeline ─────────────────────────────────────────────────────────
# StandardScaler + SVM (RBF kernel) — well-known to achieve ~97-98% on this
# dataset when properly scaled.
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma=0.1,
                probability=True,
                random_state=42,
            ),
        ),
    ]
)

# ── 4. Cross-validation ───────────────────────────────────────────────────────
print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy per fold: {np.round(cv_scores * 100, 2)}")
print(f"Mean CV Accuracy:     {cv_scores.mean() * 100:.2f}%")
print(f"Std  CV Accuracy:     {cv_scores.std() * 100:.2f}%")

# ── 5. Train on full dataset & evaluate on held-out test set ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nHeld-out test accuracy: {test_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign (2)", "Malignant (4)"]))

# ── 6. Retrain on ALL data and save ──────────────────────────────────────────
print("Retraining on full dataset and saving model.pkl...")
pipeline.fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("model.pkl saved successfully.")
print("\nFeature order expected by the model:")
for i, col in enumerate(feature_cols):
    print(f"  {i+1}. {col}")
