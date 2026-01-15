import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# =========================
# KONFIGURÁCIA
# =========================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
OUT = "outputs_zadanie2"
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# =========================
# DATASET
# =========================
def load_data() -> pd.DataFrame:
    path = os.path.join(OUT, "winequality-red.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path, sep=";")
    df["quality_bin"] = (df["quality"] >= 6).astype(int)
    return df

# =========================
# PREPROCESSING
# =========================
def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["quality", "quality_bin"])
    y = df["quality_bin"]

    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y

# =========================
# MODELY
# =========================
def get_models() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=False)
    }

# =========================
# CROSS VALIDATION
# =========================
def cross_validate(X, y, model, use_pca=False, n_components=None):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_true, y_pred = [], []
    train_scores, val_scores = [], []

    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]

        steps = [("scaler", StandardScaler())]

        if use_pca:
            steps.append(("pca", PCA(n_components=n_components)))

        steps.append(("clf", model))
        pipe = Pipeline(steps)

        pipe.fit(Xtr, ytr)

        ytr_pred = pipe.predict(Xtr)
        yval_pred = pipe.predict(Xval)

        train_scores.append(f1_score(ytr, ytr_pred))
        val_scores.append(f1_score(yval, yval_pred))

        y_true.extend(yval)
        y_pred.extend(yval_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "train_f1_mean": np.mean(train_scores),
        "val_f1_mean": np.mean(val_scores),
        "confusion": confusion_matrix(y_true, y_pred)
    }

# =========================
# PCA – VÝBER KOMPONENTOV
# =========================
def find_pca_components(X):
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    k = np.argmax(cum_var >= 0.95) + 1

    plt.figure()
    plt.plot(cum_var)
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Počet komponentov")
    plt.ylabel("Kumulatívna variancia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pca_variance.png"), dpi=150)
    plt.close()

    return k

# =========================
# HLAVNÝ PROGRAM
# =========================
def main():
    df = load_data()
    X, y = preprocess(df)

    models = get_models()
    results = []

    print("=== MODELY BEZ PCA ===")
    for name, model in models.items():
        res = cross_validate(X, y, model)
        results.append([name, "bez PCA", res["accuracy"], res["f1"], res["mcc"]])
        print(name, res)

    k = find_pca_components(X)

    print(f"\n=== MODELY S PCA (k={k}) ===")
    for name, model in models.items():
        res = cross_validate(X, y, model, use_pca=True, n_components=k)
        results.append([name, f"PCA ({k})", res["accuracy"], res["f1"], res["mcc"]])
        print(name, res)

    # =========================
    # TABUĽKA VÝSLEDKOV
    # =========================
    cols = ["Model", "Variant", "Accuracy", "F1", "MCC"]
    table = pd.DataFrame(results, columns=cols)
    table.to_csv(os.path.join(OUT, "results_summary.csv"), index=False)

    print("\n=== TABUĽKA VÝSLEDKOV ===")
    print(table)

    # =========================
    # ANALÝZA CHÝB – NAJLEPŠÍ MODEL
    # =========================
    best = table.sort_values("F1", ascending=False).iloc[0]
    print("\nNajlepší model:")
    print(best)

if __name__ == "__main__":
    main()
