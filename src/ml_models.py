import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def save_model(model, filename="results/saved_models/best_binary_model.pkl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {filename}")


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Seizure", "Seizure"],
                yticklabels=["Non-Seizure", "Seizure"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/confusion_matrix_{model_name}.png")
    plt.close()
    print(f"📊 Saved confusion matrix for {model_name}")


def plot_pca_scatter(X, y, model_name="Model"):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    plt.title(f"PCA Scatter Plot - {model_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Class")
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/pca_scatter_{model_name}.png")
    plt.close()
    print(f"📊 Saved PCA scatter plot for {model_name}")


def train_binary_models(X, y):
    # Merge classes → binary
    y_binary = np.where(y == 0, 1, 0)   # ictal=1, others=0
    print("Labels after merge:", np.unique(y_binary, return_counts=True))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Scale + PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("After PCA:", X_train_pca.shape)

    results = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(X_train_pca, y_train)
    y_pred = logreg.predict(X_test_pca)
    results["LogisticRegression"] = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, "LogisticRegression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_pca, y_train)
    y_pred = rf.predict(X_test_pca)
    results["RandomForest"] = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, "RandomForest")

    # SVM with tuning
    param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1], "kernel": ["rbf"]}
    grid = GridSearchCV(SVC(probability=True, random_state=42),
                        param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test_pca)
    results["SVM"] = accuracy_score(y_test, y_pred)
    print("Best SVM Params:", grid.best_params_)
    plot_confusion_matrix(y_test, y_pred, "SVM")

    # Stacking Ensemble
    estimators = [
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, C=10, gamma=0.1, random_state=42))
    ]
    stack = StackingClassifier(estimators=estimators,
                               final_estimator=LogisticRegression(),
                               n_jobs=-1)
    stack.fit(X_train_pca, y_train)
    y_pred = stack.predict(X_test_pca)
    results["StackingEnsemble"] = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, "StackingEnsemble")

    # Save PCA scatter only once (train set)
    plot_pca_scatter(X_train_pca, y_train, "TrainData")

    # Pick best
    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]
    print(f"\n✅ Best Model: {best_model_name} with accuracy {best_acc:.4f}")

    # Save best
    if best_model_name == "LogisticRegression":
        save_model(logreg)
    elif best_model_name == "RandomForest":
        save_model(rf)
    elif best_model_name == "SVM":
        save_model(best_svm)
    else:
        save_model(stack)

    # Build report
    df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
    return df
