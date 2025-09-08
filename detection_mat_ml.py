import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
X = np.load("Data/EEG Epilepsy Datasets/X.npy", allow_pickle=True)
y = np.load("Data/EEG Epilepsy Datasets/y.npy", allow_pickle=True)

print("Original labels:", np.unique(y))

# 2. Merging classes
y_binary = np.where(y == 1, 1, 0)  # Seizure - 1, others - 0

print("Labels after merge:", np.unique(y_binary, return_counts=True))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# 4. Scale + PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# keep 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("After PCA:", X_train_pca.shape)

# 5. Training models

# Logistic Regression
logreg = LogisticRegression(max_iter=500, random_state=42)
logreg.fit(X_train_pca, y_train)
y_pred_log = logreg.predict(X_test_pca)
print("\n🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)
print("\n🔹 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Support Vector Machine
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)
print("\n🔹 SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 6. Pick Best Model
accs = {
    "LogReg": accuracy_score(y_test, y_pred_log),
    "RandomForest": accuracy_score(y_test, y_pred_rf),
    "SVM": accuracy_score(y_test, y_pred_svm)
}
best_model = max(accs, key=accs.get)
print(" Best Model:", best_model, "with accuracy", accs[best_model])