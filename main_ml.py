from src.preprocess_mat import load_mat_dataset
from src.features import extract_features
from src.ml_models import train_binary_models

def main():
    X, y = load_mat_dataset(base_path="data/raw/mat_data")
    print("✅ Loaded dataset:", X.shape, y.shape)

    X_features = extract_features(X)
    print("✅ Features ready:", X_features.shape)

    results = train_binary_models(X_features, y)
    print("\nResults:\n", results)

    results.to_csv("results/binary_ml_results.csv")

if __name__ == "__main__":
    main()
