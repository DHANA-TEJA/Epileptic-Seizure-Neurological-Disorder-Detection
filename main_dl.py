from src.preprocess_mat import load_mat_dataset
from src.dl_models import train_cnn_lstm

def main():
    X, y = load_mat_dataset(base_path="data/raw/mat_data")
    print("✅ Loaded dataset:", X.shape, y.shape)

    model = train_cnn_lstm(X, y, epochs=25, batch_size=32)

if __name__ == "__main__":
    main()
