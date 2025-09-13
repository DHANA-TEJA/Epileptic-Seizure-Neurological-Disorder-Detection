from src.preprocess_mat import load_mat_dataset
from src.dl_prediction import train_prediction


if __name__ == "__main__":
    X, y = load_mat_dataset("data/raw/mat_data")
    train_prediction(X, y, epochs=25)
