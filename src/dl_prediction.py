import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CNN_LSTM_Prediction(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Prediction, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 classes

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  
        x = self.pool2(torch.relu(self.conv2(x)))  
        x = x.permute(0, 2, 1)  
        _, (hn, _) = self.lstm(x)
        x = torch.relu(self.fc1(hn[-1]))
        return self.fc2(x)

def train_prediction(X, y, epochs=25, batch_size=32, lr=0.001, test_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = CNN_LSTM_Prediction().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = torch.argmax(model(data), dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nPrediction Report:\n", classification_report(y_true, y_pred,
          target_names=["Interictal", "Pre-ictal", "Ictal"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Interictal", "Pre-ictal", "Ictal"],
                yticklabels=["Interictal", "Pre-ictal", "Ictal"])
    plt.title("Prediction Confusion Matrix")
    plt.savefig("results/plots/prediction_cm.png")
    plt.close()

    os.makedirs("results/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "results/saved_models/cnn_lstm_prediction.pth")
    print("✅ Prediction model saved.")
    return model
