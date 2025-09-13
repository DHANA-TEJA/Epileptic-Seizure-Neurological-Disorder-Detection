import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        # LSTM expects (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # binary output

    def forward(self, x):
        # x: (batch, 1, 256)
        x = self.pool1(torch.relu(self.conv1(x)))  # -> (batch, 32, 128)
        x = self.pool2(torch.relu(self.conv2(x)))  # -> (batch, 64, 64)

        # reshape for LSTM: (batch, seq_len=64, features=64)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (hn, _) = self.lstm(x)
        x = hn[-1]  # last hidden state

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_cnn_lstm(X, y, epochs=20, batch_size=32, lr=0.001, test_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert labels to binary (ictal=1, others=0)
    y_binary = np.where(y == 0, 1, 0)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)  # (N, 1, 256)
    y_tensor = torch.tensor(y_binary, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Train/test split
    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss
    os.makedirs("results/plots", exist_ok=True)
    plt.plot(loss_history, marker="o")
    plt.title("CNN+LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/plots/cnn_lstm_loss.png")
    plt.close()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Non-Seizure", "Seizure"])
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Seizure", "Seizure"],
                yticklabels=["Non-Seizure", "Seizure"])
    plt.title("CNN+LSTM Confusion Matrix")
    plt.savefig("results/plots/cnn_lstm_confusion_matrix.png")
    plt.close()

    # Save model
    os.makedirs("results/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "results/saved_models/cnn_lstm_model.pth")
    print("✅ CNN+LSTM model saved to results/saved_models/cnn_lstm_model.pth")

    return model
