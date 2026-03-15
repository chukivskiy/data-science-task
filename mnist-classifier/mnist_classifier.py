
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier


class SimpleFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x




class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class CNNClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = CNN()
        print("CNN model created")

    def train(self, X, y):
        print("Training CNN...")
        X_train_t = torch.from_numpy(X_train).reshape(-1, 1, 28, 28).float()
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test).reshape(-1, 1, 28, 28).float()
        y_test_t = torch.from_numpy(y_test)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 5

        for epoch in range(epochs):
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy:", 100 * correct / total)

    def predict(self, X):
        """
        Predict class labels for input data X (numpy array or tensor)
        Expects X in one of the following shapes:
        - (N, 784)          → automatically reshaped to [N, 1, 28, 28]
        - (N, 1, 28, 28)    → used as is
        - (N, 28, 28)       → channel added → [N, 1, 28, 28]

        Returns: numpy array of predicted class indices (shape: (N,))
        """
        # 1. Convert input to tensor if it is not already
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X).float()
        else:
            X_t = torch.as_tensor(X, dtype=torch.float32)

        # 2. Normalize shape → always [N, 1, 28, 28]
        if X_t.dim() == 2:  # (N, 784)
            X_t = X_t.view(-1, 1, 28, 28)
        elif X_t.dim() == 3:  # (N, 28, 28)
            X_t = X_t.unsqueeze(1)  # add channel → (N, 1, 28, 28)
        elif X_t.dim() == 4:
            if X_t.shape[1] not in (1, 3):  # check number of channels
                raise ValueError(f"Expected 1 or 3 channels, got {X_t.shape[1]}")
        else:
            raise ValueError(f"Unsupported input dimensionality: {X_t.shape}")

        # 3. Determine the device of the model
        device = next(self.model.parameters()).device

        # 4. Switch model to inference mode
        self.model.eval()

        # 5. Prediction (with automatic batching for large number of samples)
        predictions = []

        with torch.inference_mode():
            X_t = X_t.to(device)

            # Use DataLoader for large datasets
            batch_size = 256
            dataset = torch.utils.data.TensorDataset(X_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for batch in loader:
                images = batch[0]  # because TensorDataset returns a tuple
                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                predictions.append(preds.cpu())

        # 6. Combine all predictions
        final_preds = torch.cat(predictions)

        return final_preds.numpy()




class MnistRandomForestClassifier(MnistClassifierInterface):

    def __init__(self):
        print("RF model created")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        print("Training Random Forest")
        # Create Random Forest model
        # 100 trees, for better accuracy
        # Train the model
        self.rf_model.fit(X_train, y_train)

    def predict(self, X):
        print("Predict with RF")
        return self.rf_model.predict(X)


class NeuralNetClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = SimpleFFNN()
        print("NN model created")

    def train(self, X, y):
        # 2. Convert to PyTorch tensors
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test)
        y_test_t = torch.from_numpy(y_test)

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        print("Training Neural Network")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Device: {device}")

        # 4. Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 5. Training loop
        epochs = 12

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1:2d} | loss: {running_loss / len(train_loader):.4f}")

    def predict(self, X):
        """
        Returns predicted class labels (numpy array) for input data X
        """
        # 1. Convert to tensor if not already a tensor
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X).float()
        else:
            X_t = torch.tensor(X, dtype=torch.float32)

        # 2. Determine model device
        device = next(self.model.parameters()).device

        # 3. Switch model to inference mode
        self.model.eval()

        # 4. Disable gradient computation — saves memory and speeds up inference
        with torch.inference_mode():  # recommended way since PyTorch 1.9+
            X_t = X_t.to(device)

            # If needed — split into batches (especially useful for large datasets)
            if X_t.shape[0] > 10000:  # arbitrary threshold
                dataset = TensorDataset(X_t)
                loader = DataLoader(dataset, batch_size=512, shuffle=False)
                predictions = []
                for batch in loader:
                    outputs = self.model(batch[0])
                    pred = torch.argmax(outputs, dim=1)
                    predictions.append(pred.cpu())
                preds = torch.cat(predictions)
            else:
                # For small number of samples — process all at once
                outputs = self.model(X_t)
                preds = torch.argmax(outputs, dim=1)

        # 5. Return as numpy array (convenient for interface)
        return preds.cpu().numpy()


class MnistClassifier:

    def __init__(self, algorithm):

        if algorithm == "cnn":
            self.model = CNNClassifier()

        elif algorithm == "rf":
            self.model = MnistRandomForestClassifier()   # ← was wrong: RandomForestClassifier()

        elif algorithm == "nn":
            self.model = NeuralNetClassifier()

        else:
            raise ValueError("Unknown algorithm")

    def train(self, X, y):
        self.model.train(X, y)

    def predict(self, X):
        return self.model.predict(X)


# 1. Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0     # normalize to 0..1
y = mnist.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# cnn_obj = MnistClassifier("cnn")
# cnn_obj.train(X_train, y_train)
# print(cnn_obj.predict(X_test[2].reshape(1, -1)))
# print(y_test[2])

# rf_obj = MnistClassifier("rf")
#
# rf_obj.train(X_train, y_train)
# print(rf_obj.predict(X_test[2].reshape(1, -1)))
# print(y_test[2])

nn_obj = MnistClassifier("nn")
nn_obj.train(X_train, y_train)
print(nn_obj.predict(X_test[4].reshape(1, -1)))
print(y_test[4])