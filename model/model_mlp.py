from typing import List

import pandas as pd
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = torch.nn.Linear(hidden_sizes[1], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.hidden2(relu1)
        relu2 = self.relu(hidden2)
        output = self.fc3(relu2)
        output = self.sigmoid(output)
        return output


class MLPModel():
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        self.model = MLP(input_size, hidden_sizes)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = self.criterion(y_pred.squeeze(), y_tensor)
            print(f"Epoch {epoch}, train loss: {loss.item()}")
            loss.backward()
            self.optimizer.step()

    def predict(self, X_tensor: torch.Tensor):
        self.model.eval()
        y_pred = self.model(X_tensor)
        return y_pred
