import torch
import torch.nn as nn

class ConfidenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units, output_units):
        super(ConfidenceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


