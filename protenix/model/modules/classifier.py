import torch
import torch.nn as nn

class ConfidenceClassifier(nn.Module):
    def __init__(self, hidden_units=64, output_units=1, number_of_chains=2, use_intersted_atom_mask=False):
        super(ConfidenceClassifier, self).__init__()
        self.number_of_chains = number_of_chains
        if use_intersted_atom_mask:
            self.input_dim = 6 + self.number_of_chains * 9 + 2
        else:
            self.input_dim = 6 + self.number_of_chains * 9
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.fc1 = nn.Linear(self.input_dim, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.output_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


