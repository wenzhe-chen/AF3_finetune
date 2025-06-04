import torch
import torch.nn as nn

class ConfidenceClassifier(nn.Module):
    def __init__(self, ligand_length, hidden_units=1024, output_units=1, number_of_chains=2, use_intersted_atom_mask=False ):
        super(ConfidenceClassifier, self).__init__()
        self.number_of_chains = number_of_chains
        if use_intersted_atom_mask:
            self.input_dim = (ligand_length* 506 + ligand_length* (506-ligand_length))*3 + 4 + self.number_of_chains * 3 + 1
        else:
            self.input_dim = (ligand_length* 506 + ligand_length* (506-ligand_length))*3 + 4 + self.number_of_chains * 3
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.fc1 = nn.Linear(self.input_dim, self.hidden_units)
        self.ln1 = nn.LayerNorm(self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.ln2 = nn.LayerNorm(self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, self.output_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ln1(self.fc1(x))
        x = self.relu(x)
        x = self.ln2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        return x


