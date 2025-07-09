import torch

class DeepSetTransformerPooling(torch.nn.Module):
    def __init__(self, n_in: int, n_hidden_channels: int = 64, num_heads=8):
        super(DeepSetTransformerPooling, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden_channels)

        # For pooling
        self.query = torch.nn.Parameter(torch.randn(1, 1, n_hidden_channels))
        self.pooling = torch.nn.MultiheadAttention(n_hidden_channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pools a set of input vectors into a single feature vector.
        Args:
            x: Tensor of shape [batch, set_size, n_in]
        Returns:
            Tensor of shape [batch, n_hidden_channels] (or [n_hidden_channels] if batch size is 1)
        """
        x = torch.relu(self.fc1(x))
        q = self.query.expand(x.shape[0], -1, -1)
        x = self.pooling(q, x, x)[0]
        return x.squeeze(1)