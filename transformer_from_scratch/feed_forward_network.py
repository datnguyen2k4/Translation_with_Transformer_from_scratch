import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network for the Transformer model.
    This network consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, forward_expansion: int = 4):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout probability. Defaults to 0.1.
                     Note: This dropout is applied *within* the FFN block, after ReLU.
            forward_expansion (int): Factor by which to expand the hidden layer dimension.
                                   Defaults to 4.
        """
        super(FeedForwardNetwork, self).__init__()
        
        hidden_dim = d_model * forward_expansion
        
        # Define the first linear layer
        self.fc_1 = nn.Linear(d_model, hidden_dim)
        
        # Define the relu activation function
        self.relu = nn.ReLU()
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Define the second linear layer
        self.fc_2 = nn.Linear(hidden_dim, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply the first linear layer and ReLU activation
        x = self.fc_1(x)
        x = self.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply the second linear layer
        x = self.fc_2(x)
        
        return x
