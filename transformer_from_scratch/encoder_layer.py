import torch
from torch import nn
from transformer_from_scratch import MultiHeadAttention, AddNorm, FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Represents one layer of the Transformer encoder.
    It contains Multi-Head Self-Attention, followed by Add&Norm,
    then a Position-wise Feed-Forward network, followed by Add&Norm.
    Dropout is applied after each sub-layer before the residual connection.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        heads (int): The number of attention heads.
        dropout (float): The dropout probability to apply after attention and FFN.
        forward_expansion (int): The expansion factor for the hidden layer in FFN.
    """
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float = 0.1,
        forward_expansion: int = 4,
    ):
        super(EncoderLayer, self).__init__()
        
        # Initialize the Multi-Head Attention layer
        self.attention = MultiHeadAttention(d_model, heads)
        
        # Initialize the Add&Norm layers
        self.norm1 = AddNorm(d_model)
        self.norm2 = AddNorm(d_model)
        
        # Self fully connected feed forward network
        self.ffn = FeedForwardNetwork(d_model, dropout, forward_expansion)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply Multi-Head Attention
        attention_output = self.attention(query=x, key=x, value=x, mask=mask)
        
        # Apply Add&Norm
        # We dropout after the attention layer
        x = self.norm1(x, self.dropout(attention_output))
        
        # Apply Feed-Forward Network
        ffn_output = self.ffn(x)
        
        # Apply Add&Norm
        # We dropout after the FFN layer
        output = self.norm2(x, self.dropout(ffn_output))
        
        return output