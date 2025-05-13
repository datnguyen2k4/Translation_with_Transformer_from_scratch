from torch import nn
import torch  # Added for type hinting in forward

from transformer_from_scratch import EncoderLayer, PositionalEncoding


class Encoder(nn.Module):
    """
    Encoder class for Transformer in "Attention is All You Need".

    The Encoder consists of multiple layers of EncoderLayer, each of which
    contains a MultiHeadAttention layer and a FeedForward layer, both followed
    by AddNorm (add & normalization) layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the input embeddings.
        num_layers (int): The number of EncoderLayer layers.
        heads (int): The number of attention heads.
        dropout (float): The dropout rate applied after embeddings and within each EncoderLayer.
        forward_expansion (int): The expansion factor for the FFN hidden layer in EncoderLayer.
        max_len (int): The maximum length of the input sequence for PositionalEncoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
        max_len: int,
    ):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Initialize the Encoder layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        
        # Dropout layer applied after embedding + positional encoding
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the Encoder.

        Args:
            x (torch.Tensor): The input tensor (token indices) of shape (batch_size, seq_len).
            mask (torch.Tensor): The mask to apply during self-attention in EncoderLayers.

        Returns:
            torch.Tensor: The output tensor of the Encoder, shape (batch_size, seq_len, d_model).
        """
        # Embedding the input tokens
        embedding_output = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Adding positional encoding
        positional_output = self.positional_encoding(embedding_output)
        
        # Applying dropout
        out = self.dropout(positional_output)
        
        # Pass the output through each EncoderLayer
        for layer in self.layers:
            out = layer(out, mask)
            
        return out
