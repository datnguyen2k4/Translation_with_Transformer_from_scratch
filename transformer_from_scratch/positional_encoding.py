import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implement positional encoding as described in the paper "Attention is All You Need".
    Positional encoding is used in the Transformer model to give the model some information about
    the relative position of the words in the sentence.
    
    Args:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the input sequences. Default is 5000.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: The input tensor with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1)]
        return x
