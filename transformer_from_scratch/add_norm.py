import torch
from torch import nn


class AddNorm(nn.Module):
    """
    Add & Norm layer for the Transformer model.
    This layer adds a residual connection and applies layer normalization.
    """
    
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): The dimension of the model.
        """
        super(AddNorm, self).__init__()
        
        # Initialize the layer normalization layer.
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor.
            sublayer (torch.Tensor): The output of the sublayer to be added.
        
        Returns:
            torch.Tensor: The output tensor after adding the sublayer and applying layer normalization.
        """
        return self.norm(x + sublayer)
