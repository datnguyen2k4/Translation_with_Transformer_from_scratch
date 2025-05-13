import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module from the paper "Attention is All You Need".
    This module implements the scaled dot-product attention mechanism with multiple heads.
    Ref: https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model (int): The dimension of the model (must be divisible by n_heads).
            n_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Store the embed size and heads
        self.d_model = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads

        # Define the query, key, value, and output linear layers for all heads
        # These layers project the d_model input to d_model output.
        # The output is then split into heads. Bias is typically included.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculates the scaled dot-product attention scores.

        Args:
            Q (torch.Tensor): Query tensor, shape (batch_size, num_heads, query_len, head_dim).
            K (torch.Tensor): Key tensor, shape (batch_size, num_heads, key_len, head_dim).
            V (torch.Tensor): Value tensor, shape (batch_size, num_heads, value_len, head_dim).
                                Note: key_len == value_len
            mask (torch.Tensor, optional): Mask tensor, shape broadcastable to
                                           (batch_size, num_heads, query_len, key_len).
                                           Positions with 0 should be masked. Defaults to None.

        Returns:
            torch.Tensor: The context tensor after attention, shape (batch_size, num_heads, query_len, head_dim).
            torch.Tensor: The attention probabilities, shape (batch_size, num_heads, query_len, key_len).
        """
        # K shape: (batch_size, num_heads, key_len, head_dim)
        # K.transpose(-2, -1) shape: (batch_size, num_heads, head_dim, key_len)
        # Q shape: (batch_size, num_heads, query_len, head_dim)
        # attn_scores shape: (batch_size, num_heads, query_len, key_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Mask has 0 where we want to mask (e.g., padding), 1 otherwise.
            # We fill positions where mask == 0 with a large negative value.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # attn_probs shape: (batch_size, num_heads, query_len, key_len)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # attention_output shape: (batch_size, num_heads, query_len, head_dim)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension of the input tensor into (num_heads, head_dim).
        Transposes the result to shape (batch_size, num_heads, seq_len, head_dim).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with heads split, shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines the heads back into the original d_model dimension.
        Inverse operation of split_heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).

        Returns:
            torch.Tensor: Tensor with heads combined, shape (batch_size, seq_len, d_model).
        """
        batch_size, _, seq_length, head_dim = x.size()
        # Transpose back to (batch_size, seq_len, num_heads, head_dim)
        # Then reshape to (batch_size, seq_len, d_model)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, key_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, value_len, d_model).
                                  Note: key_len and value_len must be the same.
            mask (torch.Tensor, optional): Attention mask, broadcastable to
                                           (batch_size, 1, query_len, key_len) or
                                           (batch_size, 1, 1, key_len).
                                           Positions with 0 are masked. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, query_len, d_model).
        """
        # 1. Apply linear projections W_q, W_k, W_v
        # Input shape: (batch_size, seq_len, d_model)
        # Output shape: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split Q, K, V into multiple heads
        # Output shape: (batch_size, num_heads, seq_len, head_dim)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Calculate scaled dot-product attention
        # attn_output shape: (batch_size, num_heads, query_len, head_dim)
        # attn_probs shape: (batch_size, num_heads, query_len, key_len) -> Ignored here
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Combine heads
        # Output shape: (batch_size, query_len, d_model)
        output = self.combine_heads(attn_output)

        # 5. Apply final linear layer W_o
        # Output shape: (batch_size, query_len, d_model)
        output = self.W_o(output)

        return output
