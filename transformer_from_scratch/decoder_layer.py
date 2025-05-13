import torch  # Added for type hinting
from torch import nn
from transformer_from_scratch import MultiHeadAttention, AddNorm, FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    Represents one layer of the Transformer decoder.
    It contains Masked Multi-Head Self-Attention, followed by Add&Norm,
    then Multi-Head Encoder-Decoder Attention, followed by Add&Norm,
    and finally a Position-wise Feed-Forward network, followed by Add&Norm.
    Dropout is applied after each sub-layer before the residual connection.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        heads (int): The number of attention heads.
        dropout (float): The dropout probability.
        forward_expansion (int): The expansion factor for the hidden layer in FFN.
    """

    def __init__(
        self, d_model: int, heads: int, dropout: float = 0.1, forward_expansion: int = 4
    ):
        """
        Initializes the DecoderLayer.
        """
        super(DecoderLayer, self).__init__()

        # Masked Self-Attention (target sequence attends to itself)
        self.self_attention = MultiHeadAttention(d_model, heads)
        self.norm1 = AddNorm(d_model)

        # Encoder-Decoder Attention (target sequence attends to encoder output)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, heads)
        self.norm2 = AddNorm(d_model)

        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, dropout, forward_expansion)
        self.norm3 = AddNorm(d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout layer applied after sub-layers

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ):
        """
        Forward pass for the DecoderLayer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer or target embedding,
                              shape (batch_size, trg_len, d_model).
            enc_out (torch.Tensor): Output tensor from the encoder stack,
                                    shape (batch_size, src_len, d_model).
            src_mask (torch.Tensor): Mask for the encoder output (for enc-dec attention),
                                     broadcastable to (batch_size, 1, 1, src_len).
            trg_mask (torch.Tensor): Mask for the target sequence (for self-attention),
                                     broadcastable to (batch_size, 1, trg_len, trg_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, trg_len, d_model).
        """
        # 1. Masked Multi-Head Self-Attention
        # Query, Key, Value are all derived from x (decoder input)
        # Apply target mask (trg_mask)
        self_attn_output = self.self_attention(query=x, key=x, value=x, mask=trg_mask)
        
        # Apply dropout, residual connection and layer norm
        # Assumption: AddNorm applies LayerNorm(input + sublayer_output)
        x_norm1 = self.norm1(x, self.dropout(self_attn_output))

        # 2. Multi-Head Encoder-Decoder Attention
        # Query is from the previous sub-layer (x_norm1)
        # Key and Value are from the encoder output (enc_out)
        # Apply source mask (src_mask)
        enc_dec_attn_output = self.encoder_decoder_attention(
            query=x_norm1, key=enc_out, value=enc_out, mask=src_mask
        )

        # Apply dropout, residual connection and layer norm
        x_norm2 = self.norm2(
            x_norm1, self.dropout(enc_dec_attn_output)
        )  # Residual connection uses x_norm1

        # 3. Position-wise Feed-Forward Network
        ffn_output = self.ffn(x_norm2)

        # Apply dropout, residual connection and layer norm
        output = self.norm3(
            x_norm2, self.dropout(ffn_output)
        )  # Residual connection uses x_norm2

        return output
