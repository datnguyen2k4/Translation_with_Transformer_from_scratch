import torch
from torch import nn

from transformer_from_scratch import PositionalEncoding, DecoderLayer


class Decoder(nn.Module):
    """
    Decoder class for Transformer in "Attention is All You Need".

    Consists of an embedding layer, positional encoding, multiple DecoderLayers,
    and a final linear layer to project to vocabulary size.

    Args:
        vocab_size (int): The size of the target vocabulary.
        d_model (int): The dimensionality of the model embeddings.
        num_layers (int): The number of DecoderLayer layers.
        heads (int): The number of attention heads in each DecoderLayer.
        dropout (float): The dropout rate applied after embeddings and within each DecoderLayer.
        forward_expansion (int): Expansion factor for the FFN hidden layer in DecoderLayer.
        max_len (int): Maximum sequence length for PositionalEncoding.
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
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            trg (torch.Tensor): Target sequence tensor (token indices),
                                shape (batch_size, trg_len).
            enc_out (torch.Tensor): Output tensor from the encoder stack,
                                    shape (batch_size, src_len, d_model).
            src_mask (torch.Tensor): Mask for the encoder output (for enc-dec attention),
                                     broadcastable to (batch_size, 1, 1, src_len).
            trg_mask (torch.Tensor): Mask for the target sequence (for self-attention),
                                     broadcastable to (batch_size, 1, trg_len, trg_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, trg_len, vocab_size).
                         Represents raw logits for each token in the target vocabulary.
        """
        batch_size, trg_len = trg.shape

        embedding_output = self.embedding(trg)
        out = self.positional_encoding(embedding_output)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(x=out, enc_out=enc_out, src_mask=src_mask, trg_mask=trg_mask)

        return out
