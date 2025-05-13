import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    Based on the paper "Attention Is All You Need".

    The Transformer model consists of an encoder and a decoder, each composed of
    multiple layers of self-attention and feed-forward neural networks.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
    ):
        """
        Initialize the Transformer model.

        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder and decoder layers
            d_ff: Dimensionality (expansion factor * d_model) of the feed-forward layer's hidden layer.
            max_seq_length: Maximum sequence length
            dropout: Dropout rate (default: 0.1)
        """
        super(Transformer, self).__init__()
        
        # Ensure consistency between d_ff and the forward_expansion used internally
        # forward_expansion = d_ff // d_model
        # We'll pass d_ff directly assuming Encoder/Decoder expect forward_expansion factor
        # Need to ensure d_ff is passed correctly mapped to forward_expansion parameter.
        assert (
            d_ff % d_model == 0
        ), "d_ff must be divisible by d_model for standard forward_expansion calculation"
        forward_expansion = d_ff // d_model
        
        # Create the encoder and decoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            max_len=max_seq_length,
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            max_len=max_seq_length,
        )
        
        # Final linear layer to project to target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_masks(self, src, tgt):
        """
        Generate masks for the source and target sequences.

        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]

        Returns:
            src_mask: Source mask [batch_size, 1, 1, src_seq_len]
            tgt_mask: Target mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
        """
        # Source mask to avoid paying attention to padding tokens
        # We assume 0 is the padding token
        src_mask = (
            (src != 0).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, src_seq_len]
                
        # Target mask to avoid paying attention to padding tokens and future tokens
        tgt_mask = (
            (tgt != 0).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, tgt_seq_len]

        # Create a look-ahead mask to prevent attention to future tokens
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1
        ).bool()
        look_ahead_mask = look_ahead_mask.to(src.device)
        # Combine the padding mask and the look-ahead mask
        combined_mask = tgt_mask & ~look_ahead_mask.unsqueeze(0).unsqueeze(0)

        return src_mask, combined_mask
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Generate masks
        src_mask, tgt_mask = self.generate_masks(src, tgt)

        # Pass through encoder and decoder
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        
        # Project to the target vocabulary size
        out = self.fc_out(dec_out)

        return out
        