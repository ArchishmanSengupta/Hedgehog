"""
Model architectures for diffusion language models.

Based on research from:
- DiT: Diffusion Transformer (Scalable Diffusion Models with Transformers)
- MDLM: Masked Diffusion Language Model
- AR: Autoregressive baseline
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for diffusion language models."""
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    dropout: float = 0.1
    emb_dropout: float = 0.0


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal position embeddings."""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(seq_len, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive layer norm."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # Adaptive layer norm modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with adaptive layer norm.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            timestep_emb: Pre-computed timestep embeddings [batch, hidden_size]
        """
        # Ensure x has correct shape [batch, seq, hidden]
        if x.dim() == 2:
            # If 2D, it shouldn't happen, but handle gracefully
            x = x.unsqueeze(-1)

        batch_size, seq_len, hidden_size = x.shape

        # Compute modulation parameters from timestep embedding
        if timestep_emb is not None:
            modulation = self.adaLN_modulation(timestep_emb)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                modulation.chunk(6, dim=-1)
            )
            # Expand to sequence dimension [batch, seq, hidden]
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        else:
            # Use zero tensors with correct shape [batch, seq, hidden]
            zero = torch.zeros(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)
            shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = zero

        # Self-attention with adaptive norm
        x_norm = self.norm1(x)
        if timestep_emb is not None:
            x_norm = x_norm * (1 + scale_msa) + shift_msa

        # MultiheadAttention returns (output, weights) or just output depending on PyTorch version
        attn_output = self.attn(x_norm, x_norm, x_norm)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        x = x + gate_msa * attn_output

        # MLP with adaptive norm
        x_norm = self.norm2(x)
        if timestep_emb is not None:
            x_norm = x_norm * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer (DiT) for discrete token generation.

    Architecture:
    - Token embeddings + timestep embeddings
    - Series of DiT blocks with adaptive layer norm
    - Final norm + output projection
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_size)

        # Timestep embeddings (similar to DiT)
        self.timestep_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Position embeddings
        self.pos_embed = SinusoidalPositionEmbedding(hidden_size, max_seq_len)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(dropout)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens [batch, seq_len]
            timesteps: Timesteps [batch]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Token embeddings
        h = self.token_embed(x)

        # Position embeddings
        pos_emb = self.pos_embed(seq_len, device)
        h = h + pos_emb

        # Timestep embeddings
        t_emb = None
        if timesteps is not None:
            # Get timestep embeddings
            t_emb = self.get_timestep_embedding(timesteps)  # [batch, hidden_size]
            t_emb = self.timestep_embed(t_emb)  # [batch, hidden_size]
            h = h + t_emb.unsqueeze(1)  # Broadcast to [batch, seq_len, hidden_size]

        h = self.dropout(h)

        # DiT blocks - pass timestep embeddings for modulation
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        h = self.norm_final(h)
        logits = self.lm_head(h)

        return logits


class AutoregressiveTransformer(nn.Module):
    """Autoregressive Transformer baseline (like GPT).

    This is useful as a baseline for comparing diffusion models against.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for autoregressive model.

        Args:
            x: Input tokens [batch, seq_len]
            timesteps: Ignored (for compatibility)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Token and position embeddings
        h = self.token_embed(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embed(positions)

        h = self.dropout(h)

        # Causal mask (AR only sees previous tokens)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1
        )

        # Transformer blocks
        h = self.blocks(h, mask=causal_mask)

        # Output projection
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits


class MambaBlock(nn.Module):
    """Mamba block for state-space based diffusion.

    Based on "Mamba: Linear-time Sequence Modeling with Selective State Spaces"
    """

    def __init__(self, hidden_size: int, state_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Input projection
        self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(hidden_size, state_size))
        self.D = nn.Parameter(torch.ones(hidden_size))

        # Output
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using selective state space."""
        # Simplified Mamba block
        x_gate = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = x * torch.sigmoid(x_gate)

        # State projection
        s = torch.einsum('bld,dn->bln', x, self.A_log.exp())
        y = torch.einsum('bld,bd,bn->bl', s, self.D, x)

        return self.out_proj(y)


def create_model(
    model_type: str,
    vocab_size: int,
    hidden_size: int = 384,
    num_heads: int = 6,
    num_layers: int = 12,
    max_seq_len: int = 1024,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function to create models."""
    model_map = {
        "dit": DiffusionTransformer,
        "ar": AutoregressiveTransformer,
        "mamba": lambda vs, hs, nh, nl, msl, dp: DiffusionTransformer(
            vs, hs, nh, nl, msl, dp  # Use DiT for now
        ),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_map[model_type](
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
