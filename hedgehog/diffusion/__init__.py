"""
Core diffusion processes for discrete diffusion language models.

Based on research from:
- MDLM (NeurIPS 2024): Substitution-based parameterization
- D3PM: Discrete Denoising Diffusion Probabilistic Models
- SEDD: Score Entropy Discrete Diffusion
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from enum import Enum


class DiffusionType(Enum):
    """Supported diffusion types."""
    D3PM_ABSORBING = "d3pm_absorbing"  # Mask tokens during diffusion
    D3PM_UNIFORM = "d3pm_uniform"        # Uniform noise
    MDLM_SUBS = "mdlm_subs"              # Substitution-based (MDLM)
    SEDD = "sedd"                        # Score Entropy


class NoiseSchedule:
    """Noise schedule for discrete diffusion.

    Supports various schedules:
    - linear: Linear schedule
    - cosine: Cosine schedule (improved)
    - quadratic: Quadratic schedule
    """

    def __init__(self, schedule: str = "linear", num_timesteps: int = 1000):
        self.schedule = schedule
        self.num_timesteps = num_timesteps

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative noise level at timestep t."""
        if self.schedule == "linear":
            return 1 - t / self.num_timesteps
        elif self.schedule == "cosine":
            # Cosine schedule from improved ddpm
            s = 0.008
            t = t / self.num_timesteps
            return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        elif self.schedule == "quadratic":
            t = t / self.num_timesteps
            return (1 - t ** 2)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")


class DiscreteDiffusion:
    """Base class for discrete diffusion processes."""

    def __init__(
        self,
        vocab_size: int,
        diffusion_type: DiffusionType = DiffusionType.D3PM_ABSORBING,
        num_timesteps: int = 1000,
        schedule: str = "linear",
    ):
        self.vocab_size = vocab_size
        self.diffusion_type = diffusion_type
        self.num_timesteps = num_timesteps
        self.noise_schedule = NoiseSchedule(schedule, num_timesteps)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: add noise to clean data.

        Args:
            x_0: Clean tokens [batch, seq_len]
            t: Timesteps [batch]
            mask_token_id: ID of the mask token

        Returns:
            x_t: Noisy tokens
            mask: Which positions are masked
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Get noise levels per sample
        alpha_bar = self.noise_schedule.get_alpha_bar(t.float())  # [batch]
        alpha_bar = alpha_bar.to(device)

        # Create probability matrix - start with uniform
        probs = torch.full((batch_size, seq_len, self.vocab_size), fill_value=0.0, device=device)

        # Vectorized: scatter alpha_bar values at the positions of x_0
        # alpha_bar expanded to [batch, seq_len, 1]
        alpha_expanded = alpha_bar.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1] -> [batch, seq_len, vocab_size] via scatter

        # Create one-hot encoding of x_0
        one_hot = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        one_hot.scatter_(2, x_0.unsqueeze(2), 1.0)

        # Set probability for original tokens
        probs = probs + one_hot * alpha_expanded

        # Add mask token probability
        mask_prob = (1 - alpha_bar).unsqueeze(1)  # [batch, 1]
        probs[:, :, mask_token_id] = mask_prob

        # Normalize probabilities to sum to 1
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sample from distribution
        x_t = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)

        # Track which positions are masked
        mask = (x_t == mask_token_id)

        return x_t, mask

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Single reverse diffusion step.

        Args:
            model: Denoising model
            x_t: Current noisy tokens [batch, seq_len]
            t: Current timestep
            mask_token_id: ID of the mask token

        Returns:
            x_{t-1}: Denoised tokens
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Get model predictions
        with torch.no_grad():
            logits = model(x_t, timesteps=torch.full((batch_size,), t, device=device))

        # Get probabilities for non-masked positions
        probs = torch.softmax(logits, dim=-1)

        # Sample next step
        x_prev = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)

        return x_prev


class MDLMDiffusion(DiscreteDiffusion):
    """Masked Diffusion Language Model with Substitution-based parameterization.

    Based on "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
    Key insight: Substitute masked tokens with predicted distribution
    """

    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000,
        schedule: str = "linear",
    ):
        super().__init__(
            vocab_size=vocab_size,
            diffusion_type=DiffusionType.MDLM_SUBS,
            num_timesteps=num_timesteps,
            schedule=schedule,
        )

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Compute MDLM loss (simplified to masked LM).

        The substitution-based parameterization simplifies the diffusion
        loss to a mixture of masked language modeling losses.
        """
        # Forward diffusion
        x_t, mask = self.q_sample(x_0, t, mask_token_id)

        # Get model predictions
        logits = model(x_t, timesteps=t)

        # Compute loss only on masked positions
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.vocab_size), x_0.view(-1))
        loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        return loss


class D3PMDiffusion(DiscreteDiffusion):
    """D3PM (Discrete Denoising Diffusion Probabilistic Model) implementation.

    Based on "Transition-Guided Denoising Diffusion Probabilistic Models"
    Supports both absorbing and uniform transition matrices.
    """

    def __init__(
        self,
        vocab_size: int,
        diffusion_type: DiffusionType = DiffusionType.D3PM_ABSORBING,
        num_timesteps: int = 1000,
        schedule: str = "linear",
        q_type: str = "absorbing",
    ):
        super().__init__(vocab_size, diffusion_type, num_timesteps, schedule)
        self.q_type = q_type
        self._init_transition_matrix()

    def _init_transition_matrix(self):
        """Initialize transition matrix Q_t."""
        # Q_t[i,j] = probability of transitioning from state i to state j
        if self.q_type == "absorbing":
            # Absorbing state (mask) stays mask, others mix uniformly
            self.Q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size)
            for t in range(self.num_timesteps):
                beta = t / self.num_timesteps
                self.Q[t] = torch.eye(self.vocab_size) * (1 - beta)
                self.Q[t, :, self.vocab_size - 1] += beta  # mask token absorbs
        elif self.q_type == "uniform":
            # Uniform random noise
            self.Q = torch.zeros(self.num_timesteps, self.vocab_size, self.vocab_size)
            for t in range(self.num_timesteps):
                beta = t / self.num_timesteps
                self.Q[t] = torch.full((self.vocab_size, self.vocab_size), beta / self.vocab_size)
                self.Q[t] += torch.eye(self.vocab_size) * (1 - beta)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion using transition matrix."""
        batch_size, seq_len = x_0.shape
        device = x_0.device

        x_t = x_0.clone()
        mask = torch.zeros_like(x_t, dtype=torch.bool)

        for i in range(batch_size):
            t_i = t[i].item()
            Q_t = self.Q[t_i].to(device)

            for j in range(seq_len):
                probs = Q_t[x_t[i, j]]
                x_t[i, j] = torch.multinomial(probs, 1).item()
                mask[i, j] = (x_t[i, j] == mask_token_id)

        return x_t, mask


def create_diffusion(
    diffusion_type: str,
    vocab_size: int,
    num_timesteps: int = 1000,
    schedule: str = "linear",
) -> DiscreteDiffusion:
    """Factory function to create diffusion processes."""
    diffusion_map = {
        "mdlm": MDLMDiffusion,
        "d3pm_absorbing": lambda vs, nt, sc: D3PMDiffusion(
            vs, DiffusionType.D3PM_ABSORBING, nt, sc, "absorbing"
        ),
        "d3pm_uniform": lambda vs, nt, sc: D3PMDiffusion(
            vs, DiffusionType.D3PM_UNIFORM, nt, sc, "uniform"
        ),
    }

    if diffusion_type not in diffusion_map:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")

    return diffusion_map[diffusion_type](vocab_size, num_timesteps, schedule)
