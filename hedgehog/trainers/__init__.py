"""
Training infrastructure for diffusion language models.

Provides:
- Base trainer with training loop
- Checkpointing and logging
- Device management (CPU, CUDA, MPS)
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    # Model
    model_type: str = "dit"
    vocab_size: int = 32768

    # Training
    num_train_epochs: int = 3
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    warmup_start_factor: float = 0.1  # Starting LR multiplier for warmup
    warmup_end_factor: float = 1.0  # Ending LR multiplier after warmup
    lr_scheduler_type: str = "linear"  # linear, cosine, constant
    min_lr: float = 1e-6  # Minimum learning rate for cosine scheduler

    # Mixed precision
    use_amp: bool = False  # Automatic mixed precision
    amp_dtype: str = "float16"  # float16 or bfloat16

    # Model architecture
    hidden_size: int = 384
    num_heads: int = 6
    num_layers: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    mask_token_id: Optional[int] = None  # Mask token ID (defaults to vocab_size - 1)

    # Diffusion
    diffusion_type: str = "mdlm"
    num_timesteps: int = 1000
    noise_schedule: str = "linear"

    # Logging & Saving
    output_dir: str = "output"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None
    fp16: bool = False  # Legacy option for AMP (use use_amp instead)
    bf16: bool = False  # Legacy option for AMP (use use_amp and amp_dtype instead)


class Trainer:
    """Base trainer for diffusion language models.

    Handles:
    - Model initialization
    - Training loop
    - Evaluation
    - Checkpointing
    - Device management
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: Optional[nn.Module] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        self.config = config

        # Set seed
        self._set_seed(config.seed)

        # Setup device
        self.device = self._setup_device(config.device)

        # Setup AMP (Automatic Mixed Precision)
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using automatic mixed precision (AMP)")

        # Initialize model if not provided
        if model is None:
            from ..models import create_model
            self.model = create_model(
                model_type=config.model_type,
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.model = model

        self.model.to(self.device)

        # Initialize diffusion process
        from ..diffusion import create_diffusion
        self.diffusion = create_diffusion(
            diffusion_type=config.diffusion_type,
            vocab_size=config.vocab_size,
            num_timesteps=config.num_timesteps,
            schedule=config.noise_schedule,
        )

        # Get mask token id (last token is typically mask, but can be customized)
        self.mask_token_id = config.mask_token_id if config.mask_token_id is not None else config.vocab_size - 1

        # Setup datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Setup optimizer
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.loss_scale = 1.0  # For gradient accumulation

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_device(self, device: str) -> torch.device:
        """Setup training device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create dataloader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def setup_training(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

        # Scheduler - support different types
        total_steps = self._get_total_steps()
        warmup_steps = min(self.config.warmup_steps, total_steps)

        if self.config.lr_scheduler_type == "cosine":
            # Cosine scheduler with warmup
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.min_lr,
            )
            # Add warmup phase
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=self.config.warmup_end_factor,
                total_iters=warmup_steps,
            )
            from torch.optim.lr_scheduler import SequentialLR
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, self.scheduler],
                milestones=[warmup_steps],
            )
        elif self.config.lr_scheduler_type == "constant":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_end_factor,
                end_factor=self.config.warmup_end_factor,
                total_iters=total_steps,
            )
        else:  # linear
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=self.config.warmup_end_factor,
                total_iters=warmup_steps,
            )

        logger.info(f"Training device: {self.device}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Learning rate scheduler: {self.config.lr_scheduler_type}")

    def _get_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.train_dataset is None:
            return 0

        num_samples = len(self.train_dataset)
        batch_size = self.config.per_device_batch_size * self.config.gradient_accumulation_steps

        return (num_samples // batch_size) * self.config.num_train_epochs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss."""
        x_0 = batch["input_ids"].to(self.device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0,
            self.config.num_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Compute loss using diffusion
        loss = self.diffusion.compute_loss(
            self.model,
            x_0,
            t,
            self.mask_token_id,
        )

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation and AMP support."""
        self.model.train()

        # Forward pass with optional AMP
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
        else:
            loss = self.compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

        # Accumulate gradients
        self.loss_scale += 1

        # Perform optimizer step when accumulation is complete
        if self.loss_scale >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Zero gradients
            self.optimizer.zero_grad()
            self.loss_scale = 1

            # Scheduler step (now runs after each effective batch)
            if self.scheduler is not None:
                self.scheduler.step()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def evaluation(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        eval_dataloader = self._create_dataloader(
            self.eval_dataset,
            self.config.per_device_batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"eval_loss": avg_loss}

    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup training
        self.setup_training()

        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided")

        # Training loop
        train_dataloader = self._create_dataloader(
            self.train_dataset,
            self.config.per_device_batch_size,
            shuffle=True,
        )

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")

            for batch_idx, batch in enumerate(train_dataloader):
                # Training step
                metrics = self.training_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                    logger.info(
                        f"Step {self.global_step} | Loss: {metrics['loss']:.4f} | LR: {lr:.6f}"
                    )

                # Evaluation
                if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluation()
                    logger.info(f"Eval: {eval_metrics}")

                    # Save best model
                    if eval_metrics.get("eval_loss", float('inf')) < self.best_metric:
                        self.best_metric = eval_metrics["eval_loss"]
                        self.save_checkpoint("best_model")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

        # Final save
        self.save_checkpoint("final_model")
        logger.info("Training complete!")


class DiffusionTrainer(Trainer):
    """Specialized trainer for diffusion language models with sampling support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Import samplers
        from ..samplers import create_sampler

        # Default sampler
        self.sampler = create_sampler(
            sampler_type="ddpm_cache",
            diffusion=self.diffusion,
            model=self.model,
            mask_token_id=self.mask_token_id,
        )

    def sample(
        self,
        num_samples: int = 1,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples from the model."""
        self.model.eval()

        seq_len = seq_len or self.config.max_seq_len

        with torch.no_grad():
            samples = self.sampler.sample(
                num_samples=num_samples,
                seq_len=seq_len,
                device=self.device,
            )

        return samples
