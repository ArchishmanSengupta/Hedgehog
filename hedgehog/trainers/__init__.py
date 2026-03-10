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
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


FORCE_LOAD_KEYS = frozenset([
    "model_type", "hidden_size", "num_heads", "num_layers",
    "vocab_size", "max_seq_len", "diffusion_type",
])

LOAD_KEYS = frozenset([
    "noise_schedule", "num_timesteps", "mask_token_id",
    "dropout",
])

DATA_KEYS = frozenset([
    "per_device_batch_size", "dataloader_num_workers",
])


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
    warmup_start_factor: float = 0.1
    warmup_end_factor: float = 1.0
    lr_scheduler_type: str = "linear"
    min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "float16"

    # Model architecture
    hidden_size: int = 384
    num_heads: int = 6
    num_layers: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    mask_token_id: Optional[int] = None

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
    device: str = "auto"

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None
    load_args_from_checkpoint: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dict_versioned(self) -> Dict[str, Any]:
        """Return dict with hedgehog version included."""
        import hedgehog
        d = self.to_dict()
        d["hedgehog_version"] = hedgehog.__version__
        return d

    @staticmethod
    def _check_json_serializable(d: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all values are JSON-serializable, converting where needed."""
        clean = {}
        for k, v in d.items():
            if isinstance(v, Path):
                clean[k] = str(v)
            elif isinstance(v, (set, frozenset)):
                clean[k] = list(v)
            elif isinstance(v, float) and (v != v):
                clean[k] = None
            else:
                try:
                    json.dumps(v)
                    clean[k] = v
                except (TypeError, ValueError):
                    clean[k] = str(v)
        return clean

    def to_json(self, path: str) -> None:
        d = self._check_json_serializable(self.to_dict_versioned())
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def to_yaml(self, path: str) -> None:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        d = self._check_json_serializable(self.to_dict_versioned())
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "TrainerConfig":
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        return cls.from_dict(d)

    @classmethod
    def from_file(cls, path: str) -> "TrainerConfig":
        path_lower = path.lower()
        if path_lower.endswith((".yaml", ".yml")):
            return cls.from_yaml(path)
        elif path_lower.endswith(".json"):
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path}. Use .yaml, .yml, or .json")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> "TrainerConfig":
        """Reconstruct config from a checkpoint directory.

        Looks for args.json first, then falls back to reading config from
        the latest .pt checkpoint file.
        """
        checkpoint_dir = Path(checkpoint_dir)
        args_path = checkpoint_dir / "args.json"
        if args_path.exists():
            return cls.from_json(str(args_path))

        pt_files = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if pt_files:
            checkpoint = torch.load(str(pt_files[-1]), map_location="cpu", weights_only=False)
            if "config" in checkpoint and checkpoint["config"]:
                return cls.from_dict(checkpoint["config"])

        raise FileNotFoundError(
            f"No args.json or .pt checkpoint with config found in {checkpoint_dir}"
        )

    def merge(self, overrides: Dict[str, Any]) -> "TrainerConfig":
        d = self.to_dict()
        valid_fields = {f.name for f in fields(self.__class__)}
        for k, v in overrides.items():
            if k in valid_fields and v is not None:
                d[k] = v
        return self.__class__.from_dict(d)

    def apply_context_defaults(self, use_peft: bool = False,
                               peft_type: Optional[str] = None) -> "TrainerConfig":
        """Apply context-aware defaults based on training mode.

        For LoRA/DoRA/IA3 training, ms-swift uses different defaults:
        - Higher learning rate (2e-4 vs 1e-4) for adapter params
        - Fewer warmup steps
        """
        d = self.to_dict()
        defaults = self.__class__()

        if use_peft and peft_type in ("lora", "dora"):
            if d["learning_rate"] == defaults.learning_rate:
                d["learning_rate"] = 2e-4
            if d["warmup_steps"] == defaults.warmup_steps:
                d["warmup_steps"] = 100
        elif use_peft and peft_type == "ia3":
            if d["learning_rate"] == defaults.learning_rate:
                d["learning_rate"] = 1e-3
            if d["warmup_steps"] == defaults.warmup_steps:
                d["warmup_steps"] = 50

        return self.__class__.from_dict(d)

    def selective_merge(self, saved: Dict[str, Any],
                        load_data_args: bool = False) -> "TrainerConfig":
        """Merge saved config into current using selective key loading tiers.

        - FORCE_LOAD_KEYS: Always overwritten from saved config
        - LOAD_KEYS: Only loaded if current value is None or matches the default
        - DATA_KEYS: Only loaded if load_data_args is True
        """
        d = self.to_dict()
        defaults = self.__class__()
        valid_fields = {f.name for f in fields(self.__class__)}

        for k, v in saved.items():
            if k not in valid_fields or v is None:
                continue
            if k in FORCE_LOAD_KEYS:
                d[k] = v
            elif k in LOAD_KEYS:
                current = d.get(k)
                default = getattr(defaults, k, None)
                if current is None or current == default:
                    d[k] = v
            elif k in DATA_KEYS:
                if load_data_args:
                    d[k] = v

        return self.__class__.from_dict(d)


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
        self.accum_count = 0

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
            amp_dtype = torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                loss = self.compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
        else:
            loss = self.compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

        self.accum_count += 1

        if self.accum_count >= self.config.gradient_accumulation_steps:
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

            self.optimizer.zero_grad()
            self.accum_count = 0

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

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0) for distributed save guard."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return True

    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """Save model checkpoint, args.json, and update symlinks."""
        if not self._is_main_process():
            return

        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"

        import hedgehog
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
            "hedgehog_version": hedgehog.__version__,
        }

        torch.save(checkpoint, checkpoint_path)

        args_path = self.output_dir / "args.json"
        self.config.to_json(str(args_path))

        self._update_symlink("last", checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        self._cleanup_checkpoints()

    def _update_symlink(self, name: str, target: Path):
        """Create or update a symlink in output_dir pointing to target."""
        link_path = self.output_dir / f"{name}.pt"
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target.name)
        except OSError:
            pass

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str, load_args: bool = False,
                        selective: bool = False, load_data_args: bool = False):
        """Load model checkpoint and optionally restore config.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            load_args: If True, restore config from checkpoint.
            selective: If True, use tiered key loading (force/load/data keys).
                       If False (default), restore all config keys.
            load_data_args: Only used when selective=True. Load data-related keys.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if load_args:
            saved_config = checkpoint.get("config")
            if not saved_config:
                checkpoint_dir = Path(checkpoint_path).parent
                args_path = checkpoint_dir / "args.json"
                if args_path.exists():
                    saved_config = json.loads(args_path.read_text())

            if saved_config:
                if selective:
                    self.config = self.config.selective_merge(
                        saved_config, load_data_args=load_data_args,
                    )
                else:
                    self.config = TrainerConfig.from_dict(saved_config)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        self.setup_training()

        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {self.config.resume_from_checkpoint}")

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

                    if eval_metrics.get("eval_loss", float('inf')) < self.best_metric:
                        self.best_metric = eval_metrics["eval_loss"]
                        self.save_checkpoint("best_model")
                        self._update_symlink("best", self.output_dir / "best_model.pt")

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
