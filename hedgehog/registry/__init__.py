"""
Model Registry for Diffusion Language Models.

Provides:
- Pre-built model configurations
- Model loading from HuggingFace Hub
- Popular DLM model support
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


# Popular DLM Model Configurations

DLM_MODELS = {
    # MDLM-style models
    "mdlm-small": {
        "vocab_size": 32768,
        "hidden_size": 256,
        "num_heads": 8,
        "num_layers": 6,
        "max_seq_len": 512,
        "dropout": 0.1,
    },
    "mdlm-base": {
        "vocab_size": 32768,
        "hidden_size": 384,
        "num_heads": 12,
        "num_layers": 12,
        "max_seq_len": 512,
        "dropout": 0.1,
    },
    "mdlm-large": {
        "vocab_size": 32768,
        "hidden_size": 768,
        "num_heads": 16,
        "num_layers": 24,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    # Character-level models
    "char-small": {
        "vocab_size": 256,
        "hidden_size": 128,
        "num_heads": 4,
        "num_layers": 4,
        "max_seq_len": 256,
        "dropout": 0.1,
    },
    "char-base": {
        "vocab_size": 256,
        "hidden_size": 256,
        "num_heads": 8,
        "num_layers": 8,
        "max_seq_len": 512,
        "dropout": 0.1,
    },
    # Subword models
    "subword-base": {
        "vocab_size": 32000,
        "hidden_size": 512,
        "num_heads": 8,
        "num_layers": 12,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    # Transformer variants
    "dit-small": {
        "vocab_size": 32768,
        "hidden_size": 312,
        "num_heads": 12,
        "num_layers": 12,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    "dit-base": {
        "vocab_size": 32768,
        "hidden_size": 768,
        "num_heads": 16,
        "num_layers": 16,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    "dit-large": {
        "vocab_size": 32768,
        "hidden_size": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
}


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry."""
    model_name: str
    model_type: str = "dit"  # dit, ar, mamba
    pretrained: bool = False
    model_path: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast: bool = True


class ModelRegistry:
    """Registry for DLM models."""

    def __init__(self):
        self.models = DLM_MODELS.copy()
        self._loaded_models: Dict[str, nn.Module] = {}

    def register_model(self, name: str, config: Dict[str, Any]):
        """Register a new model configuration."""
        self.models[name] = config

    def get_model_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by name."""
        return self.models.get(name)

    def list_models(self) -> list:
        """List all registered models."""
        return list(self.models.keys())

    def load_model(
        self,
        name: str,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> Tuple[nn.Module, Any]:
        """Load a model from registry.

        Args:
            name: Model name or path
            model_type: Model architecture type
            **kwargs: Additional arguments for model

        Returns:
            Tuple of (model, tokenizer)
        """
        # Try to load from HuggingFace Hub first
        try:
            return self._load_from_hub(name, model_type, **kwargs)
        except Exception as e:
            print(f"Could not load from Hub: {e}")

        # Load from registry
        config = self.get_model_config(name)
        if config is None:
            raise ValueError(f"Unknown model: {name}")

        # Create model
        from ..models import create_model
        model = create_model(
            model_type=model_type or "dit",
            vocab_size=config.get("vocab_size", 32768),
            hidden_size=config.get("hidden_size", 384),
            num_heads=config.get("num_heads", 6),
            num_layers=config.get("num_layers", 12),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=config.get("dropout", 0.1),
            **kwargs,
        )

        return model, None

    def _load_from_hub(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> Tuple[nn.Module, Any]:
        """Load model from HuggingFace Hub."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required to load models from Hub")

        # Determine model class
        if model_type == "ar" or "gpt" in model_path.lower():
            model_class = AutoModelForCausalLM
        else:
            # Use auto classes
            model_class = AutoModelForCausalLM

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=kwargs.get("use_fast", True),
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
        )

        # Load model with appropriate config
        model = model_class.from_pretrained(
            model_path,
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
        )

        return model, tokenizer

    def save_model(
        self,
        model: nn.Module,
        save_path: str,
        save_tokenizer: bool = True,
        tokenizer: Optional[Any] = None,
    ):
        """Save model to disk."""
        from ..utils import safe_save_checkpoint

        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), path / "pytorch_model.bin")

        # Save config
        config = {
            "model_type": type(model).__name__,
            "config": model.config.__dict__ if hasattr(model, "config") else {},
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save tokenizer
        if save_tokenizer and tokenizer is not None:
            tokenizer.save_pretrained(str(path))


# Global registry instance
_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def register_model(name: str, config: Dict[str, Any]):
    """Register a new model."""
    registry = get_model_registry()
    registry.register_model(name, config)


def get_model(name: str, model_type: Optional[str] = None, **kwargs) -> Tuple[nn.Module, Any]:
    """Get a model from registry."""
    registry = get_model_registry()
    return registry.load_model(name, model_type, **kwargs)


def list_models() -> list:
    """List all available models."""
    registry = get_model_registry()
    return registry.list_models()


# Built-in datasets

BUILTIN_DATASETS = {
    "tiny-shakespeare": {
        "description": "Tiny Shakespeare dataset",
        "source": "builtin",
        "num_samples": 40000,
    },
    "tiny-math": {
        "description": "Mathematical expressions dataset",
        "source": "builtin",
        "num_samples": 10000,
    },
    "code-contest": {
        "description": "Code contest problems",
        "source": "openwebtext",
        "num_samples": 100000,
    },
}


def get_dataset_info(name: str) -> Optional[Dict[str, Any]]:
    """Get dataset information."""
    return BUILTIN_DATASETS.get(name)


def list_datasets() -> list:
    """List all available datasets."""
    return list(BUILTIN_DATASETS.keys())


# Training methods

TRAINING_METHODS = {
    "sft": {
        "description": "Supervised Fine-Tuning",
        "type": "full",
    },
    "lora": {
        "description": "LoRA fine-tuning",
        "type": "peft",
    },
    "qlora": {
        "description": "QLoRA (Quantized LoRA)",
        "type": "peft+quant",
    },
    "dpo": {
        "description": "Direct Preference Optimization",
        "type": "rl",
    },
    "grpo": {
        "description": "Group Relative Preference Optimization",
        "type": "rl",
    },
}


def get_training_method(name: str) -> Optional[Dict[str, Any]]:
    """Get training method configuration."""
    return TRAINING_METHODS.get(name)


def list_training_methods() -> list:
    """List all training methods."""
    return list(TRAINING_METHODS.keys())


# Sampling methods

SAMPLING_METHODS = {
    "ddpm": {
        "description": "Denoising Diffusion Probabilistic Models",
        "speed": "slow",
    },
    "ddpm_cached": {
        "description": "Efficient DDPM with caching",
        "speed": "medium",
    },
    "analytic": {
        "description": "Analytic sampling (SEDD)",
        "speed": "medium",
    },
    "semi_ar": {
        "description": "Semi-autoregressive",
        "speed": "fast",
    },
    "blockwise": {
        "description": "Blockwise parallel decoding",
        "speed": "fast",
    },
}


def get_sampling_method(name: str) -> Optional[Dict[str, Any]]:
    """Get sampling method information."""
    return SAMPLING_METHODS.get(name)


def list_sampling_methods() -> list:
    """List all sampling methods."""
    return list(SAMPLING_METHODS.keys())
