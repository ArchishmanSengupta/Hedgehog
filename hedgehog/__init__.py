# hedgehog - Diffusion Language Model Training Framework
# A lightweight framework for training and fine-tuning diffusion language models
# Inspired by MS-SWIFT but for Diffusion Language Models (DLMs)

__version__ = "0.2.1"

from . import diffusion, models, trainers, samplers, data, utils
from . import peft, distributed, quantization, inference, registry

__all__ = [
    # Core modules
    "diffusion",
    "models",
    "trainers",
    "samplers",
    "data",
    "utils",
    # PEFT modules
    "peft",
    # Distributed training
    "distributed",
    # Quantization
    "quantization",
    # Inference backends
    "inference",
    # Model registry
    "registry",
]
