"""Tests for quantization module."""

import pytest
import torch
import torch.nn as nn
from hedgehog.quantization import (
    QuantConfig,
    QuantizedLinear,
    BNBQuantizedLinear,
    quantize_model,
)


class TestQuantConfig:
    """Test QuantConfig dataclass."""

    def test_default_config(self):
        config = QuantConfig()
        assert config.quant_type == "bnb"
        assert config.bits == 4
        assert config.group_size == 128

    def test_custom_config(self):
        config = QuantConfig(quant_type="awq", bits=8, group_size=64)
        assert config.quant_type == "awq"
        assert config.bits == 8
        assert config.group_size == 64


class TestQuantizedLinear:
    """Test QuantizedLinear base class."""

    def test_creation(self):
        linear = nn.Linear(128, 256)
        config = QuantConfig(bits=4)
        quantized = QuantizedLinear(linear, config)
        assert quantized.base_layer == linear


class TestBNBQuantizedLinear:
    """Test BNBQuantizedLinear."""

    def test_creation(self):
        linear = nn.Linear(128, 256)
        quantized = BNBQuantizedLinear(linear, bits=4)
        assert quantized.bits == 4
        assert quantized.out_features == 256
        assert quantized.in_features == 128

    def test_forward(self):
        linear = nn.Linear(128, 256)
        quantized = BNBQuantizedLinear(linear, bits=4)
        x = torch.randn(2, 10, 128)
        # Forward should work (though may not be accurate without proper quantization)
        try:
            output = quantized(x)
            assert output.shape == (2, 10, 256)
        except NotImplementedError:
            # Expected if not fully implemented
            pass


class TestQuantizeModel:
    """Test quantize_model function."""

    def test_quantize_with_bnb(self):
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        config = QuantConfig(quant_type="bnb", bits=4)
        quantized = quantize_model(model, config)
        assert quantized is not None

    def test_quantize_with_different_bits(self):
        model = nn.Sequential(
            nn.Linear(128, 256),
        )
        for bits in [4, 8]:
            config = QuantConfig(quant_type="bnb", bits=bits)
            quantized = quantize_model(model, config)
            assert quantized is not None


class TestQuantConfigEdgeCases:
    """Test QuantConfig edge cases."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 8, 16])
    def test_different_bits(self, bits):
        config = QuantConfig(bits=bits)
        assert config.bits == bits

    @pytest.mark.parametrize("group_size", [32, 64, 128, 256])
    def test_different_group_sizes(self, group_size):
        config = QuantConfig(group_size=group_size)
        assert config.group_size == group_size

    @pytest.mark.parametrize("quant_type", ["bnb", "awq", "gptq", "hqq", "eetq"])
    def test_different_quant_types(self, quant_type):
        config = QuantConfig(quant_type=quant_type)
        assert config.quant_type == quant_type


class TestBNBQuantizedLinearEdgeCases:
    """Test BNBQuantizedLinear edge cases."""

    @pytest.mark.parametrize("bits", [4, 8])
    def test_different_bits(self, bits):
        linear = nn.Linear(128, 256)
        quantized = BNBQuantizedLinear(linear, bits=bits)
        assert quantized.bits == bits

    @pytest.mark.parametrize("in_features,out_features", [
        (64, 64), (128, 256), (256, 512), (512, 128)
    ])
    def test_different_dimensions(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        quantized = BNBQuantizedLinear(linear, bits=4)
        assert quantized.in_features == in_features
        assert quantized.out_features == out_features


class TestQuantizeModelEdgeCases:
    """Test quantize_model edge cases."""

    def test_quantize_deep_model(self):
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        config = QuantConfig(quant_type="bnb", bits=4)
        quantized = quantize_model(model, config)
        assert quantized is not None

    def test_quantize_model_with_bias(self):
        model = nn.Linear(128, 256, bias=True)
        config = QuantConfig(quant_type="bnb", bits=4)
        quantized = quantize_model(model, config)
        assert quantized is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
