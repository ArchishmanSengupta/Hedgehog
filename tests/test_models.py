"""Tests for models module."""

import pytest
import torch
from hedgehog.models import (
    create_model,
    DiffusionTransformer,
    AutoregressiveTransformer,
    MambaBlock,
    ModelConfig,
    DiTBlock,
    SinusoidalPositionEmbedding,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        config = ModelConfig()
        assert config.model_type == "dit"
        assert config.vocab_size == 32768
        assert config.hidden_size == 384
        assert config.num_heads == 6
        assert config.num_layers == 12
        assert config.max_seq_len == 512

    def test_custom_config(self):
        config = ModelConfig(
            model_type="ar",
            vocab_size=10000,
            hidden_size=256,
            num_heads=4,
            num_layers=6,
            max_seq_len=256,
        )
        assert config.model_type == "ar"
        assert config.vocab_size == 10000
        assert config.hidden_size == 256


class TestSinusoidalPositionEmbedding:
    """Test SinusoidalPositionEmbedding."""

    def test_embedding_creation(self):
        embedding = SinusoidalPositionEmbedding(dim=64, max_seq_len=128)
        assert embedding.dim == 64
        assert embedding.max_seq_len == 128

    def test_embedding_forward(self):
        embedding = SinusoidalPositionEmbedding(dim=64, max_seq_len=128)
        x = torch.randint(0, 128, (2, 10))  # batch_size=2, seq_len=10
        result = embedding(x)
        assert result.shape == (2, 10, 64)


class TestDiTBlock:
    """Test DiTBlock."""

    def test_dit_block_creation(self):
        config = ModelConfig(hidden_size=128, num_heads=4)
        block = DiTBlock(config)
        assert block.hidden_size == 128
        assert block.num_heads == 4

    def test_dit_block_forward(self):
        config = ModelConfig(hidden_size=128, num_heads=4)
        block = DiTBlock(config)
        x = torch.randn(2, 10, 128)  # batch_size=2, seq_len=10, hidden=128
        timestep = torch.tensor([100, 200])
        result = block(x, timestep)
        assert result.shape == (2, 10, 128)


class TestDiffusionTransformer:
    """Test DiffusionTransformer."""

    def test_dit_creation(self):
        model = DiffusionTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        assert model.vocab_size == 1000
        assert model.hidden_size == 128

    def test_dit_forward(self):
        model = DiffusionTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        x = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        timestep = torch.randint(0, 1000, (2,))
        result = model(x, timestep)
        assert result.shape == (2, 32, 1000)

    def test_dit_parameter_count(self):
        model = DiffusionTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0


class TestAutoregressiveTransformer:
    """Test AutoregressiveTransformer."""

    def test_ar_creation(self):
        model = AutoregressiveTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        assert model.vocab_size == 1000

    def test_ar_forward(self):
        model = AutoregressiveTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        x = torch.randint(0, 1000, (2, 32))
        result = model(x)
        assert result.shape == (2, 32, 1000)


class TestMambaBlock:
    """Test MambaBlock."""

    def test_mamba_creation(self):
        block = MambaBlock(d_model=128, d_state=16, expand=2)
        assert block.d_model == 128

    def test_mamba_forward(self):
        block = MambaBlock(d_model=128, d_state=16, expand=2)
        x = torch.randn(2, 10, 128)
        result = block(x)
        assert result.shape == (2, 10, 128)


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_dit(self):
        model = create_model(
            model_type="dit",
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        assert isinstance(model, DiffusionTransformer)

    def test_create_ar(self):
        model = create_model(
            model_type="ar",
            vocab_size=1000,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )
        assert isinstance(model, AutoregressiveTransformer)

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            create_model(
                model_type="invalid_model",
                vocab_size=1000,
                hidden_size=128,
                num_heads=4,
                num_layers=2,
                max_seq_len=64,
            )

    def test_model_with_different_vocab_sizes(self):
        for vocab_size in [100, 1000, 10000, 50000]:
            model = create_model(
                model_type="dit",
                vocab_size=vocab_size,
                hidden_size=64,
                num_heads=2,
                num_layers=1,
                max_seq_len=32,
            )
            x = torch.randint(0, vocab_size, (1, 10))
            timestep = torch.tensor([100])
            result = model(x, timestep)
            assert result.shape == (1, 10, vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
