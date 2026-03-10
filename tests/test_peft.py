"""Tests for PEFT module."""

import pytest
import torch
import torch.nn as nn
from hedgehog.peft import (
    create_peft_model,
    LoraConfig,
    LoraModel,
    IA3Model,
    PrefixTuningModel,
    PromptTuningModel,
    PrefixTuning,
    PromptTuning,
)
from hedgehog.models import create_model


@pytest.fixture
def base_model():
    """Create a simple model for testing."""
    return create_model(
        model_type="dit",
        vocab_size=100,
        hidden_size=64,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
    )


class TestLoraConfig:
    """Test LoraConfig dataclass."""

    def test_default_config(self):
        config = LoraConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05

    def test_custom_config(self):
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1)
        assert config.r == 4
        assert config.lora_alpha == 8
        assert config.lora_dropout == 0.1


class TestLoraModel:
    """Test LoraModel."""

    def test_lora_creation(self, base_model):
        config = LoraConfig(r=4, lora_alpha=8)
        lora_model = LoraModel(base_model, config)
        assert lora_model.base_model == base_model

    def test_lora_forward(self, base_model):
        config = LoraConfig(r=4, lora_alpha=8)
        lora_model = LoraModel(base_model, config)
        x = torch.randint(0, 100, (2, 16))
        t = torch.randint(0, 100, (2,))
        output = lora_model(x, timesteps=t)
        assert output is not None


class TestIA3Model:
    """Test IA3Model."""

    def test_ia3_creation(self, base_model):
        ia3_model = IA3Model(base_model)
        assert ia3_model.base_model == base_model

    def test_ia3_forward(self, base_model):
        ia3_model = IA3Model(base_model)
        x = torch.randint(0, 100, (2, 16))
        t = torch.randint(0, 100, (2,))
        output = ia3_model(x, timesteps=t)
        assert output is not None


class TestPrefixTuning:
    """Test PrefixTuning."""

    def test_prefix_tuning_creation(self):
        prefix = PrefixTuning(hidden_size=64, prefix_length=10, num_layers=2)
        assert prefix.hidden_size == 64
        assert prefix.prefix_length == 10


class TestPrefixTuningModel:
    """Test PrefixTuningModel."""

    def test_prefix_creation(self, base_model):
        prefix_model = PrefixTuningModel(base_model, prefix_length=10)
        assert prefix_model.base_model == base_model

    def test_prefix_forward(self, base_model):
        prefix_model = PrefixTuningModel(base_model, prefix_length=10)
        x = torch.randint(0, 100, (2, 16))
        t = torch.randint(0, 100, (2,))
        output = prefix_model(x, timesteps=t)
        assert output is not None


class TestPromptTuning:
    """Test PromptTuning."""

    def test_prompt_tuning_creation(self):
        prompt = PromptTuning(vocab_size=100, prompt_length=10, hidden_size=64)
        assert prompt.vocab_size == 100
        assert prompt.prompt_length == 10


class TestPromptTuningModel:
    """Test PromptTuningModel."""

    def test_prompt_creation(self, base_model):
        prompt_model = PromptTuningModel(base_model, vocab_size=100, prompt_length=10)
        assert prompt_model.base_model == base_model

    def test_prompt_forward(self, base_model):
        prompt_model = PromptTuningModel(base_model, vocab_size=100, prompt_length=10)
        x = torch.randint(0, 100, (2, 16))
        t = torch.randint(0, 100, (2,))
        output = prompt_model(x, timesteps=t)
        assert output is not None


class TestCreatePeftModel:
    """Test create_peft_model factory function."""

    def test_create_lora(self, base_model):
        model = create_peft_model(base_model, peft_type="lora", r=4, lora_alpha=8)
        assert isinstance(model, LoraModel)

    def test_create_dora(self, base_model):
        from hedgehog.peft import DoRAModel
        model = create_peft_model(base_model, peft_type="dora", r=4, lora_alpha=8)
        assert isinstance(model, DoRAModel)

    def test_create_ia3(self, base_model):
        model = create_peft_model(base_model, peft_type="ia3")
        assert isinstance(model, IA3Model)

    def test_create_prefix(self, base_model):
        model = create_peft_model(base_model, peft_type="prefix", prefix_length=10)
        assert isinstance(model, PrefixTuningModel)

    def test_create_prompt(self, base_model):
        model = create_peft_model(base_model, peft_type="prompt", vocab_size=100, prompt_length=10)
        assert isinstance(model, PromptTuningModel)

    def test_invalid_peft_type(self, base_model):
        with pytest.raises(ValueError) as exc_info:
            create_peft_model(base_model, peft_type="invalid")
        assert "Unknown PEFT type" in str(exc_info.value)

    @pytest.mark.parametrize("peft_type", ["lora", "dora", "ia3", "prefix", "prompt"])
    def test_all_peft_types(self, base_model, peft_type):
        kwargs = {"r": 4, "lora_alpha": 8} if peft_type in ["lora", "dora"] else {}
        if peft_type == "prefix":
            kwargs = {"prefix_length": 10}
        if peft_type == "prompt":
            kwargs = {"vocab_size": 100, "prompt_length": 10}

        model = create_peft_model(base_model, peft_type=peft_type, **kwargs)
        assert model is not None


class TestLoraConfigEdgeCases:
    """Test LoraConfig edge cases."""

    @pytest.mark.parametrize("r", [1, 2, 4, 8, 16, 32])
    def test_different_ranks(self, r):
        config = LoraConfig(r=r)
        assert config.r == r

    @pytest.mark.parametrize("lora_alpha", [8, 16, 32, 64])
    def test_different_alphas(self, lora_alpha):
        config = LoraConfig(lora_alpha=lora_alpha)
        assert config.lora_alpha == lora_alpha

    @pytest.mark.parametrize("lora_dropout", [0.0, 0.05, 0.1, 0.2, 0.5])
    def test_different_dropouts(self, lora_dropout):
        config = LoraConfig(lora_dropout=lora_dropout)
        assert config.lora_dropout == lora_dropout


class TestLoraModelEdgeCases:
    """Test LoraModel edge cases."""

    @pytest.mark.parametrize("r", [2, 4, 8])
    def test_lora_different_ranks(self, r):
        config = LoraConfig(r=r, lora_alpha=r*2)
        lora_model = LoraModel(self._get_base_model(), config)
        x = torch.randint(0, 100, (2, 16))
        t = torch.randint(0, 100, (2,))
        output = lora_model(x, timesteps=t)
        assert output is not None
        assert output.shape[0] == 2

    def _get_base_model(self):
        return create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )


class TestIA3ModelEdgeCases:
    """Test IA3Model edge cases."""

    def test_ia3_with_different_models(self):
        for model_type in ["dit", "ar"]:
            base_model = create_model(
                model_type=model_type,
                vocab_size=100,
                hidden_size=64,
                num_heads=2,
                num_layers=1,
                max_seq_len=32,
            )
            ia3_model = IA3Model(base_model)
            x = torch.randint(0, 100, (2, 16))
            t = torch.randint(0, 100, (2,))
            output = ia3_model(x, timesteps=t)
            assert output is not None


class TestPrefixTuningEdgeCases:
    """Test PrefixTuning edge cases."""

    @pytest.mark.parametrize("prefix_length", [2, 5, 10, 20, 32])
    def test_different_prefix_lengths(self, prefix_length):
        prefix = PrefixTuning(hidden_size=64, prefix_length=prefix_length, num_layers=2)
        assert prefix.prefix_length == prefix_length

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 8])
    def test_different_num_layers(self, num_layers):
        prefix = PrefixTuning(hidden_size=64, prefix_length=10, num_layers=num_layers)
        assert prefix.num_layers == num_layers


class TestPromptTuningEdgeCases:
    """Test PromptTuning edge cases."""

    @pytest.mark.parametrize("prompt_length", [2, 5, 10, 20, 32])
    def test_different_prompt_lengths(self, prompt_length):
        prompt = PromptTuning(vocab_size=100, prompt_length=prompt_length, hidden_size=64)
        assert prompt.prompt_length == prompt_length

    @pytest.mark.parametrize("vocab_size", [100, 1000, 5000])
    def test_different_vocab_sizes(self, vocab_size):
        prompt = PromptTuning(vocab_size=vocab_size, prompt_length=10, hidden_size=64)
        assert prompt.vocab_size == vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
