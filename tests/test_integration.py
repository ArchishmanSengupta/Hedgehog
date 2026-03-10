"""Integration tests for hedgehog framework.

These tests verify that multiple components work together correctly.
"""

import pytest
import torch
from hedgehog.models import create_model
from hedgehog.diffusion import create_diffusion
from hedgehog.samplers import create_sampler
from hedgehog.trainers import Trainer, DiffusionTrainer, TrainerConfig
from hedgehog.peft import create_peft_model, LoraConfig
from hedgehog.data import TextDataset, CharacterDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def __call__(self, text, return_tensors="pt", padding=False):
        if isinstance(text, str):
            text = [text]
        return {
            "input_ids": torch.randint(1, 100, (len(text), 10)),
            "attention_mask": torch.ones(len(text), 10),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [f"decoded_{i}" for i in range(len(token_ids))]


class TestModelDiffusionIntegration:
    """Test model and diffusion integration."""

    def test_model_with_mdlm_diffusion(self):
        """Test DiT model with MDLM diffusion."""
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=10)

        # Forward pass
        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        logits = model(x, t)
        assert logits.shape == (2, 16, 100)

    def test_model_with_d3pm_diffusion(self):
        """Test DiT model with D3PM diffusion."""
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        diffusion = create_diffusion("d3pm", vocab_size=100, num_timesteps=10)

        # Forward pass
        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        logits = model(x, t)
        assert logits.shape == (2, 16, 100)

    def test_ar_model_with_diffusion(self):
        """Test AR model with diffusion."""
        model = create_model(
            model_type="ar",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=10)

        # Forward pass (timesteps ignored for AR)
        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        logits = model(x, t)
        assert logits.shape == (2, 16, 100)


class TestSamplerIntegration:
    """Test sampler integration with model and diffusion."""

    def test_ddpm_sampler_integration(self):
        """Test DDPM sampler with model and diffusion."""
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=10)

        sampler = create_sampler("ddpm", diffusion=diffusion, model=model, mask_token_id=99)
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)

    def test_all_sampler_types_with_model(self):
        """Test all sampler types work with model."""
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=10)

        sampler_types = ["ddpm", "ddpm_cache", "analytic", "semi_ar", "blockwise"]
        for sampler_type in sampler_types:
            sampler = create_sampler(
                sampler_type, diffusion=diffusion, model=model, mask_token_id=99
            )
            samples = sampler.sample(num_samples=1, seq_len=8, device="cpu")
            assert samples.shape[1] == 8


class TestTrainerIntegration:
    """Test trainer integration with model and data."""

    def test_trainer_with_model_and_dataset(self):
        """Test Trainer works with model and dataset."""
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            gradient_accumulation_steps=1,
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        # Create simple dataset
        class DummyDataset:
            def __init__(self, num_samples=10):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"input_ids": torch.randint(0, 100, (32,))}

        dataset = DummyDataset(num_samples=10)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        batch = {"input_ids": torch.randint(0, 100, (2, 32))}
        metrics = trainer.training_step(batch)
        assert "loss" in metrics

    def test_diffusion_trainer_with_sampling(self):
        """Test DiffusionTrainer with sampling."""
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        class DummyDataset:
            def __init__(self, num_samples=10):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"input_ids": torch.randint(0, 100, (32,))}

        dataset = DummyDataset(num_samples=10)

        trainer = DiffusionTrainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        # Test sampling
        samples = trainer.sample(num_samples=2, seq_len=16)
        assert samples.shape == (2, 16)


class TestPEFTIntegration:
    """Test PEFT integration with model."""

    def test_lora_with_dit_model(self):
        """Test LoRA with DiT model."""
        base_model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        lora_model = create_peft_model(base_model, peft_type="lora", r=4, lora_alpha=8)

        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        output = lora_model(x, timesteps=t)
        assert output.shape == (2, 16, 100)

    def test_ia3_with_model(self):
        """Test IA3 with model."""
        base_model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        ia3_model = create_peft_model(base_model, peft_type="ia3")

        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        output = ia3_model(x, timesteps=t)
        assert output.shape == (2, 16, 100)

    def test_prefix_tuning_with_model(self):
        """Test prefix tuning with model."""
        base_model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        prefix_model = create_peft_model(base_model, peft_type="prefix", prefix_length=10)

        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        output = prefix_model(x, timesteps=t)
        assert output.shape == (2, 16, 100)


class TestDataModelIntegration:
    """Test data module integration with model."""

    def test_character_dataset_with_model(self):
        """Test CharacterDataset works with model."""
        texts = ["hello world", "test text"]
        dataset = CharacterDataset(texts=texts, max_length=32)

        # Get item from dataset
        item = dataset[0]
        assert "input_ids" in item

        # Use with model
        model = create_model(
            model_type="ar",
            vocab_size=dataset.vocab_size,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        x = item["input_ids"].unsqueeze(0)
        logits = model(x)
        assert logits.shape[2] == dataset.vocab_size


class TestFullPipelineIntegration:
    """Test full pipeline from data to sampling."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: data -> model -> diffusion -> trainer -> sample."""
        # Create model
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        # Create diffusion
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=10)

        # Create sampler
        sampler = create_sampler("ddpm", diffusion=diffusion, model=model, mask_token_id=99)

        # Sample
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)
        assert samples.min() >= 0
        assert samples.max() < 100


class TestMultipleModelTypes:
    """Test same operations across different model types."""

    @pytest.mark.parametrize("model_type", ["dit", "ar"])
    def test_forward_pass_all_models(self, model_type):
        """Test forward pass works for all model types."""
        model = create_model(
            model_type=model_type,
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        logits = model(x, t)
        assert logits.shape == (2, 16, 100)

    @pytest.mark.parametrize("model_type", ["dit", "ar"])
    def test_training_step_all_models(self, model_type):
        """Test training step works for all model types."""
        config = TrainerConfig(
            model_type=model_type,
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            gradient_accumulation_steps=1,
        )
        model = create_model(
            model_type=model_type,
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )

        class DummyDataset:
            def __init__(self):
                self.num_samples = 10

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"input_ids": torch.randint(0, 100, (32,))}

        trainer = Trainer(config=config, model=model, train_dataset=DummyDataset())
        trainer.setup_training()

        batch = {"input_ids": torch.randint(0, 100, (2, 32))}
        metrics = trainer.training_step(batch)
        assert "loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
