"""Tests for trainers module."""

import pytest
import torch
from torch.utils.data import DataLoader
from hedgehog.trainers import TrainerConfig, Trainer, DiffusionTrainer
from hedgehog.models import create_model


class DummyDataset:
    """Dummy dataset for testing."""

    def __init__(self, num_samples=10, seq_len=32, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


class TestTrainerConfig:
    """Test TrainerConfig dataclass."""

    def test_default_config(self):
        config = TrainerConfig()
        assert config.model_type == "dit"
        assert config.vocab_size == 32768
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 3
        assert config.warmup_steps == 500

    def test_custom_config(self):
        config = TrainerConfig(
            model_type="ar",
            vocab_size=10000,
            learning_rate=5e-5,
            num_train_epochs=5,
            warmup_steps=100,
            warmup_start_factor=0.2,
            warmup_end_factor=0.8,
            mask_token_id=42,
        )
        assert config.model_type == "ar"
        assert config.vocab_size == 10000
        assert config.learning_rate == 5e-5
        assert config.warmup_start_factor == 0.2
        assert config.warmup_end_factor == 0.8
        assert config.mask_token_id == 42

    def test_warmup_factors_default(self):
        config = TrainerConfig()
        assert config.warmup_start_factor == 0.1
        assert config.warmup_end_factor == 1.0


class TestTrainer:
    """Test Trainer class."""

    def test_trainer_creation(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        assert trainer.config == config
        assert trainer.model == model

    def test_trainer_with_custom_mask_token(self):
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            mask_token_id=50,
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        assert trainer.mask_token_id == 50

    def test_trainer_default_mask_token(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        # Default mask_token_id should be vocab_size - 1
        assert trainer.mask_token_id == 99

    def test_setup_training(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_training_step(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 100, (2, 32)),
        }

        metrics = trainer.training_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_gradient_accumulation(self):
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            gradient_accumulation_steps=2,
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        batch = {
            "input_ids": torch.randint(0, 100, (2, 32)),
        }

        # First accumulation step - should not update optimizer
        initial_loss_scale = trainer.loss_scale
        metrics = trainer.training_step(batch)
        assert trainer.loss_scale == initial_loss_scale + 1


class TestDiffusionTrainer:
    """Test DiffusionTrainer class."""

    def test_diffusion_trainer_creation(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = DiffusionTrainer(config=config, model=model, train_dataset=dataset)
        assert trainer.sampler is not None

    def test_sample_method(self):
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
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = DiffusionTrainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        samples = trainer.sample(num_samples=2, seq_len=16)
        assert samples.shape == (2, 16)


class TestLRScheduler:
    """Test learning rate scheduler configuration."""

    def test_cosine_scheduler(self):
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            lr_scheduler_type="cosine",
            min_lr=1e-7,
            warmup_start_factor=0.2,
            warmup_end_factor=0.5,
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        assert trainer.scheduler is not None

    def test_constant_scheduler(self):
        config = TrainerConfig(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
            device="cpu",
            lr_scheduler_type="constant",
        )
        model = create_model(
            model_type="dit",
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=32,
        )
        dataset = DummyDataset(num_samples=10, seq_len=32, vocab_size=100)

        trainer = Trainer(config=config, model=model, train_dataset=dataset)
        trainer.setup_training()

        assert trainer.scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
