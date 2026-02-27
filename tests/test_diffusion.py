"""Tests for diffusion module."""

import pytest
import torch
from hedgehog.diffusion import (
    create_diffusion,
    DiscreteDiffusion,
    MDLMDiffusion,
    D3PMDiffusion,
    DiffusionType,
    NoiseSchedule,
)


class TestNoiseSchedule:
    """Test NoiseSchedule enum."""

    def test_noise_schedule_linear(self):
        schedule = NoiseSchedule("linear", num_timesteps=100)
        assert schedule.num_timesteps == 100
        assert schedule.schedule_type == "linear"

    def test_noise_schedule_cosine(self):
        schedule = NoiseSchedule("cosine", num_timesteps=100)
        assert schedule.schedule_type == "cosine"

    def test_noise_schedule_quadratic(self):
        schedule = NoiseSchedule("quadratic", num_timesteps=100)
        assert schedule.schedule_type == "quadratic"


class TestDiffusionType:
    """Test DiffusionType enum."""

    def test_diffusion_types(self):
        assert hasattr(DiffusionType, 'MDLM_SUBS')
        assert hasattr(DiffusionType, 'MDLM_ABSORBING')
        assert hasattr(DiffusionType, 'D3PM_ABSORBING')
        assert hasattr(DiffusionType, 'D3PM_UNIFORM')


class TestMDLMDiffusion:
    """Test MDLMDiffusion."""

    def test_mdlm_creation(self):
        diffusion = MDLMDiffusion(vocab_size=1000, num_timesteps=100)
        assert diffusion.vocab_size == 1000
        assert diffusion.num_timesteps == 100

    def test_q_sample(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (2, 10))
        t = torch.tensor([5, 3])
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        assert x_t.shape == x_0.shape
        assert mask.shape == x_0.shape

    def test_compute_loss(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)

        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(128, 100)

            def forward(self, x, timestep):
                return self.lm_head(torch.randn(x.shape[0], x.shape[1], 128))

        model = MockModel()
        x_0 = torch.randint(0, 100, (2, 10))
        t = torch.tensor([5, 3])
        loss = diffusion.compute_loss(model, x_0, t, mask_token_id=99)
        assert loss.item() >= 0

    def test_forward(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (2, 10))
        t = torch.tensor([5, 3])
        x_t, mask = diffusion(x_0, t, mask_token_id=99)
        assert x_t.shape == x_0.shape


class TestD3PMDiffusion:
    """Test D3PMDiffusion."""

    def test_d3pm_absorbing_creation(self):
        diffusion = D3PMDiffusion(
            vocab_size=1000,
            diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=100,
            schedule="linear",
            transition_type="absorbing",
        )
        assert diffusion.vocab_size == 1000
        assert diffusion.diffusion_type == DiffusionType.D3PM_ABSORBING

    def test_d3pm_uniform_creation(self):
        diffusion = D3PMDiffusion(
            vocab_size=1000,
            diffusion_type=DiffusionType.D3PM_UNIFORM,
            num_timesteps=100,
            schedule="linear",
            transition_type="uniform",
        )
        assert diffusion.diffusion_type == DiffusionType.D3PM_UNIFORM


class TestCreateDiffusion:
    """Test create_diffusion factory function."""

    def test_create_mdlm(self):
        diffusion = create_diffusion("mdlm", vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, MDLMDiffusion)

    def test_create_d3pm(self):
        diffusion = create_diffusion("d3pm", vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, MDLMDiffusion)

    def test_create_sedd(self):
        diffusion = create_diffusion("sedd", vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, MDLMDiffusion)

    def test_create_d3pm_absorbing(self):
        diffusion = create_diffusion("d3pm_absorbing", vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, D3PMDiffusion)

    def test_create_d3pm_uniform(self):
        diffusion = create_diffusion("d3pm_uniform", vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, D3PMDiffusion)

    def test_invalid_diffusion_type(self):
        with pytest.raises(ValueError) as exc_info:
            create_diffusion("invalid", vocab_size=1000)
        assert "Unknown diffusion type" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    @pytest.mark.parametrize("diffusion_type", ["mdlm", "d3pm", "sedd", "d3pm_absorbing", "d3pm_uniform"])
    def test_all_diffusion_types(self, diffusion_type):
        diffusion = create_diffusion(diffusion_type, vocab_size=500, num_timesteps=50)
        assert diffusion is not None
        assert diffusion.vocab_size == 500
        assert diffusion.num_timesteps == 50


class TestDiscreteDiffusion:
    """Test DiscreteDiffusion base class."""

    def test_base_class_creation(self):
        # MDLMDiffusion inherits from DiscreteDiffusion
        diffusion = MDLMDiffusion(vocab_size=1000, num_timesteps=100)
        assert isinstance(diffusion, DiscreteDiffusion)

    def test_get_alpha_bar(self):
        diffusion = MDLMDiffusion(vocab_size=1000, num_timesteps=100)
        alpha_bar = diffusion.get_alpha_bar(50)
        assert 0 <= alpha_bar <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
