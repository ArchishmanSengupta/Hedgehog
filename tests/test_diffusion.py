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
        assert schedule.schedule == "linear"

    def test_noise_schedule_cosine(self):
        schedule = NoiseSchedule("cosine", num_timesteps=100)
        assert schedule.schedule == "cosine"

    def test_noise_schedule_quadratic(self):
        schedule = NoiseSchedule("quadratic", num_timesteps=100)
        assert schedule.schedule == "quadratic"


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

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(128, 100)

            def forward(self, x, timesteps=None):
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
        x_t = diffusion(x_0, t, mask_token_id=99)
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


class TestMDLMDiffusionEdgeCases:
    """Test MDLMDiffusion edge cases."""

    @pytest.mark.parametrize("num_timesteps", [1, 5, 10, 50, 100, 500])
    def test_different_timesteps(self, num_timesteps):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=num_timesteps)
        assert diffusion.num_timesteps == num_timesteps

    @pytest.mark.parametrize("vocab_size", [10, 50, 100, 1000, 10000])
    def test_different_vocab_sizes(self, vocab_size):
        diffusion = MDLMDiffusion(vocab_size=vocab_size, num_timesteps=10)
        assert diffusion.vocab_size == vocab_size

    def test_q_sample_t0(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (2, 10))
        t = torch.zeros(2, dtype=torch.long)  # t=0
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        assert x_t.shape == x_0.shape

    def test_q_sample_tmax(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (2, 10))
        t = torch.full((2,), 9, dtype=torch.long)  # t=max
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        assert x_t.shape == x_0.shape

    def test_q_sample_batch_size_one(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (1, 10))
        t = torch.tensor([5])
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        assert x_t.shape == x_0.shape

    def test_compute_loss_different_timesteps(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(128, 100)

            def forward(self, x, timesteps=None):
                return self.lm_head(torch.randn(x.shape[0], x.shape[1], 128))

        model = MockModel()
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_0 = torch.randint(0, 100, (2, 10))

        for t_val in [0, 1, 5, 9]:
            t = torch.full((2,), t_val, dtype=torch.long)
            loss = diffusion.compute_loss(model, x_0, t, mask_token_id=99)
            assert loss.item() >= 0


class TestD3PMDiffusionEdgeCases:
    """Test D3PMDiffusion edge cases."""

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "quadratic"])
    def test_different_schedules(self, schedule):
        diffusion = D3PMDiffusion(
            vocab_size=1000,
            diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=100,
            schedule=schedule,
        )
        assert diffusion is not None

    @pytest.mark.parametrize("transition_type", ["absorbing", "uniform"])
    def test_different_transition_types(self, transition_type):
        diffusion = D3PMDiffusion(
            vocab_size=1000,
            diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=100,
            transition_type=transition_type,
        )
        assert diffusion is not None


class TestCreateDiffusionEdgeCases:
    """Test create_diffusion edge cases."""

    @pytest.mark.parametrize("vocab_size", [100, 1000, 50000])
    @pytest.mark.parametrize("num_timesteps", [10, 50, 100])
    def test_various_configs(self, vocab_size, num_timesteps):
        diffusion = create_diffusion("mdlm", vocab_size=vocab_size, num_timesteps=num_timesteps)
        assert diffusion.vocab_size == vocab_size
        assert diffusion.num_timesteps == num_timesteps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
