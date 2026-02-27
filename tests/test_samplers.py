"""Tests for samplers module."""

import pytest
import torch
from hedgehog.samplers import (
    create_sampler,
    Sampler,
    DDPMSampler,
    DDPMCachedSampler,
    AnalyticSampler,
    SemiAutoregressiveSampler,
    BlockwiseSampler,
)
from hedgehog.diffusion import create_diffusion, MDLMDiffusion
from hedgehog.models import create_model


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return create_model(
        model_type="dit",
        vocab_size=100,
        hidden_size=64,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
    )


@pytest.fixture
def diffusion():
    """Create a simple diffusion for testing."""
    return create_diffusion("mdlm", vocab_size=100, num_timesteps=10)


@pytest.fixture
def mask_token_id():
    """Return mask token ID."""
    return 99


class TestDDPMSampler:
    """Test DDPMSampler."""

    def test_creation(self, diffusion, model, mask_token_id):
        sampler = DDPMSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert sampler.diffusion == diffusion
        assert sampler.model == model

    def test_sample_shape(self, diffusion, model, mask_token_id):
        sampler = DDPMSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)


class TestDDPMCachedSampler:
    """Test DDPMCachedSampler."""

    def test_creation(self, diffusion, model, mask_token_id):
        sampler = DDPMCachedSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert sampler.diffusion == diffusion
        assert sampler.model == model

    def test_sample_shape(self, diffusion, model, mask_token_id):
        sampler = DDPMCachedSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)


class TestAnalyticSampler:
    """Test AnalyticSampler."""

    def test_creation(self, diffusion, model, mask_token_id):
        sampler = AnalyticSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert sampler.diffusion == diffusion

    def test_sample_shape(self, diffusion, model, mask_token_id):
        sampler = AnalyticSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)


class TestSemiAutoregressiveSampler:
    """Test SemiAutoregressiveSampler."""

    def test_creation(self, diffusion, model, mask_token_id):
        sampler = SemiAutoregressiveSampler(
            diffusion=diffusion, model=model, mask_token_id=mask_token_id
        )
        assert sampler.diffusion == diffusion

    def test_sample_shape(self, diffusion, model, mask_token_id):
        sampler = SemiAutoregressiveSampler(
            diffusion=diffusion, model=model, mask_token_id=mask_token_id
        )
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)


class TestBlockwiseSampler:
    """Test BlockwiseSampler."""

    def test_creation(self, diffusion, model, mask_token_id):
        sampler = BlockwiseSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert sampler.diffusion == diffusion

    def test_sample_shape(self, diffusion, model, mask_token_id):
        sampler = BlockwiseSampler(diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        samples = sampler.sample(num_samples=2, seq_len=16, device="cpu")
        assert samples.shape == (2, 16)


class TestCreateSampler:
    """Test create_sampler factory function."""

    def test_create_ddpm(self, diffusion, model, mask_token_id):
        sampler = create_sampler("ddpm", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert isinstance(sampler, DDPMSampler)

    def test_create_ddpm_cache(self, diffusion, model, mask_token_id):
        sampler = create_sampler("ddpm_cache", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert isinstance(sampler, DDPMCachedSampler)

    def test_create_analytic(self, diffusion, model, mask_token_id):
        sampler = create_sampler("analytic", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert isinstance(sampler, AnalyticSampler)

    def test_create_semi_ar(self, diffusion, model, mask_token_id):
        sampler = create_sampler("semi_ar", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert isinstance(sampler, SemiAutoregressiveSampler)

    def test_create_blockwise(self, diffusion, model, mask_token_id):
        sampler = create_sampler("blockwise", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert isinstance(sampler, BlockwiseSampler)

    def test_invalid_sampler_type(self, diffusion, model, mask_token_id):
        with pytest.raises(ValueError) as exc_info:
            create_sampler("invalid", diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert "Unknown sampler type" in str(exc_info.value)

    @pytest.mark.parametrize("sampler_type", ["ddpm", "ddpm_cache", "analytic", "semi_ar", "blockwise"])
    def test_all_sampler_types(self, diffusion, model, mask_token_id, sampler_type):
        sampler = create_sampler(sampler_type, diffusion=diffusion, model=model, mask_token_id=mask_token_id)
        assert sampler is not None
        assert isinstance(sampler, Sampler)


class TestSampler:
    """Test Sampler base class."""

    def test_sampler_is_abstract(self):
        # Cannot instantiate Sampler directly
        with pytest.raises(TypeError):
            Sampler(diffusion=None, model=None, mask_token_id=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
