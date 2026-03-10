"""Tests for inference module."""

import pytest
import torch
import torch.nn as nn
from hedgehog.inference import (
    InferenceConfig,
    InferenceBackend,
    TransformersBackend,
    create_inference_backend,
)


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


class SimpleModel(nn.Module):
    """Simple model for testing inference."""

    def __init__(self, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lm_head = nn.Linear(128, vocab_size)

    def forward(self, input_ids, **kwargs):
        embeds = self.embedding(input_ids)
        return self.lm_head(embeds)

    def generate(self, input_ids, max_length, temperature, top_p, top_k, do_sample, pad_token_id, **kwargs):
        batch_size = input_ids.shape[0]
        extra = torch.randint(1, 100, (batch_size, max_length - input_ids.shape[1]), device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class TestInferenceConfig:
    """Test InferenceConfig dataclass."""

    def test_default_config(self):
        config = InferenceConfig()
        assert config.backend == "transformers"
        assert config.max_model_len == 4096
        assert config.max_num_seqs == 256
        assert config.dtype == "auto"
        assert config.tensor_parallel_size == 1

    def test_custom_config(self):
        config = InferenceConfig(
            backend="vllm",
            max_model_len=2048,
            max_num_seqs=128,
            dtype="float16",
            tensor_parallel_size=2,
        )
        assert config.backend == "vllm"
        assert config.max_model_len == 2048
        assert config.max_num_seqs == 128
        assert config.dtype == "float16"
        assert config.tensor_parallel_size == 2


class TestInferenceBackend:
    """Test InferenceBackend base class."""

    def test_backend_is_abstract(self):
        config = InferenceConfig()
        model = SimpleModel()

        # Cannot instantiate directly - it's abstract
        with pytest.raises(TypeError):
            InferenceBackend(model, config)


class TestTransformersBackend:
    """Test TransformersBackend class."""

    def test_creation_with_model(self):
        model = SimpleModel()
        config = InferenceConfig()
        backend = TransformersBackend(model, config)
        assert backend.model is not None
        assert backend.config == config

    def test_creation_with_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)
        assert backend.tokenizer == tokenizer

    def test_encode_with_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)

        result = backend.encode("test prompt")
        assert isinstance(result, torch.Tensor)

    def test_encode_without_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig()
        backend = TransformersBackend(model, config)

        import pytest
        with pytest.raises(ValueError, match="Tokenizer is required"):
            backend.encode("test prompt")

    def test_decode_with_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)

        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = backend.decode(token_ids)
        assert isinstance(result, list)

    def test_decode_without_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig()
        backend = TransformersBackend(model, config)

        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = backend.decode(token_ids)
        assert isinstance(result, list)

    def test_generate_single_prompt(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)

        result = backend.generate("test prompt", max_length=20)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_generate_multiple_prompts(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)

        result = backend.generate(["prompt 1", "prompt 2"], max_length=20)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_stream_generate(self):
        model = SimpleModel()
        config = InferenceConfig()
        tokenizer = MockTokenizer()
        backend = TransformersBackend(model, config, tokenizer)

        results = list(backend.stream_generate("test prompt", max_length=20))
        assert len(results) >= 1


class TestCreateInferenceBackend:
    """Test create_inference_backend factory function."""

    def test_create_transformers_backend(self):
        model = SimpleModel()
        config = InferenceConfig(backend="transformers")
        backend = create_inference_backend(model, config)
        assert isinstance(backend, TransformersBackend)

    def test_create_with_tokenizer(self):
        model = SimpleModel()
        config = InferenceConfig(backend="transformers")
        tokenizer = MockTokenizer()
        backend = create_inference_backend(model, config, tokenizer)
        assert backend.tokenizer == tokenizer

    def test_invalid_backend_raises_error(self):
        model = SimpleModel()
        config = InferenceConfig(backend="invalid_backend")
        with pytest.raises(ValueError) as exc_info:
            create_inference_backend(model, config)
        assert "Unknown inference backend" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
