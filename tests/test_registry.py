"""Tests for registry module."""

import pytest
from hedgehog.registry import (
    register_model,
    get_model,
    list_models,
    get_dataset_info,
    list_datasets,
    list_training_methods,
    list_sampling_methods,
    DLM_MODELS,
    BUILTIN_DATASETS,
    TRAINING_METHODS,
    SAMPLING_METHODS,
)


class TestBuiltinModels:
    """Test DLM_MODELS."""

    def test_builtin_models_exist(self):
        assert isinstance(DLM_MODELS, dict)
        assert len(DLM_MODELS) > 0

    def test_dit_models(self):
        assert "dit-small" in DLM_MODELS
        assert "dit-base" in DLM_MODELS
        assert "dit-large" in DLM_MODELS

    def test_mdlm_models(self):
        assert "mdlm-small" in DLM_MODELS
        assert "mdlm-base" in DLM_MODELS
        assert "mdlm-large" in DLM_MODELS

    def test_char_models(self):
        assert "char-small" in DLM_MODELS
        assert "char-base" in DLM_MODELS


class TestListModels:
    """Test list_models function."""

    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "dit-small" in models


class TestGetModel:
    """Test get_model function."""

    def test_get_builtin_model(self):
        model_config = get_model("dit-small")
        assert model_config is not None
        assert "vocab_size" in model_config
        assert "hidden_size" in model_config

    def test_get_nonexistent_model(self):
        with pytest.raises(ValueError):
            get_model("nonexistent-model")


class TestRegisterModel:
    """Test register_model function."""

    def test_register_new_model(self):
        custom_config = {
            "vocab_size": 10000,
            "hidden_size": 512,
            "num_layers": 12,
            "description": "Custom model",
        }
        # Register a unique name
        register_model("custom-test-model", custom_config)
        models = list_models()
        assert "custom-test-model" in models

        # Clean up
        if "custom-test-model" in DLM_MODELS:
            del DLM_MODELS["custom-test-model"]


class TestBuiltinDatasets:
    """Test BUILTIN_DATASETS."""

    def test_builtin_datasets_exist(self):
        assert isinstance(BUILTIN_DATASETS, dict)
        assert len(BUILTIN_DATASETS) > 0

    def test_tiny_shakespeare(self):
        assert "tiny-shakespeare" in BUILTIN_DATASETS

    def test_tiny_math(self):
        assert "tiny-math" in BUILTIN_DATASETS


class TestListDatasets:
    """Test list_datasets function."""

    def test_list_datasets(self):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "tiny-shakespeare" in datasets


class TestGetDataset:
    """Test get_dataset_info function."""

    def test_get_builtin_dataset(self):
        dataset_config = get_dataset_info("tiny-shakespeare")
        assert dataset_config is not None
        assert "description" in dataset_config

    def test_get_nonexistent_dataset(self):
        with pytest.raises(ValueError):
            get_dataset_info("nonexistent-dataset")


class TestTrainingMethods:
    """Test training methods."""

    def test_training_methods_exist(self):
        assert isinstance(TRAINING_METHODS, dict)
        assert len(TRAINING_METHODS) > 0

    def test_sft_method(self):
        assert "sft" in TRAINING_METHODS

    def test_lora_method(self):
        assert "lora" in TRAINING_METHODS

    def test_qlora_method(self):
        assert "qlora" in TRAINING_METHODS

    def test_dpo_method(self):
        assert "dpo" in TRAINING_METHODS

    def test_grpo_method(self):
        assert "grpo" in TRAINING_METHODS


class TestListTrainingMethods:
    """Test list_training_methods function."""

    def test_list_training_methods(self):
        methods = list_training_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert "sft" in methods
        assert "lora" in methods


class TestSamplingMethods:
    """Test sampling methods."""

    def test_sampling_methods_exist(self):
        assert isinstance(SAMPLING_METHODS, dict)
        assert len(SAMPLING_METHODS) > 0

    def test_ddpm_method(self):
        assert "ddpm" in SAMPLING_METHODS or "ddpm_cached" in SAMPLING_METHODS

    def test_semi_ar_method(self):
        assert "semi_ar" in SAMPLING_METHODS


class TestListSamplingMethods:
    """Test list_sampling_methods function."""

    def test_list_sampling_methods(self):
        methods = list_sampling_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
