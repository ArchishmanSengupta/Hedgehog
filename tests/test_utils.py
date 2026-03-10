"""Tests for utils module."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from hedgehog.utils import (
    setup_logging,
    get_logger,
    get_device,
    get_device_count,
    set_device,
    count_parameters,
    get_model_size,
    safe_save_checkpoint,
    safe_load_checkpoint,
    find_free_port,
    setup_distributed,
    collate_fn,
    AverageMeter,
)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        logger = setup_logging()
        assert logger.name == "hedgehog"
        assert logger.level == 20  # INFO

    def test_setup_logging_custom_level(self):
        logger = setup_logging(log_level="DEBUG")
        assert logger.level == 10  # DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        import logging
        logger = logging.getLogger("hedgehog")
        logger.handlers.clear()
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=str(log_file))
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].baseFilename == str(log_file)


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_default(self):
        logger = get_logger()
        assert logger.name == "hedgehog"

    def test_get_logger_custom_name(self):
        logger = get_logger("test")
        assert logger.name == "hedgehog.test"


class TestGetDevice:
    """Test get_device function."""

    def test_get_device_cpu(self):
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_auto_returns_valid(self):
        device = get_device("auto")
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_explicit_cuda_when_available(self):
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"


class TestGetDeviceCount:
    """Test get_device_count function."""

    def test_get_device_count_returns_positive(self):
        count = get_device_count()
        assert count >= 1


class TestSetDevice:
    """Test set_device function."""

    def test_set_device_returns_string(self):
        device = set_device()
        assert isinstance(device, str)

    def test_set_device_with_local_rank(self):
        device = set_device(local_rank=0)
        assert isinstance(device, str)


class TestCountParameters:
    """Test count_parameters function."""

    def test_count_all_parameters(self):
        model = nn.Linear(10, 5)
        count = count_parameters(model)
        assert count == 50 + 5  # weight + bias

    def test_count_trainable_only(self):
        model = nn.Linear(10, 5)
        # All parameters are trainable by default
        count = count_parameters(model, trainable_only=True)
        assert count == 55

    def test_count_with_frozen_params(self):
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        count = count_parameters(model, trainable_only=True)
        assert count == 0


class TestGetModelSize:
    """Test get_model_size function."""

    def test_model_size_calculation(self):
        model = nn.Linear(100, 50)
        size = get_model_size(model)
        assert size > 0

    def test_model_size_includes_buffers(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
        )
        size = get_model_size(model)
        assert size > 0


class TestCheckpointFunctions:
    """Test checkpoint save/load functions."""

    def test_safe_save_and_load(self, tmp_path):
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Create test checkpoint
        checkpoint = {
            "model_state_dict": {"weight": torch.randn(10, 5)},
            "epoch": 1,
            "loss": 0.5,
        }

        # Save
        safe_save_checkpoint(checkpoint, str(checkpoint_path))
        assert checkpoint_path.exists()

        # Load
        loaded = safe_load_checkpoint(str(checkpoint_path))
        assert "model_state_dict" in loaded
        assert loaded["epoch"] == 1


class TestFindFreePort:
    """Test find_free_port function."""

    def test_find_free_port_returns_integer(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


class TestSetupDistributed:
    """Test setup_distributed function."""

    def test_setup_distributed_default(self):
        env = setup_distributed()
        assert "local_rank" in env
        assert "world_size" in env
        assert "node_rank" in env


class TestCollateFn:
    """Test collate_fn function."""

    def test_collate_fn_basic(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 3)

    def test_collate_fn_different_keys(self):
        batch = [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])},
            {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 0])},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 2)
        assert result["attention_mask"].shape == (2, 2)


class TestAverageMeter:
    """Test AverageMeter class."""

    def test_average_meter_creation(self):
        meter = AverageMeter()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.count == 0

    def test_average_meter_update(self):
        meter = AverageMeter()
        meter.update(10.0)
        assert meter.val == 10.0
        assert meter.avg == 10.0
        assert meter.count == 1

    def test_average_meter_multiple_updates(self):
        meter = AverageMeter()
        meter.update(10.0)
        meter.update(20.0)
        assert meter.val == 20.0
        assert meter.avg == 15.0
        assert meter.count == 2

    def test_average_meter_update_with_n(self):
        meter = AverageMeter()
        meter.update(10.0, n=5)
        assert meter.count == 5
        assert meter.avg == 10.0

    def test_average_meter_reset(self):
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
