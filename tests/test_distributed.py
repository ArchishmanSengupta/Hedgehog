"""Tests for distributed module."""

import pytest
import torch
import torch.nn as nn
from hedgehog.distributed import (
    DistributedConfig,
    DistributedManager,
    get_distributed_manager,
    TensorParallelLinear,
    TensorParallelEmbedding,
    PipelineParallel,
    SequenceParallelLinear,
    tensor_parallelize_model,
    gather_tensor_parallel_outputs,
    split_data_for_dp,
    FSDPWrapper,
)


class TestDistributedConfig:
    """Test DistributedConfig dataclass."""

    def test_default_config(self):
        config = DistributedConfig()
        assert config.backend == "nccl"
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.zero_stage == 0

    def test_custom_config(self):
        config = DistributedConfig(
            backend="gloo",
            world_size=4,
            rank=1,
            local_rank=2,
            zero_stage=2,
        )
        assert config.backend == "gloo"
        assert config.world_size == 4
        assert config.rank == 1
        assert config.local_rank == 2
        assert config.zero_stage == 2


class TestDistributedManager:
    """Test DistributedManager class."""

    def test_singleton_pattern(self):
        manager1 = get_distributed_manager()
        manager2 = get_distributed_manager()
        assert manager1 is manager2

    def test_manager_not_initialized(self):
        manager = DistributedManager()
        assert not manager.is_initialized

    def test_manager_properties_default(self):
        manager = DistributedManager()
        assert manager.is_distributed is False
        assert manager.is_main
        assert manager.world_size == 1
        assert manager.rank == 0
        assert manager.local_rank == 0


class TestDistributedManagerSetup:
    """Test DistributedManager setup methods."""

    def test_setup_single_process(self):
        manager = DistributedManager()
        config = DistributedConfig()
        # Should not raise when world_size is 1
        manager.setup(config)
        # Note: This won't actually initialize distributed


class TestTensorParallelLinear:
    """Test TensorParallelLinear class."""

    def test_creation(self):
        base_layer = nn.Linear(128, 256)
        tp_linear = TensorParallelLinear(base_layer, tp_size=2)
        assert tp_linear.tp_size == 2

    def test_forward_without_gather(self):
        base_layer = nn.Linear(128, 256)
        tp_linear = TensorParallelLinear(base_layer, tp_size=2, gather_output=False)
        x = torch.randn(2, 10, 128)
        output = tp_linear(x)
        # Without gather, output should be half size
        assert output.shape[-1] == 128  # 256 / 2


class TestTensorParallelEmbedding:
    """Test TensorParallelEmbedding class."""

    def test_creation(self):
        base_embedding = nn.Embedding(1000, 128)
        # Note: This requires a distributed setup for full functionality
        tp_embed = TensorParallelEmbedding(base_embedding, tp_size=2)
        assert tp_embed.tp_size == 2


class TestPipelineParallel:
    """Test PipelineParallel class."""

    def test_creation(self):
        layers = [nn.Linear(128, 128) for _ in range(4)]
        pp = PipelineParallel(layers, num_stages=2, stage_id=0)
        assert pp.num_stages == 2
        assert pp.stage_id == 0

    def test_forward(self):
        layers = [nn.Linear(128, 128) for _ in range(4)]
        pp = PipelineParallel(layers, num_stages=2, stage_id=0)
        x = torch.randn(2, 10, 128)
        output = pp(x)
        assert output.shape == x.shape


class TestSequenceParallelLinear:
    """Test SequenceParallelLinear class."""

    def test_creation(self):
        base_layer = nn.Linear(64, 64)
        sp_linear = SequenceParallelLinear(base_layer, sp_size=2)
        assert sp_linear.sp_size == 2

    def test_forward(self):
        base_layer = nn.Linear(64, 64)
        sp_linear = SequenceParallelLinear(base_layer, sp_size=2)
        x = torch.randn(2, 10, 64)  # seq_len divisible by sp_size
        output = sp_linear(x)
        assert output.shape == x.shape


class TestTensorParallelizeModel:
    """Test tensor_parallelize_model function."""

    def test_tensor_parallelize_simple_model(self):
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        # Should not raise, just transform the model
        result = tensor_parallelize_model(model, tp_size=1)
        assert result is not None


class TestGatherTensorParallelOutputs:
    """Test gather_tensor_parallel_outputs function."""

    def test_gather_single_tp_size(self):
        outputs = torch.randn(2, 10, 128)
        result = gather_tensor_parallel_outputs(outputs, tp_size=1)
        assert result.shape == outputs.shape


class TestSplitDataForDP:
    """Test split_data_for_dp function."""

    def test_split_tensor(self):
        data = torch.randn(10, 128)
        result = split_data_for_dp(data, rank=0, world_size=2)
        assert result.shape[0] <= data.shape[0]

    def test_split_list(self):
        data = list(range(10))
        result = split_data_for_dp(data, rank=0, world_size=2)
        assert len(result) <= len(data)

    def test_split_other_type(self):
        result = split_data_for_dp("string", rank=0, world_size=2)
        assert result == "string"


class TestFSDPWrapper:
    """Test FSDPWrapper class."""

    def test_creation(self):
        model = nn.Linear(128, 256)
        fsdp = FSDPWrapper(model)
        assert fsdp.model is not None

    def test_forward(self):
        model = nn.Linear(128, 256)
        fsdp = FSDPWrapper(model)
        x = torch.randn(2, 10, 128)
        output = fsdp(x)
        assert output.shape == (2, 10, 256)

    def test_sharding_strategy(self):
        model = nn.Linear(128, 256)
        fsdp = FSDPWrapper(model, sharding_strategy="FULL_SHARD")
        assert fsdp.sharding_strategy == "FULL_SHARD"

    def test_mixed_precision_setting(self):
        model = nn.Linear(128, 256)
        fsdp = FSDPWrapper(model, mixed_precision=False)
        assert fsdp.mixed_precision is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
