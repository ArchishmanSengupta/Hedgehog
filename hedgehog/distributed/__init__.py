"""
Distributed training support for Diffusion Language Models.

Supports:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Sequence Parallelism (SP)
- Expert Parallelism (EP) for MoE models
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import functools
from contextlib import contextmanager


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel_size: int = 1

    # Zero
    zero_stage: int = 0  # 0, 1, 2, 3

    # Misc
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False


class DistributedManager:
    """Manages distributed training setup."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DistributedManager._initialized:
            self.config = DistributedConfig()
            self._is_initialized = False
            self._is_local_main = False
            self._world_size = 1
            self._rank = 0
            self._local_rank = 0

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_distributed(self) -> bool:
        return self._world_size > 1

    @property
    def is_main(self) -> bool:
        return self._rank == 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def setup(self, config: Optional[DistributedConfig] = None):
        """Initialize distributed training."""
        if config:
            self.config = config

        # Get distributed parameters from environment
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self._world_size > 1:
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self._world_size,
                rank=self._rank,
            )
            self._is_initialized = True
            self._is_local_main = (self._local_rank == 0)

            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)

    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self._is_initialized = False

    def barrier(self):
        """Synchronization barrier."""
        if self.is_distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All reduce operation."""
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All gather operation."""
        if not self.is_distributed:
            return [tensor]

        world_size = self._world_size
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list

    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """Gather tensors to destination rank."""
        if not self.is_distributed:
            return [tensor]

        if self._rank == dst:
            tensor_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
            dist.gather(tensor, tensor_list, dst=dst)
            return tensor_list
        else:
            dist.gather(tensor, dst=dst)
            return None

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source rank."""
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def reduce_scatter(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduce scatter operation."""
        if not self.is_distributed:
            return tensor

        world_size = self._world_size
        chunk_size = tensor.numel() // world_size
        output = torch.zeros(chunk_size, dtype=tensor.dtype, device=tensor.device)
        input_list = list(tensor.chunk(world_size))
        dist.reduce_scatter(output, input_list, op=op)
        return output


def get_distributed_manager() -> DistributedManager:
    """Get the global distributed manager instance."""
    return DistributedManager()


@contextmanager
def distributed_context(config: DistributedConfig):
    """Context manager for distributed training."""
    manager = get_distributed_manager()
    manager.setup(config)
    try:
        yield manager
    finally:
        manager.cleanup()


# Tensor Parallelism Utilities

class TensorParallelLinear(nn.Module):
    """Linear layer with tensor parallelism (column parallel)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        tp_size: int,
        gather_output: bool = True,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output

        # Split the weight along output dimension
        assert base_layer.out_features % tp_size == 0
        self.out_features_per_rank = base_layer.out_features // tp_size

        self.weight = nn.Parameter(
            base_layer.weight[:self.out_features_per_rank, :].clone()
        )
        if base_layer.bias is not None:
            self.bias = nn.Parameter(
                base_layer.bias[:self.out_features_per_rank].clone()
            )
        else:
            self.bias = None

        self.in_features = base_layer.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # Gather outputs from all ranks
            output_list = [torch.zeros_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)

        return output


class TensorParallelEmbedding(nn.Module):
    """Embedding with tensor parallelism."""

    def __init__(
        self,
        base_embedding: nn.Embedding,
        tp_size: int,
    ):
        super().__init__()
        self.tp_size = tp_size

        # Split embeddings along vocab dimension
        assert base_embedding.num_embeddings % tp_size == 0
        self.num_embeddings_per_rank = base_embedding.num_embeddings // tp_size
        self.embedding_dim = base_embedding.embedding_dim

        # Local embeddings
        start_idx = base_embedding.num_embeddings * self._rank // tp_size
        end_idx = base_embedding.num_embeddings * (self._rank + 1) // tp_size

        self.weight = nn.Parameter(
            base_embedding.weight[start_idx:end_idx].clone()
        )

    @property
    def _rank(self) -> int:
        return get_distributed_manager().rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.embedding(x, self.weight)


# Pipeline Parallelism Utilities

class PipelineParallel(nn.Module):
    """Simple pipeline parallelism wrapper."""

    def __init__(
        self,
        layers: List[nn.Module],
        num_stages: int,
        stage_id: int,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stage_id = stage_id

        # Assign layers to this stage
        layers_per_stage = len(layers) // num_stages
        start_layer = stage_id * layers_per_stage
        end_layer = start_layer + layers_per_stage

        self.layers = nn.ModuleList(layers[start_layer:end_layer])

    def forward(
        self,
        x: torch.Tensor,
        forward_step_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """Forward pass through pipeline stage."""
        for layer in self.layers:
            x = layer(x)
        return x


# Sequence Parallelism Utilities

class SequenceParallelLinear(nn.Module):
    """Linear layer with sequence parallelism."""

    def __init__(
        self,
        base_layer: nn.Linear,
        sp_size: int,
    ):
        super().__init__()
        self.sp_size = sp_size
        self.base_layer = base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sequence parallelism.

        Input x is already split along sequence dimension.
        """
        # Split along sequence dimension
        seq_len = x.shape[1]
        assert seq_len % self.sp_size == 0

        chunk_size = seq_len // self.sp_size
        x_chunks = x.split(chunk_size, dim=1)

        # Process each chunk
        outputs = []
        for chunk in x_chunks:
            output = self.base_layer(chunk)
            outputs.append(output)

        # Concatenate back
        return torch.cat(outputs, dim=1)


# Utility functions

def tensor_parallelize_model(model: nn.Module, tp_size: int) -> nn.Module:
    """Apply tensor parallelism to a model."""
    # Replace linear layers with tensor parallel versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" in name or "output" in name:
                # Output projection - gather at the end
                new_module = TensorParallelLinear(
                    module, tp_size, gather_output=True
                )
            else:
                # Hidden layers - no gathering
                new_module = TensorParallelLinear(
                    module, tp_size, gather_output=False
                )

            # Replace module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

    return model


def gather_tensor_parallel_outputs(
    outputs: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Gather outputs from tensor parallel ranks."""
    if tp_size == 1:
        return outputs

    output_list = [torch.zeros_like(outputs) for _ in range(tp_size)]
    dist.all_gather(output_list, outputs)
    return torch.cat(output_list, dim=-1)


def split_data_for_dp(
    data: Any,
    rank: int,
    world_size: int,
) -> Any:
    """Split data across data parallel ranks."""
    # Simple implementation - return slice for this rank
    if isinstance(data, torch.Tensor):
        chunk_size = len(data) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(data)
        return data[start_idx:end_idx]
    elif isinstance(data, list):
        chunk_size = len(data) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(data)
        return data[start_idx:end_idx]
    return data


class FSDPWrapper:
    """Wrapper for Fully Sharded Data Parallel (simplified)."""

    def __init__(
        self,
        model: nn.Module,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
    ):
        self.model = model
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision

        # Simple FSDP implementation - shard parameters
        self._shard_parameters()

    def _shard_parameters(self):
        """Shard model parameters across ranks."""
        world_size = get_distributed_manager().world_size
        if world_size <= 1:
            return

        rank = get_distributed_manager().rank
        for name, param in self.model.named_parameters():
            if param.numel() > 0:
                start = (param.numel() * rank) // world_size
                end = (param.numel() * (rank + 1)) // world_size
                param._original_numel = param.numel()
                param.data = param.data.flatten()[start:end].clone()
                param.data = param.data.reshape(param.shape[:-1] + (-1,)) if len(param.shape) > 1 else param.data

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Make wrapper callable."""
        return self.forward(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient synchronization."""
        loss.backward()

        # All-reduce gradients
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= get_distributed_manager().world_size
