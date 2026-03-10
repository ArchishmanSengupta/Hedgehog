# Hedgehog: Scalable Lightweight Infrastructure for Fine-Tuning Diffusion Language Models
![Screenshot 2026-02-25 at 8 51 26 PM](https://github.com/user-attachments/assets/2e416621-dbd9-43e7-a6c0-4a5564c6cb8d)

<p align="center">
    <br>
    <!-- <img src="asset/banner.png"/> -->
    <br>
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10+-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://pypi.org/project/hedgehog-dlm/"><img src="https://badge.fury.io/py/hedgehog-dlm.svg"></a>
<a href="https://github.com/ArchishmanSengupta/Hedgehog/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ArchishmanSengupta/Hedgehog"></a>
<a href="https://github.com/ArchishmanSengupta/Hedgehog/stargazers"><img src="https://img.shields.io/github/stars/ArchishmanSengupta/Hedgehog"></a>
</p>


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Training](#training)
- [Inference](#inference)
- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Examples](#examples)
- [Built-in Models](#built-in-models)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction

**Hedgehog** is a lightweight framework for training, fine-tuning, and deploying Diffusion Language Models (DLMs). Inspired by [MS-SWIFT](https://github.com/modelscope/ms-swift), it provides a comprehensive solution for working with discrete diffusion language models.

Diffusion Language Models represent a new paradigm in generative AI, where text is generated through a denoising process rather than autoregressive token-by-token prediction. This framework implements state-of-the-art techniques including MDLM, D3PM, and SEDD.

## Features

### Model Support
- **Diffusion Transformer (DiT)**: Purpose-built transformer for discrete diffusion
- **Autoregressive Transformer (AR)**: Standard autoregressive baseline
- **Custom Models**: Easy registration of new model architectures via the registry
- **Pre-trained Models**: Integration with HuggingFace Hub for model loading

### Training Methods
- **Full Parameter Training**: Traditional fine-tuning of all model parameters
- **PEFT (Parameter-Efficient Fine-Tuning)**:
  - LoRA (Low-Rank Adaptation)
  - DoRA (Weight-Decomposed LoRA)
  - IA3 (Infusion of Adapter for Attention)
  - Prefix Tuning
  - Prompt Tuning
- **Context-Aware Defaults**: Learning rate and warmup automatically adjusted for PEFT methods (e.g., LoRA uses 2e-4, IA3 uses 1e-3)

### Distributed Training
- **Data Parallelism (DP)**: Multi-GPU data parallel training
- **Tensor Parallelism (TP)**: Split model across GPUs
- **FSDP**: Fully Sharded Data Parallel
- **Distributed Save Guard**: Checkpoints only written by rank 0

### Quantization
- **AWQ**: Activation-aware Weight Quantization
- **GPTQ**: GPTQ quantization
- **BNB**: BitsAndBytes 4/8-bit quantization

### Inference Backends
- **Transformers**: Native PyTorch inference (built-in, no extra dependencies)
- **vLLM**: High-performance inference engine (`pip install vllm`)
- **SGLang**: Fast inference with custom kernels (`pip install sglang`)
- **LMDeploy**: Deployment-optimized inference (`pip install lmdeploy`)

### Sampling Strategies
- **DDPM**: Standard denoising diffusion probabilistic models
- **Cached DDPM**: Efficient caching for faster sampling (configurable cache steps and size)
- **Analytic**: Score Entropy Discrete Diffusion (SEDD)
- **Semi-Autoregressive**: Block-wise generation
- **Blockwise**: Confidence-based parallel decoding

### Diffusion Types
- **MDLM**: Masked Diffusion Language Models
- **D3PM**: Discrete Denoising Diffusion Probabilistic Models (absorbing and uniform variants)

### Configuration System
- **YAML/JSON Config Files**: Provide training/sampling/eval parameters via config files
- **CLI Override Priority**: CLI args always override config file values (argv injection, not post-parse merge)
- **Checkpoint Persistence**: `args.json` saved with every checkpoint, config embedded in `.pt` files
- **Selective Key Loading**: Three-tier checkpoint config restore (force/load/data keys, inspired by ms-swift)
- **`from_pretrained`**: Reconstruct full config from any checkpoint directory
- **Version Tracking**: `hedgehog_version` saved in every checkpoint and `args.json`
- **Checkpoint Symlinks**: `last.pt` and `best.pt` symlinks in output directory

## Installation

### From PyPI

```shell
pip install hedgehog-dlm
```

### From Source

```shell
git clone https://github.com/ArchishmanSengupta/Hedgehog.git
cd hedgehog
pip install -e .
```

### Requirements

Core dependencies (installed automatically):
- Python 3.10+
- PyTorch >= 2.0
- transformers >= 4.33.0
- datasets >= 2.14.0
- tokenizers >= 0.15.0
- numpy >= 1.24.0
- tqdm >= 4.65.0
- pyyaml >= 6.0

Optional dependencies (installed on demand):

| Package | Purpose | Install |
|---------|---------|---------|
| `accelerate` | Distributed training | `pip install accelerate` |
| `deepspeed` | ZeRO optimization | `pip install deepspeed` |
| `vllm` | vLLM inference backend | `pip install vllm` |
| `sglang` | SGLang inference backend | `pip install sglang` |
| `lmdeploy` | LMDeploy inference backend | `pip install lmdeploy` |
| `fastapi` + `pydantic` | OpenAI-compatible API server | `pip install fastapi pydantic` |
| `uvicorn` | API server runtime | `pip install uvicorn` |

You can also install groups of optional dependencies:

```shell
pip install hedgehog-dlm[train]   # accelerate, deepspeed
pip install hedgehog-dlm[infer]   # vllm, sglang, lmdeploy
pip install hedgehog-dlm[dev]     # pytest, black, ruff, mypy
```

If you try to use a feature that requires an optional package, hedgehog will raise a clear error telling you exactly what to install.

## Quick Start

### Training a Diffusion Language Model

```bash
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir output
```

### Training with LoRA

```bash
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --output_dir output/lora
```

### Training with a Config File

```bash
hedgehog train --dataset tiny-shakespeare --config config.yaml
```

Where `config.yaml` contains any TrainerConfig fields:

```yaml
model_type: dit
vocab_size: 32768
hidden_size: 384
num_heads: 6
num_layers: 12
learning_rate: 1e-4
noise_schedule: cosine
num_timesteps: 1000
use_amp: true
output_dir: output
```

CLI args always override config file values. You can also use JSON:

```bash
hedgehog train --dataset tiny-shakespeare --config config.json
```

### Generating Samples

```bash
hedgehog sample \
    --checkpoint output/final_model.pt \
    --num_samples 5 \
    --seq_len 128 \
    --sampler ddpm_cache
```

Model architecture and diffusion settings are automatically loaded from the checkpoint by default (`--load_args` is True for sample/eval). To disable this:

```bash
hedgehog sample \
    --checkpoint output/final_model.pt \
    --no-load_args \
    --model_type dit \
    --vocab_size 32768 \
    --num_samples 5
```

### Evaluating a Model

```bash
hedgehog eval \
    --checkpoint output/final_model.pt \
    --dataset tiny-shakespeare
```

### Listing Available Models

```bash
hedgehog list --models --datasets --training_methods --sampling_methods
```

### Starting Inference Server

Requires: `pip install fastapi pydantic uvicorn`

```bash
hedgehog serve \
    --checkpoint output/final_model.pt \
    --port 8000 \
    --backend transformers
```

## Configuration

### TrainerConfig

All training parameters are defined in a single `TrainerConfig` dataclass. This is the single source of truth -- CLI args are auto-generated from it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | `"dit"` | Model architecture: dit, ar |
| `vocab_size` | int | `32768` | Vocabulary size |
| `hidden_size` | int | `384` | Hidden dimension |
| `num_heads` | int | `6` | Number of attention heads |
| `num_layers` | int | `12` | Number of transformer layers |
| `max_seq_len` | int | `512` | Maximum sequence length |
| `dropout` | float | `0.1` | Dropout probability |
| `mask_token_id` | int | `None` | Mask token ID (defaults to vocab_size - 1) |
| `diffusion_type` | str | `"mdlm"` | Diffusion type: mdlm, d3pm_absorbing, d3pm_uniform |
| `num_timesteps` | int | `1000` | Number of diffusion timesteps |
| `noise_schedule` | str | `"linear"` | Noise schedule: linear, cosine, quadratic |
| `num_train_epochs` | int | `3` | Number of training epochs |
| `per_device_batch_size` | int | `32` | Batch size per device |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `learning_rate` | float | `1e-4` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping |
| `warmup_steps` | int | `500` | Warmup steps |
| `lr_scheduler_type` | str | `"linear"` | LR scheduler: linear, cosine, constant |
| `min_lr` | float | `1e-6` | Minimum LR for cosine scheduler |
| `use_amp` | bool | `False` | Enable automatic mixed precision |
| `amp_dtype` | str | `"float16"` | AMP dtype: float16, bfloat16 |
| `output_dir` | str | `"output"` | Output directory |
| `logging_steps` | int | `10` | Log every N steps |
| `save_steps` | int | `500` | Save checkpoint every N steps |
| `save_total_limit` | int | `3` | Maximum checkpoints to keep |
| `eval_steps` | int | `500` | Evaluate every N steps |
| `device` | str | `"auto"` | Device: auto, cpu, cuda, mps |
| `seed` | int | `42` | Random seed |
| `resume_from_checkpoint` | str | `None` | Path to checkpoint to resume from |

### Config File Priority

When using `--config`, the priority order is:

1. **CLI args** (highest priority -- always wins)
2. **Config file values**
3. **Defaults** (lowest priority)

This is implemented using argv injection (same approach as ms-swift), which correctly handles the edge case where a user explicitly passes a value that matches the default.

### Checkpoint Config Loading

When loading a checkpoint for sampling or evaluation, hedgehog uses selective key loading tiers (inspired by ms-swift):

| Tier | Keys | Behavior |
|------|------|----------|
| **Force Load** | `model_type`, `vocab_size`, `hidden_size`, `num_heads`, `num_layers`, `max_seq_len`, `diffusion_type` | Always restored from checkpoint |
| **Load** | `noise_schedule`, `num_timesteps`, `mask_token_id`, `dropout` | Restored only if current value is None or default |
| **Data** | `per_device_batch_size`, `dataloader_num_workers` | Only restored with `--load_data_args` |

For full training resume (`Trainer.load_checkpoint(load_args=True)`), all keys are restored.

### Context-Aware Defaults

When using PEFT, hedgehog automatically adjusts defaults:

| PEFT Type | Default LR | Default Warmup |
|-----------|-----------|----------------|
| LoRA / DoRA | 2e-4 | 100 |
| IA3 | 1e-3 | 50 |
| None (full) | 1e-4 | 500 |

These only apply when you haven't explicitly set the value.

## Training on Apple Silicon (Mac Mini, MacBook, etc.)

Hedgehog supports training on Apple Silicon using the MPS (Metal Performance Shaders) backend.

### Training on Mac Mini (16GB RAM)

For memory-constrained devices, use a smaller model with LoRA:

```bash
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --peft_type lora \
    --lora_r 4 \
    --num_train_epochs 3 \
    --per_device_batch_size 1 \
    --max_seq_len 128 \
    --gradient_accumulation_steps 8 \
    --hidden_size 256 \
    --num_layers 6 \
    --device mps \
    --output_dir output
```

### Memory-Efficient Tips

| Argument | Description | Recommended for 16GB |
|----------|-------------|---------------------|
| `--per_device_batch_size` | Batch size per device | 1 |
| `--max_seq_len` | Sequence length | 128 |
| `--hidden_size` | Model hidden size | 256 |
| `--num_layers` | Number of layers | 6 |
| `--gradient_accumulation_steps` | Effective batch size | 8 |
| `--device` | Device to use | `mps` |

## Architecture

```
hedgehog/
├── cli/            # Command-line interface (auto-generated from TrainerConfig)
├── diffusion/      # Core diffusion processes (MDLM, D3PM)
├── models/         # Model architectures (DiT, AR)
├── trainers/       # Training loops, TrainerConfig, checkpointing
├── samplers/       # Sampling strategies (DDPM, cached, analytic, semi-AR, blockwise)
├── data/           # Dataset loaders (character, text, HuggingFace)
├── peft/           # Parameter-efficient fine-tuning (LoRA, DoRA, IA3, Prefix, Prompt)
├── distributed/    # Distributed training (DP, TP, FSDP)
├── quantization/   # Model quantization (AWQ, GPTQ, BNB)
├── inference/      # Inference backends and OpenAI-compatible API server
├── registry/       # Model & dataset registry
└── _compat.py      # TrainerConfig field introspection for CLI auto-generation
```

## Training

### Training Methods

| Method | Description | Memory Usage |
|--------|-------------|---------------|
| Full | Full parameter training | High |
| LoRA | Low-rank adaptation | ~50% |
| QLoRA | Quantized LoRA | ~30% |
| DoRA | Weight-decomposed LoRA | ~50% |
| IA3 | Infused adapter scaling | ~40% |
| Prefix | Prefix tuning | ~40% |
| Prompt | Prompt tuning | ~40% |

### Example Training Scripts

See the `examples/` directory:

- `examples/train/full/` - Full parameter training
- `examples/train/lora/` - LoRA fine-tuning
- `examples/infer/` - Inference / generation

### Training Configuration (Python API)

```python
from hedgehog.trainers import TrainerConfig, DiffusionTrainer

config = TrainerConfig(
    model_type="dit",
    vocab_size=32768,
    hidden_size=384,
    num_heads=6,
    num_layers=12,
    max_seq_len=512,
    diffusion_type="mdlm",
    num_timesteps=1000,
    learning_rate=1e-4,
    num_train_epochs=3,
    per_device_batch_size=8,
    output_dir="output",
)

trainer = DiffusionTrainer(config=config, train_dataset=train_dataset)
trainer.train()
```

### Checkpointing

Every checkpoint save creates:
- `{name}.pt` - Model weights, optimizer state, scheduler state, config dict, hedgehog version
- `args.json` - Full TrainerConfig as JSON (with hedgehog version)
- `last.pt` - Symlink to the most recent checkpoint
- `best.pt` - Symlink to the best checkpoint (by eval loss)

Old checkpoints are automatically cleaned up based on `save_total_limit`.

## Inference

### OpenAI-Compatible API

Requires: `pip install fastapi pydantic uvicorn`

Start the server:
```bash
hedgehog serve --checkpoint model.pt --port 8000
```

Use the API:
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")
response = client.completions.create(
    model="hedgehog",
    prompt="Once upon a time",
    max_tokens=100
)
```

### Direct Inference

```python
from hedgehog.inference import create_inference_backend, InferenceConfig

config = InferenceConfig(backend="transformers")
backend = create_inference_backend(model, config, tokenizer)

results = backend.generate(
    prompts="Hello, world!",
    max_length=100,
    temperature=0.7
)
```

## CLI Commands

### train

Train a diffusion language model.

```bash
hedgehog train [OPTIONS]
```

All `TrainerConfig` fields are available as CLI flags (auto-generated from the dataclass). Additional flags:

| Flag | Description |
|------|-------------|
| `--dataset` (required) | HuggingFace dataset name |
| `--config` | Path to YAML/JSON config file |
| `--use_peft` | Enable PEFT training |
| `--peft_type` | PEFT method: lora, dora, ia3, prefix, prompt |
| `--lora_r` | LoRA rank (default: 8) |
| `--lora_alpha` | LoRA alpha (default: 16) |
| `--lora_dropout` | LoRA dropout (default: 0.05) |
| `--use_quantization` | Enable quantization |
| `--quant_type` | Quantization type: bnb, awq, gptq |
| `--quant_bits` | Quantization bits (default: 4) |
| `--max_train_samples` | Limit training samples |
| `--load_args` | Load args from checkpoint when resuming |

### sample

Generate samples from a trained model.

```bash
hedgehog sample [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--checkpoint` (required) | Path to model checkpoint |
| `--num_samples` | Number of samples (default: 1) |
| `--seq_len` | Sequence length (default: 128) |
| `--sampler` | Sampler: ddpm, ddpm_cache, analytic, semi_ar, blockwise |
| `--tokenizer_name` | Tokenizer for decoding output |
| `--config` | Path to YAML/JSON config file |
| `--load_args` / `--no-load_args` | Load config from checkpoint (default: True) |
| `--output` | Output file (default: samples.txt) |

### eval

Evaluate a trained model.

```bash
hedgehog eval [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--checkpoint` (required) | Path to model checkpoint |
| `--dataset` (required) | Evaluation dataset |
| `--per_device_batch_size` | Batch size (default: 8) |
| `--config` | Path to YAML/JSON config file |
| `--load_args` / `--no-load_args` | Load config from checkpoint (default: True) |

### serve

Start an OpenAI-compatible inference server.

Requires: `pip install fastapi pydantic uvicorn`

```bash
hedgehog serve [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--checkpoint` | Path to model checkpoint |
| `--backend` | Backend: transformers, vllm, sglang, lmdeploy |
| `--host` | Host (default: 0.0.0.0) |
| `--port` | Port (default: 8000) |
| `--model_name` | Model name for tokenizer loading |
| `--config` | Path to YAML/JSON config file |

### list

List available models, datasets, and methods.

```bash
hedgehog list [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--models` | List available model configs |
| `--datasets` | List built-in datasets |
| `--training_methods` | List training methods |
| `--sampling_methods` | List sampling strategies |

## Python API

### TrainerConfig Serialization

```python
from hedgehog.trainers import TrainerConfig

config = TrainerConfig(learning_rate=3e-4, noise_schedule="cosine")

# Save / load
config.to_json("config.json")
config.to_yaml("config.yaml")

loaded = TrainerConfig.from_json("config.json")
loaded = TrainerConfig.from_yaml("config.yaml")
loaded = TrainerConfig.from_file("config.yaml")  # auto-detects format

# Load from checkpoint directory
config = TrainerConfig.from_pretrained("output/")

# Merge overrides
config = config.merge({"learning_rate": 1e-3, "num_train_epochs": 5})
```

### Custom Model Registration

```python
from hedgehog.registry import register_model

register_model("my-dlm", {
    "vocab_size": 50000,
    "hidden_size": 512,
    "num_heads": 8,
    "num_layers": 12,
    "max_seq_len": 1024,
    "dropout": 0.1,
})
```

## Examples

### Fine-tuning with LoRA

```bash
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --peft_type lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --per_device_batch_size 4 \
    --output_dir output/lora
```

### QLoRA Training (Lower Memory)

```bash
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --peft_type lora \
    --use_quantization \
    --quant_type bnb \
    --quant_bits 4 \
    --per_device_batch_size 2 \
    --output_dir output/qlora
```

### Training with Cosine Schedule

```bash
hedgehog train \
    --dataset tiny-shakespeare \
    --noise_schedule cosine \
    --lr_scheduler_type cosine \
    --num_timesteps 1000 \
    --output_dir output/cosine
```

### Using a Config File

```yaml
# train_config.yaml
model_type: dit
vocab_size: 32768
hidden_size: 384
num_heads: 6
num_layers: 12
learning_rate: 3e-4
noise_schedule: cosine
num_timesteps: 1000
num_train_epochs: 5
per_device_batch_size: 8
warmup_steps: 100
lr_scheduler_type: cosine
output_dir: output/config_run
```

```bash
hedgehog train --dataset tiny-shakespeare --config train_config.yaml
```

Override any value from the command line:

```bash
hedgehog train --dataset tiny-shakespeare --config train_config.yaml --learning_rate 1e-3
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --per_device_batch_size 4 \
    --output_dir output/distributed
```

## Built-in Models

| Model | Vocab Size | Hidden Size | Heads | Layers | Max Seq Len |
|-------|------------|-------------|-------|--------|-------------|
| mdlm-small | 32768 | 256 | 8 | 6 | 512 |
| mdlm-base | 32768 | 384 | 12 | 12 | 512 |
| mdlm-large | 32768 | 768 | 16 | 24 | 1024 |
| dit-small | 32768 | 312 | 12 | 12 | 1024 |
| dit-base | 32768 | 768 | 16 | 16 | 2048 |
| dit-large | 32768 | 1024 | 16 | 24 | 2048 |
| char-small | 256 | 128 | 4 | 4 | 256 |
| char-base | 256 | 256 | 8 | 8 | 512 |
| subword-base | 32000 | 512 | 8 | 12 | 1024 |

These are configuration presets (no pre-trained weights). Use them via the registry:

```python
from hedgehog.registry import ModelRegistry

registry = ModelRegistry()
model = registry.get_model("mdlm-base")
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Hedgehog in your research, please cite:

```bibtex
@software{hedgehog2026,
  title = {Hedgehog: Scalable Lightweight Infrastructure for Fine-Tuning Diffusion Language Models},
  author = {ArchishmanSengupta},
  year = {2026},
  url = {https://github.com/ArchishmanSengupta/Hedgehog}
}
```

## Acknowledgments

Hedgehog is inspired by:
- [MS-SWIFT](https://github.com/modelscope/ms-swift) - The foundational framework this project is modeled after
- [MDLM](https://arxiv.org/abs/2210.13382) - Masked Diffusion Language Models
- [DiT](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Models with Transformers
- [Hugging Face](https://huggingface.co) - For transformers and datasets libraries

---

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ArchishmanSengupta/Hedgehog&type=Date)](https://star-history.com/#ArchishmanSengupta/Hedgehog)
