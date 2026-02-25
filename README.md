# Hedgehog: Scalable Lightweight Infrastructure for Fine-Tuning Diffusion Language Models

<p align="center">
    <br>
    <!-- <img src="asset/banner.png"/> -->
    <br>
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10+-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://pypi.org/project/hedgehog/"><img src="https://badge.fury.io/py/hedgehog.svg"></a>
<a href="https://github.com/ArchishmanSengupta/Hedgehog/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ArchishmanSengupta/Hedgehog"></a>
</p>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [Inference](#inference)
- [CLI Commands](#cli-commands)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction

**Hedgehog** is a lightweight framework for training, fine-tuning, and deploying Diffusion Language Models (DLMs). Inspired by MS-SWIFT, it provides a comprehensive solution for working with discrete diffusion language models.

Diffusion Language Models represent a new paradigm in generative AI, where text is generated through a denoising process rather than autoregressive token-by-token prediction. This framework implements state-of-the-art techniques including MDLM, D3PM, and SEDD.

## Features

### Model Support
- **600+ Model Architectures**: Support for various diffusion and transformer-based architectures
- **Custom Models**: Easy registration of new model architectures
- **Pre-trained Models**: Integration with HuggingFace Hub for model loading

### Training Methods
- **Full Parameter Training**: Traditional fine-tuning of all model parameters
- **PEFT (Parameter-Efficient Fine-Tuning)**:
  - LoRA (Low-Rank Adaptation)
  - DoRA (Weight-Decomposed LoRA)
  - IA3 (Infusion of Adapter for Attention)
  - Prefix Tuning
  - Prompt Tuning
  - LoRA+

### Distributed Training
- **Data Parallelism (DP)**: Multi-GPU data parallel training
- **Tensor Parallelism (TP)**: Split model across GPUs
- **Pipeline Parallelism (PP)**: Pipeline stages for large models
- **Sequence Parallelism (SP)**: Long sequence support
- **FSDP**: Fully Sharded Data Parallel

### Quantization
- **BNB (BitsAndBytes)**: 4/8-bit quantization
- **AWQ**: Activation-aware Weight Quantization
- **GPTQ**: GPTQ quantization
- **HQQ**: Hugging Face Quantization
- **EETQ**: Efficiently Entangled Tensor Quantization
- **FP8**: 8-bit floating point

### Inference Backends
- **Transformers**: Native PyTorch inference
- **vLLM**: High-performance inference engine
- **SGLang**: Fast inference with custom kernels
- **LMDeploy**: Deployment-optimized inference

### Sampling Strategies
- **DDPM**: Standard denoising diffusion probabilistic models
- **Cached DDPM**: Efficient caching for faster sampling
- **Analytic**: Score Entropy Discrete Diffusion (SEDD)
- **Semi-Autoregressive**: Block-wise generation
- **Blockwise**: Confidence-based parallel decoding

### Diffusion Types
- **MDLM**: Masked Diffusion Language Models (NeurIPS 2024)
- **D3PM**: Discrete Denoising Diffusion Probabilistic Models
- **SEDD**: Score Entropy Discrete Diffusion
- **Custom**: Easy integration of new diffusion processes

## Installation

### From PyPI (Coming Soon)
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
- Python 3.10+
- PyTorch 2.0+
- Transformers (for model loading)
- Additional dependencies:
  - `datasets` for HuggingFace datasets
  - `accelerate` for distributed training
  - `deepspeed` for ZeRO optimization
  - `vllm` / `sglang` / `lmdeploy` for inference

## Quick Start

### Training a Diffusion Language Model

```bash
# Basic training with LoRA
hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir output
```

### Generating Samples

```bash
# Generate samples from trained model
hedgehog sample \
    --checkpoint output/final_model.pt \
    --num_samples 5 \
    --seq_len 128 \
    --sampler ddpm_cache
```

### Listing Available Models

```bash
# List available models and datasets
hedgehog list --models --datasets
```

### Starting Inference Server

```bash
# Start OpenAI-compatible API server
hedgehog serve \
    --checkpoint output/final_model.pt \
    --port 8000 \
    --backend transformers
```

## Training on Apple Silicon (Mac Mini, MacBook, etc.)

Hedgehog supports training on Apple Silicon using the MPS (Metal Performance Shaders) backend.

### Installation

```bash
# Create a virtual environment
python3 -m venv ~/hedgehog-env
source ~/hedgehog-env/bin/activate

# Install hedgehog-dlm
pip install hedgehog-dlm
```

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
    --learning_rate 1e-4 \
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

### Minimal Training Command

```bash
hedgehog train \
    --dataset tiny-shakespeare \
    --use_peft \
    --per_device_batch_size 1 \
    --max_seq_len 128 \
    --device mps \
    --output_dir output
```

## Architecture

```
hedgehog/
├── diffusion/       # Core diffusion processes
│   ├── MDLM        # Masked Diffusion Language Models
│   ├── D3PM        # Discrete Denoising Diffusion
│   └── SEDD        # Score Entropy Discrete Diffusion
├── models/          # Model architectures
│   ├── DiT         # Diffusion Transformer
│   ├── AR          # Autoregressive baseline
│   └── Mamba       # State-space models
├── trainers/        # Training loops
├── samplers/        # Sampling strategies
├── data/           # Dataset loaders
├── peft/           # Parameter-efficient fine-tuning
├── distributed/    # Distributed training
├── quantization/   # Model quantization
├── inference/      # Inference engines
├── registry/      # Model & dataset registry
└── cli/           # Command-line interface
```

## Training

### Training Methods

| Method | Description | Memory Usage |
|--------|-------------|---------------|
| Full | Full parameter training | High |
| LoRA | Low-rank adaptation | ~50% |
| QLoRA | Quantized LoRA | ~30% |
| DoRA | Weight-decomposed LoRA | ~50% |
| Prefix | Prefix tuning | ~40% |

### Example Training Scripts

See the `examples/` directory for comprehensive training examples:

- `examples/train/full/` - Full parameter training
- `examples/train/lora/` - LoRA fine-tuning
- `examples/train/qlora/` - QLoRA training
- `examples/infer/` - Inference examples

### Training Configuration

```python
from hedgehog.trainers import TrainerConfig, DiffusionTrainer

config = TrainerConfig(
    model_type="dit",
    vocab_size=32768,
    hidden_size=768,
    num_heads=12,
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

## Inference

### OpenAI-Compatible API

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

Key options:
- `--model_type`: Model architecture (dit, ar, mamba)
- `--dataset`: Path to dataset
- `--use_peft`: Enable PEFT training
- `--peft_type`: PEFT method (lora, dora, ia3, prefix, prompt)
- `--use_quantization`: Enable quantization
- `--num_train_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--output_dir`: Output directory

### list
List available models, datasets, and methods.

```bash
hedgehog list [OPTIONS]
```

Options:
- `--models`: List available models
- `--datasets`: List built-in datasets
- `--training_methods`: List training methods
- `--sampling_methods`: List sampling strategies

### sample
Generate samples from a trained model.

```bash
hedgehog sample [OPTIONS]
```

Key options:
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of samples to generate
- `--seq_len`: Sequence length
- `--sampler`: Sampling method (ddpm, ddpm_cache, semi_ar, blockwise)

### eval
Evaluate a trained model.

```bash
hedgehog eval [OPTIONS]
```

Key options:
- `--checkpoint`: Path to model checkpoint
- `--dataset`: Evaluation dataset
- `--batch_size`: Evaluation batch size

### serve
Start an inference server.

```bash
hedgehog serve [OPTIONS]
```

Key options:
- `--checkpoint`: Path to model checkpoint
- `--backend`: Inference backend (transformers, vllm, sglang, lmdeploy)
- `--port`: Server port
- `--host`: Server host

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
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_batch_size 4 \
    --learning_rate 1e-4 \
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

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 hedgehog train \
    --model_type dit \
    --dataset tiny-shakespeare \
    --use_peft \
    --per_device_batch_size 4 \
    --output_dir output/distributed
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

## Built-in Models

| Model | Vocab Size | Hidden Size | Layers | Description |
|-------|------------|-------------|--------|-------------|
| mdlm-small | 32768 | 256 | 6 | Small MDLM |
| mdlm-base | 32768 | 384 | 12 | Base MDLM |
| mdlm-large | 32768 | 768 | 24 | Large MDLM |
| dit-small | 32768 | 312 | 12 | Small DiT |
| dit-base | 32768 | 768 | 16 | Base DiT |
| dit-large | 32768 | 1024 | 24 | Large DiT |
| char-small | 256 | 128 | 4 | Character-level |
| char-base | 256 | 256 | 8 | Character-level base |

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
@software{hedgehog2025,
  title = {Hedgehog: Scalable Lightweight Infrastructure for Fine-Tuning Diffusion Language Models},
  author = {ArchishmanSengupta},
  year = {2025},
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