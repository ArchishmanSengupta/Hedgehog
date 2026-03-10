"""
CLI interface for hedgehog.

Inspired by ms-swift but simplified for diffusion language models.
Commands:
- train: Train a diffusion language model
- sample: Generate samples from a trained model
- eval: Evaluate a model
- serve: Serve model for inference
"""

import argparse
import json
import sys
from typing import Optional, Dict, Any


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="hedgehog",
        description="Diffusion Language Model Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a diffusion language model")
    add_train_args(train_parser)

    # List command
    list_parser = subparsers.add_parser("list", help="List available models and datasets")
    add_list_args(list_parser)

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate samples")
    add_sample_args(sample_parser)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    add_eval_args(eval_parser)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve model for inference")
    add_serve_args(serve_parser)

    return parser


def _add_config_arg(parser: argparse.ArgumentParser):
    """Add --config argument to a parser."""
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML or JSON config file. CLI args override config file values.")


TRAINER_CONFIG_HELP = {
    "model_type": "Model type: dit, ar, mamba",
    "vocab_size": "Vocabulary size",
    "num_train_epochs": "Number of training epochs",
    "per_device_batch_size": "Batch size per device",
    "gradient_accumulation_steps": "Gradient accumulation steps",
    "learning_rate": "Learning rate",
    "weight_decay": "Weight decay",
    "max_grad_norm": "Maximum gradient norm",
    "warmup_steps": "Warmup steps",
    "warmup_start_factor": "Warmup starting LR factor",
    "warmup_end_factor": "Warmup ending LR factor",
    "lr_scheduler_type": "LR scheduler: linear, cosine, constant",
    "min_lr": "Minimum learning rate for cosine scheduler",
    "use_amp": "Use automatic mixed precision (AMP)",
    "amp_dtype": "AMP dtype: float16, bfloat16",
    "hidden_size": "Hidden size",
    "num_heads": "Number of attention heads",
    "num_layers": "Number of layers",
    "max_seq_len": "Maximum sequence length",
    "dropout": "Dropout probability",
    "mask_token_id": "Mask token ID (defaults to vocab_size - 1)",
    "diffusion_type": "Diffusion type: mdlm, d3pm_absorbing, d3pm_uniform",
    "num_timesteps": "Number of diffusion timesteps",
    "noise_schedule": "Noise schedule: linear, cosine, quadratic",
    "output_dir": "Output directory",
    "logging_steps": "Log every N steps",
    "save_steps": "Save checkpoint every N steps",
    "save_total_limit": "Maximum number of checkpoints to keep",
    "eval_steps": "Evaluate every N steps",
    "device": "Device: auto, cpu, cuda, mps",
    "seed": "Random seed",
    "dataloader_num_workers": "Number of dataloader workers",
    "resume_from_checkpoint": "Resume from checkpoint path",
    "load_args_from_checkpoint": "Load training args from checkpoint",
}


def _add_trainer_config_args(parser: argparse.ArgumentParser,
                             include: Optional[set] = None,
                             exclude: Optional[set] = None):
    """Auto-generate argparse args from TrainerConfig dataclass fields.

    This is the single source of truth: TrainerConfig fields drive CLI args,
    eliminating duplication and drift risk.
    """
    from .._compat import get_trainer_config_fields
    for name, ftype, default in get_trainer_config_fields():
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue

        help_text = TRAINER_CONFIG_HELP.get(name, name.replace("_", " ").capitalize())

        if ftype is bool:
            if default is True:
                parser.add_argument(f"--{name}", default=True,
                                    action=argparse.BooleanOptionalAction, help=help_text)
            else:
                parser.add_argument(f"--{name}", action="store_true", help=help_text)
        elif ftype is Optional[int] or (default is None and "int" in str(ftype)):
            parser.add_argument(f"--{name}", type=int, default=None, help=help_text)
        elif ftype is Optional[str] or (default is None and "str" in str(ftype)):
            parser.add_argument(f"--{name}", type=str, default=None, help=help_text)
        elif ftype is int:
            parser.add_argument(f"--{name}", type=int, default=default, help=help_text)
        elif ftype is float:
            parser.add_argument(f"--{name}", type=float, default=default, help=help_text)
        elif ftype is str:
            parser.add_argument(f"--{name}", type=str, default=default, help=help_text)
        else:
            parser.add_argument(f"--{name}", default=default, help=help_text)


def add_list_args(parser: argparse.ArgumentParser):
    """Add list arguments."""
    parser.add_argument("--models", action="store_true",
                        help="List available models")
    parser.add_argument("--datasets", action="store_true",
                        help="List available datasets")
    parser.add_argument("--training_methods", action="store_true",
                        help="List training methods")
    parser.add_argument("--sampling_methods", action="store_true",
                        help="List sampling methods")


def add_train_args(parser: argparse.ArgumentParser):
    """Add training arguments.

    TrainerConfig fields are auto-generated from the dataclass (single source of truth).
    PEFT, quantization, distributed, and dataset args are added manually.
    """
    _add_config_arg(parser)

    _add_trainer_config_args(parser, exclude={"load_args_from_checkpoint"})

    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name from registry")

    parser.add_argument("--lr_scheduler", type=str, default=None,
                        help="Alias for --lr_scheduler_type")

    # PEFT
    parser.add_argument("--use_peft", action="store_true",
                        help="Use PEFT for efficient training")
    parser.add_argument("--peft_type", type=str, default="lora",
                        help="PEFT type: lora, dora, ia3, prefix, prompt")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Quantization
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for training")
    parser.add_argument("--quant_type", type=str, default="bnb",
                        help="Quantization type: bnb, awq, gptq")
    parser.add_argument("--quant_bits", type=int, default=4,
                        help="Quantization bits")

    # Distributed
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Pipeline parallel size")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path or name")
    parser.add_argument("--dataset_type", type=str, default="text",
                        help="Dataset type: text, character, huggingface")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for dataset")
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Data cache directory")

    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples to use (None for all)")
    parser.add_argument("--load_args", action="store_true",
                        help="Load training args from checkpoint's args.json when resuming")


def add_sample_args(parser: argparse.ArgumentParser):
    """Add sampling arguments."""
    _add_config_arg(parser)

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="dit",
                        help="Model type")
    parser.add_argument("--vocab_size", type=int, default=32768,
                        help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")

    # Diffusion
    parser.add_argument("--diffusion_type", type=str, default="mdlm",
                        help="Diffusion type: mdlm, d3pm, d3pm_absorbing, d3pm_uniform, sedd")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        help="Noise schedule: linear, cosine, quadratic")
    parser.add_argument("--mask_token_id", type=int, default=None,
                        help="Mask token ID (defaults to vocab_size - 1)")

    # Sampling
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length to generate")
    parser.add_argument("--sampler", type=str, default="ddpm_cache",
                        help="Sampler type: ddpm, ddpm_cache, analytic, semi_ar, blockwise")

    # Tokenizer
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name or path for decoding output")
    parser.add_argument("--load_args", default=True, action=argparse.BooleanOptionalAction,
                        help="Load training args from checkpoint (default: True for sample/eval)")

    # Output
    parser.add_argument("--output", type=str, default="samples.txt",
                        help="Output file for samples")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")


def add_eval_args(parser: argparse.ArgumentParser):
    """Add evaluation arguments."""
    _add_config_arg(parser)

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="dit",
                        help="Model type")
    parser.add_argument("--vocab_size", type=int, default=32768,
                        help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")

    # Diffusion
    parser.add_argument("--diffusion_type", type=str, default="mdlm",
                        help="Diffusion type: mdlm, d3pm, d3pm_absorbing, d3pm_uniform, sedd")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        help="Noise schedule: linear, cosine, quadratic")
    parser.add_argument("--mask_token_id", type=int, default=None,
                        help="Mask token ID (defaults to vocab_size - 1)")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path or name")
    parser.add_argument("--per_device_batch_size", type=int, default=8,
                        help="Batch size per device for evaluation")
    parser.add_argument("--load_args", default=True, action=argparse.BooleanOptionalAction,
                        help="Load training args from checkpoint (default: True for sample/eval)")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Output directory")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")


def add_serve_args(parser: argparse.ArgumentParser):
    """Add serve arguments."""
    _add_config_arg(parser)

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="dit",
                        help="Model type")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32768,
                        help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")

    # Inference
    parser.add_argument("--backend", type=str, default="transformers",
                        help="Inference backend: transformers, vllm, sglang, lmdeploy")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")


def _load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file and return as dict."""
    path_lower = path.lower()
    if path_lower.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if data else {}
    elif path_lower.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .yaml, .yml, or .json")



def _extract_config_path(argv: list) -> Optional[str]:
    """Extract --config path from argv without consuming other args."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _config_to_argv(config_dict: Dict[str, Any], sub_parser: Optional[argparse.ArgumentParser] = None) -> list:
    """Convert config dict to argv-style list for injection before CLI args.

    Only injects keys that correspond to known argparse destinations.
    Handles bool (store_true) actions correctly.
    """
    known_actions = {}
    if sub_parser:
        for action in sub_parser._actions:
            if action.dest != "help":
                known_actions[action.dest] = action

    injected = []
    for key, value in config_dict.items():
        if key == "config":
            continue
        action = known_actions.get(key)
        if action is None:
            continue

        if isinstance(action, argparse._StoreTrueAction):
            if value is True:
                injected.append(f"--{key}")
        elif isinstance(action, argparse._StoreFalseAction):
            if value is False:
                injected.append(f"--{key}")
        else:
            if value is not None:
                injected.append(f"--{key}")
                injected.append(str(value))
    return injected


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments with config file support.

    Priority: CLI args > config file > defaults.

    Uses argv injection (ms-swift approach): config file values are injected
    into argv *before* the actual CLI args, so argparse's natural
    "last occurrence wins" behavior ensures CLI args always take priority.
    This correctly handles the case where a user explicitly passes a value
    that matches the default.
    """
    parser = create_parser()
    raw_args = args if args is not None else sys.argv[1:]

    config_path = _extract_config_path(raw_args)
    if config_path is not None:
        config_dict = _load_config_file(config_path)

        temp_args, _ = parser.parse_known_args(raw_args)
        sub_parser = None
        if temp_args.command:
            for action in parser._subparsers._actions:
                if isinstance(action, argparse._SubParsersAction):
                    sub_parser = action.choices.get(temp_args.command)
                    break

        config_argv = _config_to_argv(config_dict, sub_parser)

        merged_argv = []
        command_found = False
        for a in raw_args:
            merged_argv.append(a)
            if not command_found and not a.startswith("-"):
                command_found = True
                merged_argv.extend(config_argv)

        cli_args = parser.parse_args(merged_argv)
    else:
        cli_args = parser.parse_args(raw_args)

    return cli_args


def run_list(args: argparse.Namespace):
    """List available models, datasets, etc."""
    from ..registry import (
        list_models, list_datasets, list_training_methods, list_sampling_methods,
    )

    if args.models or not any([args.datasets, args.training_methods, args.sampling_methods]):
        print("\nAvailable DLM Models:")
        print("-" * 40)
        for model_name in list_models():
            print(f"  - {model_name}")

    if args.datasets:
        print("\nAvailable Built-in Datasets:")
        print("-" * 40)
        for dataset_name in list_datasets():
            print(f"  - {dataset_name}")

    if args.training_methods:
        print("\nAvailable Training Methods:")
        print("-" * 40)
        for method_name in list_training_methods():
            print(f"  - {method_name}")

    if args.sampling_methods:
        print("\nAvailable Sampling Methods:")
        print("-" * 40)
        for method_name in list_sampling_methods():
            print(f"  - {method_name}")


def run_train(args: argparse.Namespace):
    """Run training."""
    from ..trainers import TrainerConfig, DiffusionTrainer
    from ..models import create_model
    from ..data import create_dataset, TokenizerWrapper
    from ..utils import setup_logging, get_device
    from ..peft import create_peft_model, LoraConfig
    from ..quantization import quantize_model, QuantConfig

    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting training...")

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for training. "
            "Install with: pip install transformers"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.dataset)
        tokenizer = TokenizerWrapper(tokenizer)
    except Exception:
        logger.warning("Could not load tokenizer from HuggingFace, using simple character tokenizer")

    if tokenizer:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for loading HuggingFace datasets. "
                "Install with: pip install datasets"
            )
        try:
            dataset = load_dataset(args.dataset)
            train_data = [item["text"] for item in dataset["train"]]
            if args.max_train_samples is not None:
                train_data = train_data[:args.max_train_samples]
            train_dataset = create_dataset(
                dataset_type="text",
                texts=train_data,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
        except Exception as e:
            logger.warning(f"Failed to load dataset '{args.dataset}': {e}")
            train_dataset = None
    else:
        train_dataset = None

    if train_dataset is None:
        raise ValueError(
            f"Could not load dataset '{args.dataset}'. "
            "Ensure the dataset path is valid and a tokenizer is available."
        )

    # Create base model
    logger.info(f"Creating model: {args.model_type}")
    model = create_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    # Apply PEFT if requested
    if args.use_peft:
        logger.info(f"Applying PEFT: {args.peft_type}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        from ..peft import LoraModel
        model = LoraModel(model, lora_config)

    # Apply quantization if requested
    if args.use_quantization:
        logger.info(f"Applying quantization: {args.quant_type}")
        quant_config = QuantConfig(
            quant_type=args.quant_type,
            bits=args.quant_bits,
        )
        model = quantize_model(model, quant_config)

    if getattr(args, "lr_scheduler", None) is not None:
        args.lr_scheduler_type = args.lr_scheduler

    config_kwargs = {}
    from dataclasses import fields as dc_fields
    for f in dc_fields(TrainerConfig):
        if hasattr(args, f.name):
            config_kwargs[f.name] = getattr(args, f.name)
    config_kwargs["device"] = str(device)
    config = TrainerConfig(**config_kwargs)
    config = config.apply_context_defaults(
        use_peft=args.use_peft,
        peft_type=args.peft_type if args.use_peft else None,
    )

    # Create trainer
    trainer = DiffusionTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()
    logger.info("Training complete!")


def run_sample(args: argparse.Namespace):
    """Run sampling."""
    from ..models import create_model
    from ..diffusion import create_diffusion
    from ..samplers import create_sampler
    from ..utils import setup_logging, get_device, safe_load_checkpoint
    from ..trainers import TrainerConfig

    logger = setup_logging()
    logger.info("Starting sampling...")

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    checkpoint = safe_load_checkpoint(args.checkpoint, device)

    if args.load_args:
        saved_config = checkpoint.get("config")
        if not saved_config:
            from pathlib import Path as _Path
            args_path = _Path(args.checkpoint).parent / "args.json"
            if args_path.exists():
                saved_config = json.loads(args_path.read_text())
        if saved_config:
            from ..trainers import FORCE_LOAD_KEYS, LOAD_KEYS
            for key in FORCE_LOAD_KEYS:
                if key in saved_config and saved_config[key] is not None:
                    setattr(args, key, saved_config[key])
            for key in LOAD_KEYS:
                if key in saved_config and saved_config[key] is not None:
                    if getattr(args, key, None) is None:
                        setattr(args, key, saved_config[key])
            logger.info("Loaded model config from checkpoint (selective keys)")

    model = create_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    diffusion = create_diffusion(
        diffusion_type=args.diffusion_type,
        vocab_size=args.vocab_size,
        num_timesteps=args.num_timesteps,
        schedule=args.noise_schedule,
    )

    mask_token_id = args.mask_token_id if args.mask_token_id is not None else args.vocab_size - 1

    sampler = create_sampler(
        sampler_type=args.sampler,
        diffusion=diffusion,
        model=model,
        mask_token_id=mask_token_id,
    )

    tokenizer = None
    if args.tokenizer_name:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for tokenizer loading. "
                "Install with: pip install transformers"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer '{args.tokenizer_name}': {e}")

    logger.info(f"Generating {args.num_samples} samples...")
    samples = sampler.sample(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        device=device,
    )

    logger.info(f"Saving samples to {args.output}")
    with open(args.output, "w") as f:
        for i, sample in enumerate(samples):
            if tokenizer is not None:
                text = tokenizer.decode(sample.tolist(), skip_special_tokens=True)
            else:
                text = str(sample.tolist())
            f.write(f"Sample {i+1}: {text}\n")

    logger.info("Sampling complete!")


def run_eval(args: argparse.Namespace):
    """Run evaluation."""
    from ..models import create_model
    from ..diffusion import create_diffusion
    from ..utils import setup_logging, get_device, safe_load_checkpoint
    from ..data import create_dataset
    from ..trainers import TrainerConfig
    import torch
    import os

    logger = setup_logging()
    logger.info("Starting evaluation...")

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    checkpoint = safe_load_checkpoint(args.checkpoint, device)

    if args.load_args:
        saved_config = checkpoint.get("config")
        if not saved_config:
            from pathlib import Path as _Path
            args_path = _Path(args.checkpoint).parent / "args.json"
            if args_path.exists():
                saved_config = json.loads(args_path.read_text())
        if saved_config:
            from ..trainers import FORCE_LOAD_KEYS, LOAD_KEYS
            for key in FORCE_LOAD_KEYS:
                if key in saved_config and saved_config[key] is not None:
                    setattr(args, key, saved_config[key])
            for key in LOAD_KEYS:
                if key in saved_config and saved_config[key] is not None:
                    if getattr(args, key, None) is None:
                        setattr(args, key, saved_config[key])
            logger.info("Loaded model config from checkpoint (selective keys)")

    model = create_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    diffusion = create_diffusion(
        diffusion_type=args.diffusion_type,
        vocab_size=args.vocab_size,
        num_timesteps=args.num_timesteps,
        schedule=args.noise_schedule,
    )

    from ..data import TokenizerWrapper
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for evaluation. "
            "Install with: pip install transformers"
        )
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.dataset)
        tokenizer = TokenizerWrapper(tokenizer)
    except Exception:
        logger.warning("Could not load tokenizer from dataset name, using character-level fallback")

    eval_data = None
    if tokenizer:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for loading HuggingFace datasets. "
                "Install with: pip install datasets"
            )
        try:
            hf_dataset = load_dataset(args.dataset)
            split = hf_dataset.get("test") or hf_dataset.get("validation")
            if split is not None:
                eval_data = [item["text"] for item in split]
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace dataset: {e}")

    if eval_data is None:
        raise ValueError(
            f"Could not load evaluation dataset '{args.dataset}'. "
            "Ensure the dataset path is valid and a tokenizer is available."
        )

    eval_dataset = create_dataset(
        dataset_type="text",
        texts=eval_data,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )

    total_loss = 0.0
    num_batches = 0

    mask_token_id = args.mask_token_id if args.mask_token_id is not None else args.vocab_size - 1
    from torch.utils.data import DataLoader

    dataloader = DataLoader(eval_dataset, batch_size=args.per_device_batch_size, shuffle=False)

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            x_0 = batch["input_ids"].to(device)
            batch_size = x_0.shape[0]

            t = torch.randint(
                0,
                args.num_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            loss = diffusion.compute_loss(model, x_0, t, mask_token_id)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Evaluation complete!")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Perplexity: {perplexity:.4f}")

    results = {
        "checkpoint": args.checkpoint,
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_timesteps": args.num_timesteps,
        "diffusion_type": args.diffusion_type,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def run_serve(args: argparse.Namespace):
    """Run model server."""
    from ..models import create_model
    from ..utils import setup_logging, get_device, safe_load_checkpoint
    from ..inference import InferenceConfig, create_inference_backend, OpenAICompatibleServer
    import torch

    # Setup logging
    logger = setup_logging()
    logger.info("Starting inference server...")

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model = create_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )

    # Load checkpoint
    if args.checkpoint:
        checkpoint = safe_load_checkpoint(args.checkpoint, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    model.to(device)
    model.eval()

    # Create inference config
    inference_config = InferenceConfig(
        backend=args.backend,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    tokenizer = None
    if args.model_name:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for tokenizer loading. "
                "Install with: pip install transformers"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer for '{args.model_name}': {e}")
    else:
        logger.warning("No --model_name provided, running without tokenizer")

    backend = create_inference_backend(model, inference_config, tokenizer)

    # Create and run server
    server = OpenAICompatibleServer(
        backend=backend,
        host=args.host,
        port=args.port,
    )

    logger.info(f"Server starting on {args.host}:{args.port}")
    server.run()


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "list":
        run_list(args)
    elif args.command == "sample":
        run_sample(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


# For direct execution
if __name__ == "__main__":
    main()
