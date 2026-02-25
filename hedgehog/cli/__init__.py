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
    """Add training arguments."""
    # Model
    parser.add_argument("--model_type", type=str, default="dit",
                        help="Model type: dit, ar, mamba")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name from registry")
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

    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_batch_size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")

    # Diffusion
    parser.add_argument("--diffusion_type", type=str, default="mdlm",
                        help="Diffusion type: mdlm, d3pm_absorbing, d3pm_uniform")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        help="Noise schedule: linear, cosine, quadratic")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path or name")
    parser.add_argument("--dataset_type", type=str, default="text",
                        help="Dataset type: text, character, huggingface")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for dataset")
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Data cache directory")

    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")


def add_sample_args(parser: argparse.ArgumentParser):
    """Add sampling arguments."""
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

    # Sampling
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length to generate")
    parser.add_argument("--sampler", type=str, default="ddpm_cache",
                        help="Sampler type: ddpm, ddpm_cache, analytic, semi_ar, blockwise")

    # Output
    parser.add_argument("--output", type=str, default="samples.txt",
                        help="Output file for samples")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")


def add_eval_args(parser: argparse.ArgumentParser):
    """Add evaluation arguments."""
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

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path or name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Output directory")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")


def add_serve_args(parser: argparse.ArgumentParser):
    """Add serve arguments."""
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


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(args)


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

    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.dataset)
        tokenizer = TokenizerWrapper(tokenizer)
    except:
        logger.warning("Could not load tokenizer from HuggingFace, using simple character tokenizer")

    # Create dataset
    if tokenizer:
        try:
            from datasets import load_dataset
            dataset = load_dataset(args.dataset)
            train_data = [item["text"] for item in dataset["train"]]
            train_dataset = create_dataset(
                dataset_type=args.dataset_type,
                texts=train_data[:1000],  # Limit for demo
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
        except:
            train_dataset = None
    else:
        train_dataset = None

    # Fallback dataset
    if train_dataset is None:
        sample_texts = [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "diffusion models are awesome",
        ] * 100
        train_dataset = create_dataset(
            dataset_type="character",
            texts=sample_texts,
            max_length=args.max_length,
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

    # Create config
    config = TrainerConfig(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        num_train_epochs=args.num_train_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        diffusion_type=args.diffusion_type,
        num_timesteps=args.num_timesteps,
        noise_schedule=args.noise_schedule,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        device=str(device),
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
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

    # Setup logging
    logger = setup_logging()
    logger.info("Starting sampling...")

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
    checkpoint = safe_load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create diffusion
    diffusion = create_diffusion(
        diffusion_type="mdlm",
        vocab_size=args.vocab_size,
        num_timesteps=1000,
    )

    # Get mask token id
    mask_token_id = args.vocab_size - 1

    # Create sampler
    sampler = create_sampler(
        sampler_type=args.sampler,
        diffusion=diffusion,
        model=model,
        mask_token_id=mask_token_id,
    )

    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = sampler.sample(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        device=device,
    )

    # Decode and save
    logger.info(f"Saving samples to {args.output}")
    with open(args.output, "w") as f:
        for i, sample in enumerate(samples):
            text = tokenizer.decode(sample.tolist()) if hasattr(tokenizer, 'decode') else str(sample.tolist())
            f.write(f"Sample {i+1}: {text}\n")

    logger.info("Sampling complete!")


def run_eval(args: argparse.Namespace):
    """Run evaluation."""
    from ..models import create_model
    from ..diffusion import create_diffusion
    from ..utils import setup_logging, get_device, safe_load_checkpoint
    from ..data import create_dataset
    import torch

    # Setup logging
    logger = setup_logging()
    logger.info("Starting evaluation...")

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
    checkpoint = safe_load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create diffusion
    diffusion = create_diffusion(
        diffusion_type="mdlm",
        vocab_size=args.vocab_size,
        num_timesteps=1000,
    )

    # Load dataset
    from ..data import TokenizerWrapper
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.dataset)
        tokenizer = TokenizerWrapper(tokenizer)
    except:
        tokenizer = None
        logger.warning("Could not load tokenizer, using simple text data")

    if tokenizer:
        try:
            from datasets import load_dataset
            dataset = load_dataset(args.dataset)
            eval_data = [item["text"] for item in dataset.get("test", dataset.get("validation", []))[:100]]
        except:
            eval_data = ["sample text for evaluation"] * 100
    else:
        eval_data = ["hello world", "test data", "evaluation sample"] * 33

    eval_dataset = create_dataset(
        dataset_type="text",
        texts=eval_data,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )

    # Evaluate
    total_loss = 0.0
    num_batches = 0

    mask_token_id = args.vocab_size - 1
    from torch.utils.data import DataLoader

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size or 8, shuffle=False)

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            x_0 = batch["input_ids"].to(device)
            batch_size = x_0.shape[0]

            # Sample random timesteps
            t = torch.randint(
                0,
                1000,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            # Compute loss
            loss = diffusion.compute_loss(model, x_0, t, mask_token_id)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Evaluation complete!")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Perplexity: {perplexity:.4f}")

    # Save results
    import json
    results = {
        "checkpoint": args.checkpoint,
        "loss": avg_loss,
        "perplexity": perplexity,
    }
    output_path = args.output_dir or "eval_results.json"
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

    # Create inference backend
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name or "gpt2")
    except:
        logger.warning("Could not load tokenizer")

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
