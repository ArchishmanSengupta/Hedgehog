#!/usr/bin/env python3
"""
Inference example for Diffusion Language Models.
This script demonstrates generating text from a trained DLM.
"""

import torch
from hedgehog.models import create_model
from hedgehog.diffusion import create_diffusion
from hedgehog.samplers import create_sampler
from hedgehog.utils import safe_load_checkpoint


def generate_text(
    checkpoint_path: str,
    prompt: str = "",
    num_samples: int = 5,
    seq_len: int = 128,
    sampler_type: str = "ddpm_cache",
    device: str = "auto",
):
    """Generate text from a trained DLM."""

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    # Model configuration
    vocab_size = 32768
    hidden_size = 384
    num_heads = 6
    num_layers = 12
    max_seq_len = 512

    # Create model
    print("Creating model...")
    model = create_model(
        model_type="dit",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create diffusion process
    print("Creating diffusion process...")
    diffusion = create_diffusion(
        diffusion_type="mdlm",
        vocab_size=vocab_size,
        num_timesteps=1000,
    )

    # Create sampler
    mask_token_id = vocab_size - 1
    sampler = create_sampler(
        sampler_type=sampler_type,
        diffusion=diffusion,
        model=model,
        mask_token_id=mask_token_id,
    )

    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples = sampler.sample(
        num_samples=num_samples,
        seq_len=seq_len,
        device=device,
    )

    # Convert to text (simplified)
    print("\n" + "=" * 50)
    print("Generated samples:")
    print("=" * 50)
    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print(f"  Tokens: {sample[:20].tolist()}...")  # Print first 20 tokens

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from DLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--sampler", type=str, default="ddpm_cache",
                        help="Sampler type")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")

    args = parser.parse_args()

    generate_text(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        sampler_type=args.sampler,
        device=args.device,
    )
