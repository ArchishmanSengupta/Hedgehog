"""Tests for CLI module."""

import pytest
import argparse
from hedgehog.cli import (
    create_parser,
    add_train_args,
    add_sample_args,
    add_eval_args,
    add_list_args,
    add_serve_args,
)


class TestCreateParser:
    """Test create_parser function."""

    def test_parser_creation(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_subcommands(self):
        parser = create_parser()
        # Parser should have subparsers
        assert parser._subparsers is not None


class TestAddTrainArgs:
    """Test add_train_args function."""

    def test_train_args_exist(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        # Parse with some args
        args = parser.parse_args([
            "--model_type", "dit",
            "--dataset", "tiny-shakespeare",
            "--output_dir", "test_output",
        ])

        assert args.model_type == "dit"
        assert args.dataset == "tiny-shakespeare"
        assert args.output_dir == "test_output"

    def test_train_default_values(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        args = parser.parse_args(["--dataset", "test"])

        # Check some defaults
        assert args.model_type == "dit"
        assert args.vocab_size == 32768
        assert args.hidden_size == 384
        assert args.num_heads == 6
        assert args.num_layers == 12
        assert args.learning_rate == 1e-4
        assert args.num_train_epochs == 3

    def test_new_configurable_args(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        args = parser.parse_args([
            "--dataset", "test",
            "--warmup_start_factor", "0.2",
            "--warmup_end_factor", "0.8",
            "--mask_token_id", "42",
        ])

        assert args.warmup_start_factor == 0.2
        assert args.warmup_end_factor == 0.8
        assert args.mask_token_id == 42


class TestAddSampleArgs:
    """Test add_sample_args function."""

    def test_sample_args_exist(self):
        parser = argparse.ArgumentParser()
        add_sample_args(parser)

        args = parser.parse_args([
            "--checkpoint", "test.pt",
            "--num_samples", "5",
            "--seq_len", "128",
        ])

        assert args.checkpoint == "test.pt"
        assert args.num_samples == 5
        assert args.seq_len == 128


class TestAddEvalArgs:
    """Test add_eval_args function."""

    def test_eval_args_exist(self):
        parser = argparse.ArgumentParser()
        add_eval_args(parser)

        args = parser.parse_args([
            "--checkpoint", "test.pt",
            "--dataset", "tiny-shakespeare",
        ])

        assert args.checkpoint == "test.pt"
        assert args.dataset == "tiny-shakespeare"


class TestAddListArgs:
    """Test add_list_args function."""

    def test_list_args_exist(self):
        parser = argparse.ArgumentParser()
        add_list_args(parser)

        args = parser.parse_args(["--models"])
        assert args.models == True

    def test_list_multiple_args(self):
        parser = argparse.ArgumentParser()
        add_list_args(parser)

        args = parser.parse_args(["--models", "--datasets", "--training_methods"])
        assert args.models == True
        assert args.datasets == True
        assert args.training_methods == True


class TestAddServeArgs:
    """Test add_serve_args function."""

    def test_serve_args_exist(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)

        args = parser.parse_args([
            "--checkpoint", "test.pt",
            "--host", "0.0.0.0",
            "--port", "8000",
        ])

        assert args.checkpoint == "test.pt"
        assert args.host == "0.0.0.0"
        assert args.port == 8000


class TestTrainArgs:
    """Test complete training argument parsing."""

    def test_all_peft_args(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        args = parser.parse_args([
            "--dataset", "test",
            "--use_peft",
            "--peft_type", "lora",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--lora_dropout", "0.1",
        ])

        assert args.use_peft == True
        assert args.peft_type == "lora"
        assert args.lora_r == 16
        assert args.lora_alpha == 32
        assert args.lora_dropout == 0.1

    def test_quantization_args(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        args = parser.parse_args([
            "--dataset", "test",
            "--use_quantization",
            "--quant_type", "bnb",
            "--quant_bits", "4",
        ])

        assert args.use_quantization == True
        assert args.quant_type == "bnb"
        assert args.quant_bits == 4

    def test_lr_scheduler_args(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        args = parser.parse_args([
            "--dataset", "test",
            "--lr_scheduler", "cosine",
            "--min_lr", "1e-7",
            "--warmup_steps", "1000",
        ])

        assert args.lr_scheduler == "cosine"
        assert args.min_lr == 1e-7
        assert args.warmup_steps == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
