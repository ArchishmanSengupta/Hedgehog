"""
Data loading utilities for diffusion language models.

Supports:
- Text datasets with character-level and subword tokenization
- HuggingFace datasets integration
- Custom datasets
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import json
import logging


class TextDataset(Dataset):
    """Simple text dataset for diffusion language modeling.

    Supports character-level and token-level tokenization.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        return_mask: bool = True,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer (character-level or subword)
            max_length: Maximum sequence length
            return_mask: Whether to return attention mask
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_mask = return_mask

        # Tokenize all texts
        self.encodings = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.encodings[idx]

        result = {
            "input_ids": encoding["input_ids"],
        }

        if self.return_mask:
            result["attention_mask"] = encoding["attention_mask"]

        return result


class CharacterDataset(Dataset):
    """Character-level dataset for diffusion language modeling.

    Useful for small-scale experiments like tiny-diffusion.
    """

    def __init__(
        self,
        texts: List[str],
        max_length: int = 512,
        padding_token_id: int = 0,
        mask_token_id: Optional[int] = None,
    ):
        """
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding_token_id: ID for padding token
            mask_token_id: ID for mask token (last in vocab)
        """
        self.texts = texts
        self.max_length = max_length

        # Build character vocabulary
        chars = set()
        for text in texts:
            chars.update(text)
        self.chars = sorted(list(chars))
        self.char_to_id = {c: i + 2 for i, c in enumerate(self.chars)}  # Reserve 0 for padding, 1 for mask
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.char_to_id['<pad>'] = 0
        self.char_to_id['<mask>'] = mask_token_id or (len(self.chars) + 2)
        self.padding_token_id = padding_token_id
        self.mask_token_id = self.char_to_id['<mask>']

        # Pre-tokenize
        self.encodings = []
        for text in texts:
            ids = [self.char_to_id.get(c, 0) for c in text[:max_length]]
            # Pad to max_length
            if len(ids) < max_length:
                ids += [padding_token_id] * (max_length - len(ids))
            self.encodings.append(torch.tensor(ids))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings[idx],
        }


class HuggingFaceDataset(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        max_length: int = 512,
        text_column: str = "text",
        tokenizer: Any = None,
    ):
        """
        Args:
            dataset_path: Path to HuggingFace dataset or local path
            split: Dataset split (train, test, etc.)
            max_length: Maximum sequence length
            text_column: Name of text column
            tokenizer: Tokenizer for processing (required for proper dataset)
        """
        from datasets import load_dataset

        # Load the dataset
        self.dataset = load_dataset(dataset_path, split=split)
        self.max_length = max_length
        self.text_column = text_column
        self.tokenizer = tokenizer

        # If no tokenizer provided, we need to handle it differently
        if tokenizer is None:
            # Build a simple character-level tokenizer fallback
            self._build_char_tokenizer()

    def _build_char_tokenizer(self):
        """Build simple character-level tokenizer if no tokenizer provided."""
        # Just use character-level encoding
        self.use_char_tokenizer = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        text = item[self.text_column]

        if self.tokenizer is not None:
            # Use provided tokenizer
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        else:
            # Fallback: simple character-level tokenization
            # This is a simple implementation - users should provide a tokenizer
            chars = list(text)[:self.max_length]
            if len(chars) < self.max_length:
                chars = chars + ['<pad>'] * (self.max_length - len(chars))
            # Return text for later processing
            return {
                "text": text,
            }


class StreamingDataset(Dataset):
    """Streaming dataset for large-scale training.

    Loads data on-the-fly to save memory.
    Uses efficient line indexing to avoid O(n²) complexity.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        max_length: int = 512,
        file_format: str = "jsonl",
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            file_path: Path to data file
            tokenizer: Tokenizer for processing
            max_length: Maximum sequence length
            file_format: Format of data file (jsonl, txt)
            num_samples: Number of samples to load (for streaming, we need to limit)
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_format = file_format

        # Build line index for efficient random access
        self._build_line_index(num_samples)

    def _build_line_index(self, num_samples: Optional[int] = None):
        """Build an index of line offsets for efficient access."""
        self.line_offsets = [0]
        with open(self.file_path, 'r', encoding='utf-8') as f:
            position = 0
            for line in f:
                position += len(line.encode('utf-8'))
                self.line_offsets.append(position)
                if num_samples and len(self.line_offsets) > num_samples:
                    break

        self.num_lines = len(self.line_offsets) - 1

    def __len__(self) -> int:
        return self.num_lines

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= self.num_lines:
            raise IndexError(f"Index {idx} out of range")

        # Seek to the correct position
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()

        if self.file_format == "jsonl":
            item = json.loads(line)
            text = item.get("text", "")
        else:
            text = line.strip()

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_dataset(
    dataset_type: str,
    texts: Optional[list] = None,
    max_length: int = 512,
    tokenizer: Any = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    file_path: Optional[str] = None,
    file_format: str = "jsonl",
    **kwargs,
) -> Dataset:
    """Factory function to create datasets.

    Args:
        dataset_type: Type of dataset (text, character, huggingface, streaming)
        texts: List of text samples (for text and character datasets)
        max_length: Maximum sequence length
        tokenizer: Tokenizer for processing
        dataset_path: Path to HuggingFace dataset (for huggingface dataset)
        split: Dataset split (for huggingface dataset)
        text_column: Name of text column (for huggingface dataset)
        file_path: Path to data file (for streaming dataset)
        file_format: Format of data file (for streaming dataset)
        **kwargs: Additional arguments for specific dataset types
    """
    dataset_map = {
        "text": TextDataset,
        "character": CharacterDataset,
        "huggingface": HuggingFaceDataset,
        "streaming": StreamingDataset,
    }

    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(dataset_map.keys())}")

    # Build kwargs based on dataset type
    # Only pass tokenizer to datasets that accept it
    dataset_kwargs = {"max_length": max_length}

    # For text dataset, tokenizer is required - fallback to character if not provided
    if dataset_type == "text":
        if tokenizer is None:
            # Automatically fallback to character-level dataset
            logger = logging.getLogger(__name__)
            logger.warning("No tokenizer provided for text dataset, falling back to character-level dataset")
            dataset_type = "character"
        else:
            dataset_kwargs["tokenizer"] = tokenizer

    if dataset_type in ["huggingface", "streaming"]:
        dataset_kwargs["tokenizer"] = tokenizer

    if dataset_type == "text" or dataset_type == "character":
        if texts is not None:
            dataset_kwargs["texts"] = texts
        else:
            raise ValueError(f"Dataset type '{dataset_type}' requires 'texts' parameter")
    elif dataset_type == "huggingface":
        if dataset_path is not None:
            dataset_kwargs["dataset_path"] = dataset_path
            dataset_kwargs["split"] = split
            dataset_kwargs["text_column"] = text_column
        else:
            raise ValueError("Dataset type 'huggingface' requires 'dataset_path' parameter")
    elif dataset_type == "streaming":
        if file_path is not None:
            dataset_kwargs["file_path"] = file_path
            dataset_kwargs["file_format"] = file_format
        else:
            raise ValueError("Dataset type 'streaming' requires 'file_path' parameter")

    # Add any additional kwargs
    dataset_kwargs.update(kwargs)

    return dataset_map[dataset_type](**dataset_kwargs)


class TokenizerWrapper:
    """Wrapper for tokenizers to provide consistent interface."""

    def __init__(self, tokenizer: Any, add_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) or tokenizer.pad_token_id
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None) or (len(tokenizer) - 1)

    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, add_special_tokens=self.add_special_tokens, **kwargs)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
