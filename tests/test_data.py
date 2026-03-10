"""Tests for data module."""

import pytest
import torch
from hedgehog.data import (
    create_dataset,
    TextDataset,
    CharacterDataset,
    HuggingFaceDataset,
    StreamingDataset,
    TokenizerWrapper,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.mask_token_id = 3

    def __call__(self, text, max_length=512, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True):
        if isinstance(text, str):
            text = [text]
        ids = [[i % self.vocab_size for i in range(len(t))] for t in text]
        # Pad to max_length
        for i in range(len(ids)):
            if len(ids[i]) < max_length:
                ids[i] = ids[i] + [self.pad_token_id] * (max_length - len(ids[i]))
            else:
                ids[i] = ids[i][:max_length]
        result = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones_like(torch.tensor(ids)),
        }
        if return_tensors == "pt":
            if len(result["input_ids"]) == 1:
                result["input_ids"] = result["input_ids"].squeeze(0)
                result["attention_mask"] = result["attention_mask"].squeeze(0)
        return result

    def decode(self, ids, skip_special_tokens=True):
        return f"decoded_{ids[:5]}"

    def __len__(self):
        return self.vocab_size


class TestTextDataset:
    """Test TextDataset."""

    def test_creation_with_tokenizer(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test text"]
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=32)
        assert len(dataset) == 2

    def test_getitem(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test text"]
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=32)
        item = dataset[0]
        assert "input_ids" in item
        assert item["input_ids"].shape == (32,)

    def test_with_return_mask(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=32, return_mask=True)
        item = dataset[0]
        assert "attention_mask" in item


class TestCharacterDataset:
    """Test CharacterDataset."""

    def test_creation(self):
        texts = ["hello", "world"]
        dataset = CharacterDataset(texts=texts, max_length=32)
        assert len(dataset) == 2

    def test_getitem(self):
        texts = ["hello", "world"]
        dataset = CharacterDataset(texts=texts, max_length=32)
        item = dataset[0]
        assert "input_ids" in item
        assert item["input_ids"].shape == (32,)

    def test_char_vocabulary(self):
        texts = ["hello", "world"]
        dataset = CharacterDataset(texts=texts, max_length=32)
        assert len(dataset.chars) > 0


class TestTokenizerWrapper:
    """Test TokenizerWrapper."""

    def test_creation(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)
        assert wrapper.tokenizer == tokenizer

    def test_call(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)
        result = wrapper("test text")
        assert "input_ids" in result

    def test_decode(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)
        text = wrapper.decode([1, 2, 3, 4, 5])
        assert text == "decoded_[1, 2, 3, 4, 5]"

    def test_len(self):
        tokenizer = MockTokenizer(vocab_size=5000)
        wrapper = TokenizerWrapper(tokenizer)
        assert len(wrapper) == 5000


class TestCreateDataset:
    """Test create_dataset factory function."""

    def test_create_character_dataset(self):
        texts = ["hello world", "test data"]
        dataset = create_dataset("character", texts=texts, max_length=32)
        assert isinstance(dataset, CharacterDataset)
        assert len(dataset) == 2

    def test_create_text_dataset_with_tokenizer(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test data"]
        dataset = create_dataset("text", texts=texts, tokenizer=tokenizer, max_length=32)
        assert len(dataset) == 2
        item = dataset[0]
        assert "input_ids" in item

    def test_create_text_dataset_fallback_without_tokenizer(self):
        texts = ["hello world", "test data"]
        dataset = create_dataset("text", texts=texts, max_length=32)
        # Should fallback to character dataset
        assert isinstance(dataset, CharacterDataset)

    def test_invalid_dataset_type(self):
        with pytest.raises(ValueError) as exc_info:
            create_dataset("invalid_type", texts=["test"])
        assert "Unknown dataset type" in str(exc_info.value)

    def test_character_dataset_requires_texts(self):
        with pytest.raises(ValueError) as exc_info:
            create_dataset("character", max_length=32)
        assert "requires 'texts'" in str(exc_info.value)


class TestStreamingDataset:
    """Test StreamingDataset."""

    def test_creation_with_mock_file(self, tmp_path):
        # Create a temporary jsonl file
        file_path = tmp_path / "test.jsonl"
        file_path.write_text('{"text": "hello"}\n{"text": "world"}\n')

        tokenizer = MockTokenizer()
        dataset = StreamingDataset(
            file_path=str(file_path),
            tokenizer=tokenizer,
            max_length=32,
            file_format="jsonl",
        )
        assert len(dataset) == 2

    def test_getitem(self, tmp_path):
        # Create a temporary jsonl file
        file_path = tmp_path / "test.jsonl"
        file_path.write_text('{"text": "hello"}\n{"text": "world"}\n')

        tokenizer = MockTokenizer()
        dataset = StreamingDataset(
            file_path=str(file_path),
            tokenizer=tokenizer,
            max_length=32,
            file_format="jsonl",
        )
        item = dataset[0]
        assert "input_ids" in item


class TestTextDatasetEdgeCases:
    """Test TextDataset edge cases."""

    @pytest.mark.parametrize("max_length", [8, 16, 32, 64, 128])
    def test_different_max_lengths(self, max_length):
        tokenizer = MockTokenizer()
        texts = ["hello world test"]
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=max_length)
        item = dataset[0]
        assert item["input_ids"].shape[0] == max_length

    def test_empty_text_list(self):
        tokenizer = MockTokenizer()
        texts = []
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=32)
        assert len(dataset) == 0

    @pytest.mark.parametrize("num_texts", [1, 5, 10, 20])
    def test_different_num_texts(self, num_texts):
        tokenizer = MockTokenizer()
        texts = [f"text {i}" for i in range(num_texts)]
        dataset = TextDataset(texts=texts, tokenizer=tokenizer, max_length=32)
        assert len(dataset) == num_texts


class TestCharacterDatasetEdgeCases:
    """Test CharacterDataset edge cases."""

    @pytest.mark.parametrize("max_length", [8, 16, 32, 64])
    def test_different_max_lengths(self, max_length):
        texts = ["hello"]
        dataset = CharacterDataset(texts=texts, max_length=max_length)
        item = dataset[0]
        assert item["input_ids"].shape[0] == max_length

    @pytest.mark.parametrize("text_len", [1, 5, 10, 50])
    def test_different_text_lengths(self, text_len):
        text = "a" * text_len
        texts = [text]
        dataset = CharacterDataset(texts=texts, max_length=100)
        item = dataset[0]
        # Should have at least text_len tokens (padded to max_length)
        assert item["input_ids"].shape[0] == 100


class TestTokenizerWrapperEdgeCases:
    """Test TokenizerWrapper edge cases."""

    def test_wrapper_with_special_tokens(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)
        # Test various input types
        result = wrapper("test")
        assert "input_ids" in result

        result = wrapper(["test1", "test2"])
        assert "input_ids" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
