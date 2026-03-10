"""
Inference backends for Diffusion Language Models.

Supports:
- Transformers (native PyTorch)
- vLLM
- SGLang
- LMDeploy
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    backend: str = "transformers"
    model_name_or_path: Optional[str] = None
    max_model_len: int = 4096
    max_num_seqs: int = 256
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    vocab_size: int = 32768


class InferenceBackend(ABC):
    """Base class for inference backends."""

    def __init__(self, model: nn.Module, config: InferenceConfig):
        self.model = model
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> List[str]:
        """Generate text from prompts."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ):
        """Stream generated text."""
        pass

    @abstractmethod
    def encode(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode prompts to tokens."""
        pass

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text."""
        pass


class TransformersBackend(InferenceBackend):
    """Native PyTorch/Transformers inference backend."""

    def __init__(self, model: nn.Module, config: InferenceConfig, tokenizer: Any = None):
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.model.eval()

        # Move to appropriate device
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        elif torch.backends.mps.is_available():
            self.model = self.model.to("mps")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> List[str]:
        """Generate text using transformers."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.tokenizer:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        else:
            raise ValueError("Tokenizer is required for generation. Provide a tokenizer when creating the backend.")

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            )

        # Decode
        if self.tokenizer:
            results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            results = [[str(tok) for tok in seq] for seq in outputs]

        return results

    def stream_generate(self, prompts, max_length=100, temperature=1.0, top_p=1.0, top_k=-1, **kwargs):
        """Stream generation (simplified - yields full output)."""
        results = self.generate(prompts, max_length, temperature, top_p, top_k, **kwargs)
        for result in results:
            yield result

    def encode(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode prompts."""
        if self.tokenizer:
            return self.tokenizer(prompts, return_tensors="pt")["input_ids"]
        raise ValueError("Tokenizer is required for encoding.")

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs."""
        if self.tokenizer:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [str(ids.tolist()) for ids in token_ids]


class VLLMBackend(InferenceBackend):
    """vLLM inference backend."""

    def __init__(self, model: nn.Module, config: InferenceConfig, tokenizer: Any = None):
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.llm = None

        # Try to initialize vLLM
        try:
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
            self._init_vllm()
        except ImportError:
            raise ImportError(
                "vLLM is required for the vllm backend. "
                "Install with: pip install vllm  (or: pip install hedgehog-dlm[infer])"
            )

    def _init_vllm(self):
        """Initialize vLLM engine."""
        try:
            from vllm import LLM

            if self.config.model_name_or_path is None:
                raise ValueError("model_name_or_path is required for vLLM backend")
            self.llm = LLM(
                model=self.config.model_name_or_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                enforce_eager=self.config.enforce_eager,
            )
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            self.llm = None

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> List[str]:
        """Generate using vLLM."""
        if self.llm is None:
            return self._fallback.generate(prompts, max_length, temperature, top_p, top_k, **kwargs)

        if isinstance(prompts, str):
            prompts = [prompts]

        # Sampling params
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 100,
            max_tokens=max_length,
        )

        # Generate
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract text
        return [output.outputs[0].text for output in outputs]

    def stream_generate(self, prompts, max_length=100, temperature=1.0, top_p=1.0, top_k=-1, **kwargs):
        """Stream generation from vLLM."""
        if self.llm is None:
            yield from self._fallback.stream_generate(prompts, max_length, temperature, top_p, top_k, **kwargs)
            return

        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 100,
            max_tokens=max_length,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        for output in outputs:
            yield output.outputs[0].text

    def encode(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode prompts."""
        if self.tokenizer:
            return self.tokenizer(prompts, return_tensors="pt")["input_ids"]
        raise ValueError("Tokenizer is required for encoding.")

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs."""
        if self.tokenizer:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [str(ids.tolist()) for ids in token_ids]


class SGLangBackend(InferenceBackend):
    """SGLang inference backend."""

    def __init__(self, model: nn.Module, config: InferenceConfig, tokenizer: Any = None):
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.client = None

        # Try to connect to SGLang backend
        self._init_sglang()

    def _init_sglang(self):
        """Initialize SGLang client."""
        try:
            from sglang import sgl

            self.client = sgl
        except ImportError:
            raise ImportError(
                "SGLang is required for the sglang backend. "
                "Install with: pip install sglang  (or: pip install hedgehog-dlm[infer])"
            )

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> List[str]:
        """Generate using SGLang."""
        if self.client is None:
            return self._fallback.generate(prompts, max_length, temperature, top_p, top_k, **kwargs)

        # SGLang API call
        raise NotImplementedError("SGLang generation not yet implemented")

    def stream_generate(self, prompts, max_length=100, temperature=1.0, top_p=1.0, top_k=-1, **kwargs):
        """Stream generation from SGLang."""
        if self.client is None:
            yield from self._fallback.stream_generate(prompts, max_length, temperature, top_p, top_k, **kwargs)
            return

        raise NotImplementedError("SGLang streaming not yet implemented")

    def encode(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode prompts."""
        if self.tokenizer:
            return self.tokenizer(prompts, return_tensors="pt")["input_ids"]
        raise ValueError("Tokenizer is required for encoding.")

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs."""
        if self.tokenizer:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [str(ids.tolist()) for ids in token_ids]


class LMDeployBackend(InferenceBackend):
    """LMDeploy inference backend."""

    def __init__(self, model: nn.Module, config: InferenceConfig, tokenizer: Any = None):
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.pipeline = None

        # Try to initialize LMDeploy
        self._init_lmdeploy()

    def _init_lmdeploy(self):
        """Initialize LMDeploy pipeline."""
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig

            engine_config = TurbomindEngineConfig(
                tp=self.config.tensor_parallel_size,
                session_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
            )
            if self.config.model_name_or_path is None:
                raise ValueError("model_name_or_path is required for LMDeploy backend")
            self.pipeline = pipeline(
                self.config.model_name_or_path,
                engine_config=engine_config,
            )
        except ImportError:
            raise ImportError(
                "LMDeploy is required for the lmdeploy backend. "
                "Install with: pip install lmdeploy  (or: pip install hedgehog-dlm[infer])"
            )

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> List[str]:
        """Generate using LMDeploy."""
        if self.pipeline is None:
            return self._fallback.generate(prompts, max_length, temperature, top_p, top_k, **kwargs)

        # LMDeploy generation
        outputs = self.pipeline(
            prompts,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
        )

        if isinstance(outputs, list):
            return [out.text for out in outputs]
        return [outputs.text]

    def stream_generate(self, prompts, max_length=100, temperature=1.0, top_p=1.0, top_k=-1, **kwargs):
        """Stream generation from LMDeploy."""
        if self.pipeline is None:
            yield from self._fallback.stream_generate(prompts, max_length, temperature, top_p, top_k, **kwargs)
            return

        # Streaming not directly supported, yield full output
        results = self.generate(prompts, max_length, temperature, top_p, top_k, **kwargs)
        for result in results:
            yield result

    def encode(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode prompts."""
        if self.tokenizer:
            return self.tokenizer(prompts, return_tensors="pt")["input_ids"]
        raise ValueError("Tokenizer is required for encoding.")

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs."""
        if self.tokenizer:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [str(ids.tolist()) for ids in token_ids]


def create_inference_backend(
    model: nn.Module,
    config: InferenceConfig,
    tokenizer: Any = None,
) -> InferenceBackend:
    """Factory function to create inference backend.

    Args:
        model: Model to use for inference
        config: Inference configuration
        tokenizer: Tokenizer for the model

    Returns:
        Inference backend
    """
    backend_type = config.backend.lower()

    if backend_type == "transformers":
        return TransformersBackend(model, config, tokenizer)
    elif backend_type == "vllm":
        return VLLMBackend(model, config, tokenizer)
    elif backend_type == "sglang":
        return SGLangBackend(model, config, tokenizer)
    elif backend_type == "lmdeploy":
        return LMDeployBackend(model, config, tokenizer)
    else:
        raise ValueError(f"Unknown inference backend: {backend_type}")


# OpenAI-compatible API server

class OpenAICompatibleServer:
    """OpenAI-compatible API server for diffusion models."""

    def __init__(self, backend: InferenceBackend, host: str = "0.0.0.0", port: int = 8000):
        self.backend = backend
        self.host = host
        self.port = port
        self.app = None

        # Try to create FastAPI app
        self._init_app()

    def _init_app(self):
        """Initialize FastAPI app."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import Optional, List, Union

            app = FastAPI(title="Hedgehog DLM API")

            class CompletionRequest(BaseModel):
                model: str
                prompt: Union[str, List[str]]
                max_tokens: int = 100
                temperature: float = 1.0
                top_p: float = 1.0
                n: int = 1
                stream: bool = False

            class CompletionResponse(BaseModel):
                id: str
                object: str = "text_completion"
                created: int
                model: str
                choices: List[dict]

            @app.post("/v1/completions")
            async def completions(request: CompletionRequest):
                results = self.backend.generate(
                    prompts=request.prompt,
                    max_length=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )

                return CompletionResponse(
                    id=f"cmpl-{hash(str(results))}",
                    created=0,
                    model=request.model,
                    choices=[
                        {"text": text, "index": i, "finish_reason": "length"}
                        for i, text in enumerate(results)
                    ]
                )

            self.app = app

        except ImportError:
            raise ImportError(
                "FastAPI and pydantic are required for the API server. "
                "Install with: pip install fastapi pydantic"
            )

    def run(self):
        """Run the server."""
        if self.app is None:
            raise RuntimeError(
                "Server not initialized. Install FastAPI to enable: pip install fastapi pydantic"
            )

        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run the API server. "
                "Install with: pip install uvicorn"
            )
        uvicorn.run(self.app, host=self.host, port=self.port)
