"""
Parameter-Efficient Fine-Tuning (PEFT) for Diffusion Language Models.

Supports:
- LoRA: Low-Rank Adaptation
- LoRA+: LoRA with learned learning rates
- DoRA: Weight-Decomposed Low-Rank Adaptation
- IA3: Infusion of Adapter for Attention
- Prefix Tuning
- Prompt Tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
import math


@dataclass
class LoraConfig:
    """Configuration for LoRA."""
    r: int = 8  # Rank
    lora_alpha: int = 16  # Alpha for scaling
    lora_dropout: float = 0.05  # Dropout probability
    target_modules: Optional[List[str]] = None  # Modules to apply LoRA
    bias: str = "none"  # Bias type: none, all, lora_only
    task_type: str = "CAUSAL_LM"  # Task type


class LoRALayer(nn.Module):
    """LoRA layer implementation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        h = self.lora_dropout(x)
        h = torch.matmul(h, self.lora_A.T)  # (batch, r)
        h = torch.matmul(h, self.lora_B.T)  # (batch, out_features)
        return h * self.scaling


class LoRAEmbedding(nn.Module):
    """LoRA for embedding layers."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Original embedding (frozen)
        self.base_embedding = nn.Embedding(num_embeddings, embedding_dim)

        # LoRA for embedding
        self.lora_A = nn.Parameter(torch.zeros(r, embedding_dim))
        self.lora_B = nn.Parameter(torch.zeros(num_embeddings, r))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings with LoRA adaptation."""
        base_embeds = self.base_embedding(x)
        after_A = F.embedding(x, self.lora_B)
        lora_embeds = torch.matmul(after_A, self.lora_A) * self.scaling
        return base_embeds + lora_embeds


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.fan_in_fan_out = fan_in_fan_out

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get base layer output
        base_output = self.base_layer(x)

        # Get LoRA adaptation
        lora_output = self.lora(x)

        return base_output + lora_output


class IA3Layer(nn.Module):
    """IA3 (Infusion of Adapter for Attention) layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        target_modules: Optional[List[str]] = None,
        ia3_alpha: float = 0.5,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.ia3_alpha = ia3_alpha
        self.target_modules = target_modules or []

        # Get dimensions
        out_features = base_layer.out_features
        in_features = base_layer.in_features

        # IA3 scaling vectors (learnable)
        self.ia3_vector = nn.Parameter(torch.zeros(out_features))

        if hasattr(base_layer, 'weight'):
            self.weight = base_layer.weight
        if hasattr(base_layer, 'bias'):
            self.bias = base_layer.bias

        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with IA3 scaling."""
        base_output = self.base_layer(x)
        # Scale by learned vector
        scale = (self.ia3_vector * self.ia3_alpha).exp()
        return base_output * scale


class PrefixTuning(nn.Module):
    """Prefix tuning for text generation."""

    def __init__(
        self,
        hidden_size: int,
        prefix_length: int = 10,
        num_layers: int = 12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_length = prefix_length
        self.num_layers = num_layers

        # Prefix tokens (learnable)
        self.prefix_embeddings = nn.Parameter(
            torch.zeros(num_layers, prefix_length, hidden_size)
        )

        # Optimization: use MLP instead of direct embeddings
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size * num_layers),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.prefix_embeddings, std=0.02)

    def get_prefix(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get prefix embeddings for the batch."""
        prefix_input = self.prefix_embeddings.mean(dim=1).mean(dim=0, keepdim=True)
        prefix_flat = self.mlp(prefix_input)
        prefix = prefix_flat.view(self.num_layers, 1, self.hidden_size)
        prefix = prefix.expand(-1, batch_size, -1)
        return prefix


class PromptTuning(nn.Module):
    """Prompt tuning (soft prompts)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        prompt_length: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length

        # Soft prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get prompt embeddings."""
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class DoRALayer(nn.Module):
    """DoRA (Weight-Decomposed Low-Rank Adaptation)."""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.magnitude = nn.Parameter(torch.ones(out_features))

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._reset_parameters()

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        lora_output = self.lora_dropout(x)
        lora_output = torch.matmul(lora_output, self.lora_A.T)
        lora_output = torch.matmul(lora_output, self.lora_B.T)

        base_norm = torch.norm(self.base_layer.weight, dim=1)
        lora_weight = torch.matmul(self.lora_B, self.lora_A)
        combined_weight = self.base_layer.weight + lora_weight * self.scaling
        combined_norm = torch.norm(combined_weight, dim=1)

        scale = (self.magnitude * combined_norm) / (base_norm + 1e-8)
        return base_output * scale + lora_output * self.scaling


class LoraModel(nn.Module):
    """Wrapper model with LoRA/PEFT adapters applied."""

    def __init__(
        self,
        base_model: nn.Module,
        config: LoraConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.peft_config = config

        # Find target modules and apply LoRA
        self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to target modules in the model."""
        target_modules = self.config.target_modules or self._get_default_target_modules()

        replacements = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        replacements.append((name, module))
                        break

        for name, module in replacements:
            self._replace_with_lora(module, name)

    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules for LoRA."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2", "attention.qkv",
            "mlp", "lm_head", "out_proj", "in_proj",
        ]

    def _replace_with_lora(self, module: nn.Module, name: str):
        """Replace a module with LoRA version."""
        if isinstance(module, nn.Linear):
            lora_linear = LoRALinear(
                base_layer=module,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            # Set the parent module
            parts = name.split('.')
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_linear)

        elif isinstance(module, nn.Embedding):
            lora_emb = LoRAEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
            )
            parts = name.split('.')
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_emb)

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.base_model(*args, **kwargs)

    def get_trainable_parameters(self):
        """Get only trainable parameters (LoRA parameters)."""
        return [p for p in self.parameters() if p.requires_grad]

    def merge_lora(self):
        """Merge LoRA weights into base model."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALinear):
                base = module.base_layer
                lora = module.lora
                delta = (lora.lora_B @ lora.lora_A) * lora.scaling
                base.weight.data += delta
                parts = name.split('.')
                parent = self.base_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], base)

    def save_peft_checkpoint(self, path: str):
        """Save only PEFT (LoRA) parameters."""
        state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data

        torch.save(state_dict, path)

    def load_peft_checkpoint(self, path: str):
        """Load PEFT (LoRA) parameters."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)


class DoRAModel(nn.Module):
    """Wrapper model with DoRA adapters applied."""

    def __init__(self, base_model: nn.Module, config: LoraConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.peft_config = config
        self._apply_dora()

    def _apply_dora(self):
        target_modules = self.config.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2", "mlp", "lm_head", "out_proj", "in_proj",
        ]

        replacements = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        replacements.append((name, module))
                        break

        for name, module in replacements:
            dora_layer = DoRALayer(
                base_layer=module,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            parts = name.split('.')
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], dora_layer)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def create_peft_model(
    base_model: nn.Module,
    peft_type: str = "lora",
    **kwargs,
) -> nn.Module:
    """Factory function to create PEFT model.

    Args:
        base_model: Base model to apply PEFT to
        peft_type: Type of PEFT: lora, dora, ia3, prefix, prompt
        **kwargs: Additional arguments for PEFT config

    Returns:
        Model with PEFT applied
    """
    if peft_type == "lora":
        config = LoraConfig(**kwargs)
        return LoraModel(base_model, config)
    elif peft_type == "dora":
        config = LoraConfig(**kwargs)
        return DoRAModel(base_model, config)
    elif peft_type == "ia3":
        return IA3Model(base_model, **kwargs)
    elif peft_type == "prefix":
        return PrefixTuningModel(base_model, **kwargs)
    elif peft_type == "prompt":
        return PromptTuningModel(base_model, **kwargs)
    else:
        raise ValueError(f"Unknown PEFT type: {peft_type}")


class IA3Model(nn.Module):
    """Model with IA3 adapters applied."""

    def __init__(self, base_model: nn.Module, **kwargs):
        super().__init__()
        self.base_model = base_model
        # Apply IA3 to attention layers
        self._apply_ia3()

    def _apply_ia3(self):
        """Apply IA3 to target modules."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(
                t in name for t in ["q_proj", "v_proj", "k_proj", "out_proj"]
            ):
                ia3_layer = IA3Layer(module)
                parts = name.split('.')
                parent = self.base_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], ia3_layer)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class PrefixTuningModel(nn.Module):
    """Model with prefix tuning applied."""

    def __init__(self, base_model: nn.Module, prefix_length: int = 10):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length

        if not hasattr(base_model, 'hidden_size'):
            raise ValueError("base_model must have a 'hidden_size' attribute for PrefixTuning")
        hidden_size = base_model.hidden_size

        num_layers = None
        if hasattr(base_model, 'config') and base_model.config is not None:
            if hasattr(base_model.config, 'num_layers'):
                num_layers = base_model.config.num_layers
            elif hasattr(base_model.config, 'num_hidden_layers'):
                num_layers = base_model.config.num_hidden_layers
        if num_layers is None and hasattr(base_model, 'blocks'):
            num_layers = len(base_model.blocks)
        if num_layers is None:
            raise ValueError("Cannot determine num_layers from base_model for PrefixTuning")

        self.prefix_tuning = PrefixTuning(
            hidden_size=hidden_size,
            prefix_length=prefix_length,
            num_layers=num_layers,
        )

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        device = x.device

        if hasattr(self.base_model, 'token_embed'):
            h = self.base_model.token_embed(x)

            prefix = self.prefix_tuning.prefix_embeddings[0, :, :].unsqueeze(0)
            prefix = prefix.expand(batch_size, -1, -1).to(device)
            h = torch.cat([prefix, h], dim=1)

            if hasattr(self.base_model, 'pos_embed'):
                seq_len = h.shape[1]
                pos_emb = self.base_model.pos_embed(seq_len, device)
                h = h + pos_emb

            timesteps = kwargs.get('timesteps', None)
            t_emb = None
            if timesteps is not None and hasattr(self.base_model, 'get_timestep_embedding'):
                t_emb = self.base_model.get_timestep_embedding(timesteps)
                t_emb = self.base_model.timestep_embed(t_emb)
                h = h + t_emb.unsqueeze(1)

            if hasattr(self.base_model, 'dropout'):
                h = self.base_model.dropout(h)

            for block in self.base_model.blocks:
                h = block(h, t_emb)

            h = self.base_model.norm_final(h)
            logits = self.base_model.lm_head(h)
            return logits[:, self.prefix_length:, :]

        return self.base_model(x, **kwargs)


class PromptTuningModel(nn.Module):
    """Model with prompt tuning applied."""

    def __init__(
        self,
        base_model: nn.Module,
        vocab_size: Optional[int] = None,
        prompt_length: int = 20,
    ):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length

        if not hasattr(base_model, 'hidden_size'):
            raise ValueError("base_model must have a 'hidden_size' attribute for PromptTuning")
        hidden_size = base_model.hidden_size

        if vocab_size is None:
            if hasattr(base_model, 'vocab_size'):
                vocab_size = base_model.vocab_size
            elif hasattr(base_model, 'config') and hasattr(base_model.config, 'vocab_size'):
                vocab_size = base_model.config.vocab_size
            else:
                raise ValueError("Cannot determine vocab_size from base_model for PromptTuning. Pass vocab_size explicitly.")

        self.prompt_tuning = PromptTuning(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            prompt_length=prompt_length,
        )

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        device = x.device

        if hasattr(self.base_model, 'token_embed'):
            h = self.base_model.token_embed(x)

            prompt = self.prompt_tuning(batch_size, device)
            h = torch.cat([prompt, h], dim=1)

            if hasattr(self.base_model, 'pos_embed'):
                seq_len = h.shape[1]
                pos_emb = self.base_model.pos_embed(seq_len, device)
                h = h + pos_emb

            timesteps = kwargs.get('timesteps', None)
            t_emb = None
            if timesteps is not None and hasattr(self.base_model, 'get_timestep_embedding'):
                t_emb = self.base_model.get_timestep_embedding(timesteps)
                t_emb = self.base_model.timestep_embed(t_emb)
                h = h + t_emb.unsqueeze(1)

            if hasattr(self.base_model, 'dropout'):
                h = self.base_model.dropout(h)

            for block in self.base_model.blocks:
                h = block(h, t_emb)

            h = self.base_model.norm_final(h)
            logits = self.base_model.lm_head(h)
            return logits[:, self.prompt_length:, :]

        return self.base_model(x, **kwargs)
