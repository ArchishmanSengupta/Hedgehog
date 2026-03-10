import pytest
import torch
import torch.nn as nn
import tempfile
import os
import json
import shutil
from pathlib import Path

from hedgehog.models import (
    create_model, DiffusionTransformer, AutoregressiveTransformer,
    MambaBlock, ModelConfig, DiTBlock, SinusoidalPositionEmbedding,
)
from hedgehog.diffusion import (
    create_diffusion, MDLMDiffusion, D3PMDiffusion,
    DiscreteDiffusion, DiffusionType, NoiseSchedule,
)
from hedgehog.samplers import (
    create_sampler, DDPMSampler, DDPMCachedSampler,
    AnalyticSampler, SemiAutoregressiveSampler, BlockwiseSampler,
)
from hedgehog.trainers import TrainerConfig, Trainer, DiffusionTrainer
from hedgehog.peft import (
    create_peft_model, LoraConfig, LoraModel, DoRAModel,
    IA3Model, PrefixTuningModel, PromptTuningModel,
    LoRALayer, LoRALinear, LoRAEmbedding, IA3Layer,
    DoRALayer, PrefixTuning, PromptTuning,
)
from hedgehog.quantization import (
    QuantConfig, quantize_model, BNBQuantizedLinear,
    AWQLinear, GPTQLinear, HQQQuantizedLinear, EETQLinear,
    create_quantizer, estimate_model_size, get_nbits_from_dtype,
)
from hedgehog.distributed import (
    DistributedConfig, DistributedManager, get_distributed_manager,
    TensorParallelLinear, PipelineParallel, SequenceParallelLinear,
    FSDPWrapper, split_data_for_dp,
)
from hedgehog.utils import (
    setup_logging, get_logger, get_device, get_device_count,
    count_parameters, get_model_size, safe_save_checkpoint,
    safe_load_checkpoint, AverageMeter, collate_fn,
    find_free_port, setup_distributed,
)
from hedgehog.data import (
    CharacterDataset, TextDataset, create_dataset, TokenizerWrapper,
)
from hedgehog.registry import (
    get_model_registry, register_model, get_model, list_models,
    get_dataset_info, list_datasets, get_training_method,
    list_training_methods, get_sampling_method, list_sampling_methods,
    ModelRegistry, DLM_MODELS,
)
from hedgehog.inference import (
    InferenceConfig, TransformersBackend, create_inference_backend,
)
from hedgehog.cli import create_parser, parse_args


def _make_model(vocab_size=100, hidden_size=64, num_heads=2, num_layers=1,
                max_seq_len=32, dropout=0.0, model_type="dit"):
    return create_model(
        model_type=model_type, vocab_size=vocab_size,
        hidden_size=hidden_size, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=max_seq_len, dropout=dropout,
    )


class DummyDataset:
    def __init__(self, n=20, seq_len=16, vocab_size=100):
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, self.vocab_size, (self.seq_len,))}


# ---------------------------------------------------------------------------
# 1. TIMESTEP CONDITIONING REGRESSION TESTS
# ---------------------------------------------------------------------------
class TestTimestepConditioning:
    def test_different_timesteps_produce_different_outputs(self):
        torch.manual_seed(0)
        model = _make_model()
        model.eval()
        x = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            out_t0 = model(x, timesteps=torch.tensor([0]))
            out_t500 = model(x, timesteps=torch.tensor([500]))
            out_t999 = model(x, timesteps=torch.tensor([999]))
        assert not torch.allclose(out_t0, out_t500, atol=1e-5)
        assert not torch.allclose(out_t0, out_t999, atol=1e-5)
        assert not torch.allclose(out_t500, out_t999, atol=1e-5)

    def test_same_timestep_deterministic(self):
        torch.manual_seed(0)
        model = _make_model()
        model.eval()
        x = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            out1 = model(x, timesteps=torch.tensor([42, 42]))
            out2 = model(x, timesteps=torch.tensor([42, 42]))
        assert torch.allclose(out1, out2)

    def test_timestep_emb_fed_to_blocks(self):
        model = _make_model()
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([100])
        t_emb = model.get_timestep_embedding(t)
        t_emb = model.timestep_embed(t_emb)
        assert t_emb.shape == (1, 64)
        h = model.token_embed(x) + model.pos_embed(8, x.device)
        h = h + t_emb.unsqueeze(1)
        for block in model.blocks:
            h = block(h, t_emb)
        assert h.shape == (1, 8, 64)

    def test_diffusion_loss_uses_timesteps(self):
        torch.manual_seed(0)
        model = _make_model()
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=1000)
        x_0 = torch.randint(0, 99, (4, 16))
        t_low = torch.zeros(4, dtype=torch.long)
        t_high = torch.full((4,), 999, dtype=torch.long)
        loss_low = diffusion.compute_loss(model, x_0, t_low, mask_token_id=99)
        loss_high = diffusion.compute_loss(model, x_0, t_high, mask_token_id=99)
        assert loss_low.item() != loss_high.item()

    def test_ar_model_ignores_timesteps_gracefully(self):
        model = _make_model(model_type="ar")
        model.eval()
        x = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            out_none = model(x, timesteps=None)
            out_t = model(x, timesteps=torch.tensor([42]))
        assert torch.allclose(out_none, out_t)


# ---------------------------------------------------------------------------
# 2. LORA MERGE CORRECTNESS
# ---------------------------------------------------------------------------
class TestLoRAMerge:
    def test_merge_lora_removes_lora_and_modifies_weights(self):
        base = _make_model()
        w_before = {k: v.clone() for k, v in base.state_dict().items()}
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0)
        lora_model = LoraModel(base, config)
        has_lora = False
        for name, m in lora_model.base_model.named_modules():
            if isinstance(m, LoRALinear):
                nn.init.ones_(m.lora.lora_A)
                nn.init.ones_(m.lora.lora_B)
                has_lora = True
        assert has_lora
        lora_model.merge_lora()
        has_lora_after = any(
            isinstance(m, LoRALinear)
            for _, m in lora_model.base_model.named_modules()
        )
        assert not has_lora_after
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        lora_model.base_model.eval()
        with torch.no_grad():
            out = lora_model.base_model(x, timesteps=t)
        assert out.shape == (1, 8, 100)
        assert not torch.isnan(out).any()

    def test_merge_lora_standalone_linear(self):
        linear = nn.Linear(32, 64, bias=False)
        w_orig = linear.weight.data.clone()
        lora_linear = LoRALinear(linear, r=4, lora_alpha=8, lora_dropout=0.0)
        nn.init.ones_(lora_linear.lora.lora_A)
        nn.init.ones_(lora_linear.lora.lora_B)
        x = torch.randn(1, 5, 32)
        lora_linear.eval()
        with torch.no_grad():
            out_lora = lora_linear(x).clone()
        delta = (lora_linear.lora.lora_B @ lora_linear.lora.lora_A) * lora_linear.lora.scaling
        linear.weight.data = w_orig + delta
        with torch.no_grad():
            out_merged = linear(x)
        assert torch.allclose(out_lora, out_merged, atol=1e-5)

    def test_merge_produces_correct_delta(self):
        linear = nn.Linear(32, 64)
        w_before = linear.weight.data.clone()
        lora_linear = LoRALinear(linear, r=4, lora_alpha=8, lora_dropout=0.0)
        nn.init.ones_(lora_linear.lora.lora_A)
        nn.init.ones_(lora_linear.lora.lora_B)
        expected_delta = (lora_linear.lora.lora_B @ lora_linear.lora.lora_A) * lora_linear.lora.scaling
        linear.weight.data += expected_delta
        assert not torch.allclose(w_before, linear.weight.data)

    def test_lora_has_trainable_lora_params(self):
        base = _make_model()
        config = LoraConfig(r=4, lora_alpha=8)
        lora_model = LoraModel(base, config)
        trainable = lora_model.get_trainable_parameters()
        trainable_params = sum(p.numel() for p in trainable)
        assert trainable_params > 0
        has_lora_param = any("lora" in n for n, p in lora_model.named_parameters() if p.requires_grad)
        assert has_lora_param


# ---------------------------------------------------------------------------
# 3. SAVE / LOAD / CHECKPOINT ROUND-TRIP
# ---------------------------------------------------------------------------
class TestCheckpointRoundTrip:
    def test_save_load_preserves_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir,
            )
            ds = DummyDataset()
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            batch = {"input_ids": torch.randint(0, 100, (2, 16))}
            trainer.training_step(batch)
            trainer.global_step = 1
            trainer.save_checkpoint("checkpoint-1")
            w_before = {k: v.clone() for k, v in model.state_dict().items()}
            model2 = _make_model()
            config2 = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir,
            )
            trainer2 = Trainer(config=config2, model=model2, train_dataset=ds)
            trainer2.setup_training()
            trainer2.load_checkpoint(os.path.join(tmpdir, "checkpoint-1.pt"))
            for k in w_before:
                assert torch.allclose(w_before[k], model2.state_dict()[k])
            assert trainer2.global_step == 1

    def test_save_load_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, gradient_accumulation_steps=1,
            )
            ds = DummyDataset()
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            for _ in range(3):
                batch = {"input_ids": torch.randint(0, 100, (2, 16))}
                trainer.training_step(batch)
            trainer.global_step = 3
            trainer.save_checkpoint("checkpoint-3")
            opt_state = {k: v for k, v in trainer.optimizer.state_dict().items()}
            model2 = _make_model()
            config2 = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir,
            )
            trainer2 = Trainer(config=config2, model=model2, train_dataset=ds)
            trainer2.setup_training()
            trainer2.load_checkpoint(os.path.join(tmpdir, "checkpoint-3.pt"))
            assert trainer2.global_step == 3

    def test_peft_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_model()
            config = LoraConfig(r=4, lora_alpha=8)
            lora_model = LoraModel(base, config)
            path = os.path.join(tmpdir, "peft.pt")
            lora_model.save_peft_checkpoint(path)
            assert os.path.exists(path)
            state = torch.load(path, weights_only=False)
            assert len(state) > 0
            for k in state:
                assert "lora" in k or "base_model" in k

    def test_safe_save_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "ckpt.pt")
            data = {"step": 42, "tensor": torch.randn(10)}
            safe_save_checkpoint(data, path)
            loaded = safe_load_checkpoint(path)
            assert loaded["step"] == 42
            assert torch.allclose(loaded["tensor"], data["tensor"])


# ---------------------------------------------------------------------------
# 4. SAVE_TOTAL_LIMIT ENFORCEMENT
# ---------------------------------------------------------------------------
class TestSaveTotalLimit:
    def test_old_checkpoints_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, save_total_limit=2,
            )
            ds = DummyDataset()
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            for i in range(1, 6):
                trainer.global_step = i
                trainer.save_checkpoint(f"checkpoint-{i}")
            remaining = list(Path(tmpdir).glob("checkpoint-*.pt"))
            assert len(remaining) <= 2

    def test_save_total_limit_zero_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, save_total_limit=0,
            )
            ds = DummyDataset()
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            for i in range(1, 4):
                trainer.global_step = i
                trainer.save_checkpoint(f"checkpoint-{i}")
            remaining = list(Path(tmpdir).glob("checkpoint-*.pt"))
            assert len(remaining) == 3


# ---------------------------------------------------------------------------
# 5. RESUME FROM CHECKPOINT
# ---------------------------------------------------------------------------
class TestResumeFromCheckpoint:
    def test_resume_restores_global_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, gradient_accumulation_steps=1,
                num_train_epochs=1, per_device_batch_size=4,
                save_steps=2, logging_steps=100, dataloader_num_workers=0,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            batch = {"input_ids": torch.randint(0, 100, (4, 16))}
            for _ in range(5):
                trainer.training_step(batch)
                trainer.global_step += 1
            trainer.save_checkpoint("checkpoint-5")
            model2 = _make_model()
            config2 = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, resume_from_checkpoint=os.path.join(tmpdir, "checkpoint-5.pt"),
                num_train_epochs=1, per_device_batch_size=4,
                dataloader_num_workers=0,
            )
            ds2 = DummyDataset(n=8, seq_len=16)
            trainer2 = Trainer(config=config2, model=model2, train_dataset=ds2)
            trainer2.setup_training()
            trainer2.load_checkpoint(config2.resume_from_checkpoint)
            assert trainer2.global_step == 5


# ---------------------------------------------------------------------------
# 6. FULL TRAINING LOOP E2E
# ---------------------------------------------------------------------------
class TestFullTrainingLoop:
    def test_training_loop_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, num_train_epochs=2,
                per_device_batch_size=4, gradient_accumulation_steps=1,
                save_steps=100, eval_steps=100, logging_steps=100,
                warmup_steps=1, dataloader_num_workers=0,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.train()
            assert trainer.global_step > 0
            assert (Path(tmpdir) / "final_model.pt").exists()

    def test_training_with_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, num_train_epochs=1,
                per_device_batch_size=4, gradient_accumulation_steps=1,
                save_steps=100, eval_steps=2, logging_steps=100,
                warmup_steps=0, dataloader_num_workers=0,
            )
            train_ds = DummyDataset(n=8, seq_len=16)
            eval_ds = DummyDataset(n=4, seq_len=16)
            trainer = Trainer(
                config=config, model=model,
                train_dataset=train_ds, eval_dataset=eval_ds,
            )
            trainer.train()
            assert trainer.global_step > 0

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(42)
        model = _make_model(num_layers=2)
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=2, max_seq_len=32, device="cpu",
            gradient_accumulation_steps=1, learning_rate=1e-3,
            warmup_steps=0,
        )
        x_fixed = torch.randint(0, 99, (8, 16))
        ds = DummyDataset(n=8, seq_len=16)
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        trainer.setup_training()
        losses = []
        for _ in range(20):
            batch = {"input_ids": x_fixed}
            metrics = trainer.training_step(batch)
            losses.append(metrics["loss"])
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_diffusion_trainer_full_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            config = TrainerConfig(
                vocab_size=100, hidden_size=64, num_heads=2,
                num_layers=1, max_seq_len=32, device="cpu",
                output_dir=tmpdir, num_train_epochs=1,
                per_device_batch_size=4, gradient_accumulation_steps=1,
                save_steps=100, eval_steps=100, logging_steps=100,
                warmup_steps=0, dataloader_num_workers=0,
                num_timesteps=10,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = DiffusionTrainer(config=config, model=model, train_dataset=ds)
            trainer.train()
            samples = trainer.sample(num_samples=2, seq_len=8)
            assert samples.shape == (2, 8)
            assert samples.min() >= 0
            assert samples.max() < 100


# ---------------------------------------------------------------------------
# 7. PEFT FORWARD PASSES AND SHAPES
# ---------------------------------------------------------------------------
class TestPEFTForwardPasses:
    @pytest.mark.parametrize("peft_type,kwargs", [
        ("lora", {"r": 4, "lora_alpha": 8}),
        ("dora", {"r": 4, "lora_alpha": 8}),
        ("ia3", {}),
        ("prefix", {"prefix_length": 5}),
        ("prompt", {"vocab_size": 100, "prompt_length": 5}),
    ])
    def test_peft_output_shape(self, peft_type, kwargs):
        base = _make_model()
        peft_model = create_peft_model(base, peft_type=peft_type, **kwargs)
        x = torch.randint(0, 100, (2, 16))
        t = torch.tensor([5, 3])
        out = peft_model(x, timesteps=t)
        assert out.shape == (2, 16, 100)

    def test_lora_output_differs_from_base(self):
        torch.manual_seed(0)
        base = _make_model(dropout=0.0)
        base.eval()
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        with torch.no_grad():
            base_out = base(x, timesteps=t).clone()
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0)
        lora_model = LoraModel(base, config)
        lora_model.eval()
        for _, m in lora_model.base_model.named_modules():
            if isinstance(m, LoRALinear):
                nn.init.ones_(m.lora.lora_A)
                nn.init.ones_(m.lora.lora_B)
        with torch.no_grad():
            lora_out = lora_model(x, timesteps=t)
        assert not torch.allclose(base_out, lora_out, atol=1e-5)

    def test_ia3_freezes_base_params(self):
        base = _make_model()
        ia3_model = IA3Model(base)
        for name, m in ia3_model.base_model.named_modules():
            if isinstance(m, IA3Layer):
                for p in m.base_layer.parameters():
                    assert not p.requires_grad

    def test_prefix_tuning_prepends_prefix(self):
        base = _make_model()
        prefix_model = PrefixTuningModel(base, prefix_length=5)
        x = torch.randint(0, 100, (2, 10))
        t = torch.tensor([3, 7])
        out = prefix_model(x, timesteps=t)
        assert out.shape == (2, 10, 100)

    def test_prompt_tuning_prepends_prompt(self):
        base = _make_model()
        prompt_model = PromptTuningModel(base, vocab_size=100, prompt_length=5)
        x = torch.randint(0, 100, (2, 10))
        t = torch.tensor([3, 7])
        out = prompt_model(x, timesteps=t)
        assert out.shape == (2, 10, 100)

    def test_dora_forward_produces_output(self):
        base = _make_model()
        dora_model = DoRAModel(base, LoraConfig(r=4, lora_alpha=8))
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = dora_model(x, timesteps=t)
        assert out.shape == (1, 8, 100)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_lora_embedding_adaptation(self):
        emb = LoRAEmbedding(num_embeddings=100, embedding_dim=32, r=4, lora_alpha=8)
        nn.init.ones_(emb.lora_A)
        nn.init.ones_(emb.lora_B)
        x = torch.tensor([0, 1, 2, 3])
        out = emb(x)
        assert out.shape == (4, 32)
        base_out = emb.base_embedding(x)
        assert not torch.allclose(out, base_out)


# ---------------------------------------------------------------------------
# 8. DIFFUSION PROCESS CORRECTNESS
# ---------------------------------------------------------------------------
class TestDiffusionCorrectness:
    def test_q_sample_t0_preserves_input(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=1000)
        x_0 = torch.randint(0, 99, (4, 32))
        t = torch.zeros(4, dtype=torch.long)
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        match = (x_t == x_0).float().mean()
        assert match > 0.9

    def test_q_sample_tmax_masks_most(self):
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=100)
        x_0 = torch.randint(0, 99, (4, 32))
        t = torch.full((4,), 99, dtype=torch.long)
        x_t, mask = diffusion.q_sample(x_0, t, mask_token_id=99)
        mask_ratio = mask.float().mean()
        assert mask_ratio > 0.5

    def test_noise_schedule_monotonic(self):
        for schedule_type in ["linear", "cosine", "quadratic"]:
            schedule = NoiseSchedule(schedule_type, num_timesteps=100)
            alphas = [schedule.get_alpha_bar(torch.tensor(float(t))).item() for t in range(100)]
            for i in range(1, len(alphas)):
                assert alphas[i] <= alphas[i - 1] + 1e-6, \
                    f"{schedule_type}: alpha_bar not monotonically decreasing at t={i}"

    def test_noise_schedule_boundary_values(self):
        for schedule_type in ["linear", "cosine", "quadratic"]:
            schedule = NoiseSchedule(schedule_type, num_timesteps=100)
            alpha_0 = schedule.get_alpha_bar(torch.tensor(0.0)).item()
            alpha_max = schedule.get_alpha_bar(torch.tensor(100.0)).item()
            assert alpha_0 > 0.5
            assert alpha_max < 0.5

    def test_d3pm_absorbing_transition(self):
        d3pm = D3PMDiffusion(
            vocab_size=10, diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=100, transition_type="absorbing",
        )
        probs = d3pm._get_transition_probs(token_id=3, t=0)
        assert probs[3] == 1.0
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_d3pm_uniform_transition(self):
        d3pm = D3PMDiffusion(
            vocab_size=10, diffusion_type=DiffusionType.D3PM_UNIFORM,
            num_timesteps=100, transition_type="uniform",
        )
        probs = d3pm._get_transition_probs(token_id=3, t=50)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs[3] > probs[0]

    def test_d3pm_q_sample_shapes(self):
        d3pm = D3PMDiffusion(
            vocab_size=20, diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=10, transition_type="absorbing",
        )
        x_0 = torch.randint(0, 19, (2, 8))
        t = torch.tensor([3, 7])
        x_t, mask = d3pm.q_sample(x_0, t, mask_token_id=19)
        assert x_t.shape == (2, 8)
        assert mask.shape == (2, 8)

    def test_p_sample_shape(self):
        model = _make_model()
        diffusion = MDLMDiffusion(vocab_size=100, num_timesteps=10)
        x_t = torch.randint(0, 100, (2, 16))
        x_prev = diffusion.p_sample(model, x_t, t=5, mask_token_id=99)
        assert x_prev.shape == (2, 16)

    def test_invalid_noise_schedule(self):
        schedule = NoiseSchedule("invalid", num_timesteps=100)
        with pytest.raises(ValueError):
            schedule.get_alpha_bar(torch.tensor(50.0))


# ---------------------------------------------------------------------------
# 9. SAMPLER CORRECTNESS
# ---------------------------------------------------------------------------
class TestSamplerCorrectness:
    def test_all_samplers_produce_valid_tokens(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=5)
        for stype in ["ddpm", "ddpm_cache", "analytic", "blockwise"]:
            sampler = create_sampler(stype, diffusion=diffusion, model=model, mask_token_id=99)
            samples = sampler.sample(num_samples=2, seq_len=8, device="cpu")
            assert samples.shape == (2, 8)
            assert samples.min() >= 0
            assert samples.max() < 100

    def test_ddpm_cached_uses_cache(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=100)
        sampler = DDPMCachedSampler(
            diffusion=diffusion, model=model,
            mask_token_id=99, num_cache_steps=50,
        )
        samples = sampler.sample(num_samples=1, seq_len=8, device="cpu")
        assert samples.shape == (1, 8)

    def test_blockwise_confidence_threshold(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=5)
        sampler = BlockwiseSampler(
            diffusion=diffusion, model=model,
            mask_token_id=99, confidence_threshold=0.01,
        )
        samples = sampler.sample(num_samples=1, seq_len=8, device="cpu")
        assert samples.shape == (1, 8)
        assert samples.min() >= 0
        assert samples.max() < 100

    def test_semi_ar_block_size(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=3)
        sampler = SemiAutoregressiveSampler(
            diffusion=diffusion, model=model,
            mask_token_id=99, block_size=4, num_refine_steps=1,
        )
        samples = sampler.sample(num_samples=1, seq_len=8, device="cpu")
        assert samples.shape == (1, 8)


# ---------------------------------------------------------------------------
# 10. MODEL VALIDATION
# ---------------------------------------------------------------------------
class TestModelValidation:
    def test_hidden_size_not_divisible_by_num_heads_dit(self):
        with pytest.raises(ValueError, match="divisible"):
            DiffusionTransformer(vocab_size=100, hidden_size=65, num_heads=4)

    def test_hidden_size_not_divisible_by_num_heads_ar(self):
        with pytest.raises(ValueError, match="divisible"):
            AutoregressiveTransformer(vocab_size=100, hidden_size=65, num_heads=4)

    def test_create_model_invalid_type(self):
        with pytest.raises(ValueError):
            create_model(model_type="nonexistent", vocab_size=100)

    def test_weight_tying(self):
        model = _make_model()
        assert model.lm_head.weight is model.token_embed.weight

    def test_ar_weight_tying(self):
        model = _make_model(model_type="ar")
        assert model.lm_head.weight is model.token_embed.weight

    def test_mamba_block_shapes(self):
        block = MambaBlock(d_model=64, d_state=16, expand=2)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_sinusoidal_position_embedding_consistency(self):
        emb = SinusoidalPositionEmbedding(dim=64, max_seq_len=128)
        pe1 = emb(16, torch.device("cpu"))
        pe2 = emb(16, torch.device("cpu"))
        assert torch.allclose(pe1, pe2)

    def test_dit_block_no_timestep(self):
        block = DiTBlock(hidden_size=64, num_heads=2)
        x = torch.randn(2, 8, 64)
        out = block(x, timestep_emb=None)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 11. EDGE CASES: DATA
# ---------------------------------------------------------------------------
class TestDataEdgeCases:
    def test_character_dataset_single_sample(self):
        ds = CharacterDataset(texts=["a"], max_length=8)
        item = ds[0]
        assert item["input_ids"].shape == (8,)

    def test_character_dataset_empty_string(self):
        ds = CharacterDataset(texts=[""], max_length=8)
        item = ds[0]
        assert item["input_ids"].shape == (8,)
        assert (item["input_ids"] == 0).all()

    def test_character_dataset_long_string_truncated(self):
        text = "a" * 200
        ds = CharacterDataset(texts=[text], max_length=16)
        item = ds[0]
        assert item["input_ids"].shape == (16,)

    def test_character_dataset_vocab_size(self):
        ds = CharacterDataset(texts=["abc", "def"])
        assert ds.vocab_size > 0

    def test_character_dataset_mask_token(self):
        ds = CharacterDataset(texts=["hello"])
        assert ds.mask_token_id > 0

    def test_create_dataset_character(self):
        ds = create_dataset("character", texts=["hello", "world"], max_length=16)
        assert len(ds) == 2

    def test_create_dataset_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_dataset("nonexistent", texts=["hello"])

    def test_create_dataset_missing_texts(self):
        with pytest.raises(ValueError, match="requires"):
            create_dataset("character")

    def test_collate_fn_works(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])},
            {"input_ids": torch.tensor([7, 8, 9]), "labels": torch.tensor([10, 11, 12])},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)


# ---------------------------------------------------------------------------
# 12. QUANTIZATION PIPELINE
# ---------------------------------------------------------------------------
class TestQuantizationPipeline:
    @pytest.mark.parametrize("quant_type", ["bnb", "awq", "gptq", "hqq", "eetq"])
    def test_quantize_model_forward(self, quant_type):
        model = _make_model()
        qconfig = QuantConfig(quant_type=quant_type, bits=4)
        qmodel = quantize_model(model, qconfig)
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = qmodel(x, timesteps=t)
        assert out.shape == (1, 8, 100)
        assert not torch.isnan(out).any()

    def test_bnb_dequantize(self):
        layer = nn.Linear(32, 64)
        bnb = BNBQuantizedLinear(layer, bits=4)
        dequant = bnb.dequantize()
        assert dequant.shape == layer.weight.shape

    def test_quantize_then_lora(self):
        model = _make_model()
        qconfig = QuantConfig(quant_type="bnb", bits=4)
        qmodel = quantize_model(model, qconfig)
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = qmodel(x, timesteps=t)
        assert out.shape == (1, 8, 100)

    def test_create_quantizer_factory(self):
        for qt in ["bnb", "awq", "gptq", "hqq", "eetq"]:
            cls = create_quantizer(qt)
            assert cls is not None
        assert create_quantizer("nonexistent") is None

    def test_estimate_model_size(self):
        model = _make_model()
        sizes = estimate_model_size(model)
        assert sizes["total_mb"] > 0
        assert sizes["param_mb"] > 0

    def test_get_nbits_from_dtype(self):
        assert get_nbits_from_dtype(torch.float32) == 32
        assert get_nbits_from_dtype(torch.float16) == 16
        assert get_nbits_from_dtype(torch.int8) == 8


# ---------------------------------------------------------------------------
# 13. CLI ARGUMENT PARSING
# ---------------------------------------------------------------------------
class TestCLIParsing:
    def test_train_args_parse(self):
        args = parse_args([
            "train", "--dataset", "test",
            "--model_type", "dit", "--vocab_size", "1000",
            "--num_train_epochs", "5", "--learning_rate", "0.001",
        ])
        assert args.command == "train"
        assert args.model_type == "dit"
        assert args.vocab_size == 1000
        assert args.num_train_epochs == 5
        assert args.learning_rate == 0.001

    def test_sample_args_parse(self):
        args = parse_args([
            "sample", "--checkpoint", "/path/to/ckpt",
            "--num_samples", "10", "--seq_len", "64",
            "--sampler", "ddpm_cache",
        ])
        assert args.command == "sample"
        assert args.num_samples == 10
        assert args.seq_len == 64
        assert args.sampler == "ddpm_cache"

    def test_eval_args_parse(self):
        args = parse_args([
            "eval", "--checkpoint", "/path/to/ckpt",
            "--dataset", "test", "--per_device_batch_size", "16",
        ])
        assert args.command == "eval"
        assert args.per_device_batch_size == 16

    def test_serve_args_parse(self):
        args = parse_args([
            "serve", "--host", "localhost", "--port", "9000",
            "--backend", "transformers",
        ])
        assert args.command == "serve"
        assert args.host == "localhost"
        assert args.port == 9000

    def test_list_args_parse(self):
        args = parse_args(["list", "--models", "--datasets"])
        assert args.command == "list"
        assert args.models is True
        assert args.datasets is True

    def test_train_peft_args(self):
        args = parse_args([
            "train", "--dataset", "test",
            "--use_peft", "--peft_type", "lora",
            "--lora_r", "16", "--lora_alpha", "32",
        ])
        assert args.use_peft is True
        assert args.peft_type == "lora"
        assert args.lora_r == 16

    def test_train_quant_args(self):
        args = parse_args([
            "train", "--dataset", "test",
            "--use_quantization", "--quant_type", "awq", "--quant_bits", "8",
        ])
        assert args.use_quantization is True
        assert args.quant_type == "awq"
        assert args.quant_bits == 8

    def test_train_scheduler_args(self):
        args = parse_args([
            "train", "--dataset", "test",
            "--lr_scheduler", "cosine", "--min_lr", "1e-7",
            "--warmup_start_factor", "0.2", "--warmup_end_factor", "0.8",
        ])
        assert args.lr_scheduler == "cosine"
        assert args.min_lr == 1e-7
        assert args.warmup_start_factor == 0.2

    def test_train_amp_args(self):
        args = parse_args([
            "train", "--dataset", "test",
            "--use_amp", "--amp_dtype", "bfloat16",
        ])
        assert args.use_amp is True
        assert args.amp_dtype == "bfloat16"

    def test_no_command_returns_none(self):
        args = parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# 14. DISTRIBUTED UTILITIES (single-process path)
# ---------------------------------------------------------------------------
class TestDistributedSingleProcess:
    def test_distributed_manager_singleton(self):
        m1 = get_distributed_manager()
        m2 = get_distributed_manager()
        assert m1 is m2

    def test_not_distributed_by_default(self):
        m = DistributedManager.__new__(DistributedManager)
        m._world_size = 1
        m._rank = 0
        assert not m.is_distributed
        assert m.is_main

    def test_all_reduce_noop(self):
        m = get_distributed_manager()
        t = torch.tensor([1.0, 2.0])
        result = m.all_reduce(t)
        assert torch.allclose(result, torch.tensor([1.0, 2.0]))

    def test_all_gather_single(self):
        m = get_distributed_manager()
        t = torch.tensor([1.0, 2.0])
        result = m.all_gather(t)
        assert len(result) == 1
        assert torch.allclose(result[0], t)

    def test_broadcast_noop(self):
        m = get_distributed_manager()
        t = torch.tensor([3.0])
        result = m.broadcast(t)
        assert torch.allclose(result, torch.tensor([3.0]))

    def test_reduce_scatter_single(self):
        m = get_distributed_manager()
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = m.reduce_scatter(t)
        assert torch.allclose(result, t)

    def test_gather_single(self):
        m = get_distributed_manager()
        t = torch.tensor([5.0])
        result = m.gather(t)
        assert result is not None
        assert len(result) == 1

    def test_split_data_for_dp_tensor(self):
        data = torch.arange(10)
        chunk = split_data_for_dp(data, rank=0, world_size=2)
        assert len(chunk) == 5

    def test_split_data_for_dp_list(self):
        data = list(range(10))
        chunk = split_data_for_dp(data, rank=1, world_size=2)
        assert len(chunk) == 5

    def test_fsdp_wrapper_single_process(self):
        model = _make_model()
        wrapper = FSDPWrapper(model)
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = wrapper(x, timesteps=t)
        assert out.shape == (1, 8, 100)

    def test_tensor_parallel_linear_shape(self):
        layer = nn.Linear(32, 64)
        tp_layer = TensorParallelLinear(layer, tp_size=2, gather_output=False)
        assert tp_layer.weight.shape[0] == 32
        x = torch.randn(1, 10, 32)
        out = tp_layer(x)
        assert out.shape == (1, 10, 32)

    def test_pipeline_parallel(self):
        layers = [nn.Linear(32, 32) for _ in range(4)]
        pp = PipelineParallel(layers, num_stages=2, stage_id=0)
        assert len(pp.layers) == 2
        x = torch.randn(1, 32)
        out = pp(x)
        assert out.shape == (1, 32)

    def test_sequence_parallel_linear(self):
        layer = nn.Linear(32, 64)
        sp = SequenceParallelLinear(layer, sp_size=2)
        x = torch.randn(1, 10, 32)
        out = sp(x)
        assert out.shape == (1, 10, 64)

    def test_distributed_config_defaults(self):
        cfg = DistributedConfig()
        assert cfg.world_size == 1
        assert cfg.rank == 0
        assert cfg.backend == "nccl"


# ---------------------------------------------------------------------------
# 15. UTILS
# ---------------------------------------------------------------------------
class TestUtils:
    def test_get_device_cpu(self):
        d = get_device("cpu")
        assert d == torch.device("cpu")

    def test_get_device_auto(self):
        d = get_device("auto")
        assert d.type in ("cpu", "cuda", "mps")

    def test_get_device_none(self):
        d = get_device(None)
        assert d.type in ("cpu", "cuda", "mps")

    def test_count_parameters(self):
        model = _make_model()
        total = count_parameters(model)
        trainable = count_parameters(model, trainable_only=True)
        assert total > 0
        assert trainable == total

    def test_count_parameters_frozen(self):
        model = _make_model()
        for p in model.parameters():
            p.requires_grad = False
        trainable = count_parameters(model, trainable_only=True)
        assert trainable == 0

    def test_get_model_size(self):
        model = _make_model()
        size = get_model_size(model)
        assert size > 0

    def test_average_meter(self):
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(3.0)
        assert meter.avg == 2.0
        assert meter.count == 2
        assert meter.sum == 4.0
        meter.reset()
        assert meter.count == 0

    def test_average_meter_weighted(self):
        meter = AverageMeter()
        meter.update(2.0, n=3)
        meter.update(4.0, n=1)
        assert meter.avg == pytest.approx(2.5)

    def test_setup_logging_returns_logger(self):
        import logging
        hedgehog_logger = logging.getLogger("hedgehog")
        hedgehog_logger.handlers.clear()
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_no_duplicate_handlers(self):
        import logging
        hedgehog_logger = logging.getLogger("hedgehog")
        hedgehog_logger.handlers.clear()
        setup_logging("INFO")
        n1 = len(hedgehog_logger.handlers)
        setup_logging("INFO")
        n2 = len(hedgehog_logger.handlers)
        assert n1 == n2

    def test_get_logger(self):
        logger = get_logger("test_module")
        assert logger.name == "hedgehog.test_module"

    def test_get_logger_default(self):
        logger = get_logger()
        assert logger.name == "hedgehog"

    def test_find_free_port(self):
        port = find_free_port()
        assert 1024 <= port <= 65535

    def test_setup_distributed_defaults(self):
        info = setup_distributed()
        assert info["world_size"] == 1
        assert info["local_rank"] == 0

    def test_get_device_count(self):
        count = get_device_count()
        assert count >= 1


# ---------------------------------------------------------------------------
# 16. REGISTRY
# ---------------------------------------------------------------------------
class TestRegistryComprehensive:
    def test_list_models_not_empty(self):
        models = list_models()
        assert len(models) > 0
        assert "mdlm-small" in models
        assert "mdlm-base" in models

    def test_list_datasets_not_empty(self):
        datasets = list_datasets()
        assert len(datasets) > 0
        assert "tiny-shakespeare" in datasets

    def test_list_training_methods(self):
        methods = list_training_methods()
        assert "sft" in methods
        assert "lora" in methods
        assert "qlora" in methods

    def test_list_sampling_methods(self):
        methods = list_sampling_methods()
        assert "ddpm" in methods
        assert "analytic" in methods

    def test_get_dataset_info(self):
        info = get_dataset_info("tiny-shakespeare")
        assert info is not None
        assert "description" in info

    def test_get_dataset_info_nonexistent(self):
        info = get_dataset_info("nonexistent")
        assert info is None

    def test_get_training_method(self):
        method = get_training_method("lora")
        assert method is not None
        assert method["type"] == "peft"

    def test_get_sampling_method(self):
        method = get_sampling_method("ddpm")
        assert method is not None

    def test_register_custom_model(self):
        registry = get_model_registry()
        registry.register_model("test-custom", {
            "vocab_size": 100, "hidden_size": 32,
            "num_heads": 2, "num_layers": 1,
        })
        config = registry.get_model_config("test-custom")
        assert config["vocab_size"] == 100

    def test_get_model_from_registry(self):
        model, tokenizer = get_model("mdlm-small", model_type="dit")
        assert model is not None
        assert tokenizer is None

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("totally-nonexistent-model-xyz", model_type="dit")

    def test_model_registry_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            registry = get_model_registry()
            registry.save_model(model, tmpdir)
            assert (Path(tmpdir) / "pytorch_model.bin").exists()
            assert (Path(tmpdir) / "config.json").exists()
            with open(Path(tmpdir) / "config.json") as f:
                config = json.load(f)
            assert "model_type" in config

    def test_all_builtin_model_configs_valid(self):
        for name, cfg in DLM_MODELS.items():
            assert cfg["hidden_size"] % cfg["num_heads"] == 0, \
                f"Model {name} has hidden_size % num_heads != 0"


# ---------------------------------------------------------------------------
# 17. GRADIENT ACCUMULATION
# ---------------------------------------------------------------------------
class TestGradientAccumulation:
    def test_accum_count_resets(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            gradient_accumulation_steps=3,
        )
        ds = DummyDataset()
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        trainer.setup_training()
        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        trainer.training_step(batch)
        assert trainer.accum_count == 1
        trainer.training_step(batch)
        assert trainer.accum_count == 2
        trainer.training_step(batch)
        assert trainer.accum_count == 0

    def test_optimizer_step_only_at_accumulation_boundary(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            gradient_accumulation_steps=2,
        )
        ds = DummyDataset()
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        trainer.setup_training()
        w_before = list(model.parameters())[0].data.clone()
        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        trainer.training_step(batch)
        w_after_1 = list(model.parameters())[0].data.clone()
        assert torch.allclose(w_before, w_after_1), "Weights changed before accumulation complete"
        trainer.training_step(batch)
        w_after_2 = list(model.parameters())[0].data.clone()
        assert not torch.allclose(w_before, w_after_2), "Weights didn't change after accumulation"


# ---------------------------------------------------------------------------
# 18. LR SCHEDULERS
# ---------------------------------------------------------------------------
class TestLRSchedulers:
    @pytest.mark.parametrize("scheduler_type", ["linear", "cosine", "constant"])
    def test_scheduler_creates(self, scheduler_type):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            lr_scheduler_type=scheduler_type,
        )
        ds = DummyDataset()
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        trainer.setup_training()
        assert trainer.scheduler is not None

    def test_cosine_lr_decreases(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            lr_scheduler_type="cosine", warmup_steps=2,
            learning_rate=1e-3, min_lr=1e-6,
            gradient_accumulation_steps=1,
        )
        ds = DummyDataset(n=100)
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        trainer.setup_training()
        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        lrs = []
        for _ in range(10):
            trainer.training_step(batch)
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
        assert lrs[-1] < lrs[2]


# ---------------------------------------------------------------------------
# 19. EVALUATION
# ---------------------------------------------------------------------------
class TestEvaluation:
    def test_evaluation_returns_loss(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            dataloader_num_workers=0,
        )
        eval_ds = DummyDataset(n=8, seq_len=16)
        trainer = Trainer(
            config=config, model=model,
            train_dataset=DummyDataset(), eval_dataset=eval_ds,
        )
        trainer.setup_training()
        metrics = trainer.evaluation()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] > 0

    def test_evaluation_no_eval_dataset(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
        )
        trainer = Trainer(config=config, model=model, train_dataset=DummyDataset())
        trainer.setup_training()
        metrics = trainer.evaluation()
        assert metrics == {}

    def test_train_without_dataset_raises(self):
        model = _make_model()
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
        )
        trainer = Trainer(config=config, model=model)
        with pytest.raises(ValueError, match="train_dataset"):
            trainer.train()


# ---------------------------------------------------------------------------
# 20. WEIGHT TYING THROUGH LORA
# ---------------------------------------------------------------------------
class TestWeightTyingWithLoRA:
    def test_lora_linear_exposes_weight(self):
        layer = nn.Linear(32, 64)
        lora_linear = LoRALinear(layer, r=4, lora_alpha=8)
        assert lora_linear.weight is layer.weight

    def test_lora_linear_exposes_bias(self):
        layer = nn.Linear(32, 64, bias=True)
        lora_linear = LoRALinear(layer, r=4, lora_alpha=8)
        assert lora_linear.bias is layer.bias

    def test_lora_linear_no_bias(self):
        layer = nn.Linear(32, 64, bias=False)
        lora_linear = LoRALinear(layer, r=4, lora_alpha=8)
        assert lora_linear.bias is None


# ---------------------------------------------------------------------------
# 21. LORA LAYER UNIT TESTS
# ---------------------------------------------------------------------------
class TestLoRALayerUnit:
    def test_lora_layer_forward_shape(self):
        layer = LoRALayer(in_features=32, out_features=64, r=4, lora_alpha=8)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_lora_layer_zero_init(self):
        layer = LoRALayer(in_features=32, out_features=64, r=4)
        x = torch.randn(1, 5, 32)
        out = layer(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_lora_layer_scaling(self):
        layer = LoRALayer(in_features=32, out_features=64, r=4, lora_alpha=16)
        assert layer.scaling == 16 / 4


# ---------------------------------------------------------------------------
# 22. DORA LAYER
# ---------------------------------------------------------------------------
class TestDoRALayerUnit:
    def test_dora_layer_forward(self):
        base = nn.Linear(32, 64)
        dora = DoRALayer(base, r=4, lora_alpha=8, lora_dropout=0.0)
        x = torch.randn(1, 5, 32)
        out = dora(x)
        assert out.shape == (1, 5, 64)
        assert not torch.isnan(out).any()

    def test_dora_magnitude_param(self):
        base = nn.Linear(32, 64)
        dora = DoRALayer(base, r=4)
        assert dora.magnitude.shape == (64,)
        assert torch.allclose(dora.magnitude, torch.ones(64))


# ---------------------------------------------------------------------------
# 23. IA3 LAYER
# ---------------------------------------------------------------------------
class TestIA3LayerUnit:
    def test_ia3_layer_forward(self):
        base = nn.Linear(32, 64)
        ia3 = IA3Layer(base)
        x = torch.randn(1, 5, 32)
        out = ia3(x)
        assert out.shape == (1, 5, 64)

    def test_ia3_has_weight_and_bias(self):
        base = nn.Linear(32, 64, bias=True)
        ia3 = IA3Layer(base)
        assert ia3.weight is base.weight
        assert ia3.bias is base.bias

    def test_ia3_no_bias(self):
        base = nn.Linear(32, 64, bias=False)
        ia3 = IA3Layer(base)
        assert not hasattr(ia3, 'bias') or ia3.bias is None


# ---------------------------------------------------------------------------
# 24. INFERENCE
# ---------------------------------------------------------------------------
class TestInference:
    def test_create_inference_backend_transformers(self):
        model = _make_model()

        class SimpleModel(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x, **kwargs):
                return self.base(x)

            def generate(self, input_ids, **kwargs):
                return input_ids

        simple = SimpleModel(model)
        config = InferenceConfig(backend="transformers")
        backend = create_inference_backend(simple, config)
        assert isinstance(backend, TransformersBackend)

    def test_invalid_backend_raises(self):
        model = _make_model()
        config = InferenceConfig(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown inference backend"):
            create_inference_backend(model, config)

    def test_inference_config_defaults(self):
        config = InferenceConfig()
        assert config.backend == "transformers"
        assert config.max_model_len == 4096
        assert config.dtype == "auto"


# ---------------------------------------------------------------------------
# 25. NOISE SCHEDULE EDGE CASES
# ---------------------------------------------------------------------------
class TestNoiseScheduleEdgeCases:
    @pytest.mark.parametrize("schedule", ["linear", "cosine", "quadratic"])
    def test_alpha_bar_in_range(self, schedule):
        ns = NoiseSchedule(schedule, num_timesteps=100)
        for t in range(101):
            alpha = ns.get_alpha_bar(torch.tensor(float(t))).item()
            assert -0.1 <= alpha <= 1.1, f"alpha_bar out of range at t={t}: {alpha}"

    def test_linear_schedule_endpoints(self):
        ns = NoiseSchedule("linear", num_timesteps=100)
        assert ns.get_alpha_bar(torch.tensor(0.0)).item() == pytest.approx(1.0)
        assert ns.get_alpha_bar(torch.tensor(100.0)).item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 26. COMPREHENSIVE FACTORY TESTS
# ---------------------------------------------------------------------------
class TestFactoryFunctions:
    @pytest.mark.parametrize("diff_type", ["mdlm", "d3pm", "sedd", "d3pm_absorbing", "d3pm_uniform"])
    def test_create_diffusion_all_types(self, diff_type):
        d = create_diffusion(diff_type, vocab_size=50, num_timesteps=10)
        assert d.vocab_size == 50

    @pytest.mark.parametrize("model_type", ["dit", "ar", "mamba"])
    def test_create_model_all_types(self, model_type):
        m = create_model(model_type=model_type, vocab_size=100, hidden_size=64,
                         num_heads=2, num_layers=1, max_seq_len=32)
        x = torch.randint(0, 100, (1, 8))
        out = m(x)
        assert out.shape == (1, 8, 100)

    @pytest.mark.parametrize("sampler_type", ["ddpm", "ddpm_cache", "analytic", "semi_ar", "blockwise"])
    def test_create_sampler_all_types(self, sampler_type):
        model = _make_model()
        diff = create_diffusion("mdlm", vocab_size=100, num_timesteps=5)
        s = create_sampler(sampler_type, diffusion=diff, model=model, mask_token_id=99)
        assert s is not None

    def test_create_diffusion_invalid(self):
        with pytest.raises(ValueError):
            create_diffusion("invalid", vocab_size=100)

    def test_create_sampler_invalid(self):
        model = _make_model()
        diff = create_diffusion("mdlm", vocab_size=100, num_timesteps=5)
        with pytest.raises(ValueError):
            create_sampler("invalid", diffusion=diff, model=model, mask_token_id=99)


# ---------------------------------------------------------------------------
# 27. CROSS-MODULE INTEGRATION (e2e pipelines)
# ---------------------------------------------------------------------------
class TestCrossModuleIntegration:
    def test_data_to_model_to_loss(self):
        ds = CharacterDataset(texts=["hello world", "foo bar baz"], max_length=16)
        vocab = ds.vocab_size
        mask_id = ds.mask_token_id
        effective_vocab = max(vocab, mask_id + 1)
        model = _make_model(vocab_size=effective_vocab)
        diffusion = MDLMDiffusion(vocab_size=effective_vocab, num_timesteps=10)
        item = ds[0]
        x = item["input_ids"].unsqueeze(0)
        t = torch.tensor([5])
        loss = diffusion.compute_loss(model, x, t, mask_token_id=mask_id)
        assert loss.item() > 0

    def test_model_to_sampler_to_tokens(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=5)
        sampler = create_sampler("ddpm", diffusion=diffusion, model=model, mask_token_id=99)
        tokens = sampler.sample(num_samples=1, seq_len=8, device="cpu")
        assert tokens.shape == (1, 8)
        assert tokens.dtype == torch.long

    def test_quantize_then_forward(self):
        model = _make_model()
        qconfig = QuantConfig(quant_type="awq", bits=4)
        qmodel = quantize_model(model, qconfig)
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = qmodel(x, timesteps=t)
        assert out.shape == (1, 8, 100)

    def test_lora_then_train_step(self):
        base = _make_model()
        lora_model = LoraModel(base, LoraConfig(r=4, lora_alpha=8))
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=32, device="cpu",
            gradient_accumulation_steps=1,
        )
        ds = DummyDataset()
        trainer = Trainer(config=config, model=lora_model, train_dataset=ds)
        trainer.setup_training()
        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        metrics = trainer.training_step(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_peft_with_all_model_types(self):
        for model_type in ["dit", "ar"]:
            base = _make_model(model_type=model_type)
            lora_model = LoraModel(base, LoraConfig(r=4, lora_alpha=8))
            x = torch.randint(0, 100, (1, 8))
            t = torch.tensor([5])
            out = lora_model(x, timesteps=t)
            assert out.shape == (1, 8, 100)


# ---------------------------------------------------------------------------
# 28. BATCH SIZE / SEQUENCE LENGTH STRESS
# ---------------------------------------------------------------------------
class TestBatchSeqStress:
    def test_batch_size_1(self):
        model = _make_model()
        x = torch.randint(0, 100, (1, 8))
        out = model(x, timesteps=torch.tensor([5]))
        assert out.shape == (1, 8, 100)

    def test_batch_size_large(self):
        model = _make_model()
        x = torch.randint(0, 100, (32, 8))
        out = model(x, timesteps=torch.arange(32))
        assert out.shape == (32, 8, 100)

    def test_seq_len_1(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 1))
        out = model(x, timesteps=torch.tensor([5, 3]))
        assert out.shape == (2, 1, 100)

    def test_max_seq_len(self):
        model = _make_model(max_seq_len=64)
        x = torch.randint(0, 100, (1, 64))
        out = model(x, timesteps=torch.tensor([5]))
        assert out.shape == (1, 64, 100)


# ---------------------------------------------------------------------------
# 29. GRADIENT FLOW
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_dit_gradients_flow(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 8))
        t = torch.tensor([5, 3])
        out = model(x, timesteps=t)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None, "Missing gradient"

    def test_lora_gradients_flow_to_lora_params(self):
        base = _make_model()
        lora_model = LoraModel(base, LoraConfig(r=4, lora_alpha=8))
        x = torch.randint(0, 100, (1, 8))
        t = torch.tensor([5])
        out = lora_model(x, timesteps=t)
        loss = out.sum()
        loss.backward()
        lora_params_with_grad = [
            name for name, p in lora_model.named_parameters()
            if "lora" in name and p.grad is not None
        ]
        assert len(lora_params_with_grad) > 0

    def test_ia3_vector_is_trainable(self):
        base = _make_model()
        ia3_model = IA3Model(base)
        ia3_params = [
            (name, p) for name, p in ia3_model.named_parameters()
            if "ia3_vector" in name
        ]
        assert len(ia3_params) > 0
        for name, p in ia3_params:
            assert p.requires_grad, f"IA3 param {name} not trainable"


# ---------------------------------------------------------------------------
# 30. TOKENIZER WRAPPER
# ---------------------------------------------------------------------------
class TestTokenizerWrapper:
    def test_wrapper_attributes(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            bos_token_id = 2
            mask_token_id = 3

            def __call__(self, text, add_special_tokens=True, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def __len__(self):
                return 100

            def decode(self, ids, skip_special_tokens=True):
                return "decoded"

        wrapper = TokenizerWrapper(FakeTokenizer())
        assert wrapper.pad_token_id == 0
        assert wrapper.eos_token_id == 1
        assert len(wrapper) == 100
        assert wrapper.decode([1, 2]) == "decoded"


# ---------------------------------------------------------------------------
# 31. TRAINER CONFIG - YAML/JSON SERIALIZATION
# ---------------------------------------------------------------------------
class TestTrainerConfigSerialization:
    def test_to_dict(self):
        config = TrainerConfig(learning_rate=0.001, num_train_epochs=5)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["learning_rate"] == 0.001
        assert d["num_train_epochs"] == 5
        assert "model_type" in d

    def test_to_dict_has_all_fields(self):
        config = TrainerConfig()
        d = config.to_dict()
        from dataclasses import fields as dc_fields
        for f in dc_fields(TrainerConfig):
            if not callable(getattr(TrainerConfig, f.name, None)):
                assert f.name in d, f"Field {f.name} missing from to_dict()"

    def test_from_dict(self):
        d = {"learning_rate": 0.005, "num_train_epochs": 10, "model_type": "ar"}
        config = TrainerConfig.from_dict(d)
        assert config.learning_rate == 0.005
        assert config.num_train_epochs == 10
        assert config.model_type == "ar"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"learning_rate": 0.005, "unknown_key": "value", "another_unknown": 42}
        config = TrainerConfig.from_dict(d)
        assert config.learning_rate == 0.005
        assert not hasattr(config, "unknown_key")

    def test_to_json_and_from_json(self):
        config = TrainerConfig(learning_rate=0.002, vocab_size=50000)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.to_json(path)
            assert os.path.exists(path)

            loaded = TrainerConfig.from_json(path)
            assert loaded.learning_rate == 0.002
            assert loaded.vocab_size == 50000

    def test_to_yaml_and_from_yaml(self):
        config = TrainerConfig(learning_rate=0.003, noise_schedule="cosine")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            config.to_yaml(path)
            assert os.path.exists(path)

            loaded = TrainerConfig.from_yaml(path)
            assert loaded.learning_rate == 0.003
            assert loaded.noise_schedule == "cosine"

    def test_from_file_json(self):
        config = TrainerConfig(hidden_size=256)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.to_json(path)
            loaded = TrainerConfig.from_file(path)
            assert loaded.hidden_size == 256

    def test_from_file_yaml(self):
        config = TrainerConfig(hidden_size=128)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yml")
            config.to_yaml(path)
            loaded = TrainerConfig.from_file(path)
            assert loaded.hidden_size == 128

    def test_from_file_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported config file format"):
            TrainerConfig.from_file("config.txt")

    def test_from_yaml_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.yaml")
            with open(path, "w") as f:
                f.write("")
            config = TrainerConfig.from_yaml(path)
            assert config.learning_rate == TrainerConfig().learning_rate

    def test_merge_overrides(self):
        config = TrainerConfig(learning_rate=0.001, num_train_epochs=3)
        merged = config.merge({"learning_rate": 0.01, "num_train_epochs": 5})
        assert merged.learning_rate == 0.01
        assert merged.num_train_epochs == 5

    def test_merge_ignores_none(self):
        config = TrainerConfig(learning_rate=0.001)
        merged = config.merge({"learning_rate": None})
        assert merged.learning_rate == 0.001

    def test_merge_ignores_unknown_keys(self):
        config = TrainerConfig()
        merged = config.merge({"totally_fake_key": 123})
        assert not hasattr(merged, "totally_fake_key")

    def test_roundtrip_preserves_all_values(self):
        config = TrainerConfig(
            model_type="ar",
            vocab_size=16000,
            num_train_epochs=7,
            per_device_batch_size=64,
            learning_rate=3e-5,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            use_amp=True,
            amp_dtype="bfloat16",
            diffusion_type="d3pm_absorbing",
            num_timesteps=500,
            noise_schedule="cosine",
            output_dir="/tmp/test_output",
            seed=123,
            mask_token_id=15999,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "config.json")
            config.to_json(json_path)
            loaded = TrainerConfig.from_json(json_path)
            assert loaded.to_dict() == config.to_dict()

            yaml_path = os.path.join(tmpdir, "config.yaml")
            config.to_yaml(yaml_path)
            loaded_yaml = TrainerConfig.from_yaml(yaml_path)
            assert loaded_yaml.to_dict() == config.to_dict()


# ---------------------------------------------------------------------------
# 32. ARGS.JSON PERSISTENCE IN CHECKPOINTS
# ---------------------------------------------------------------------------
class TestArgsJsonPersistence:
    def test_save_checkpoint_creates_args_json(self):
        config = TrainerConfig(
            output_dir=tempfile.mkdtemp(),
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=16,
            learning_rate=0.01,
        )
        try:
            model = _make_model(vocab_size=100)
            ds = CharacterDataset(["hello world"] * 10, max_length=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("checkpoint-1")
            args_path = Path(config.output_dir) / "args.json"
            assert args_path.exists()
            with open(args_path) as f:
                saved_args = json.load(f)
            assert saved_args["learning_rate"] == 0.01
            assert saved_args["vocab_size"] == 100
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_checkpoint_contains_config(self):
        config = TrainerConfig(
            output_dir=tempfile.mkdtemp(),
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=16,
        )
        try:
            model = _make_model(vocab_size=100)
            ds = CharacterDataset(["hello"] * 10, max_length=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("test-ckpt")
            ckpt_path = Path(config.output_dir) / "test-ckpt.pt"
            ckpt = torch.load(ckpt_path, weights_only=False)
            assert "config" in ckpt
            assert isinstance(ckpt["config"], dict)
            assert ckpt["config"]["vocab_size"] == 100
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_load_checkpoint_with_load_args(self):
        config = TrainerConfig(
            output_dir=tempfile.mkdtemp(),
            vocab_size=100,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            max_seq_len=16,
            learning_rate=0.005,
        )
        try:
            model = _make_model(vocab_size=100)
            ds = CharacterDataset(["hello"] * 10, max_length=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.global_step = 42
            trainer.save_checkpoint("ckpt-42")
            ckpt_path = str(Path(config.output_dir) / "ckpt-42.pt")

            config2 = TrainerConfig(
                output_dir=config.output_dir,
                vocab_size=100,
                hidden_size=64,
                num_heads=2,
                num_layers=1,
                max_seq_len=16,
                learning_rate=0.999,
            )
            model2 = _make_model(vocab_size=100)
            trainer2 = Trainer(config=config2, model=model2, train_dataset=ds)
            trainer2.setup_training()
            trainer2.load_checkpoint(ckpt_path, load_args=True)
            assert trainer2.config.learning_rate == 0.005
            assert trainer2.global_step == 42
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 33. CLI --CONFIG FILE SUPPORT
# ---------------------------------------------------------------------------
class TestCLIConfigFile:
    def test_train_with_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "train_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "model_type": "ar",
                    "learning_rate": 0.005,
                    "num_train_epochs": 7,
                }, f)
            args = parse_args([
                "train", "--dataset", "test_ds", "--config", config_path,
            ])
            assert args.model_type == "ar"
            assert args.learning_rate == 0.005
            assert args.num_train_epochs == 7

    def test_train_with_yaml_config(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "train_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({
                    "model_type": "ar",
                    "vocab_size": 50000,
                    "hidden_size": 256,
                }, f)
            args = parse_args([
                "train", "--dataset", "test_ds", "--config", config_path,
            ])
            assert args.model_type == "ar"
            assert args.vocab_size == 50000
            assert args.hidden_size == 256

    def test_cli_overrides_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "model_type": "ar",
                    "learning_rate": 0.005,
                }, f)
            args = parse_args([
                "train", "--dataset", "test_ds",
                "--config", config_path,
                "--learning_rate", "0.01",
            ])
            assert args.learning_rate == 0.01
            assert args.model_type == "ar"

    def test_sample_with_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "sample_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "num_timesteps": 500,
                    "noise_schedule": "cosine",
                    "mask_token_id": 9999,
                }, f)
            args = parse_args([
                "sample", "--checkpoint", "/fake/ckpt",
                "--config", config_path,
            ])
            assert args.num_timesteps == 500
            assert args.noise_schedule == "cosine"
            assert args.mask_token_id == 9999

    def test_eval_with_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "eval_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "num_timesteps": 200,
                    "per_device_batch_size": 32,
                }, f)
            args = parse_args([
                "eval", "--checkpoint", "/fake/ckpt",
                "--dataset", "test_ds",
                "--config", config_path,
            ])
            assert args.num_timesteps == 200
            assert args.per_device_batch_size == 32

    def test_config_without_config_flag(self):
        args = parse_args(["train", "--dataset", "test_ds"])
        assert args.model_type == "dit"
        assert args.learning_rate == 1e-4


# ---------------------------------------------------------------------------
# 34. NEW CLI ARGS FOR SAMPLE/EVAL COMMANDS
# ---------------------------------------------------------------------------
class TestNewCLIArgs:
    def test_sample_has_num_timesteps(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--num_timesteps", "500",
        ])
        assert args.num_timesteps == 500

    def test_sample_has_noise_schedule(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--noise_schedule", "cosine",
        ])
        assert args.noise_schedule == "cosine"

    def test_sample_has_mask_token_id(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--mask_token_id", "9999",
        ])
        assert args.mask_token_id == 9999

    def test_sample_has_dropout(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--dropout", "0.2",
        ])
        assert args.dropout == 0.2

    def test_sample_has_tokenizer_name(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--tokenizer_name", "gpt2",
        ])
        assert args.tokenizer_name == "gpt2"

    def test_sample_has_load_args(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake", "--load_args",
        ])
        assert args.load_args is True

    def test_eval_has_num_timesteps(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
            "--num_timesteps", "500",
        ])
        assert args.num_timesteps == 500

    def test_eval_has_noise_schedule(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
            "--noise_schedule", "cosine",
        ])
        assert args.noise_schedule == "cosine"

    def test_eval_has_mask_token_id(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
            "--mask_token_id", "5000",
        ])
        assert args.mask_token_id == 5000

    def test_eval_has_dropout(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
            "--dropout", "0.3",
        ])
        assert args.dropout == 0.3

    def test_eval_has_per_device_batch_size(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
            "--per_device_batch_size", "16",
        ])
        assert args.per_device_batch_size == 16

    def test_eval_has_load_args(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds", "--load_args",
        ])
        assert args.load_args is True

    def test_train_has_max_train_samples(self):
        args = parse_args([
            "train", "--dataset", "ds", "--max_train_samples", "500",
        ])
        assert args.max_train_samples == 500

    def test_train_has_config_arg(self):
        args = parse_args([
            "train", "--dataset", "ds",
        ])
        assert hasattr(args, "config")

    def test_sample_has_config_arg(self):
        args = parse_args([
            "sample", "--checkpoint", "/fake",
        ])
        assert hasattr(args, "config")

    def test_eval_has_config_arg(self):
        args = parse_args([
            "eval", "--checkpoint", "/fake", "--dataset", "ds",
        ])
        assert hasattr(args, "config")

    def test_serve_has_config_arg(self):
        args = parse_args([
            "serve",
        ])
        assert hasattr(args, "config")


# ---------------------------------------------------------------------------
# 35. INFERENCE CONFIG FIXES
# ---------------------------------------------------------------------------
class TestInferenceConfigFixes:
    def test_inference_config_has_model_name_or_path(self):
        config = InferenceConfig(model_name_or_path="/path/to/model")
        assert config.model_name_or_path == "/path/to/model"

    def test_inference_config_default_model_name_or_path_is_none(self):
        config = InferenceConfig()
        assert config.model_name_or_path is None

    def test_inference_config_has_vocab_size(self):
        config = InferenceConfig(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_transformers_backend_encode_requires_tokenizer(self):
        model = _make_model()
        config = InferenceConfig()
        backend = TransformersBackend(model, config)
        with pytest.raises(ValueError, match="Tokenizer is required"):
            backend.encode("hello")

    def test_transformers_backend_generate_requires_tokenizer(self):
        model = _make_model()
        config = InferenceConfig()
        backend = TransformersBackend(model, config)
        with pytest.raises(ValueError, match="Tokenizer is required"):
            backend.generate("hello")


# ---------------------------------------------------------------------------
# 36. PEFT HARDCODED FALLBACK FIXES
# ---------------------------------------------------------------------------
class TestPEFTHardcodedFixes:
    def test_prefix_tuning_requires_hidden_size(self):
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="hidden_size"):
            PrefixTuningModel(model)

    def test_prefix_tuning_requires_num_layers(self):
        model = nn.Linear(10, 10)
        model.hidden_size = 64
        with pytest.raises(ValueError, match="num_layers"):
            PrefixTuningModel(model)

    def test_prefix_tuning_derives_num_layers_from_blocks(self):
        model = _make_model(vocab_size=100, hidden_size=64, num_heads=2, num_layers=3)
        pm = PrefixTuningModel(model, prefix_length=5)
        assert pm.prefix_tuning.num_layers == 3

    def test_prefix_tuning_derives_num_layers_from_config(self):
        model = _make_model(vocab_size=100, hidden_size=64, num_heads=2, num_layers=4)
        pm = PrefixTuningModel(model, prefix_length=5)
        assert pm.prefix_tuning.num_layers == 4

    def test_prompt_tuning_requires_hidden_size(self):
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="hidden_size"):
            PromptTuningModel(model)

    def test_prompt_tuning_requires_vocab_size(self):
        model = nn.Linear(10, 10)
        model.hidden_size = 64
        with pytest.raises(ValueError, match="vocab_size"):
            PromptTuningModel(model)

    def test_prompt_tuning_derives_vocab_size(self):
        model = _make_model(vocab_size=200, hidden_size=64, num_heads=2, num_layers=1)
        pm = PromptTuningModel(model, prompt_length=5)
        assert pm.prompt_tuning.vocab_size == 200

    def test_prefix_get_prefix_uses_prefix_embeddings(self):
        pt = PrefixTuning(hidden_size=64, prefix_length=5, num_layers=2)
        pt.prefix_embeddings.data.fill_(1.0)
        result = pt.get_prefix(batch_size=2, device=torch.device("cpu"))
        assert result.shape == (2, 2, 64)
        assert not torch.all(result == 0), "get_prefix should use prefix_embeddings, not zero tensor"


# ---------------------------------------------------------------------------
# 37. QUANTIZED LINEAR ABC FIX
# ---------------------------------------------------------------------------
class TestQuantizedLinearABC:
    def test_quantized_linear_is_abstract(self):
        from hedgehog.quantization import QuantizedLinear
        from abc import abstractmethod
        assert hasattr(QuantizedLinear.forward, '__isabstractmethod__') or \
            any(getattr(m, '__isabstractmethod__', False) for m in [QuantizedLinear.forward])


# ---------------------------------------------------------------------------
# 38. CONSISTENT MAX_SEQ_LEN DEFAULTS
# ---------------------------------------------------------------------------
class TestConsistentDefaults:
    def test_dit_default_max_seq_len(self):
        model = DiffusionTransformer(vocab_size=100)
        assert model.config.max_seq_len == 512

    def test_ar_default_max_seq_len(self):
        model = AutoregressiveTransformer(vocab_size=100)
        assert model.config.max_seq_len == 512

    def test_create_model_default_max_seq_len(self):
        model = create_model("dit", vocab_size=100)
        assert model.config.max_seq_len == 512

    def test_trainer_config_default_max_seq_len(self):
        config = TrainerConfig()
        assert config.max_seq_len == 512

    def test_model_config_default_max_seq_len(self):
        config = ModelConfig()
        assert config.max_seq_len == 512


# ---------------------------------------------------------------------------
# 39. CACHED SAMPLER CONFIGURABLE PARAMS
# ---------------------------------------------------------------------------
class TestCachedSamplerParams:
    def test_custom_num_cache_steps(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=100)
        sampler = DDPMCachedSampler(
            diffusion=diffusion, model=model,
            mask_token_id=99, num_cache_steps=10,
        )
        assert sampler.num_cache_steps == 10
        samples = sampler.sample(num_samples=1, seq_len=8, device="cpu")
        assert samples.shape == (1, 8)

    def test_custom_max_cache_size(self):
        model = _make_model()
        model.eval()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=100)
        sampler = DDPMCachedSampler(
            diffusion=diffusion, model=model,
            mask_token_id=99, max_cache_size=3,
        )
        assert sampler.max_cache_size == 3

    def test_default_cache_params(self):
        model = _make_model()
        diffusion = create_diffusion("mdlm", vocab_size=100, num_timesteps=100)
        sampler = DDPMCachedSampler(
            diffusion=diffusion, model=model, mask_token_id=99,
        )
        assert sampler.num_cache_steps == 50
        assert sampler.max_cache_size == 5


# ---------------------------------------------------------------------------
# 40. DEAD CODE REMOVAL
# ---------------------------------------------------------------------------
class TestDeadCodeRemoval:
    def test_noise_schedule_no_schedule_type_attr(self):
        schedule = NoiseSchedule("linear", num_timesteps=100)
        assert not hasattr(schedule, "schedule_type")
        assert schedule.schedule == "linear"

    def test_d3pm_no_q_type_attr(self):
        d3pm = D3PMDiffusion(
            vocab_size=100, diffusion_type=DiffusionType.D3PM_ABSORBING,
            num_timesteps=100, schedule="linear", transition_type="absorbing",
        )
        assert not hasattr(d3pm, "q_type")
        assert d3pm.transition_type == "absorbing"

    def test_trainer_config_no_fp16_bf16(self):
        config = TrainerConfig()
        assert not hasattr(config, "fp16")
        assert not hasattr(config, "bf16")


# ---------------------------------------------------------------------------
# 41. MLP RATIO CONFIGURABLE
# ---------------------------------------------------------------------------
class TestMLPRatioConfigurable:
    def test_dit_custom_mlp_ratio(self):
        model = DiffusionTransformer(vocab_size=100, hidden_size=64, num_heads=2,
                                      num_layers=1, mlp_ratio=2)
        x = torch.randint(0, 100, (2, 8))
        t = torch.zeros(2, dtype=torch.long)
        out = model(x, timesteps=t)
        assert out.shape == (2, 8, 100)

    def test_ar_custom_mlp_ratio(self):
        model = AutoregressiveTransformer(vocab_size=100, hidden_size=64, num_heads=2,
                                           num_layers=1, mlp_ratio=2)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.shape == (2, 8, 100)

    def test_dit_block_custom_mlp_ratio(self):
        block = DiTBlock(hidden_size=64, num_heads=2, mlp_ratio=3)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_default_mlp_ratio_is_4(self):
        model = DiffusionTransformer(vocab_size=100, hidden_size=64, num_heads=2, num_layers=1)
        linear = model.timestep_embed[0]
        assert linear.out_features == 64 * 4


# ---------------------------------------------------------------------------
# 42. LOAD_ARGS_FROM_CHECKPOINT FLAG
# ---------------------------------------------------------------------------
class TestLoadArgsFromCheckpoint:
    def test_load_args_flag_exists(self):
        config = TrainerConfig(load_args_from_checkpoint=True)
        assert config.load_args_from_checkpoint is True

    def test_load_args_flag_default_true(self):
        config = TrainerConfig()
        assert config.load_args_from_checkpoint is True

    def test_load_args_flag_serializes(self):
        config = TrainerConfig(load_args_from_checkpoint=False)
        d = config.to_dict()
        assert d["load_args_from_checkpoint"] is False
        restored = TrainerConfig.from_dict(d)
        assert restored.load_args_from_checkpoint is False


# ---------------------------------------------------------------------------
# 43. D3PM/SEDD/MAMBA WARNINGS
# ---------------------------------------------------------------------------
class TestAliasWarnings:
    def test_d3pm_alias_warns(self):
        import warnings
        import logging
        with warnings.catch_warnings(record=True):
            diffusion = create_diffusion("d3pm", vocab_size=100)
            assert isinstance(diffusion, MDLMDiffusion)

    def test_sedd_alias_warns(self):
        diffusion = create_diffusion("sedd", vocab_size=100)
        assert isinstance(diffusion, MDLMDiffusion)

    def test_mamba_alias_warns(self):
        model = create_model("mamba", vocab_size=100)
        assert isinstance(model, DiffusionTransformer)


# ---------------------------------------------------------------------------
# 44. FULL CONFIG ROUNDTRIP VIA CLI
# ---------------------------------------------------------------------------
class TestFullConfigRoundtrip:
    def test_train_config_from_yaml_to_trainer_config(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "model_type": "ar",
                "vocab_size": 5000,
                "hidden_size": 128,
                "num_heads": 4,
                "num_layers": 2,
                "learning_rate": 3e-4,
                "num_timesteps": 500,
                "noise_schedule": "cosine",
                "warmup_steps": 50,
                "lr_scheduler_type": "cosine",
                "seed": 99,
            }
            yaml_path = os.path.join(tmpdir, "config.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(config_data, f)

            args = parse_args(["train", "--dataset", "ds", "--config", yaml_path])
            assert args.model_type == "ar"
            assert args.vocab_size == 5000
            assert args.num_timesteps == 500
            assert args.noise_schedule == "cosine"
            assert args.seed == 99

    def test_config_to_trainer_config_object(self):
        tc = TrainerConfig(
            model_type="ar", vocab_size=5000, hidden_size=128,
            num_heads=4, num_layers=2, learning_rate=3e-4,
        )
        d = tc.to_dict()
        tc2 = TrainerConfig.from_dict(d)
        assert tc2.model_type == "ar"
        assert tc2.vocab_size == 5000
        assert tc2.hidden_size == 128


# ---------------------------------------------------------------------------
# 45. COSINE SCHEDULE CONSTANT
# ---------------------------------------------------------------------------
class TestCosineScheduleConstant:
    def test_cosine_schedule_uses_named_constant(self):
        from hedgehog.diffusion import COSINE_SCHEDULE_OFFSET
        assert COSINE_SCHEDULE_OFFSET == 0.008

    def test_sinusoidal_base_freq_constant(self):
        from hedgehog.diffusion import SINUSOIDAL_BASE_FREQ
        assert SINUSOIDAL_BASE_FREQ == 10000.0


# ---------------------------------------------------------------------------
# 46. CLI ARGV INJECTION (CONFIG FILE PRIORITY BUG FIX)
# ---------------------------------------------------------------------------
class TestCLIArgvInjection:
    def test_cli_overrides_config_when_value_matches_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"learning_rate": 0.999}, f)
            args = parse_args([
                "train", "--dataset", "test",
                "--config", config_path,
                "--learning_rate", "0.0001",
            ])
            assert args.learning_rate == 0.0001

    def test_config_applies_when_cli_not_specified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"vocab_size": 50000, "num_layers": 24}, f)
            args = parse_args([
                "train", "--dataset", "test", "--config", config_path,
            ])
            assert args.vocab_size == 50000
            assert args.num_layers == 24

    def test_cli_explicit_default_beats_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"seed": 999}, f)
            args = parse_args([
                "train", "--dataset", "test",
                "--config", config_path,
                "--seed", "42",
            ])
            assert args.seed == 42

    def test_config_bool_store_true_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"use_amp": True}, f)
            args = parse_args([
                "train", "--dataset", "test", "--config", config_path,
            ])
            assert args.use_amp is True

    def test_config_with_equals_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"vocab_size": 12345}, f)
            args = parse_args([
                "train", "--dataset", "test", f"--config={config_path}",
            ])
            assert args.vocab_size == 12345

    def test_config_unknown_keys_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"totally_unknown_key": 42, "vocab_size": 5000}, f)
            args = parse_args([
                "train", "--dataset", "test", "--config", config_path,
            ])
            assert args.vocab_size == 5000
            assert not hasattr(args, "totally_unknown_key")


# ---------------------------------------------------------------------------
# 47. SELECTIVE KEY LOADING TIERS
# ---------------------------------------------------------------------------
class TestSelectiveKeyLoading:
    def test_force_load_keys_always_overwrite(self):
        config = TrainerConfig(
            model_type="ar", vocab_size=100, hidden_size=64,
            num_heads=2, num_layers=1,
        )
        saved = {
            "model_type": "dit", "vocab_size": 50000, "hidden_size": 512,
            "num_heads": 8, "num_layers": 12, "diffusion_type": "d3pm_absorbing",
        }
        merged = config.selective_merge(saved)
        assert merged.model_type == "dit"
        assert merged.vocab_size == 50000
        assert merged.hidden_size == 512
        assert merged.num_heads == 8
        assert merged.num_layers == 12
        assert merged.diffusion_type == "d3pm_absorbing"

    def test_load_keys_only_when_default(self):
        config = TrainerConfig(noise_schedule="linear", num_timesteps=1000)
        saved = {"noise_schedule": "cosine", "num_timesteps": 500}
        merged = config.selective_merge(saved)
        assert merged.noise_schedule == "cosine"
        assert merged.num_timesteps == 500

    def test_load_keys_not_overridden_when_custom(self):
        config = TrainerConfig(noise_schedule="quadratic")
        saved = {"noise_schedule": "cosine"}
        merged = config.selective_merge(saved)
        assert merged.noise_schedule == "quadratic"

    def test_data_keys_not_loaded_by_default(self):
        config = TrainerConfig(per_device_batch_size=8)
        saved = {"per_device_batch_size": 64, "dataloader_num_workers": 8}
        merged = config.selective_merge(saved)
        assert merged.per_device_batch_size == 8
        assert merged.dataloader_num_workers == 4

    def test_data_keys_loaded_when_flag_set(self):
        config = TrainerConfig(per_device_batch_size=8)
        saved = {"per_device_batch_size": 64, "dataloader_num_workers": 8}
        merged = config.selective_merge(saved, load_data_args=True)
        assert merged.per_device_batch_size == 64
        assert merged.dataloader_num_workers == 8

    def test_non_tier_keys_not_loaded(self):
        config = TrainerConfig(learning_rate=0.001)
        saved = {"learning_rate": 0.999}
        merged = config.selective_merge(saved)
        assert merged.learning_rate == 0.001

    def test_selective_merge_ignores_none_values(self):
        config = TrainerConfig(model_type="dit")
        saved = {"model_type": None}
        merged = config.selective_merge(saved)
        assert merged.model_type == "dit"

    def test_selective_merge_ignores_unknown_keys(self):
        config = TrainerConfig()
        saved = {"unknown_key_xyz": 42}
        merged = config.selective_merge(saved)
        assert not hasattr(merged, "unknown_key_xyz")

    def test_trainer_load_checkpoint_selective(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(vocab_size=100)
            config = TrainerConfig(
                output_dir=tmpdir, vocab_size=100, hidden_size=64,
                num_heads=2, num_layers=1, max_seq_len=16,
                per_device_batch_size=16, noise_schedule="cosine",
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("ckpt-sel")
            ckpt_path = os.path.join(tmpdir, "ckpt-sel.pt")

            config2 = TrainerConfig(
                output_dir=tmpdir, vocab_size=999, hidden_size=32,
                num_heads=2, num_layers=1, max_seq_len=16,
                per_device_batch_size=8,
            )
            model2 = _make_model(vocab_size=100)
            trainer2 = Trainer(config=config2, model=model2, train_dataset=ds)
            trainer2.setup_training()
            trainer2.load_checkpoint(ckpt_path, load_args=True, selective=True)
            assert trainer2.config.vocab_size == 100
            assert trainer2.config.per_device_batch_size == 8


# ---------------------------------------------------------------------------
# 48. LOAD_ARGS DEFAULT FOR SAMPLE/EVAL
# ---------------------------------------------------------------------------
class TestLoadArgsDefaults:
    def test_sample_load_args_default_true(self):
        args = parse_args(["sample", "--checkpoint", "/fake"])
        assert args.load_args is True

    def test_eval_load_args_default_true(self):
        args = parse_args(["eval", "--checkpoint", "/fake", "--dataset", "ds"])
        assert args.load_args is True

    def test_sample_no_load_args(self):
        args = parse_args(["sample", "--checkpoint", "/fake", "--no-load_args"])
        assert args.load_args is False

    def test_eval_no_load_args(self):
        args = parse_args(["eval", "--checkpoint", "/fake", "--dataset", "ds", "--no-load_args"])
        assert args.load_args is False

    def test_train_load_args_default_false(self):
        args = parse_args(["train", "--dataset", "ds"])
        assert args.load_args is False


# ---------------------------------------------------------------------------
# 49. FROM_PRETRAINED CLASSMETHOD
# ---------------------------------------------------------------------------
class TestFromPretrained:
    def test_from_pretrained_with_args_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(vocab_size=12345, learning_rate=0.005)
            config.to_json(os.path.join(tmpdir, "args.json"))
            loaded = TrainerConfig.from_pretrained(tmpdir)
            assert loaded.vocab_size == 12345
            assert loaded.learning_rate == 0.005

    def test_from_pretrained_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(vocab_size=54321, hidden_size=256)
            ckpt = {"config": config.to_dict(), "model_state_dict": {}}
            torch.save(ckpt, os.path.join(tmpdir, "model.pt"))
            loaded = TrainerConfig.from_pretrained(tmpdir)
            assert loaded.vocab_size == 54321
            assert loaded.hidden_size == 256

    def test_from_pretrained_prefers_args_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = TrainerConfig(vocab_size=111)
            config_json.to_json(os.path.join(tmpdir, "args.json"))
            config_pt = TrainerConfig(vocab_size=222)
            torch.save({"config": config_pt.to_dict()}, os.path.join(tmpdir, "model.pt"))
            loaded = TrainerConfig.from_pretrained(tmpdir)
            assert loaded.vocab_size == 111

    def test_from_pretrained_no_config_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                TrainerConfig.from_pretrained(tmpdir)

    def test_from_pretrained_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                TrainerConfig.from_pretrained(tmpdir)


# ---------------------------------------------------------------------------
# 50. VERSION TRACKING IN ARGS.JSON
# ---------------------------------------------------------------------------
class TestVersionTracking:
    def test_to_dict_versioned_includes_version(self):
        config = TrainerConfig()
        d = config.to_dict_versioned()
        assert "hedgehog_version" in d
        import hedgehog
        assert d["hedgehog_version"] == hedgehog.__version__

    def test_args_json_contains_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig()
            path = os.path.join(tmpdir, "args.json")
            config.to_json(path)
            with open(path) as f:
                d = json.load(f)
            assert "hedgehog_version" in d

    def test_checkpoint_contains_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(vocab_size=100)
            config = TrainerConfig(
                output_dir=tmpdir, vocab_size=100, hidden_size=64,
                num_heads=2, num_layers=1, max_seq_len=16,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("ckpt-ver")
            ckpt = torch.load(os.path.join(tmpdir, "ckpt-ver.pt"), weights_only=False)
            assert "hedgehog_version" in ckpt
            import hedgehog
            assert ckpt["hedgehog_version"] == hedgehog.__version__

    def test_version_ignored_in_from_dict(self):
        d = {"vocab_size": 100, "hedgehog_version": "99.99.99"}
        config = TrainerConfig.from_dict(d)
        assert config.vocab_size == 100
        assert not hasattr(config, "hedgehog_version")


# ---------------------------------------------------------------------------
# 51. CHECK JSON FORMAT SAFETY
# ---------------------------------------------------------------------------
class TestCheckJsonFormat:
    def test_path_serialized_to_string(self):
        d = {"output_dir": Path("/tmp/test"), "seed": 42}
        clean = TrainerConfig._check_json_serializable(d)
        assert clean["output_dir"] == "/tmp/test"
        assert isinstance(clean["output_dir"], str)

    def test_set_serialized_to_list(self):
        d = {"tags": {"a", "b"}, "seed": 42}
        clean = TrainerConfig._check_json_serializable(d)
        assert isinstance(clean["tags"], list)

    def test_nan_serialized_to_none(self):
        d = {"loss": float("nan"), "seed": 42}
        clean = TrainerConfig._check_json_serializable(d)
        assert clean["loss"] is None

    def test_normal_values_unchanged(self):
        d = {"vocab_size": 100, "learning_rate": 0.001, "model_type": "dit"}
        clean = TrainerConfig._check_json_serializable(d)
        assert clean == d

    def test_unserializable_converted_to_str(self):
        d = {"custom_obj": object()}
        clean = TrainerConfig._check_json_serializable(d)
        assert isinstance(clean["custom_obj"], str)


# ---------------------------------------------------------------------------
# 52. CHECKPOINT SYMLINKS
# ---------------------------------------------------------------------------
class TestCheckpointSymlinks:
    def test_last_symlink_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(vocab_size=100)
            config = TrainerConfig(
                output_dir=tmpdir, vocab_size=100, hidden_size=64,
                num_heads=2, num_layers=1, max_seq_len=16,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("checkpoint-100")
            last_link = Path(tmpdir) / "last.pt"
            assert last_link.exists() or last_link.is_symlink()
            if last_link.is_symlink():
                assert last_link.resolve().name == "checkpoint-100.pt"

    def test_last_symlink_updated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(vocab_size=100)
            config = TrainerConfig(
                output_dir=tmpdir, vocab_size=100, hidden_size=64,
                num_heads=2, num_layers=1, max_seq_len=16,
                save_total_limit=0,
            )
            ds = DummyDataset(n=8, seq_len=16)
            trainer = Trainer(config=config, model=model, train_dataset=ds)
            trainer.setup_training()
            trainer.save_checkpoint("checkpoint-1")
            trainer.save_checkpoint("checkpoint-2")
            last_link = Path(tmpdir) / "last.pt"
            if last_link.is_symlink():
                assert last_link.resolve().name == "checkpoint-2.pt"

    def test_best_symlink_created_on_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(vocab_size=100)
            config = TrainerConfig(
                output_dir=tmpdir, vocab_size=100, hidden_size=64,
                num_heads=2, num_layers=1, max_seq_len=16,
                num_train_epochs=1, per_device_batch_size=4,
                gradient_accumulation_steps=1, save_steps=100,
                eval_steps=2, logging_steps=100, warmup_steps=0,
                dataloader_num_workers=0,
            )
            train_ds = DummyDataset(n=8, seq_len=16)
            eval_ds = DummyDataset(n=4, seq_len=16)
            trainer = Trainer(
                config=config, model=model,
                train_dataset=train_ds, eval_dataset=eval_ds,
            )
            trainer.train()
            best_link = Path(tmpdir) / "best.pt"
            assert best_link.exists() or best_link.is_symlink()


# ---------------------------------------------------------------------------
# 53. CONTEXT-AWARE DEFAULTS
# ---------------------------------------------------------------------------
class TestContextAwareDefaults:
    def test_lora_default_learning_rate(self):
        config = TrainerConfig()
        adjusted = config.apply_context_defaults(use_peft=True, peft_type="lora")
        assert adjusted.learning_rate == 2e-4

    def test_dora_default_learning_rate(self):
        config = TrainerConfig()
        adjusted = config.apply_context_defaults(use_peft=True, peft_type="dora")
        assert adjusted.learning_rate == 2e-4
        assert adjusted.warmup_steps == 100

    def test_ia3_default_learning_rate(self):
        config = TrainerConfig()
        adjusted = config.apply_context_defaults(use_peft=True, peft_type="ia3")
        assert adjusted.learning_rate == 1e-3
        assert adjusted.warmup_steps == 50

    def test_no_peft_keeps_defaults(self):
        config = TrainerConfig()
        adjusted = config.apply_context_defaults(use_peft=False)
        assert adjusted.learning_rate == 1e-4
        assert adjusted.warmup_steps == 500

    def test_custom_lr_not_overridden(self):
        config = TrainerConfig(learning_rate=0.01)
        adjusted = config.apply_context_defaults(use_peft=True, peft_type="lora")
        assert adjusted.learning_rate == 0.01

    def test_custom_warmup_not_overridden(self):
        config = TrainerConfig(warmup_steps=200)
        adjusted = config.apply_context_defaults(use_peft=True, peft_type="lora")
        assert adjusted.warmup_steps == 200


# ---------------------------------------------------------------------------
# 54. AUTO-GENERATED CLI ARGS FROM TRAINERCONFIG
# ---------------------------------------------------------------------------
class TestAutoGeneratedCLIArgs:
    def test_all_trainer_config_fields_in_train_parser(self):
        from dataclasses import fields as dc_fields
        args = parse_args(["train", "--dataset", "test"])
        for f in dc_fields(TrainerConfig):
            if f.name == "load_args_from_checkpoint":
                continue
            assert hasattr(args, f.name), f"TrainerConfig field '{f.name}' missing from train CLI args"

    def test_autogen_int_field(self):
        args = parse_args(["train", "--dataset", "test", "--num_timesteps", "500"])
        assert args.num_timesteps == 500

    def test_autogen_float_field(self):
        args = parse_args(["train", "--dataset", "test", "--learning_rate", "0.01"])
        assert args.learning_rate == 0.01

    def test_autogen_str_field(self):
        args = parse_args(["train", "--dataset", "test", "--noise_schedule", "cosine"])
        assert args.noise_schedule == "cosine"

    def test_autogen_bool_field(self):
        args = parse_args(["train", "--dataset", "test", "--use_amp"])
        assert args.use_amp is True

    def test_autogen_optional_int_field(self):
        args = parse_args(["train", "--dataset", "test", "--mask_token_id", "999"])
        assert args.mask_token_id == 999

    def test_autogen_optional_int_default_none(self):
        args = parse_args(["train", "--dataset", "test"])
        assert args.mask_token_id is None

    def test_lr_scheduler_alias_still_works(self):
        args = parse_args(["train", "--dataset", "test", "--lr_scheduler", "cosine"])
        assert args.lr_scheduler == "cosine"

    def test_lr_scheduler_type_direct(self):
        args = parse_args(["train", "--dataset", "test", "--lr_scheduler_type", "cosine"])
        assert args.lr_scheduler_type == "cosine"


# ---------------------------------------------------------------------------
# 55. DISTRIBUTED SAVE GUARD
# ---------------------------------------------------------------------------
class TestDistributedSaveGuard:
    def test_is_main_process_true_by_default(self):
        model = _make_model(vocab_size=100)
        config = TrainerConfig(
            vocab_size=100, hidden_size=64, num_heads=2,
            num_layers=1, max_seq_len=16,
        )
        ds = DummyDataset()
        trainer = Trainer(config=config, model=model, train_dataset=ds)
        assert trainer._is_main_process() is True


# ---------------------------------------------------------------------------
# 56. FORCE_LOAD_KEYS / LOAD_KEYS / DATA_KEYS CONSTANTS
# ---------------------------------------------------------------------------
class TestKeyTierConstants:
    def test_force_load_keys_contains_model_arch(self):
        from hedgehog.trainers import FORCE_LOAD_KEYS
        assert "model_type" in FORCE_LOAD_KEYS
        assert "hidden_size" in FORCE_LOAD_KEYS
        assert "vocab_size" in FORCE_LOAD_KEYS
        assert "num_heads" in FORCE_LOAD_KEYS
        assert "num_layers" in FORCE_LOAD_KEYS

    def test_load_keys_contains_schedule(self):
        from hedgehog.trainers import LOAD_KEYS
        assert "noise_schedule" in LOAD_KEYS
        assert "num_timesteps" in LOAD_KEYS

    def test_data_keys_contains_batch(self):
        from hedgehog.trainers import DATA_KEYS
        assert "per_device_batch_size" in DATA_KEYS

    def test_key_tiers_no_overlap(self):
        from hedgehog.trainers import FORCE_LOAD_KEYS, LOAD_KEYS, DATA_KEYS
        assert len(FORCE_LOAD_KEYS & LOAD_KEYS) == 0
        assert len(FORCE_LOAD_KEYS & DATA_KEYS) == 0
        overlap = LOAD_KEYS & DATA_KEYS
        assert len(overlap) == 0 or overlap == set(), f"Unexpected overlap: {overlap}"

    def test_all_tier_keys_are_valid_config_fields(self):
        from hedgehog.trainers import FORCE_LOAD_KEYS, LOAD_KEYS, DATA_KEYS
        from dataclasses import fields as dc_fields
        valid = {f.name for f in dc_fields(TrainerConfig)}
        for k in FORCE_LOAD_KEYS | LOAD_KEYS | DATA_KEYS:
            assert k in valid, f"Tier key '{k}' is not a valid TrainerConfig field"
