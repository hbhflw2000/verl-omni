# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Catch FSDP dispatch and stale V0 config keys before allocating Megatron GPUs."""

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import McoreActorConfig, McoreEngineConfig

from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync
from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

CONFIG = Path(__file__).parents[3] / "examples/gspo_trainer/qwen3_omni/config"


def test_public_recipe_selects_megatron_and_v1_separate_async(monkeypatch, tmp_path):
    tb_dir = str(tmp_path / "tensorboard")
    monkeypatch.setenv("TENSORBOARD_DIR", tb_dir)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        config = compose(config_name="audiomcq_megatron_separate_async")
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor, McoreActorConfig)
    assert isinstance(actor.engine, McoreEngineConfig)
    assert config.actor_rollout_ref.model.model_type == "omni_model"
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert not actor.engine.use_remove_padding
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.model.lora.rank == 0
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt == {
        "audio": 1,
        "image": 1,
        "video": 0,
    }
    assert config.trainer.v1.trainer_mode == "omni_separate_async"
    trainer = OmniPPOTrainerSeparateAsync(config)
    assert trainer.parameter_sync_step == 1
    assert config.data.train_batch_size == trainer.parameter_sync_step * actor.ppo_mini_batch_size
    assert not config.algorithm.rollout_correction.bypass_mode
    env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    assert env["TENSORBOARD_DIR"] == tb_dir
    assert env["VERL_USE_EXTERNAL_MODULES"] == "verl_omni"
    assert all(isinstance(value, str) for value in env.values())


def test_megatron_detach_preserves_shard_list_protocol():
    worker = object.__new__(OmniDetachActorWorker)
    worker._strategy_handlers = None
    modules = [object(), object()]
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=modules))
    worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="megatron"))
    snapshots = [object(), object()]
    # Exercise the real strategy dispatcher without requiring Megatron/CUDA in
    # CPU CI. Native tensor copies are covered by the Megatron GPU smoke.
    helpers = ModuleType("verl.utils.megatron_utils")
    helpers.copy_megatron_model_to_cpu = save = Mock(return_value=snapshots)
    helpers.restore_megatron_model_from_cpu = restore = Mock()
    with patch.dict(sys.modules, {helpers.__name__: helpers}):
        worker.save_model_to_cpu(1)
        worker.restore_model_from_cpu(1)
        worker.clear_cpu_model(1)
    save.assert_called_once_with(modules)
    restore.assert_called_once_with(modules, snapshots)
    assert not worker.cpu_saved_models


def test_megatron_worker_initialization_selects_ppo_loss(monkeypatch):
    import verl_omni.workers.engine_workers as workers

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        config = compose(config_name="audiomcq_megatron_separate_async")
    worker = object.__new__(workers.ActorRolloutRefWorker)
    worker.config = config.actor_rollout_ref
    worker.role = "actor"
    worker.distillation_enabled = False
    model_config = OmegaConf.create({"model_type": "omni_model", "use_remove_padding": False})
    original_convert = workers.omega_conf_to_dataclass
    monkeypatch.setattr(
        workers,
        "omega_conf_to_dataclass",
        lambda value: model_config if value is worker.config.model else original_convert(value),
    )

    class EngineBoundaryReached(Exception):
        pass

    def build_engine(config):
        assert isinstance(config.engine_config, McoreEngineConfig)
        assert config.model_type == "omni_model"
        raise EngineBoundaryReached

    monkeypatch.setattr(workers, "TrainingWorker", build_engine)
    with pytest.raises(EngineBoundaryReached):
        inspect.unwrap(workers.ActorRolloutRefWorker.init_model)(worker)
    assert worker.loss_fn.func is workers.ppo_loss
    assert model_config.trainer_type == "policy_gradient"


def test_native_megatron_adapter_dispatch_and_forward_binding(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from verl.workers.engine.base import EngineRegistry
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    monkeypatch.setenv("VERL_ENGINE_DEVICE", "cuda")
    assert EngineRegistry.get_engine_cls("omni_model", "megatron") is OmniMegatronEngine
    assert issubclass(OmniMegatronEngine, MegatronEngineWithLMHead)
    assert OmniMegatronEngine.optimizer_step is MegatronEngineWithLMHead.optimizer_step
    assert OmniMegatronEngine.get_per_tensor_param is MegatronEngineWithLMHead.get_per_tensor_param

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            self.seen = kwargs
            return kwargs["input_features"].sum()

    features = torch.ones(1, 128, 5, requires_grad=True)
    batch = {"multi_modal_inputs": [{"input_features": features}]}
    batches = []

    def upstream_forward(_engine, batch_iter, model, *_args):
        batches.append(next(batch_iter))
        return model(input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=torch.arange(4))

    monkeypatch.setattr(MegatronEngineWithLMHead, "forward_step", upstream_forward)
    model = Model()
    engine = object.__new__(OmniMegatronEngine)
    engine.forward_step(iter([batch]), model, None, None).backward()
    assert batches == [batch]
    assert model.seen["position_ids"] is None
    assert torch.equal(features.grad, torch.ones_like(features))
    assert not model._forward_pre_hooks


def test_native_megatron_config_view_does_not_mutate_rollout_config(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from transformers import Qwen3OmniMoeConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    model_config = SimpleNamespace(hf_config=Qwen3OmniMoeConfig(), model_stage="thinker")
    engine_config = SimpleNamespace(
        use_remove_padding=False,
        use_fused_kernels=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )

    def parent_init(engine, model_config, *_args):
        engine.model_config = model_config

    monkeypatch.setattr(MegatronEngineWithLMHead, "__init__", parent_init)
    engine = OmniMegatronEngine(model_config, engine_config, None, None)
    assert engine.model_config is not model_config
    assert engine.model_config.hf_config is not model_config.hf_config
    assert (
        engine.model_config.hf_config.text_config.hidden_size
        == model_config.hf_config.thinker_config.text_config.hidden_size
    )
    engine.model_config.hf_config.text_config.hidden_size = 128
    assert not hasattr(model_config.hf_config, "text_config")
    assert model_config.hf_config.thinker_config.text_config.hidden_size != 128
