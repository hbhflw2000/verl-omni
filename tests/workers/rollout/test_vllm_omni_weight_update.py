# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest
import torch

from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

pytestmark = pytest.mark.cpu


def _make_worker():
    worker = object.__new__(rollout_utils.vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._get_zmq_handle = lambda: "ipc:///tmp/test.sock"
    return worker


def test_update_weights_from_ipc_accumulates_lora_buckets(monkeypatch):
    received = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("a", torch.tensor([1]))])
            on_bucket_received([("b", torch.tensor([2]))])

    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "BucketedWeightReceiver", FakeReceiver)

    worker = _make_worker()
    worker.remove_lora = lambda _adapter_id: None
    worker._update_weights = lambda weights, peft_config, base_sync_done: received.append(
        (list(weights), peft_config, base_sync_done)
    )

    worker.update_weights_from_ipc(peft_config={"r": 16}, base_sync_done=True)

    assert len(received) == 1
    assert [name for name, _ in received[0][0]] == ["a", "b"]
    assert received[0][1] == {"r": 16}
    assert received[0][2] is True


def test_update_weights_releases_lora_tensor_reference(monkeypatch):
    requests = []
    worker = _make_worker()
    worker.add_lora = requests.append

    weights = [("a", torch.tensor([1])), ("b", torch.tensor([2]))]
    worker._update_weights(weights, peft_config={"r": 16}, base_sync_done=True)

    assert len(requests) == 1
    assert requests[0].peft_config == {"r": 16}
    assert requests[0].lora_tensors is None
