# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""The shared toy checkpoint must actually encode audio, not just load for text."""

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tests.special_e2e.build_qwen3_omni_multimodal_tiny_random import build


def test_toy_audio_processor_and_forward(tmp_path):
    model_path = str(tmp_path / "model")
    build(model_path, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, attn_implementation="sdpa").eval()
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": "unused.wav"}, {"type": "text", "text": "Listen."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    waveform = np.sin(np.arange(8160, dtype=np.float32) / 10)
    inputs = processor(text=[text], audio=[waveform], sampling_rate=16000, return_tensors="pt", padding=True)
    assert inputs["input_features"].shape[1] == model.config.thinker_config.audio_config.num_mel_bins == 128
    audio_id = model.config.thinker_config.audio_token_id
    assert (inputs["input_ids"] == audio_id).sum() > 0
    with torch.no_grad():
        output = model.thinker(**inputs)
    assert output.logits.shape[:2] == inputs["input_ids"].shape
    assert torch.isfinite(output.logits).all()
