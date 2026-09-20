# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni Thinker inputs at verl's Megatron BSHD model-call boundary."""

from typing import Callable

import torch
from verl.models.mcore.util import build_vlm_attn_mask_bshd, postprocess_bshd_engine, preprocess_bshd_engine
from verl.utils.megatron_utils import unwrap_model

_MULTIMODAL_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)


def qwen3_omni_forward_model_engine(
    model,
    input_ids: torch.Tensor,
    multi_modal_inputs: dict,
    *,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    vision_model: bool = False,
    pad_token_id: int | None = None,
    forced_max_seqlen: int | None = None,
):
    """Follow verl's BSHD forward, passing audio directly and letting Thinker build M-RoPE.

    MTP, fused kernels, THD, PP and CP are rejected by ``OmniMegatronEngine``.
    Logits processing and BSHD postprocessing match pinned verl's model forward.
    """
    unwrapped_model = unwrap_model(model)
    post_process = unwrapped_model.post_process
    use_fp8_padding = unwrapped_model.config.fp8 in ("e4m3", "hybrid")
    input_ids_bshd, attention_mask_bshd, _ = preprocess_bshd_engine(
        input_ids,
        pre_process=unwrapped_model.pre_process,
        use_fp8_padding=use_fp8_padding,
        forced_max_seqlen=forced_max_seqlen,
    )
    if vision_model:
        input_ids_bshd, attention_mask = build_vlm_attn_mask_bshd(
            input_ids, input_ids.shape[0], pad_token_id, forced_max_seqlen=forced_max_seqlen
        )
    else:
        attention_mask = attention_mask_bshd

    model_kwargs = {
        key: multi_modal_inputs[key].to(input_ids.device)
        for key in _MULTIMODAL_KEYS
        if key in multi_modal_inputs and multi_modal_inputs[key] is not None
    }
    output_orig = model(
        input_ids=input_ids_bshd,
        attention_mask=attention_mask,
        position_ids=None,
        **model_kwargs,
    )

    if post_process and logits_processor is not None:
        processor_args = {
            key: preprocess_bshd_engine(
                value,
                pre_process=True,
                need_roll=(key == "label"),
                use_fp8_padding=use_fp8_padding,
                forced_max_seqlen=forced_max_seqlen,
            )[0]
            for key, value in (logits_processor_args or {}).items()
            if key not in ("loss_mask", "response_attention_mask")
        }
        output_dict = logits_processor(output_orig, **processor_args)
        return {
            key: postprocess_bshd_engine(value, attention_mask_bshd, post_process=True)
            for key, value in output_dict.items()
        }
    return postprocess_bshd_engine(output_orig, attention_mask_bshd, post_process=post_process)
