# Qwen3-Omni Talker codec-0 reference

Review draft for the codec-0 scope of [RFC #800](https://github.com/verl-project/verl-omni/issues/800). This is actor-math reference code for the FSDP implementer, not a runnable FSDP pipeline or a PR.

## Scope

- Train the Talker backbone and codec-0 head; freeze the residual predictor and conditioning projections.
- Score one codec-0 action per audio frame, plus natural EOS when present. Truncation does not invent an EOS action.
- Retain residual codes as conditioning for subsequent frames. They are not response actions or loss terms.
- Use the existing trainer's GRPO/GSPO loss; no algorithm-specific loss is introduced here.

`codec0_forward` accepts a prepared native conditioning prefix, trailing conditioning and padding embedding. The caller must preserve trainable prefix embeddings in the graph. It performs a causal teacher-forced forward and returns action logits and labels. `align_codec0_logits` places those logits at the trainer's next-token prediction positions.

The structure follows the existing Qwen3-TTS codec-0 teacher-forcing pattern. Model-specific conditioning and token support must be checked against the Qwen3-Omni checkpoint and actual rollout sampler, rather than copied from TTS unchanged.

## Validation

Run with PyTorch and pytest installed:

```bash
python -m pytest -q test_codec0_reference.py
```

20 CPU tests passed on 2026-10-26. Tests use a tiny FP64 recurrent causal model, comparing batched teacher forcing against independently recomputed causal prefixes. They cover logits and trainable gradients, frozen parameters across an optimizer update, residual conditioning, natural EOS, EOS-only responses, truncation, response alignment, finite entropy and malformed-input rejection. Ruff check and format passed. The local test environment emitted a missing-NumPy warning; these tests do not use NumPy.

## Integration still required

- Agree on the rollout replay payload and implement native Thinker embedding/hidden-state conditioning with the FSDP adapter owner.
- Register the adapter and wire it into the standard engine; this reference makes no shared-engine changes.
- Validate actual checkpoint vocab masking, sampling transforms, temperature and behavior/replay probability agreement.
- Validate FSDP wrapping and EOS-only/mixed-length rank behavior, stage-specific weight synchronization, and a real GPU E2E step.

These CPU results do not establish BF16 replay parity, distributed correctness, training quality, or production V1/vLLM compatibility. No GPU result is claimed for this reference.
