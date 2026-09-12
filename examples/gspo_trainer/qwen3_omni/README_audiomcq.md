# AudioMCQ with Megatron and V1 separate-async rollout

This recipe trains all Thinker language-model parameters (LoRA rank zero),
freezes the vision/audio towers, and generates text conditioned on audio with
standalone vLLM-Omni replicas. It uses `trainer.v1.trainer_mode=omni_separate_async`.
The toy smoke and full-model run share the same configuration and reward.
`OmniMegatronEngine` reuses verl's Megatron LM engine and adds model-scoped audio
inputs and M-RoPE handling at the module-call boundary. It uses BSHD, PP1 and CP1;
the development audio bridge does not implement packed sequences. Optimizer,
old-policy snapshots, losses and weight export remain upstream implementations.

## Environment and data

Use the repository's pinned verl/vLLM-Omni runtime and install the audio extra
(`pip install -e '.[audio]'`). Megatron also requires a compatible Megatron-Core,
Transformer Engine, and Megatron-Bridge with Qwen3-Omni audio forward/export
support. The recipe selects `use_mbridge=true`, `vanilla_mbridge=false`.
The development audio bridge is
[`hbhflw2000/Megatron-Bridge@fe22f9d2`](https://github.com/hbhflw2000/Megatron-Bridge/commit/fe22f9d20bc32d3f09fd08dd58d9ad701d885d42),
with Megatron-Core `e41b37002cd8df1cd97c93e3e0876cf0850f72f8`.
With Transformers 5.13+, also apply the registration fix from upstream
[Megatron-Bridge #4876](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/039156328f9587ccb5a9c8c9e6adf30e63f2cf6a)
to that older bridge revision; otherwise native ASR auto-registration collides
while importing the bridge, before any Omni model is initialized.
These are external prerequisites, not implementations vendored by this recipe;
install them into the same environment as verl. Native Transformer Engine and
FlashAttention extensions must be built for that environment's PyTorch version.

Download AudioMCQ and its audio assets separately, respecting their licenses.
Prepare a local `data.jsonl` containing `question`, `choices`, `answer`,
`audio_path`, and optional `source_dataset`/`id` fields:

```bash
python examples/gspo_trainer/data_process/audiomcq.py \
  --input-jsonl /data/AudioMCQ/data.jsonl \
  --audio-root /data/AudioMCQ \
  --output-dir /data/audiomcq-prepared \
  --validation-size 256 --seed 42
```

Conversion checks file existence, labels and path containment; it does not
decode the entire audio corpus. Missing/invalid rows are counted in
`dataset_info.json`. Validation holds out 256 unique audio assets; questions
sharing an asset stay in the same split. Existing output files are never
overwritten; use a new output directory for each conversion. Audio paths must
be accessible at the same location on all nodes.

Previously audited AudioMCQ parquets with `prompt`, `audios`, and structured
`reward_model.ground_truth` can be used directly to preserve their exact split.
The scorer accepts exact option text or an option letter inside `<answer>` and
reports `content_correct` and `format_valid`. It preserves the development
recipe's reward semantics, including its handling of repeated answer tags.

## Toy smoke (4 GPUs)

```bash
bash tests/special_e2e/run_qwen3_omni_megatron_audiomcq_smoke.sh
```

Builds the existing multimodal tiny-random Qwen3-Omni checkpoint and short
synthetic PCM WAVs locally. Uses 2 Megatron training GPUs plus a standalone TP2
replica on 2 GPUs, four optimizer steps, sync every two steps, and validation
before training and every two steps. Keeping two updates per sync exercises
V1's old-policy parameter save/restore. Synthetic tones test audio transport
only. Random-model correctness and nonzero reward are **not** acceptance gates.
Inspect finite losses/logprobs, successful optimizer steps, weight transfers,
and validation completion. A zero gradient is permitted when every reward and
advantage is zero.

## Full-model run (32 GPUs)

After allocating four 8-GPU nodes and starting a Ray cluster, run once on the
head. The defaults request 4 training and 4 standalone rollout GPUs per node,
actor TP4, rollout TP4, and 150 steps with validation every 10 steps:

```bash
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/data/audiomcq-prepared/train.parquet \
VAL_FILE=/data/audiomcq-prepared/validation.parquet \
OUTPUT_DIR=/persistent/audiomcq \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_audiomcq_separate_async.sh \
  ray_kwargs.ray_init.address=auto
```

The launcher records the command, Git revision, resolved configuration, console
log and TensorBoard events in a unique run directory. Use local scratch for
high-frequency writes and archive once afterward on fragile shared filesystems.
TensorBoard and worker bootstrap environment variables are explicitly forwarded
through Ray's per-job runtime environment, including for pre-started clusters.
Hydra overrides are forwarded unchanged; keep
`data.train_batch_size == parameter_sync_step * actor.ppo_mini_batch_size`.

V1 also uses hybrid replicas on the training pool for its initial sampling
window and validation. Although hybrid switching during training is disabled,
sleep/wake and colocated weight loading still need to work. Prefix caching is
disabled. The old development `fully_async_policy` run does not validate these
V1 lifecycle paths or Decoupled PPO (`bypass_mode=false`). A successful toy smoke
establishes structural coverage, not full-model learning or TP4 numerical
parity; evaluate the need for a new full-model run after reviewing the changes.
