# Qwen3-Omni Megatron LoRA GRPO

These examples run text-only GSM8K GRPO with Qwen3-Omni, Megatron actor/ref
workers, and vLLM-Omni rollout on one 4-GPU node.

Set the model and data paths before launching:

```bash
export MODEL_PATH=/path/to/Qwen3-Omni
export TRAIN_FILES=/path/to/gsm8k/train.parquet
export VAL_FILES=/path/to/gsm8k/test.parquet
```

Run the 1-step smoke:

```bash
bash examples/grpo_trainer/run_qwen3_omni_megatron_lora_4gpu_smoke.sh
```

Run the GSPO/DAPO-aligned recipe:

```bash
bash examples/grpo_trainer/run_qwen3_omni_megatron_lora_4gpu_gspo_aligned.sh
```

The validated scope is text-only rollout and LoRA adapter refresh. Full
visual/audio LoRA coverage needs a separate multimodal smoke.
