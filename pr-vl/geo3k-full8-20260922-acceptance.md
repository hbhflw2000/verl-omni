### What does this PR do?

Adds a single-node, eight-GPU Qwen3-Omni Thinker Geo3K recipe using the mainstream V1 separate-async trainer and the shared Megatron multimodal adapter. Four GPUs train all Thinker language-model parameters; four GPUs run standalone rollout. Vision/audio towers remain frozen, and Talker is outside this text-output task. The eight-H200 configuration uses the supported Megatron hybrid optimizer to keep half of FP32 Adam states and updates on CPU; host RAM is part of this configuration.

This updates the existing VL Draft rather than introducing a second implementation. It builds on #413's shared adapter and dependency fixes, with the V1 integration from the merged #380/#497 work. The VL-specific delta is data preparation, a public launcher, documentation, and regression coverage; model-specific forward logic stays in the shared pipeline adapter.

### Formulation

- Geo3K rule reward: 0.9 answer accuracy + 0.1 explicit reasoning/boxed-answer format.
- Group-normalized GRPO advantages; vanilla token-level PPO clipping (0.2), token-mean loss, reference KL coefficient 0.001, with MoE load-balancing auxiliary loss explicitly disabled (matching the verl engine default).
- Attention/hidden dropout are explicitly zero. The transferred Bridge provider otherwise initializes attention dropout to 0.1, producing a train/eval discrepancy before the first policy update.
- 16 prompts per update, eight responses per prompt; one PPO epoch and one 128-response minibatch per update. Actor old log-probs are recomputed immediately before training, and weights synchronize every update. With dropout disabled, zero pre-update PPO KL/clipping is expected in this setup; reference KL and rollout/train mismatch track policy drift.
- Importance sampling and rejection sampling are not enabled in this baseline. The V1 sampler's default version-span cap is eight; measured staleness and rollout/train mismatch are reported below.
- Prompt limit 1,024; response limit 2,048. The full 2,101-row training set fits the prompt cap including image tokens (maximum observed 911).

### Test

**Eight-H200 acceptance passed: 30/30 full-model updates**, with full 601-example validation at steps 10/20/30 and wrapper exit code 0. Training loop plus validation took 4:02:54; startup added about 13 minutes.

| Update | Composite reward | Answer accuracy | Format compliance |
| --- | --- | --- | --- |
| 10 | 0.610982 | 349/601 (58.07%) | 531/601 (88.35%) |
| 20 | 0.651248 | 374/601 (62.23%) | 548/601 (91.18%) |
| 30 | 0.641265 | 365/601 (60.73%) | 569/601 (94.68%) |

All 5,643 saved training/validation completions were independently regraded with the official Geo3K reward: zero scoring mismatches and zero empty answers. Final accuracy is below step 20; this run does not establish monotonic improvement or convergence.

- All recorded scalar values are finite. Gradient norm range 0.305–0.624, final 0.540; reference KL rises from zero to 0.01394.
- Rollout/recomputed k3 KL range 0.00345–0.00470, final 0.00411; final probability correlation 0.98803.
- Actual trajectory staleness stays at one update; all 30 weight synchronization operations completed.
- Mean training reward: first ten updates 0.49055, last ten 0.60445. Final response length 436.8 tokens, clipping 4.69%, aborted ratio zero.
- Peak sampled GPU memory 116,907 MiB (114.17 GiB), sampled every ten seconds.
- A multiprocessing DataLoader-worker `atexit` cleanup RuntimeError was logged after all 30 updates, final validation and TensorBoard flush. The wrapper subsequently returned 0; this cleanup warning remains recorded rather than described as an exception-free run.

![30-update eight-H200 Geo3K acceptance](https://raw.githubusercontent.com/hbhflw2000/verl-omni/pr-assets/pr-vl/geo3k-full8-20260922-training-metrics.svg)

[Per-update metrics CSV](https://github.com/hbhflw2000/verl-omni/blob/pr-assets/pr-vl/geo3k-full8-20260922-training-metrics.csv)

The current acceptance run uses source commit `5d8d8b2ffb794e46d499b66aaa38b9bedb5e1d3a`, tree `8748e58fb55c9fc13758a556eb5fcebee4b458ec`, on eight H200 GPUs and a host with 2 TiB RAM. Runtime overrides use eager rollout, 32 Ray CPUs, a 16 GiB Ray object store, four data/reward workers, and eight agent workers. Initial validation is disabled in this run; full 601-example evaluations occur at steps 10/20/30. The earlier pre-training evaluation came from a prior attempt and is not a same-source initial baseline.

The TensorBoard `acc/mean@1` field is the composite rule reward, not answer accuracy. Report answer accuracy and format compliance separately from direct regrading of saved completions. Thirty updates establish an E2E smoke test, not convergence. No full-model checkpoint is saved by this recipe (`save_freq=-1`).

The H200 runtime rebuilds the same FlashAttention 2.8.3 source for SM90: the transferred binary contained only SM80 kernels. Fixed/variable-length attention forward/backward and a three-step hybrid-Adam comparison passed; these probes are separate from full-model GPU acceptance.

The current tested source passes 104 focused CPU tests (no skips). CPU coverage includes byte-preserving data conversion, reward dispatch, real tiny-Thinker image conditioning/backward, adapter logits/gradient parity, and the launcher's actual resolved Hydra configuration. Targeted Ruff 0.12.2, mypy 1.17.0, generated configurations, and repository sanity checks pass. Full pre-commit invocation could not fetch its GitHub hooks; do not label it passed.

The development runtime uses the dependency overrides documented by #413. Successful E2E here does not establish that unchanged public dependency pins work. Runtime: Python 3.12.13, Torch 2.13.0+cu129, Transformers 5.13.1, vLLM 0.28.0+cu129, vLLM-Omni 0.28.0rc1, Ray 2.55.1, TransferQueue 0.1.8, Transformer Engine 2.13 and FlashAttention 2.8.3 (SM90 build). Source snapshots include verl `fefb080` plus its multimodal position-ID fix, Megatron-Bridge `fe22f9d` plus Qwen3-Omni registration/Transformers 5 compatibility, and Megatron-Core `e41b370`.

Publication commit `785dbf33ec1fde3f260ed9899f2d41a72736e9d4` has exactly the tested tree. It retains the existing PR history and stacks on #413 (`da50a44`), which is still unmerged. The focused VL delta relative to #413 is seven files; the current diff against main also contains that prerequisite backend work. Rebase/deduplicate after #413 lands.

### Usage

```bash
python examples/gspo_trainer/data_process/geo3k.py --local_save_dir /data/geo3k
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/data/geo3k/train.parquet \
VAL_FILE=/data/geo3k/test.parquet \
OUTPUT_DIR=/outputs/geo3k \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_geo3k_separate_async.sh
```

The launcher records the command, commit, resolved configuration, logs and TensorBoard data in a unique directory. Default acceptance is 30 updates with full validation at steps 0/10/20/30. Internal deployment paths and environment restoration scripts are not part of the public recipe.

### Review status

- [ ] Human line-by-line review.
- [x] Final GPU acceptance evidence and runtime limitations recorded.
- [ ] Full pre-commit/CI.
- [x] Documentation and focused tests.

AI assistance: implemented and tested with OpenAI Codex; human review remains pending. Intern contributions are retained through `Co-authored-by: mart1n <97656152+martinzhang03@users.noreply.github.com>`.
