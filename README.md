# PR724 review-candidate video E2E: 20 steps

Run: `avqa_pr724_video_review_full8_20step_20261104a` (Nov 5, 2026), 8 H800 GPUs.
Aggregate metrics only; no media, prompts or generated answers are included.

Three complete validations (256 examples each) and 320 training outputs matched official reward re-evaluation. All 96 recorded TensorBoard scalar tags were finite. Terminal exit 0, no owned processes, and eight GPUs released were verified before acceptance.

Validation accuracy: 228/256 -> 221/256 -> 224/256 (steps 0/10/20). Valid answer choice means an output parsed to A/B/C/D; it is not a strict formatting score. Character lengths are decoded Unicode lengths; training token lengths come from the TensorBoard response_length/mean metric. Validation and training distributions are separate. Reward on this validation is binary correctness and thus matches accuracy; no duplicate reward curve is plotted. CSVs preserve raw observations; plots are not smoothed. The elapsed-time column measures from completion of initial validation and includes subsequent validation overhead.

Steps 6, 10, 12 and 20 had zero gradients: each actual reward group was constant, with zero native GRPO advantages. These are not reported as nonzero-gradient updates. Policy loss values near zero do not imply zero gradients in the other steps.

Scope: short integration regression for the review candidate whose fixes were subsequently published in PR724 (3a0ed32). Development dependency combination, not clean public pins; DataLoader shutdown traceback and AccumulateGrad warnings remain recorded. Not a quality improvement, convergence, multi-seed, reuse-parity or distributed fault-recovery claim. Historical 100-step evidence is a separate run.

Figures prepared with OpenAI Codex assistance from the SHA-bound accepted audit. SOURCE_METRICS_SHA256.txt records the source and archive hashes.
