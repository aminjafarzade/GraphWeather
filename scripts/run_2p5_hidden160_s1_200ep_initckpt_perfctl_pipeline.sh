#!/usr/bin/env bash
# =============================================================================
# run_2p5_hidden160_s1_200ep_initckpt_perfctl_pipeline.sh
#
# PERF CONTROL twin of run_2p5_hidden160_s1_200ep_initckpt_pipeline.sh.
#
# Same architecture, loss, LR schedule and effective_batch_size (12) as the
# 200-epoch hidden-160 CTL. Changes:
#   1. attention_impl: elementwise -- the matmul reformulation was implemented and
#      is selectable, but measured 0.45x eager elementwise at the L0 block shape,
#      so the config pins the faster path. See scripts/dev/bench_attention_impl.py
#      and ATTENTION_IMPL_DEFAULT in src/layers.py.
#   2. rollout-scoped cache for the static edge projections (numerically equivalent)
#   3. batch_size 4 x accum 3  ->  batch_size 12 x accum 1
#   4. compile_processor: true -- fusing the elementwise attention is what the
#      matmul was meant to achieve and did not: it removes the [B,N,k,H,d]
#      transient instead of trading it for a gemv.
#   5. ema.enabled: true (decay 0.999)
# Rollout checkpointing is off; see the memory probe below.
#
# CAUTION: with EMA on this is NO LONGER a pure perf control. Validation,
# best-checkpoint selection and the saved model_state all use the EMA weights, so
# valid_* and every downstream eval measure a different model than the 200-epoch
# CTL did. Changes 1-4 are performance-only and preserve the control; change 5 is
# a skill intervention. Judge the skill deltas accordingly -- a difference vs the
# old CTL is expected here, not a regression signal.
#
# Measured 2026-07-31, one B200, 240 samples / 20 optimizer steps per config:
#   S1    old CTL (B4 x accum3)              0.1421 s/step    3.2 GB
#   S1    this config                        0.0537 s/step    7.5 GB   2.65x
#   S10   eager  (compile off)               0.8150 s/step   81.5 GB
#   S10   this config                        0.5007 s/step   61.5 GB   1.63x
# torch.compile only wraps model.processor (9 of the 11 attention blocks); the
# encoder/decoder blocks stay eager, and compilation failure falls back to eager
# rather than killing the run. Checkpoints are saved through
# Trainer._canonical_model_state and the EMA shadow through
# Trainer._canonical_named_parameters, both of which strip the compile
# `_orig_mod.` prefix -- so eval, resume and the curriculum warm-start load them
# unchanged. (compile + EMA used to raise; the shadow keys are now
# compile-invariant, which is what that guard was protecting.)
#
# Usage (mirrors the non-perfctl script):
#   nohup env GPU=3 bash scripts/run_2p5_hidden160_s1_200ep_initckpt_perfctl_pipeline.sh \
#     > runs/2p5_hidden160_s1_200ep_perfctl_pipeline.out 2>&1 &
# Flags: FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun training;
#        RUN_EVAL/RUN_PLOT/RUN_QUAL/RUN_DIAG/RUN_DASHMAPS=0 skip a stage.
#
# Memory probe, measured 2026-07-31 on one B200 (183 GB), 20 train batches,
# batch 12 x accum 1, rollout checkpointing off:
#     S1   9.8 GB      S4  33.7 GB      S7  57.6 GB      S10  81.5 GB
#     S10 with checkpoint_rollout_steps: true -> 10.6 GB, but 1.45x the step time.
# 81.5 GB is under the 120 GB abort gate, so the curriculum runs with rollout
# checkpointing off. Re-probe before changing batch size, hidden_dim or resolution:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python scripts/train.py \
#     --config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_s1_200ep_perfctl.yaml \
#     --config_name 2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_s1x200_perfctl \
#     --run_num probe --max_train_batches 20 --max_epochs 1
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# The pyramid produces variable-shape tensors; expandable segments measurably
# reduce fragmentation and are assumed by the memory estimates in the brief.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGES="train"
[[ "${RUN_EVAL:-1}"     == "1" ]] && STAGES+=",eval"
[[ "${RUN_PLOT:-1}"     == "1" ]] && STAGES+=",plot"
[[ "${RUN_QUAL:-1}"     == "1" ]] && STAGES+=",qual"
[[ "${RUN_DIAG:-1}"     == "1" ]] && STAGES+=",diag"
[[ "${RUN_DASHMAPS:-1}" == "1" ]] && STAGES+=",dashmaps"

exec bash "${SCRIPT_DIR}/run_pipeline.sh" \
  --resolution 2p5 --gpu "${GPU:-3}" \
  --s1-config configs/experiments/config_2p5_l3_hidden160_base_S1_200epoch_dense_l3k24_perfctl.yaml \
  --s1-name   2p5_l3_h160_densel3k24_s1x200_perfctl \
  --curr-config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_s1_200ep_perfctl.yaml \
  --curr-name   2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_s1x200_perfctl \
  --primary-label hidden160_s1_200ep_perfctl \
  --compare "runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_s1x200 runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt runs/2p5_l3_h128_densel3k24_currS2toS10x3_initckpt" \
  --compare-labels "ctl_hidden160_s1_200ep, hidden160_s1_100ep, initckpt_h128" \
  --stages "${STAGES}"
