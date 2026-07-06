#!/bin/bash
# Local run harness (NOT part of the PR). Single-node GB200 (4 GPUs) e2e
# determinism check on Moonshot AI's Moonlight-16B-A3B (a DeepSeek-V3-arch MoE:
# MLA + fine-grained MoE), at TP2 / PP1 / EP2.
#
# Architecture is transcribed verbatim from Megatron-Bridge's authoritative
# recipe (../Megatron-Bridge/src/megatron/bridge/recipes/moonlight/moonlight_16b.py):
#   27 layers (1 dense + 26 MoE), hidden 2048, ffn 11264, 16 heads,
#   MLA: q_lora_rank=None (NO Q-LoRA compression), kv_lora_rank=512,
#        qk_head_dim 128 + qk_pos_emb 64, v_head_dim 128,
#   MoE: 64 routed experts (top-6) + 2 shared (moe_ffn 1408, shared 2816),
#        sigmoid score fn + expert bias, topk-scaling 2.446, aux_loss 0.001,
#   rotary_base 50000, RMSNorm eps 1e-5, no MTP, vocab 163842.
# "never lora" note: Moonlight has NO Q-LoRA (q_lora_rank=None -> direct Q
# projection). kv_lora_rank is the KV latent compression that DEFINES MLA and
# cannot be removed without ceasing to be a DeepSeek-arch model.
#
#   Grid on one GB200 node = 4 GPUs:
#     TP=2, PP=1, CP=1  -> DP = 4/(2*1*1) = 2
#     EP=2, ETP=1       -> expert-DP = 2   (num-experts 64 % EP == 0)
#
# Deliberately does NOT export NCCL_ALGO / NVTE_* / CUBLAS_* — the branch's
# apply_determinism_to_args() sets them (NCCL_ALGO=Ring,
# NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, CUBLAS_WORKSPACE_CONFIG=:4096:8) and calls
# torch.use_deterministic_algorithms(True). Config is chosen to PASS that guard:
# the Bridge recipe's --cross-entropy-loss-fusion is OMITTED (guard rejects it),
# and MoE/attention use bit-exact-safe paths (alltoall dispatcher, no
# grad-accum/rope fusion). Training hyperparams (seq/iters/gbs) are shrunk for a
# fast local check; the model architecture is unchanged.
#
# Step 1 = one valid e2e run; Step 2 = a second run, then assert per-step lm
# loss is bit-identical across the two runs.
#
#   sbatch run_determinism_moonlight.slurm
#SBATCH --job-name=det-moonlight
#SBATCH --account=nemotron_sw_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/logs/%x-%j.out

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/Megatron-LM
SQSH=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/megatron_rl/adlr+megatron-rl+arm64-260407.sqsh
OUT=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/moonlight-${SLURM_JOB_ID}
mkdir -p "$OUT"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29607
export NNODES=${SLURM_NNODES}
export GPUS_PER_NODE=4

# Moonlight-16B-A3B. --mock-data + NullTokenizer so there is no dataset/HF
# tokenizer dependency (vocab_size matches the real model).
export MOONLIGHT_ARGS="\
--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
--expert-model-parallel-size 2 --expert-tensor-parallel-size 1 \
--context-parallel-size 1 --sequence-parallel --use-distributed-optimizer \
--overlap-grad-reduce --overlap-param-gather \
--use-mcore-models --transformer-impl transformer_engine --bf16 \
--deterministic-mode --no-gradient-accumulation-fusion --no-rope-fusion \
--attention-softmax-in-fp32 --distributed-backend nccl --seed 1234 \
--manual-gc --manual-gc-interval 5 \
--num-layers 27 --hidden-size 2048 --ffn-hidden-size 11264 \
--num-attention-heads 16 --kv-channels 128 \
--seq-length 4096 --max-position-embeddings 4096 \
--position-embedding-type rope --rotary-base 50000 --rotary-percent 1.0 \
--normalization RMSNorm --norm-epsilon 1e-5 --swiglu \
--disable-bias-linear --untie-embeddings-and-output-weights --qk-layernorm \
--attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02 \
--micro-batch-size 1 --global-batch-size 16 --train-iters 30 \
--lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-fraction .01 \
--weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
--adam-eps 1e-5 \
--use-precision-aware-optimizer --main-grads-dtype bf16 \
--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 \
--multi-latent-attention \
--kv-lora-rank 512 --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128 \
--rotary-scaling-factor 1.0 --mscale 1.0 --mscale-all-dim 1.0 \
--num-experts 64 --moe-layer-freq '([0]*1+[1]*26)' \
--moe-ffn-hidden-size 1408 --moe-shared-expert-intermediate-size 2816 \
--moe-router-load-balancing-type seq_aux_loss --moe-router-topk 6 \
--moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
--moe-aux-loss-coeff 0.001 --moe-router-num-groups 1 --moe-router-group-topk 1 \
--moe-router-topk-scaling-factor 2.446 --moe-router-score-function sigmoid \
--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3 \
--moe-router-dtype fp32 \
--mock-data --tokenizer-type NullTokenizer --vocab-size 163842 \
--make-vocab-size-divisible-by 1280 --split 1,0,0 \
--no-check-for-nan-in-loss-and-grad \
--log-interval 1 --eval-iters 0 --eval-interval 100000"

run_once () {
  local tag="$1"
  srun --ntasks="${NNODES}" --ntasks-per-node=1 \
       --container-image="$SQSH" \
       --container-mounts="/lustre:/lustre,$WORKDIR:/opt/megatron-lm" \
       --container-workdir=/opt/megatron-lm \
       --no-container-mount-home --export=ALL,HOME=/tmp \
       bash -c '
         export PYTHONPATH=/opt/megatron-lm
         python -m torch.distributed.run \
           --nnodes=${NNODES} --nproc-per-node=${GPUS_PER_NODE} \
           --node-rank=${SLURM_NODEID} \
           --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} \
           pretrain_gpt.py '"${MOONLIGHT_ARGS}"'
       ' 2>&1 | tee "$OUT/$tag.log"
}

echo "########## STEP 1: RUN 1 (valid e2e) ##########"; run_once run1
echo "########## STEP 2: RUN 2 (determinism repeat) ##########"; run_once run2

echo "########## COMPARE ##########"
grep -oE "lm loss: [0-9.eE+-]+" "$OUT/run1.log" > "$OUT/loss1.txt" || true
grep -oE "lm loss: [0-9.eE+-]+" "$OUT/run2.log" > "$OUT/loss2.txt" || true
echo "run1 loss lines: $(wc -l < "$OUT/loss1.txt"), run2: $(wc -l < "$OUT/loss2.txt")"
echo "--- run1 first/last ---"; head -3 "$OUT/loss1.txt"; tail -1 "$OUT/loss1.txt"
if [[ -s "$OUT/loss1.txt" ]] && diff -q "$OUT/loss1.txt" "$OUT/loss2.txt" >/dev/null; then
  echo "RESULT: DETERMINISM PASS — per-step lm loss bit-identical across the two Moonlight runs"
else
  echo "RESULT: DETERMINISM FAIL — losses differ (or empty); first diffs:"
  diff "$OUT/loss1.txt" "$OUT/loss2.txt" | head -20
fi
echo "verify the branch set the deterministic env itself:"
grep -iE "NCCL_ALGO|deterministic|CUBLAS_WORKSPACE" "$OUT/run1.log" | head -5 || true
