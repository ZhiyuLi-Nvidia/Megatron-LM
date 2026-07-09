#!/bin/bash
# Local run harness (NOT part of the PR). Two-node GB200 (2 GPUs/node) e2e
# determinism check on Moonshot AI's Moonlight-16B-A3B (a DeepSeek-V3-arch MoE:
# MLA + fine-grained MoE), at TP2 / PP1 / EP2. run1 and run2 run IN PARALLEL,
# one per node (2 GPUs each), so wall time is a single run's, not two.
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
# projection). kv_lora_rank is the KV latent compression that DEFINES MLA.
#
#   Grid per run = 2 GPUs on one node:
#     TP=2, PP=1, CP=1  -> DP = 2/(2*1*1) = 1
#     EP=2, ETP=1       -> expert-DP = 1   (num-experts 64 % EP == 0)
#   Determinism is preserved across the two parallel runs: each is a separate
#   process on its own node/GPUs/NCCL-communicator with no shared state.
#
# Deliberately does NOT export NCCL_ALGO / NVTE_* / CUBLAS_* — the branch's
# apply_determinism_to_args() sets them and calls
# torch.use_deterministic_algorithms(True). --cross-entropy-loss-fusion is
# OMITTED (the guard rejects it); MoE/attention use bit-exact-safe paths.
#
#   sbatch run_determinism_moonlight.slurm
#SBATCH --job-name=det-moonlight
#SBATCH --account=nemotron_sw_pre
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# --gres (not --gpus-per-node, which can under-allocate to 1 GPU on this cluster)
# gives 2 GPUs per node; --exclusive reserves the whole node per run.
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/logs/%x-%j.out

set -uo pipefail

WORKDIR=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/Megatron-LM
SQSH=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/megatron_rl/adlr+megatron-rl+arm64-260407.sqsh
OUT=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/moonlight-${SLURM_JOB_ID}
mkdir -p "$OUT"

# wandb + secrets. Real keys live OUTSIDE the repo in ../secrets.sh and are
# sourced INSIDE the container (below), because enroot does not forward host env.
WANDB_PROJECT=mlm-determinism-zhiyul
SECRETS=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh
PORT=29601                          # same port is fine — the two runs are on different nodes

# Moonlight-16B-A3B, 2-GPU grid. --mock-data + NullTokenizer (no dataset needed).
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

# One independent 2-GPU run pinned to a single node.
run_once () {
  local tag="$1" node="$2"
  srun --nodes=1 --ntasks=1 --nodelist="$node" --overlap \
       --container-image="$SQSH" \
       --container-mounts="/lustre:/lustre,$WORKDIR:/opt/megatron-lm" \
       --container-workdir=/opt/megatron-lm \
       --no-container-mount-home --export=ALL,HOME=/tmp \
       bash -c '
         export PYTHONPATH=/opt/megatron-lm
         # Determinism defense-in-depth + fragmentation guard (core determinism
         # env NCCL_ALGO/NVTE_*/CUBLAS_* is set by the branch under test).
         export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
         # Secrets (WANDB_API_KEY/HF_TOKEN) MUST be set inside the container:
         [ -f '"$SECRETS"' ] && source '"$SECRETS"' || echo "[warn] '"$SECRETS"' missing; wandb may be disabled"
         export WANDB_MODE=${WANDB_MODE:-online}   # set WANDB_MODE=offline if the node has no internet
         python -m torch.distributed.run --nproc-per-node=2 \
           --master-addr=127.0.0.1 --master-port='"$PORT"' \
           pretrain_gpt.py '"${MOONLIGHT_ARGS}"' \
           --tensorboard-dir '"$OUT/tb-$tag"' \
           --wandb-project '"$WANDB_PROJECT"' --wandb-exp-name '"moonlight-$tag"' \
           --wandb-save-dir '"$OUT/wandb-$tag"'
       ' 2>&1 | tee "$OUT/$tag.log"
}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "run1 -> ${nodes[0]}   run2 -> ${nodes[1]}   (parallel, 2 GPUs each)"
run_once run1 "${nodes[0]}" & p1=$!
run_once run2 "${nodes[1]}" & p2=$!
wait "$p1" || echo "[warn] run1 exited non-zero"
wait "$p2" || echo "[warn] run2 exited non-zero"

echo "########## COMPARE ##########"
grep -oE "lm loss: [0-9.eE+-]+" "$OUT/run1.log" | sed 's/lm loss: //' > "$OUT/loss1.txt" || true
grep -oE "lm loss: [0-9.eE+-]+" "$OUT/run2.log" | sed 's/lm loss: //' > "$OUT/loss2.txt" || true
n1=$(wc -l < "$OUT/loss1.txt"); n2=$(wc -l < "$OUT/loss2.txt")
echo "run1 loss lines: $n1, run2: $n2"
# full per-step table (printed and saved) for manual tracking
paste "$OUT/loss1.txt" "$OUT/loss2.txt" \
  | awk 'BEGIN{printf "%-4s %-18s %-18s %s\n","it","run1_lm_loss","run2_lm_loss","match"}
         {m=($1==$2)?"ok":"DIFF"; printf "%-4d %-18s %-18s %s\n", NR, $1, ($2==""?"-":$2), m}' \
  | tee "$OUT/loss_compare.tsv"
if [[ -s "$OUT/loss1.txt" ]] && diff -q "$OUT/loss1.txt" "$OUT/loss2.txt" >/dev/null; then
  echo "RESULT: DETERMINISM PASS — per-step lm loss bit-identical across the two Moonlight runs ($n1 steps)"
else
  echo "RESULT: DETERMINISM FAIL — losses differ (or empty)"
fi
echo "saved: $OUT/loss_compare.tsv | $OUT/loss1.txt $OUT/loss2.txt"

echo "########## WANDB (project: $WANDB_PROJECT) ##########"
for tag in run1 run2; do
  url=$(grep -oE "https://wandb\.ai/\S+/runs/\S+" "$OUT/$tag.log" | head -1)
  echo "  $tag: ${url:-<offline or no key — run: wandb sync $OUT/wandb-$tag>}"
done

echo "verify the branch set the deterministic env itself:"
grep -iE "NCCL_ALGO|deterministic|CUBLAS_WORKSPACE" "$OUT/run1.log" | head -5 || true
