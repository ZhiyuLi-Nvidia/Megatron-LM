# Moonlight-16B-A3B single-node determinism check

A bit-exact-reproducibility check for a DeepSeek-V3-architecture MoE
(Moonshot AI's **Moonlight-16B-A3B**: Multi-Latent Attention + fine-grained MoE)
on **2 GB200 nodes (2 GPUs each)** at **TP2 / PP1 / EP2**, with run1 and run2
running **in parallel** (one per node), so wall time is a single run's.

- **Step 1** — one valid end-to-end run.
- **Step 2** — a second run; assert the per-step `lm loss` is bit-identical across the two.

## Files

| file | purpose |
| --- | --- |
| `run_determinism_moonlight.slurm` | the harness — submits `pretrain_gpt.py` twice and diffs the loss |
| `reproduce_determinism_moonlight.sh` | one-shot: submit, wait, print PASS/FAIL |

## Reproduce

From a shell that can reach the Slurm controller (a normal login shell):

```bash
./reproduce_determinism_moonlight.sh
```

It submits the job, waits for both runs plus the loss diff, and prints:

```
RESULT: DETERMINISM PASS — per-step lm loss bit-identical across the two Moonlight runs
```

To submit manually and inspect instead:

```bash
sbatch run_determinism_moonlight.slurm
# then, once it finishes:
grep RESULT /lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/logs/det-moonlight-<JOBID>.out
cmp  /lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/moonlight-<JOBID>/loss1.txt \
     /lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/moonlight-<JOBID>/loss2.txt   # no output = identical
```

## Configuration

Edit these for your environment:

- **SLURM account** — set the account in `run_determinism_moonlight.slurm`:
  ```
  #SBATCH --account=<your-account>      # e.g. nemotron_sw_pre
  ```

- **Secrets (wandb / HF) — never commit these.** Copy `secrets.sh.example` to
  `../secrets.sh` (one level above the repo, so it stays untracked) and fill in
  your keys; it exports `WANDB_API_KEY` and `HF_TOKEN`. Do not paste keys into
  any tracked file.

  **The keys must be set *inside* the container**, so the harness sources the
  file within the container step — the file is reachable there through the
  mounted `/lustre`:
  ```bash
  # inside run_once()'s  bash -c  (this runs in the container):
  source /lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh
  ```
  Sourcing on the login shell then `sbatch` is **not** reliable — enroot/pyxis
  does not automatically forward host env vars into the container. (Alternative:
  add `--container-env=WANDB_API_KEY,HF_TOKEN` to the `srun` line to whitelist
  them explicitly.)

- **wandb project** — runs log to project `mlm-determinism-zhiyul` (set via
  `--wandb-project`); change the name in the harness to use your own. On compute
  nodes without internet, set `WANDB_MODE=offline` and `wandb sync` the run
  directories afterward.

## Model (from Megatron-Bridge's `moonlight_16b` recipe)

27 layers (1 dense + 26 MoE), hidden 2048, ffn 11264, 16 heads. MLA with
**no Q-LoRA** (`q_lora_rank=None` → direct Q projection) and `kv_lora_rank=512`.
MoE: 64 routed experts (top-6) + 2 shared, sigmoid router with expert bias,
grouped GEMM, `alltoall` dispatcher. No MTP. rotary base 50000, RMSNorm.
Training args (seq/iters/batch) are shrunk for a fast local check; the
architecture is unchanged. `--mock-data` + `NullTokenizer`, so no dataset needed.

## Parallelism (2 nodes × 2 GPUs, parallel)

Each run uses 2 GPUs on its own node:

```
TP=2, PP=1, CP=1  ->  DP = 2 / (2*1*1) = 1
EP=2, ETP=1       ->  expert-DP = 1        (num-experts 64 % EP == 0)
```

The job books `--nodes=2 --gres=gpu:2 --exclusive` (use `--gres`, not
`--gpus-per-node`, which can under-allocate to a single GPU on this cluster) and
launches run1 on node 0 and run2 on node 1 **simultaneously** (`srun --nodelist
--overlap … &` then `wait`). Determinism is preserved: the two runs are separate
processes on separate nodes/GPUs/NCCL-communicators with no shared state.

## What makes it deterministic

`--deterministic-mode` runs the branch's `apply_determinism_to_args`, which sets
`NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and calls `torch.use_deterministic_algorithms(True)`.
The config is chosen to pass that guard and stay on bit-exact-safe paths:

- **omit** `--cross-entropy-loss-fusion` (the guard rejects it),
- `alltoall` MoE dispatcher, `--no-gradient-accumulation-fusion`, `--no-rope-fusion`,
- `NCCL_NVLS_ENABLE=0` (keeps NCCL on the deterministic Ring allreduce),
- `NCCL_ALGO` / `NVTE_*` / `CUBLAS_*` are **not** exported by the harness — the
  code under test sets them, which is the behavior being validated.

The determinism check itself is just: run twice, `diff` the per-step `lm loss`.
