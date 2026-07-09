#!/bin/bash
# Reproduce the Moonlight determinism check. Run from a Slurm login shell.
# Polls every 30s and prints progress: queue state -> live iteration/loss ->
# final PASS/FAIL. Exit 0 = PASS. Ctrl-C stops watching (the job keeps running;
# re-attach later with: tail -f <log>).
set -uo pipefail
cd "$(dirname "$0")"

jobid=$(sbatch --parsable run_determinism_moonlight.slurm)
log=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/logs/det-moonlight-$jobid.out
echo "submitted $jobid -> $log"

while :; do
  if result=$(grep -m1 "RESULT: DETERMINISM" "$log" 2>/dev/null); then
    echo; echo "$result"
    links=$(grep -oE "https://wandb\.ai/\S+/runs/\S+" "$log" 2>/dev/null | sort -u)
    if [ -n "$links" ]; then echo "wandb runs:"; echo "$links" | sed 's/^/  /'; else echo "wandb: offline/disabled"; fi
    [[ "$result" == *PASS* ]]; exit
  fi
  ts=$(date +%H:%M:%S)
  if [ -f "$log" ]; then
    prog=$(grep -oE "iteration +[0-9]+/ *[0-9]+.*lm loss: [0-9.eE+-]+" "$log" | tail -1)
    echo "[$ts] ${prog:-initializing...}"
  else
    echo "[$ts] queued: $(squeue -j "$jobid" -h -o '%T (%r)' 2>/dev/null || echo '?')"
  fi
  sleep 30
done
