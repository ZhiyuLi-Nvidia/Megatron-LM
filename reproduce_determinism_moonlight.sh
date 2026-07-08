#!/bin/bash
# Reproduce the Moonlight-16B-A3B determinism check. Run from a Slurm login shell.
# Submits the harness, waits for the RESULT line, prints it. Exit 0 = PASS.
set -euo pipefail
cd "$(dirname "$0")"

jobid=$(sbatch --parsable run_determinism_moonlight.slurm)
log=/lustre/fsw/portfolios/llmservice/users/zhiyul/det-adhoc/logs/det-moonlight-$jobid.out
echo "submitted $jobid; waiting for $log"

until result=$(grep -m1 "RESULT: DETERMINISM" "$log" 2>/dev/null); do sleep 30; done
echo "$result"
[[ "$result" == *PASS* ]]
