#!/bin/sh
# Evaluate every cached meeting under the three caption conditions, sequentially per
# meeting (Whisper is GPU-bound; parallel Whisper processes just contend). Usage:
#   sh bench/run_eval.sh [meeting ...]
cd "$(dirname "$0")/.."
M="$*"; [ -z "$M" ] && M=$(ls bench/ami/cache/*.probs.npz | sed 's|.*/||; s|.probs.npz||')
for m in $M; do
  for c in none words enrolled; do
    [ -s "bench/ami/results/$m.$c.json" ] && continue
    PYTHONPATH=. .venv/bin/python bench/ami_bench.py eval "$m" "$c" > "bench/ami/logs/$m.$c.log" 2>&1
    grep -h "^\[$m/" "bench/ami/logs/$m.$c.log" || echo "[$m/$c] FAILED (see bench/ami/logs/$m.$c.log)"
  done
done
