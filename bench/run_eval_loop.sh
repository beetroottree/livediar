#!/bin/sh
# Keep scoring meetings as their caches appear until diarization is done and all 48 results exist.
cd "$(dirname "$0")/.."
while :; do
  sh bench/run_eval.sh
  n=$(ls bench/ami/results/*.json 2>/dev/null | wc -l | tr -d ' ')
  if grep -q "DIARIZE DONE" bench/ami/logs/diarize.batch.log 2>/dev/null && [ "$n" -ge 48 ]; then break; fi
  sleep 60
done
PYTHONPATH=. .venv/bin/python bench/ami_bench.py report > bench/ami/logs/report.log 2>&1
echo "EVAL DONE ($n results)"
