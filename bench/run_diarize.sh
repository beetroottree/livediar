#!/bin/sh
# Diarize every AMI test meeting once (Sortformer, preset low), N in parallel. Usage: sh bench/run_diarize.sh [N]
cd "$(dirname "$0")/.."
N=${1:-5}
printf "%s\n" IS1009a ES2004a TS3003a EN2002b IS1009c IS1009d IS1009b EN2002a EN2002d ES2004d ES2004c TS3003b ES2004b EN2002c TS3003c TS3003d \
 | xargs -P "$N" -I{} sh -c '[ -s bench/ami/cache/{}.probs.npz ] || PYTHONPATH=. .venv/bin/python bench/ami_bench.py diarize {} > bench/ami/logs/{}.diarize.log 2>&1; grep -h "^\[{}\]" bench/ami/logs/{}.diarize.log'
echo "DIARIZE DONE"
