#!/bin/sh
# Keep N diarization workers busy until every meeting is cached. Usage: sh bench/run_pool.sh [N]
cd "$(dirname "$0")/.."
N=${1:-3}
for m in EN2002a EN2002d ES2004d ES2004c TS3003b ES2004b EN2002c TS3003c TS3003d IS1009a ES2004a TS3003a EN2002b IS1009c IS1009d IS1009b; do
  [ -s "bench/ami/cache/$m.probs.npz" ] && continue
  pgrep -f "ami_bench.py diarize $m\$" >/dev/null && continue
  while [ "$(ps -eo command | grep -c '^\.venv/bin/python bench/ami_bench.py diarize')" -ge "$N" ]; do sleep 15; done
  ( PYTHONPATH=. .venv/bin/python bench/ami_bench.py diarize "$m" > "bench/ami/logs/$m.diarize.log" 2>&1; grep -h "^\[$m\]" "bench/ami/logs/$m.diarize.log" ) &
  sleep 5
done
wait
echo "POOL DONE"
