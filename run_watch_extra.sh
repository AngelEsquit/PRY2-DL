#!/usr/bin/env bash
# Vigilante extra: evalúa los snapshots de iter08_detlong_s6 en hitos adicionales
# (51-54M y 56-60M) según aparezcan. Complementa a run_queue4.sh (que ya cubre
# 5, 8, 10, 12, 15, 20, ..., 50, 55, 60, 65M). Mismo protocolo: 100 episodios,
# semillas desde 5000, ambos settings. Termina cuando el entrenamiento finaliza.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
IT="iter08_detlong_s6"
MILESTONES="51 52 53 54 56 57 58 59"

for _ in $(seq 1 720); do  # máx. ~60 h
  for m in $MILESTONES; do
    f="checkpoints/${IT}_${m}000k.pt"
    out="logs/robustness_long/${m}M.csv"
    if [ -f "$f" ] && [ ! -f "$out" ]; then
      sleep 20
      python evaluate_robustness.py --checkpoints "$f" --episodes 100 --seed 5000 \
        --output "$out" > "logs/robustness_long/${m}M_stdout.log" 2>&1
      echo "=== $(date) evaluado snapshot ${m}M"
    fi
  done
  grep -q "entrenamiento finalizado" "logs/${IT}_stdout.log" 2>/dev/null && break
  sleep 300
done
echo "=== $(date) vigilante extra terminado"
