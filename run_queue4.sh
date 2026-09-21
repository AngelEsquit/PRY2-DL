#!/usr/bin/env bash
# Cola 4: corrida larga (objetivo 50M pasos, misma config que iter04_clean5m) con
# snapshots livianos cada 1M; evalúa los snapshots de una lista de hitos
# (sticky y determinista, 100 episodios) en logs/robustness_long/.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
IT="iter08_detlong_s6"
mkdir -p logs/robustness_long

python -m dqn.train --iteration "$IT" --total-steps 100000000 \
  --architecture dueling --double-dqn --gamma 0.99 --lr 2.5e-4 \
  --buffer-size 150000 --epsilon-start 1.0 --epsilon-end 0.02 \
  --epsilon-decay-steps 1000000 --snapshot-freq 1000000 --snapshot-light \
  --seed 6 > "logs/${IT}_stdout.log" 2>&1 &
TRAIN_PID=$!
echo "=== $(date) inicio $IT (pid $TRAIN_PID)"

MILESTONES="5 8 10 12 15 20 25 30 35 40 45 50 55 60 65"
while kill -0 $TRAIN_PID 2>/dev/null; do
  for m in $MILESTONES; do
    f="checkpoints/${IT}_${m}000k.pt"
    out="logs/robustness_long/${m}M.csv"
    if [ -f "$f" ] && [ ! -f "$out" ]; then
      sleep 20  # asegurar que el archivo terminó de escribirse
      python evaluate_robustness.py --checkpoints "$f" --episodes 100 --seed 5000 \
        --output "$out" > "logs/robustness_long/${m}M_stdout.log" 2>&1
      echo "=== $(date) evaluado snapshot ${m}M"
    fi
  done
  sleep 300
done
echo "=== $(date) COLA 4 COMPLETA (entrenamiento terminó)"
