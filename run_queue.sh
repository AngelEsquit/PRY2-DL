#!/usr/bin/env bash
# Cola de entrenamiento desatendida: 2 semillas de Dueling+Double DQN con sticky
# actions (0.25), 5M pasos, con snapshots cada 1M; luego evaluación de robustez
# de todos los snapshots. Secuencial (cada buffer usa ~8.5 GB de RAM).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

for seed in 0 1; do
  it="iter05_sticky5m_s${seed}"
  echo "=== $(date) inicio $it"
  python -m dqn.train --iteration "$it" --total-steps 5000000 \
    --architecture dueling --double-dqn --buffer-size 150000 \
    --epsilon-end 0.02 --sticky-prob 0.25 --snapshot-freq 1000000 \
    --seed "$seed" > "logs/${it}_stdout.log" 2>&1
  rc=$?
  echo "=== $(date) fin $it (exit $rc)"
done

echo "=== $(date) evaluación de robustez"
python evaluate_robustness.py \
  --checkpoints checkpoints/iter05_sticky5m_s0_*k.pt checkpoints/iter05_sticky5m_s1_*k.pt \
  --episodes 50 --output logs/robustness_iter05.csv > logs/robustness_iter05_stdout.log 2>&1
echo "=== $(date) COLA COMPLETA"
