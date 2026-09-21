#!/usr/bin/env bash
# Cola 2: espera a que termine iter05_sticky5m_s2, luego entrena 2 semillas más
# de la config de iter04_clean5m (Dueling+Double, entrenamiento determinista,
# 5M pasos) y evalúa cada checkpoint (final: 200 episodios; snapshots 3M/4M: 50).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

# Esperar (máx. 6 h) a que seed 2 termine.
for _ in $(seq 1 720); do
  grep -q "entrenamiento finalizado" logs/iter05_sticky5m_s2_stdout.log 2>/dev/null && break
  sleep 30
done
echo "=== $(date) seed 2 terminó (o timeout); inicio cola 2"

for seed in 3 4; do
  it="iter06_det5m_s${seed}"
  echo "=== $(date) inicio $it"
  python -m dqn.train --iteration "$it" --total-steps 5000000 \
    --architecture dueling --double-dqn --gamma 0.99 --lr 2.5e-4 \
    --buffer-size 150000 --epsilon-start 1.0 --epsilon-end 0.02 \
    --epsilon-decay-steps 1000000 --snapshot-freq 1000000 \
    --seed "$seed" > "logs/${it}_stdout.log" 2>&1
  rc=$?
  echo "=== $(date) fin $it (exit $rc)"
  python evaluate_robustness.py --checkpoints "checkpoints/${it}.pt" \
    --episodes 200 --seed 5000 --output "logs/robustness_${it}_final.csv" \
    > "logs/robustness_${it}_final_stdout.log" 2>&1
  python evaluate_robustness.py --checkpoints "checkpoints/${it}_3000k.pt" "checkpoints/${it}_4000k.pt" \
    --episodes 50 --output "logs/robustness_${it}_snaps.csv" \
    > "logs/robustness_${it}_snaps_stdout.log" 2>&1
done

echo "=== $(date) COLA 2 COMPLETA"
