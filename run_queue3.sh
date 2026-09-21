#!/usr/bin/env bash
# Cola 3: espera a que termine iter06_det5m_s3, lo evalúa, y lanza una corrida
# larga (objetivo 50M pasos, misma config que iter04_clean5m) con snapshots
# livianos cada 1M. Mientras entrena, evalúa los snapshots de una lista de hitos
# (sticky y determinista, 100 episodios) en logs/robustness_long/.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
IT="iter08_det50m_s6"
mkdir -p logs/robustness_long

# 1) esperar a seed 3 (máx. 4 h)
for _ in $(seq 1 480); do
  grep -q "entrenamiento finalizado" logs/iter06_det5m_s3_stdout.log 2>/dev/null && break
  sleep 30
done
echo "=== $(date) seed 3 terminó (o timeout)"
python evaluate_robustness.py --checkpoints checkpoints/iter06_det5m_s3.pt \
  --episodes 200 --seed 5000 --output logs/robustness_iter06_det5m_s3_final.csv \
  > logs/robustness_iter06_det5m_s3_final_stdout.log 2>&1
python evaluate_robustness.py --checkpoints checkpoints/iter06_det5m_s3_3000k.pt checkpoints/iter06_det5m_s3_4000k.pt \
  --episodes 50 --output logs/robustness_iter06_det5m_s3_snaps.csv \
  > logs/robustness_iter06_det5m_s3_snaps_stdout.log 2>&1
echo "=== $(date) seed 3 evaluado"

# 2) corrida larga en segundo plano
python -m dqn.train --iteration "$IT" --total-steps 50000000 \
  --architecture dueling --double-dqn --gamma 0.99 --lr 2.5e-4 \
  --buffer-size 150000 --epsilon-start 1.0 --epsilon-end 0.02 \
  --epsilon-decay-steps 1000000 --snapshot-freq 1000000 --snapshot-light \
  --seed 6 > "logs/${IT}_stdout.log" 2>&1 &
TRAIN_PID=$!
echo "=== $(date) inicio $IT (pid $TRAIN_PID)"

# 3) evaluar snapshots hito según aparezcan
MILESTONES="5 8 10 12 15 20 25 30 35 40 45 50"
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
echo "=== $(date) COLA 3 COMPLETA (entrenamiento terminó)"
