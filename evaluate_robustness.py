"""Evaluación de robustez de checkpoints: muchos episodios greedy con semillas
variadas, bajo dos configuraciones del entorno:

    - determinista: repeat_action_probability=0.0 (la usada en entrenamiento y
      en evaluate.py).
    - sticky: repeat_action_probability=0.25 (el valor por defecto de
      ALE/SpaceInvaders-v5 en Gymnasium).

Con 5 episodios greedy en un entorno casi determinista solo hay 2-3
trayectorias distintas, así que el promedio/máximo de 5 es muy ruidoso. Aquí
se reportan media, mediana, desviación, mínimo, máximo y nº de puntajes
distintos sobre N episodios, para elegir el checkpoint de la competencia.

Uso:
    python evaluate_robustness.py --checkpoints checkpoints/best_run.pt checkpoints/iter04_clean5m.pt \
        --episodes 50 --output logs/robustness_eval.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ale_utils import ejecutar_episodio
from dqn.agent import DQNAgent
from dqn.wrappers import crear_entorno_dqn

SETTINGS = {"determinista": 0.0, "sticky": 0.25}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1000, help="Semilla base; el episodio i usa seed + i.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="logs/robustness_eval.csv")
    return p.parse_args()


def evaluar(policy, repeat_action_probability, episodes, seed):
    env = crear_entorno_dqn(
        terminal_on_life_loss=False,
        clip_reward=False,
        repeat_action_probability=repeat_action_probability,
    )
    try:
        return np.array(
            [ejecutar_episodio(env, policy, seed=seed + i)["recompensa_total"] for i in range(episodes)]
        )
    finally:
        env.close()


def main():
    args = parse_args()
    filas = []
    for ckpt in args.checkpoints:
        nombre = Path(ckpt).stem
        policy = DQNAgent.load(ckpt, device=args.device).policy_fn()
        for setting, prob in SETTINGS.items():
            r = evaluar(policy, prob, args.episodes, args.seed)
            fila = {
                "checkpoint": nombre,
                "setting": setting,
                "episodes": len(r),
                "mean": r.mean(),
                "median": np.median(r),
                "std": r.std(),
                "min": r.min(),
                "max": r.max(),
                "distinct_scores": len(np.unique(r)),
            }
            filas.append(fila)
            print(
                f"{nombre:24s} {setting:13s} mean={fila['mean']:7.1f} median={fila['median']:6.1f} "
                f"std={fila['std']:6.1f} min={fila['min']:6.1f} max={fila['max']:6.1f} "
                f"distintos={fila['distinct_scores']}/{len(r)}",
                flush=True,
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
