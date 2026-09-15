"""Evalúa un agente DQN entrenado sobre ALE/SpaceInvaders-v5 con política
greedy (sin exploración), siguiendo el protocolo de la competencia:

    - 5 episodios de evaluación.
    - Se reporta la recompensa total real de cada episodio (sin recortar)
      y el máximo de los 5 (criterio de ranking de la competencia).
    - Se graba un video del agente jugando, con la misma configuración de
      preprocesamiento usada en entrenamiento (salvo por clip_reward y
      terminal_on_life_loss, que solo aplican durante entrenamiento).

Reutiliza `ejecutar_episodio` del Laboratorio #5 (ale_utils.py),
reemplazando la función de agente aleatorio/regla simple por la política
aprendida del agente DQN (ver dqn/agent.py: DQNAgent.policy_fn).

Uso:
    python evaluate.py --checkpoint checkpoints/iter01.pt --episodes 5 --video-folder videos
"""

from __future__ import annotations

import argparse

import numpy as np

from ale_utils import ejecutar_episodio
from dqn.agent import DQNAgent
from dqn.wrappers import crear_entorno_dqn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pt guardado por dqn/train.py.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video-folder", default="videos")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default=None, help="cpu o cuda. Por defecto usa el dispositivo guardado en el checkpoint.")
    p.add_argument("--full-action-space", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    agent = DQNAgent.load(args.checkpoint, device=args.device)
    policy = agent.policy_fn()

    env = crear_entorno_dqn(
        video_folder=args.video_folder,
        # RecordVideo por defecto solo graba episode_id 0, 1, 8, 27, ...
        # (cubos perfectos); forzamos grabar los `episodes` episodios.
        episode_trigger=lambda episode_id: True,
        terminal_on_life_loss=False,  # recompensa/episodio real de la partida completa
        clip_reward=False,  # puntaje real, sin recortar (solo se recorta en entrenamiento)
        full_action_space=args.full_action_space,
    )

    resultados = []
    try:
        for i in range(args.episodes):
            resultado = ejecutar_episodio(env, policy, seed=args.seed + i)
            resultados.append(resultado)
            print(f"  episodio {i + 1}/{args.episodes}: recompensa_total={resultado['recompensa_total']:.1f} pasos={resultado['pasos']}")
    finally:
        env.close()

    recompensas = np.array([r["recompensa_total"] for r in resultados])
    print()
    print(f"Recompensa promedio: {recompensas.mean():.2f}")
    print(f"Recompensa máxima (criterio de ranking): {recompensas.max():.2f}")
    print(f"Recompensas por episodio: {recompensas.tolist()}")
    print(f"Videos guardados en: {args.video_folder}/")


if __name__ == "__main__":
    main()
