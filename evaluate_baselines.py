"""Evalúa los agentes baseline del Laboratorio #5 (aleatorio y regla
simple) sobre ALE/SpaceInvaders-v5, con el mismo protocolo usado para
evaluar el agente DQN (`evaluate.py`): 5 episodios, recompensa real de la
partida completa (sin clipping, sin terminar por vida perdida), reporte
del promedio y del máximo (criterio de ranking de la competencia).

Sirven como referencia ("baseline", análogo a una regresión lineal en un
problema supervisado) contra la cual comparar el desempeño del agente
DQN entrenado: si el DQN no supera claramente a estos baselines, algo
falló en el entrenamiento.

Uso:
    python evaluate_baselines.py --episodes 5 --video-folder videos
"""

from __future__ import annotations

import argparse

import numpy as np

from ale_utils import agente_aleatorio, agente_regla_simple, crear_entorno, ejecutar_episodio
from dqn.wrappers import ENV_ID

BASELINES = {
    "aleatorio": agente_aleatorio,
    "regla_simple": agente_regla_simple,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video-folder", default="videos")
    p.add_argument("--seed", type=int, default=123, help="Misma semilla base que evaluate.py, para comparar en igualdad de condiciones.")
    p.add_argument("--no-video", action="store_true", help="Solo calcular puntajes, sin grabar video.")
    return p.parse_args()


def evaluar_baseline(nombre, funcion_agente, episodes, seed, video_folder):
    # Subcarpeta por baseline para no pisar los videos entre sí (RecordVideo
    # siempre nombra sus archivos rl-video-episode-N.mp4 por defecto).
    video_folder_arg = None if video_folder is None else f"{video_folder}/{nombre}"
    env = crear_entorno(
        ENV_ID,
        video_folder=video_folder_arg,
        episode_trigger=(lambda episode_id: True) if video_folder_arg else None,
        render_mode="rgb_array" if video_folder_arg else None,
    )

    resultados = []
    try:
        for i in range(episodes):
            resultado = ejecutar_episodio(env, funcion_agente, seed=seed + i)
            resultados.append(resultado)
            print(f"  [{nombre}] episodio {i + 1}/{episodes}: recompensa_total={resultado['recompensa_total']:.1f} pasos={resultado['pasos']}")
    finally:
        env.close()

    return np.array([r["recompensa_total"] for r in resultados])


def main():
    args = parse_args()
    video_folder = None if args.no_video else args.video_folder

    resumen = {}
    for nombre, funcion_agente in BASELINES.items():
        print(f"Evaluando baseline: {nombre}")
        recompensas = evaluar_baseline(nombre, funcion_agente, args.episodes, args.seed, video_folder)
        resumen[nombre] = recompensas
        print(f"  -> promedio={recompensas.mean():.2f}  máximo={recompensas.max():.2f}  episodios={recompensas.tolist()}")
        print()

    print("Resumen (para la tabla de resultados / comparación con el agente DQN):")
    print(f"{'baseline':<15}{'promedio':>12}{'máximo':>12}")
    for nombre, recompensas in resumen.items():
        print(f"{nombre:<15}{recompensas.mean():>12.2f}{recompensas.max():>12.2f}")

    if video_folder:
        print(f"\nVideos guardados en: {video_folder}/ (prefijo por defecto de RecordVideo: rl-video-episode-*)")


if __name__ == "__main__":
    main()
