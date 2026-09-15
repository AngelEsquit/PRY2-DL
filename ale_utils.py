"""Módulo de funciones reutilizables para interactuar con entornos de
Gymnasium / Arcade Learning Environment (ALE), ejecutar agentes y grabar
video de las partidas.

Funciones:
    crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, **kwargs)
    agente_aleatorio(observation, env)
    agente_regla_simple(observation, env)
    ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None)
    generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1, **kwargs)
"""

from __future__ import annotations

import gymnasium as gym

try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass


def crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, **kwargs):
    """Crea y retorna un entorno de Gymnasium.

    Funciona tanto para entornos de Atari/ALE (p. ej. "ALE/SpaceInvaders-v5")
    como para cualquier otro entorno de Gymnasium (p. ej. "CartPole-v1").

    Args:
        nombre_entorno: id del entorno a crear con gym.make.
        video_folder: si se especifica, envuelve el entorno con
            gymnasium.wrappers.RecordVideo para grabar episodios en esa carpeta.
            Fuerza render_mode="rgb_array" si el usuario no indicó uno.
        episode_trigger: función episodio -> bool que decide qué episodios
            grabar. Si video_folder está definido y no se pasa esta función,
            se graban todos los episodios (lambda ep: True).
        **kwargs: argumentos adicionales pasados directamente a gym.make
            (p. ej. obs_type="ram", frameskip=4, repeat_action_probability=0.0,
            full_action_space=False).

    Returns:
        Instancia de gymnasium.Env (envuelta en RecordVideo si aplica).
    """
    if video_folder is not None:
        kwargs.setdefault("render_mode", "rgb_array")

    env = gym.make(nombre_entorno, **kwargs)

    if video_folder is not None:
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_folder, episode_trigger=episode_trigger
        )

    return env


def agente_aleatorio(observation, env):
    """Agente baseline sin entrenamiento: muestrea una acción del
    espacio de acciones del entorno, ignorando la observación.

    Args:
        observation: observación actual del entorno (no se usa).
        env: entorno de Gymnasium, del cual se toma action_space.

    Returns:
        Una acción válida muestreada de env.action_space.sample().
    """
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """Agente basado en una regla simple para ALE/SpaceInvaders-v5.

    Heurística: dispara siempre que puede (acción FIRE) intercalando el
    movimiento lateral, para mantener presión de disparo sin quedarse
    quieto. Si el entorno no expone las acciones esperadas (NOOP, FIRE,
    RIGHT, LEFT, RIGHTFIRE, LEFTFIRE), recurre a una acción aleatoria.

    Args:
        observation: observación actual del entorno (no se usa; la regla
            no depende de la posición de las naves ni del jugador).
        env: entorno de Gymnasium, usado para consultar los significados
            de las acciones disponibles (get_action_meanings).

    Returns:
        Una acción entera válida para env.action_space.
    """
    try:
        significados = env.unwrapped.get_action_meanings()
    except AttributeError:
        return env.action_space.sample()

    preferidas = ["RIGHTFIRE", "LEFTFIRE", "FIRE"]
    for nombre in preferidas:
        if nombre in significados:
            return significados.index(nombre)

    return env.action_space.sample()


def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None):
    """Ejecuta un episodio completo usando la función de agente indicada.

    Args:
        env: entorno de Gymnasium ya creado (típicamente con crear_entorno).
        funcion_agente: función (observation, env) -> action.
        max_steps: número máximo de pasos antes de detener el episodio
            aunque no haya terminado ni sido truncado.
        seed: semilla opcional para env.reset.

    Returns:
        dict con al menos:
            "pasos": número de pasos ejecutados en el episodio.
            "recompensa_total": recompensa (return) acumulada del episodio.
            "terminated": bool, si el episodio terminó por la dinámica del entorno.
            "truncated": bool, si el episodio fue truncado (límite externo).
    """
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    pasos = 0
    recompensa_total = 0.0

    while not (terminated or truncated) and pasos < max_steps:
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        recompensa_total += reward
        pasos += 1

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": terminated,
        "truncated": truncated,
    }


def generar_video_agente(
    nombre_entorno,
    funcion_agente,
    video_folder,
    name_prefix,
    n_episodios=1,
    max_steps=10000,
    seed=0,
    **kwargs,
):
    """Función de alto nivel: crea el entorno con grabación de video
    habilitada, ejecuta n_episodios episodios completos con la función de
    agente indicada, cierra el entorno correctamente y retorna las rutas
    de los videos generados junto con las métricas de cada episodio.

    Args:
        nombre_entorno: id del entorno (p. ej. "ALE/SpaceInvaders-v5").
        funcion_agente: función (observation, env) -> action a usar en
            cada episodio (p. ej. agente_aleatorio o agente_regla_simple).
        video_folder: carpeta donde RecordVideo escribirá los .mp4.
        name_prefix: prefijo de nombre para los archivos de video generados.
        n_episodios: número de episodios completos a ejecutar y grabar.
        max_steps: límite de pasos por episodio (ver ejecutar_episodio).
        seed: semilla base; el episodio i se ejecuta con seed + i.
        **kwargs: argumentos adicionales pasados a crear_entorno / gym.make.

    Returns:
        dict con:
            "videos": lista de rutas (str) a los archivos .mp4 generados.
            "metricas": lista de dicts (uno por episodio) con "pasos" y
                "recompensa_total" (además de "terminated"/"truncated").
    """
    # gym.make no acepta name_prefix; se pasa directamente a RecordVideo.
    make_kwargs = dict(kwargs)
    make_kwargs.setdefault("render_mode", "rgb_array")
    base_env = gym.make(nombre_entorno, **make_kwargs)
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: True,
    )

    metricas = []
    try:
        for i in range(n_episodios):
            resultado = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps, seed=seed + i
            )
            metricas.append(resultado)
    finally:
        env.close()

    videos = sorted(str(p) for p in _listar_videos(video_folder, name_prefix))

    return {"videos": videos, "metricas": metricas}


def _listar_videos(video_folder, name_prefix):
    """Devuelve las rutas .mp4 en video_folder que empiezan con name_prefix."""
    import pathlib

    carpeta = pathlib.Path(video_folder)
    if not carpeta.exists():
        return []
    return list(carpeta.glob(f"{name_prefix}*.mp4"))
