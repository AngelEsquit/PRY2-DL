"""Preprocesamiento de observaciones para ALE/SpaceInvaders-v5, pensado
para entrenar un agente DQN.

Reutiliza `crear_entorno` del Laboratorio #5 (ale_utils.py) y le agrega,
sobre el entorno base ya creado, el preprocesamiento estándar de la
literatura de DQN (Mnih et al., 2015):

    - AtariPreprocessing: escala de grises, resize a 84x84, max-pooling de
      frames consecutivos, frame-skip, y (opcionalmente) fin de episodio al
      perder una vida.
    - FrameStackObservation: apila los últimos N frames para darle al
      agente información de movimiento (velocidad/dirección de balas y
      naves), ya que una sola imagen es un estado parcialmente observable.
    - ClipReward: recorta la recompensa a {-1, 0, 1}, como en el paper
      original de DQN, para estabilizar el entrenamiento entre distintos
      juegos de Atari con escalas de puntaje muy distintas.

El entorno base se crea con frameskip=1 y repeat_action_probability=0.0
porque AtariPreprocessing hace su propio frame-skipping (por defecto 4) y
así evitamos aplicar frame-skip dos veces.
"""

from __future__ import annotations

import gymnasium as gym

from ale_utils import crear_entorno

ENV_ID = "ALE/SpaceInvaders-v5"


class EpisodicLifeWrapper(gym.Wrapper):
    """Reporta `terminated=True` cada vez que el agente pierde una vida,
    sin reiniciar realmente la partida (el juego real solo termina cuando
    se agotan las 3 vidas).

    Esto le da al agente una señal de aprendizaje más clara ("perder una
    vida es malo") sin tener que esperar el fin completo del episodio real
    para propagar esa información a través de la ecuación de Bellman.
    Basado en el wrapper equivalente de OpenAI Baselines / Stable-Baselines3
    (EpisodicLifeEnv).

    Importante: por diseño, esto SOLO debe usarse durante entrenamiento.
    Durante evaluación/competencia se debe reportar la recompensa real
    acumulada de la partida completa (3 vidas), por lo que este wrapper
    no debe aplicarse al construir el entorno de evaluación.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._lives = 0
        self._was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._was_real_done = terminated or truncated

        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Continúa la partida en la siguiente vida en lugar de reiniciar
            # el juego completo: un paso NOOP para que aparezca la nave.
            obs, _, _, _, info = self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        return obs, info

# Número de frames preprocesados apilados como estado del agente.
N_FRAME_STACK = 4
# Lado (en píxeles) del frame preprocesado (cuadrado 84x84, como en DQN).
FRAME_SIZE = 84
# Frame-skip aplicado por AtariPreprocessing (se repite la acción N pasos
# del emulador y se toma el max-pool de los últimos 2 frames crudos).
FRAME_SKIP = 4
# Pasos NOOP aleatorios al iniciar cada episodio, para variar el estado
# inicial y evitar que el agente memorice la secuencia exacta de frames.
NOOP_MAX = 30


def crear_entorno_dqn(
    video_folder=None,
    episode_trigger=None,
    terminal_on_life_loss=True,
    clip_reward=True,
    render_mode=None,
    **kwargs,
):
    """Crea el entorno ALE/SpaceInvaders-v5 con el preprocesamiento usado
    para entrenar y evaluar el agente DQN.

    Args:
        video_folder: si se especifica, graba video con RecordVideo (igual
            que en `ale_utils.crear_entorno`). Nota: el video grabado será
            del frame preprocesado en escala de grises 84x84, no de la
            imagen RGB original a menos que se use `generar_video_agente_dqn`.
        episode_trigger: ver `ale_utils.crear_entorno`.
        terminal_on_life_loss: si True (recomendado durante entrenamiento),
            cada vida perdida se reporta como fin de episodio para la señal
            de aprendizaje (terminated=True), aunque el juego real continúe
            con las vidas restantes. Esto ayuda a que el agente asocie
            perder una vida con una señal de terminación clara. Debe
            desactivarse (False) durante evaluación/competencia, ya que la
            recompensa reportada por el entorno base sigue acumulándose
            durante todas las vidas de un episodio real.
        clip_reward: si True, recorta la recompensa a {-1, 0, 1} (solo debe
            usarse para el entrenamiento; en evaluación se debe reportar la
            recompensa real sin recortar).
        render_mode: modo de render pasado a gym.make. AtariPreprocessing
            requiere "rgb_array" internamente si se usa grabación de video
            en RGB; crear_entorno ya fuerza esto si video_folder está
            definido.
        **kwargs: argumentos adicionales para gym.make (ver
            ale_utils.crear_entorno), p.ej. full_action_space=True.

    Returns:
        gymnasium.Env cuyas observaciones son arreglos (N_FRAME_STACK,
        FRAME_SIZE, FRAME_SIZE) en escala de grises normalizados 0-255
        (uint8), y cuyo espacio de acciones es Discrete (el del entorno
        base de Space Invaders).
    """
    make_kwargs = dict(kwargs)
    make_kwargs.setdefault("frameskip", 1)
    make_kwargs.setdefault("repeat_action_probability", 0.0)
    if render_mode is not None:
        make_kwargs.setdefault("render_mode", render_mode)

    env = crear_entorno(
        ENV_ID,
        video_folder=video_folder,
        episode_trigger=episode_trigger,
        **make_kwargs,
    )

    if terminal_on_life_loss:
        env = EpisodicLifeWrapper(env)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=NOOP_MAX,
        frame_skip=FRAME_SKIP,
        screen_size=FRAME_SIZE,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=N_FRAME_STACK)

    if clip_reward:
        env = gym.wrappers.ClipReward(env, min_reward=-1.0, max_reward=1.0)

    return env
