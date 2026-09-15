# PRY2-DL — Agente DQN para ALE/SpaceInvaders-v5

CC3092 Deep Learning y Sistemas Inteligentes — Proyecto 2: Competencia de Agentes en Space Invaders.

Construido sobre las utilidades del [Laboratorio #5](https://github.com/AngelEsquit/Lab5-DL) (`ale_utils.py`:
`crear_entorno`, `ejecutar_episodio`, `generar_video_agente`), reemplazando el agente aleatorio / de regla
simple por una política aprendida con **DQN / Double DQN / Dueling DQN**.

## Estructura del repositorio

```
ale_utils.py          # Utilidades del Lab 5 (sin modificar), base de todo el proyecto
dqn/
  wrappers.py          # Preprocesamiento Atari: grises, resize 84x84, frame-stack,
                        # frame-skip, fin de episodio por vida perdida, reward clipping
  model.py             # Arquitecturas de red: NatureCNN, QNetwork, DuelingQNetwork
  replay_buffer.py      # Replay buffer (numpy, uint8)
  agent.py              # DQNAgent: epsilon-greedy, target network, DQN/Double DQN, save/load
  train.py              # Script de entrenamiento (CLI)
evaluate.py              # Evaluación greedy (5 episodios) + generación de video
evaluate_baselines.py     # Baselines de referencia (agente aleatorio y de regla simple del Lab 5)
notebooks/
  Proyecto2_SpaceInvaders_DQN.ipynb   # Análisis del entorno, metodología, resultados
checkpoints/             # Pesos guardados por iteración (*.pt, ignorados por git salvo README)
logs/                     # Una fila por episodio por corrida (*.csv, ignorados por git)
videos/                   # Videos de evaluación (*.mp4, ignorados por git)
requirements.txt
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`torch` en `requirements.txt` instalará la versión CPU por defecto en algunos entornos; si hay GPU NVIDIA
disponible, instalar la build con CUDA siguiendo https://pytorch.org/get-started/locally/ **antes** de
`pip install -r requirements.txt` (o reinstalar torch después) para acelerar el entrenamiento
significativamente.

## Reproducir el entrenamiento

Cada corrida es una "iteración" documentada en el trabajo escrito (sección 2.3):

```bash
python -m dqn.train --iteration iter01 --total-steps 2000000 --architecture dqn --double-dqn
python -m dqn.train --iteration iter02 --total-steps 2000000 --architecture dueling --double-dqn
```

Esto escribe `logs/<iteration>.csv` (recompensa, longitud, epsilon y pérdida por episodio) y
`checkpoints/<iteration>.pt` (actualizado cada `--checkpoint-freq` pasos y al finalizar). Ver
`python -m dqn.train --help` para todos los hiperparámetros configurables (gamma, learning rate,
tamaño de batch, tamaño del replay buffer, frecuencia de actualización de la red objetivo,
decaimiento de epsilon, etc.).

El notebook `notebooks/Proyecto2_SpaceInvaders_DQN.ipynb` carga estos CSV para graficar las curvas de
entrenamiento y compara iteraciones.

## Cargar el modelo final y evaluar (protocolo de competencia)

```bash
python evaluate.py --checkpoint checkpoints/<iteration>.pt --episodes 5 --video-folder videos
```

Esto:
1. Reconstruye la arquitectura desde la configuración guardada en el checkpoint.
2. Ejecuta 5 episodios completos con política **greedy** (sin exploración) sobre el entorno real
   (recompensa sin recortar, episodio = partida completa de 3 vidas — no se corta por vida perdida).
3. Imprime la recompensa de cada episodio, el promedio y el máximo (criterio de ranking de la competencia).
4. Graba un video `.mp4` por episodio en `videos/`, con la misma configuración de preprocesamiento
   (frame-skip, resize, frame-stack) usada en el entrenamiento.

## Baselines de referencia

El profesor pidió comparar contra un baseline (análogo a una regresión lineal en un problema supervisado:
no busca ser competitivo, solo dar un punto de referencia). Se usan los dos agentes sin aprendizaje ya
implementados en el Laboratorio #5:

```bash
python evaluate_baselines.py --episodes 5 --video-folder videos
```

- **`agente_aleatorio`**: acción aleatoria en cada paso (baseline estándar en RL, equivalente a "predecir la
  media" en regresión).
- **`agente_regla_simple`**: política fija (dispara siempre que puede) sin ningún tipo de aprendizaje.

Reporta promedio y máximo de 5 episodios con el mismo protocolo que `evaluate.py` (recompensa real, sin
recortar, episodio = partida completa), para poder compararlos directamente en la tabla de resultados del
trabajo escrito junto a las iteraciones del agente DQN.

## Notas de diseño relevantes para el trabajo escrito

- **Preprocesamiento** (`dqn/wrappers.py`): escala de grises + resize a 84×84, `frame_skip=4` con
  max-pooling de los últimos 2 frames crudos (estándar de Mnih et al., 2015), y apilado de 4 frames para
  darle al agente información de movimiento.
- **Fin de episodio por vida perdida**: implementado con un wrapper propio (`EpisodicLifeWrapper`, basado en
  `EpisodicLifeEnv` de Baselines/SB3) — reporta `terminated=True` cada vez que se pierde una vida (para
  propagar antes la señal de "esto fue malo"), pero **sin** reiniciar la partida real; solo se usa durante
  **entrenamiento**. En evaluación (`evaluate.py`) se desactiva para reportar la recompensa real de la
  partida completa, como exige el protocolo de competencia.
- **Reward clipping** a `{-1, 0, 1}`: solo durante entrenamiento, para estabilizar la magnitud del error de
  TD; en evaluación se usa la recompensa real sin recortar.
- **DQN vs Double DQN**: aislado en `DQNAgent._compute_targets` — Double DQN usa la red online para elegir
  la mejor acción del siguiente estado y la red objetivo solo para evaluarla (reduce sobreestimación de Q).
- **Dueling DQN**: arquitectura alternativa (`--architecture dueling`) que separa V(s) y A(s,a); se puede
  combinar con Double DQN (son ortogonales).
