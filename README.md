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

Para continuar entrenando desde un checkpoint existente en vez de empezar de cero (pesos, optimizador y
contador de pasos), usar `--resume-from`. Si el checkpoint es de antes de que se guardara `env_step` (como
`best_run`), hay que indicar `--start-step` explícitamente para no reiniciar el decaimiento de ε:

```bash
python -m dqn.train --iteration iter03 --total-steps 5000000 \
    --resume-from checkpoints/best_run.pt --start-step 3000000
```

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

## Resultados actuales

| Agente | Reward promedio (5 ep.) | Reward máximo |
|---|---|---|
| Baseline aleatorio | 185.0 | 260.0 |
| Baseline regla simple | 270.0 | 270.0 |
| **DQN (`best_run`: Dueling + Double DQN, 3M pasos)** | **710.0** | **800.0** |
| DQN (`iter03`: `best_run` continuado a 5M pasos) | 634.0 | 655.0 |
| DQN (`iter_plain_dqn`: sin Double DQN ni Dueling, 1.5M pasos) | 602.0 | 755.0 |
| DQN (`iter_double_only`: Double DQN sin Dueling, 1.5M pasos) | 385.0 | 490.0 |
| DQN (`iter_dueling_only`: Dueling sin Double DQN, 1.5M pasos) | 360.0 | 460.0 |

`best_run`: arquitectura Dueling DQN, Double DQN activado, γ=0.99, lr=2.5e-4, replay buffer de 150k,
ε decae de 1.0 a 0.02 en el primer millón de pasos, 3,000,000 pasos de entorno totales
(~153 min en una GTX 1660 Super). Episodios de evaluación: `[605, 715, 800, 715, 715]`. La pérdida se
mantuvo estable y baja (~0.01-0.02) durante todo el entrenamiento, sin señales de divergencia.

`iter03`: mismos hiperparámetros, se reanudó `best_run` (`--resume-from` + `--start-step 3000000`) hasta
5,000,000 pasos totales para probar si más entrenamiento seguía mejorando el agente. Episodios de
evaluación: `[605, 655, 600, 655, 655]` — **el resultado empeoró** respecto a `best_run` (710→634 promedio,
800→655 máximo) en vez de mejorar. La pérdida siguió baja y estable (sin divergencia visible en la curva),
por lo que no parece un colapso catastrófico clásico; una explicación plausible es que el replay buffer no
se persiste entre corridas, así que al reanudar se rellenó desde cero con ε ya en 0.02 (casi greedy) — es
decir, con transiciones muy correlacionadas y poco diversas del propio agente, en vez de la mezcla más
exploratoria que tuvo el buffer original durante el decaimiento de ε de 1.0 a 0.02. Esto sugiere que "más
pasos" no es una mejora gratuita en DQN: la calidad/diversidad del buffer en el momento de reanudar importa
tanto como el número total de pasos. `best_run` (3M) se mantiene como el mejor agente reportado.

`iter_plain_dqn`: mismos hiperparámetros e igual número de pasos de warmup/decaimiento que `best_run`, pero
con arquitectura `dqn` (sin Dueling) y `--no-double-dqn` (DQN vanilla), entrenado 1,500,000 pasos (la mitad
que `best_run`, por tiempo disponible — la comparación no es 100% controlada por esto). Episodios de
evaluación: `[260, 755, 485, 755, 755]`. Pérdida estable (~0.01-0.017), sin divergencia. Promedio (602.0) y
máximo (755.0) quedan por debajo de `best_run`, y la varianza entre episodios es notablemente mayor
(260-755 vs. 605-800 en `best_run`), consistente con lo esperado: sin Double DQN la sobreestimación de
valores Q produce una política menos consistente, y sin Dueling la red generaliza peor entre acciones en
estados donde el valor no depende mucho de cuál se tome. Sugiere que ambas mejoras (Double + Dueling)
contribuyen tanto al puntaje como a la estabilidad de la política, aunque con solo la mitad de los pasos de
entrenamiento de `best_run` no se puede aislar completamente su efecto del de la duración del entrenamiento.

`iter_double_only` e `iter_dueling_only`: mismos hiperparámetros y 1,500,000 pasos (igual que
`iter_plain_dqn`), pero activando **solo una** mejora cada vez — Double DQN sin Dueling, y Dueling sin
Double DQN, respectivamente — para intentar aislar el efecto individual de cada una. Episodios de
evaluación: `iter_double_only` = `[460, 325, 490, 325, 325]`, `iter_dueling_only` = `[120, 460, 300, 460,
460]`. Resultado contraintuitivo: **ambas variantes con una sola mejora activada quedan por debajo de
`iter_plain_dqn`** (que no tiene ninguna), y muy por debajo de `best_run` (que tiene ambas, pero a 3M
pasos). Esto no respalda una historia simple de "cada mejora ayuda de forma independiente y monotónica";
es más consistente con que, a 1.5M pasos, ninguna configuración individual ha convergido todavía y el
resultado está dominado por varianza de una sola semilla, no por el efecto real de cada componente.
Además, esta comparación **no está controlada por número de pasos** frente a `best_run` (3M) — extender
estas dos corridas a 3M pasos requeriría reentrenar desde cero (no reanudar desde el checkpoint de 1.5M),
porque reanudar después de que ε ya decayó reproduciría el mismo artefacto de buffer de baja diversidad
documentado arriba para `iter03`. Por limitaciones de tiempo, esa corrida de control queda pendiente como
trabajo futuro; los resultados de `iter_double_only` e `iter_dueling_only` deben leerse como preliminares,
no como evidencia concluyente del efecto aislado de cada mejora.

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
