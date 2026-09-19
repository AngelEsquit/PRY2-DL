"""Script de entrenamiento de un agente DQN / Double DQN / Dueling DQN
sobre ALE/SpaceInvaders-v5.

Uso:
    python -m dqn.train --iteration iter01 --total-steps 2000000

Cada corrida escribe:
    logs/<iteration>.csv       -- una fila por episodio (para graficar curvas)
    checkpoints/<iteration>.pt -- pesos del agente (actualizados periódicamente)

Ver README.md para una descripción de los hiperparámetros y su efecto.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ale_utils import ejecutar_episodio
from dqn.agent import DQNAgent, DQNConfig
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import FRAME_SIZE, N_FRAME_STACK, crear_entorno_dqn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iteration", required=True, help="Identificador de la corrida (para logs/ y checkpoints/).")
    p.add_argument("--total-steps", type=int, default=2_000_000, help="Pasos de entorno totales a entrenar (absoluto, incluyendo los ya hechos si se usa --resume-from).")
    p.add_argument("--resume-from", default=None, help="Ruta a un checkpoint (.pt) desde el cual continuar entrenando (pesos, optimizador y contador de pasos).")
    p.add_argument("--start-step", type=int, default=None, help="Env_step desde el cual continuar (sobreescribe el guardado en el checkpoint). Obligatorio con --resume-from si el checkpoint es de antes de que se guardara env_step (dará error explícito en ese caso).")
    p.add_argument("--architecture", choices=["dqn", "dueling"], default="dqn")
    p.add_argument("--double-dqn", action="store_true", default=True)
    p.add_argument("--no-double-dqn", dest="double_dqn", action="store_false")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=200_000, help="Capacidad del replay buffer (en transiciones).")
    p.add_argument("--learning-starts", type=int, default=50_000, help="Pasos de exploración aleatoria antes de empezar a entrenar.")
    p.add_argument("--train-freq", type=int, default=4, help="Entrenar cada N pasos de entorno.")
    p.add_argument("--target-update-freq", type=int, default=10_000, help="Actualizar la red objetivo cada N pasos de GRADIENTE.")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.01)
    p.add_argument("--epsilon-decay-steps", type=int, default=1_000_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000, help="Guardar checkpoint cada N pasos de entorno.")
    p.add_argument("--snapshot-freq", type=int, default=0, help="Si > 0, guarda además un checkpoint permanente <iteration>_<N>k.pt cada N pasos de entorno (sin sobreescribir), para evaluar la curva de puntaje vs. pasos.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-action-space", action="store_true", help="Usar el espacio de acciones completo de ALE (18) en vez del subconjunto mínimo del juego.")
    p.add_argument("--sticky-prob", type=float, default=0.0, help="repeat_action_probability del entorno durante entrenamiento (0.0 = determinista, como en las corridas previas; 0.25 = valor por defecto de ALE/SpaceInvaders-v5).")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = crear_entorno_dqn(
        terminal_on_life_loss=True,
        clip_reward=True,
        full_action_space=args.full_action_space,
        repeat_action_probability=args.sticky_prob,
    )
    n_actions = env.action_space.n
    obs_shape = (N_FRAME_STACK, FRAME_SIZE, FRAME_SIZE)

    if args.resume_from:
        agent = DQNAgent.load(args.resume_from, for_training=True)
        start_step, episode_idx = DQNAgent.resume_counters(args.resume_from)
        if start_step == 0 and args.start_step is None:
            raise SystemExit(
                f"[{args.iteration}] el checkpoint {args.resume_from} no tiene env_step guardado "
                "(fue entrenado antes de que train.py lo registrara). Pasa --start-step explícitamente "
                "(p.ej. --start-step 3000000) para continuar el schedule de epsilon correctamente en vez "
                "de reiniciar la exploración desde epsilon_start."
            )
        if args.start_step is not None:
            start_step = args.start_step
        print(f"[{args.iteration}] reanudando desde {args.resume_from} (env_step={start_step}, episodio={episode_idx})")
    else:
        config = DQNConfig(
            n_actions=n_actions,
            n_frames=N_FRAME_STACK,
            architecture=args.architecture,
            double_dqn=args.double_dqn,
            gamma=args.gamma,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            seed=args.seed,
        )
        agent = DQNAgent(config)
        start_step, episode_idx = 0, 0

    buffer = ReplayBuffer(args.buffer_size, obs_shape, seed=args.seed)

    print(f"[{args.iteration}] device={agent.device} n_actions={n_actions} arch={agent.config.architecture} double_dqn={agent.config.double_dqn}")

    log_path = log_dir / f"{args.iteration}.csv"
    if not args.resume_from or not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "env_step", "reward_total", "length", "epsilon", "avg_loss", "elapsed_s"]
            )

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_losses = []
    start_time = time.time()

    # El buffer siempre arranca vacío al reanudar (no se persiste), así que el
    # warmup de `learning_starts` se cuenta desde el inicio de esta corrida,
    # no desde el env_step absoluto reanudado.
    for local_step, env_step in enumerate(range(start_step + 1, args.total_steps + 1), start=1):
        action = agent.select_action(obs, env_step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(np.asarray(obs), action, reward, np.asarray(next_obs), done)
        obs = next_obs
        episode_reward += reward
        episode_length += 1

        if local_step >= args.learning_starts and local_step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size)
            loss = agent.train_step(batch)
            episode_losses.append(loss)

        if done:
            episode_idx += 1
            avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
            elapsed = time.time() - start_time
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [episode_idx, env_step, episode_reward, episode_length,
                     agent.epsilon(env_step), avg_loss, round(elapsed, 1)]
                )
            if episode_idx % 10 == 0:
                print(
                    f"[{args.iteration}] ep={episode_idx} step={env_step}/{args.total_steps} "
                    f"reward={episode_reward:.1f} len={episode_length} eps={agent.epsilon(env_step):.3f} "
                    f"avg_loss={avg_loss:.4f} elapsed={elapsed/60:.1f}min"
                )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_losses = []

        if env_step % args.checkpoint_freq == 0:
            agent.save(str(ckpt_dir / f"{args.iteration}.pt"), env_step=env_step, episode_idx=episode_idx)

        if args.snapshot_freq > 0 and env_step % args.snapshot_freq == 0:
            agent.save(
                str(ckpt_dir / f"{args.iteration}_{env_step // 1000}k.pt"),
                env_step=env_step, episode_idx=episode_idx,
            )

    agent.save(str(ckpt_dir / f"{args.iteration}.pt"), env_step=args.total_steps, episode_idx=episode_idx)
    env.close()
    print(f"[{args.iteration}] entrenamiento finalizado. Checkpoint final: {ckpt_dir / f'{args.iteration}.pt'}")


if __name__ == "__main__":
    main()
