"""Agente DQN / Double DQN con red objetivo (target network) y
exploración epsilon-greedy con decaimiento lineal.

La diferencia entre DQN y Double DQN está aislada en `_compute_targets`:
DQN usa la red objetivo tanto para seleccionar como para evaluar la mejor
acción del siguiente estado; Double DQN usa la red online para seleccionar
la acción y la red objetivo solo para evaluarla, lo que reduce la
sobreestimación de los valores Q reportada por Van Hasselt et al. (2016).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from dqn.model import build_network


@dataclass
class DQNConfig:
    n_actions: int
    n_frames: int = 4
    architecture: str = "dqn"  # "dqn" o "dueling"
    double_dqn: bool = True
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    batch_size: int = 32
    target_update_freq: int = 10_000  # en pasos de entrenamiento (gradiente)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int | None = None


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)

        self.online_net = build_network(config.architecture, config.n_actions, config.n_frames).to(self.device)
        self.target_net = build_network(config.architecture, config.n_actions, config.n_frames).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: menos sensible a outliers que MSE.

        self._train_steps = 0

    def epsilon(self, env_step: int) -> float:
        """Epsilon con decaimiento lineal de epsilon_start a epsilon_end a
        lo largo de epsilon_decay_steps pasos de entorno."""
        cfg = self.config
        fraction = min(1.0, env_step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, obs: np.ndarray, env_step: int, greedy: bool = False) -> int:
        """Selecciona una acción con política epsilon-greedy.

        Args:
            obs: observación apilada (n_frames, 84, 84), uint8.
            env_step: paso global de entorno, usado para calcular epsilon.
            greedy: si True, ignora epsilon y siempre actúa de forma
                greedy (usado en evaluación).
        """
        eps = 0.0 if greedy else self.epsilon(env_step)
        if random.random() < eps:
            return random.randrange(self.config.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
            q_values = self.online_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def _compute_targets(self, rewards, next_obs, dones) -> torch.Tensor:
        cfg = self.config
        with torch.no_grad():
            if cfg.double_dqn:
                next_actions = self.online_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            targets = rewards + cfg.gamma * next_q * (1.0 - dones)
        return targets

    def train_step(self, batch) -> float:
        """Un paso de descenso de gradiente sobre un batch muestreado del
        replay buffer. Retorna el valor escalar de la pérdida."""
        obs, actions, rewards, next_obs, dones = batch

        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self._compute_targets(rewards, next_obs, dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.config.target_update_freq == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAgent":
        """Carga un agente entrenado desde un checkpoint guardado con
        `save`. Reconstruye la arquitectura a partir de la configuración
        guardada, para asegurar consistencia con el entrenamiento."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        config: DQNConfig = checkpoint["config"]
        if device is not None:
            config.device = device
        agent = cls(config)
        agent.online_net.load_state_dict(checkpoint["online_state_dict"])
        agent.target_net.load_state_dict(checkpoint["online_state_dict"])
        agent.online_net.eval()
        return agent

    def policy_fn(self):
        """Retorna una función (observation, env) -> action compatible con
        `ejecutar_episodio` / `generar_video_agente` de ale_utils.py,
        actuando siempre de forma greedy (sin exploración)."""

        def _policy(observation, env):
            return self.select_action(observation, env_step=0, greedy=True)

        return _policy
