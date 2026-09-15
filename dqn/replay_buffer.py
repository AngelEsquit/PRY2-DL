"""Replay buffer de repetición de experiencias, con almacenamiento en
arreglos numpy pre-asignados (más eficiente en memoria que una deque de
tuplas para observaciones tipo imagen)."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Buffer circular de transiciones (s, a, r, s', done).

    Los estados se guardan como uint8 (4, 84, 84) para ahorrar memoria;
    la normalización a [0, 1] ocurre dentro de la red (ver dqn/model.py).
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._rng = np.random.default_rng(seed)

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def add(self, obs, action, reward, next_obs, done) -> None:
        idx = self._pos
        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self._pos += 1
        if self._pos == self.capacity:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int):
        """Muestrea un batch uniforme de transiciones.

        Returns:
            Tupla (obs, actions, rewards, next_obs, dones) de tensores
            numpy listos para convertir a torch.Tensor.
        """
        max_idx = len(self)
        indices = self._rng.integers(0, max_idx, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )
