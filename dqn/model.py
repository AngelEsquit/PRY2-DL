"""Arquitecturas de red para aproximar Q(s, a) sobre observaciones
preprocesadas de ALE/SpaceInvaders-v5 (4, 84, 84).

Incluye la arquitectura convolucional clásica de Mnih et al. (2015)
("Nature DQN") y una variante Dueling DQN (Wang et al., 2016) que separa
la estimación de V(s) y la ventaja A(s, a).
"""

from __future__ import annotations

import torch
from torch import nn


class NatureCNN(nn.Module):
    """Tronco convolucional compartido por DQN / Double DQN / Dueling DQN.

    Espera entradas (batch, n_frames, 84, 84) en el rango [0, 255] (uint8
    o float) y las normaliza internamente a [0, 1].
    """

    def __init__(self, n_frames: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Para entrada 84x84: 8/4 -> 20x20, 4/2 -> 9x9, 3/1 -> 7x7.
        self.output_dim = 64 * 7 * 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        return self.conv(x)


class QNetwork(nn.Module):
    """Red Q estándar: tronco convolucional + MLP -> Q(s, a) para cada
    acción discreta. Usada por DQN y Double DQN (la diferencia entre ambos
    está en cómo se calcula el target, no en la arquitectura)."""

    def __init__(self, n_actions: int, n_frames: int = 4):
        super().__init__()
        self.encoder = NatureCNN(n_frames)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class DuelingQNetwork(nn.Module):
    """Red Dueling DQN: separa V(s) y A(s, a), combinados como

        Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))

    para que la red pueda aprender el valor de un estado sin necesidad de
    aprender el efecto de cada acción por separado (útil cuando muchas
    acciones tienen impacto similar en el corto plazo).
    """

    def __init__(self, n_actions: int, n_frames: int = 4):
        super().__init__()
        self.encoder = NatureCNN(n_frames)
        self.value_stream = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def build_network(architecture: str, n_actions: int, n_frames: int = 4) -> nn.Module:
    """Fábrica simple para elegir arquitectura desde configuración/CLI."""
    architecture = architecture.lower()
    if architecture == "dqn":
        return QNetwork(n_actions, n_frames)
    if architecture == "dueling":
        return DuelingQNetwork(n_actions, n_frames)
    raise ValueError(f"Arquitectura desconocida: {architecture!r} (usar 'dqn' o 'dueling')")
