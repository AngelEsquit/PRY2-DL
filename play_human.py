"""Juega ALE/SpaceInvaders-v5 tú mismo con el teclado (solo para uso personal,
no forma parte del proyecto/entrega).

Controles: A = izquierda, D = derecha, ESPACIO = disparar
           (D+ESPACIO / A+ESPACIO = moverte disparando)

Uso:
    python play_human.py
"""

from gymnasium.utils.play import play

from ale_utils import crear_entorno

env = crear_entorno("ALE/SpaceInvaders-v5", render_mode="rgb_array")

print("Controles: A = izquierda | D = derecha | ESPACIO = disparar")
play(env, fps=30, zoom=3)
