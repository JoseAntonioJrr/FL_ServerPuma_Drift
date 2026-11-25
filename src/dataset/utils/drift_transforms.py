"""Transformações de Concept Drift compartilhadas pelos geradores *_drift.

Extraídas de ``generate_Cifar10_drift.py`` sem mudança de comportamento.
``apply_gaussian_noise`` aceita um ``rng`` opcional; sem ele usa o RNG global
do numpy (comportamento legado).
"""

import numpy as np

# Swap semântico de pares em CIFAR-10:
# 0 (airplane) <-> 2 (bird), 1 (automobile) <-> 9 (truck), 3 (cat) <-> 5 (dog),
# 4 (deer) <-> 7 (horse), 6 (frog) <-> 8 (ship)
PAIRWISE_SWAP = {
    0: 2, 2: 0,
    1: 9, 9: 1,
    3: 5, 5: 3,
    4: 7, 7: 4,
    6: 8, 8: 6,
}


def apply_pairwise_label_swap(y_array, num_classes=10):
    """Swap semântico de pares:
    - CIFAR-10 (num_classes=10): pares semânticos definidos em ``PAIRWISE_SWAP``.
    - CIFAR-100 (num_classes=100) ou outro número par de classes: i <-> (i + K//2) % K.
    """
    y_array = np.asarray(y_array)
    if num_classes == 10:
        y_drift = np.copy(y_array)
        for old_val, new_val in PAIRWISE_SWAP.items():
            y_drift[y_array == old_val] = new_val
        return y_drift
    if num_classes % 2 != 0:
        raise ValueError(f"apply_pairwise_label_swap requer num_classes par; recebido {num_classes}")
    half = num_classes // 2
    return (np.copy(y_array) + half) % num_classes


def apply_cyclic_shift(y_array, shift=1, num_classes=10):
    """Deslocamento circular dos rótulos: y' = (y + shift) % num_classes"""
    return (np.copy(y_array) + shift) % num_classes


def apply_gaussian_noise(x_array, std=0.25, rng=None):
    """Adiciona ruído Gaussiano nas imagens normalizadas em [-1, 1]"""
    if rng is None:
        noise = np.random.normal(0, std, size=x_array.shape).astype(np.float32)
    else:
        noise = rng.normal(0, std, size=x_array.shape).astype(np.float32)
    x_noisy = np.clip(x_array + noise, -1.0, 1.0)
    return x_noisy
