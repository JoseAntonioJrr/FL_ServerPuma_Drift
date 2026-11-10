"""Particionamento Non-IID estático idêntico ao ASTRA-FL.

Reproduz exatamente ``scripts/setup/prepare_data.py`` do ASTRA-FL
(https://github.com/IcsPingu/ASTRA-FL): para cada classe ``k`` embaralha os
índices, sorteia ``Dirichlet(alpha * 1_N)`` e faz ``np.split`` nas proporções
acumuladas — sem balanceamento e sem tamanho mínimo (diferente do Dirichlet do
PFLlib em ``dataset_utils.separate_data``).

Convenções de ID:
- ASTRA: chaves ``"1".."N"`` (``client_001`` -> ``"1"``), base 1.
- PUMA:  ``client_idx`` ``0..N-1``, base 0.
Regra única: ``puma_idx = astra_id - 1``.
"""

import numpy as np


def astra_dirichlet_partition(labels, num_clients=20, alpha=0.1, seed=42, num_classes=10):
    """Retorna ``{astra_id (1..N): [índices]}`` com a mesma sequência de RNG do ASTRA.

    Usa o RNG global legado do numpy (``np.random.seed``), exatamente como o
    ASTRA, para que a partição seja bit-a-bit idêntica dada a mesma ordem de
    ``labels`` (``torchvision.datasets.CIFAR10(train=True).targets``).
    """
    if num_clients < 1:
        raise ValueError("num_clients deve ser >= 1")
    if alpha <= 0:
        raise ValueError("alpha deve ser > 0")

    np.random.seed(seed)
    labels = np.asarray(labels)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_partitions = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        img_idx_k = class_indices[k]
        np.random.shuffle(img_idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions_cumsum = (np.cumsum(proportions) * len(img_idx_k)).astype(int)[:-1]
        split_indices = np.split(img_idx_k, proportions_cumsum)
        for i in range(num_clients):
            client_partitions[i].extend(split_indices[i].tolist())

    return {i + 1: indices for i, indices in enumerate(client_partitions)}


def astra_to_puma_ids(partition):
    """Converte ``{astra_id: idx}`` (chaves int ou str, base 1) para ``{puma_idx: idx}`` (base 0).

    Falha se as chaves não forem exatamente ``1..N``.
    """
    keys = sorted(int(k) for k in partition.keys())
    n = len(keys)
    if keys != list(range(1, n + 1)):
        raise ValueError(f"IDs ASTRA devem ser contínuos 1..{n}; recebido {keys}")
    return {int(k) - 1: list(v) for k, v in partition.items()}


def puma_to_astra_id(puma_idx):
    return int(puma_idx) + 1


def astra_to_puma_id(astra_id):
    return int(str(astra_id).split('_')[-1]) - 1

