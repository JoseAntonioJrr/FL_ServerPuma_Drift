"""Testes sintéticos para o gerador ASTRA-FL + Concept Drift do CIFAR-100."""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, SRC / "dataset"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset.generate_Cifar10_astra_drift import build_astra_drift_dataset, save_dataset
from dataset.generate_Cifar100_astra_drift import default_output_name
from dataset.utils.drift_transforms import apply_pairwise_label_swap


def _synthetic_cifar100(n_train_per_class=20, n_test_per_class=6):
    rng = np.random.default_rng(7)
    y_tr = np.repeat(np.arange(100, dtype=np.int64), n_train_per_class)
    y_te = np.repeat(np.arange(100, dtype=np.int64), n_test_per_class)
    X_tr = rng.uniform(-1.0, 1.0, size=(len(y_tr), 3, 4, 4)).astype(np.float32)
    X_te = rng.uniform(-1.0, 1.0, size=(len(y_te), 3, 4, 4)).astype(np.float32)
    # Marca distinta no pixel [0,0,0] para provar zero mistura entre treino e teste
    X_tr[:, 0, 0, 0] = 0.25
    X_te[:, 0, 0, 0] = -0.75
    return X_tr, y_tr, X_te, y_te


def test_cifar100_pairwise_label_swap_involution():
    y = np.arange(100, dtype=np.int64)
    y_swapped = apply_pairwise_label_swap(y, num_classes=100)
    assert np.array_equal(y_swapped[:50], np.arange(50, 100))
    assert np.array_equal(y_swapped[50:], np.arange(0, 50))
    assert np.array_equal(apply_pairwise_label_swap(y_swapped, num_classes=100), y)


def test_cifar100_default_output_name():
    name = default_output_name(0.1, 42, "label_swap", set(range(20)), 20)
    assert name == "Cifar100_astra_a0.1_s42_label_swap_dcall"
    name_none = default_output_name(0.5, 42, "none", set(), 20)
    assert name_none == "Cifar100_astra_a0.5_s42"


@pytest.mark.parametrize("transform", ["label_swap", "cyclic_shift", "gaussian_noise", "class_shift"])
def test_cifar100_astra_zero_leakage_and_save(tmp_path, transform):
    X_tr, y_tr, X_te, y_te = _synthetic_cifar100()
    res = build_astra_drift_dataset(
        X_tr, y_tr, X_te, y_te,
        num_clients=10,
        alpha=0.1,
        partition_seed=42,
        drift_transform=transform,
        drift_clients=list(range(10)),
        drift_seed=1234,
        num_classes=100,
        allow_empty=True,
        log=lambda *a, **k: None,
    )

    # Todos os índices de treino da fase 1 são disjuntos e cobrem todo o treino oficial
    all_tr_idx = np.concatenate([res["clients"][c]["train_idx_p1"] for c in range(10)])
    assert len(all_tr_idx) == len(y_tr)
    assert len(np.unique(all_tr_idx)) == len(y_tr)

    # Zero leakage: nenhum dado de teste entra no treino (e vice-versa), em nenhuma fase
    for c in range(10):
        cl = res["clients"][c]
        if transform != "gaussian_noise":
            assert np.allclose(cl["train_p1"]["x"][:, 0, 0, 0], 0.25)
            assert np.allclose(cl["train_p2"]["x"][:, 0, 0, 0], 0.25)
            assert np.allclose(cl["test_p1"]["x"][:, 0, 0, 0], -0.75)
            assert np.allclose(cl["test_p2"]["x"][:, 0, 0, 0], -0.75)

    if transform == "label_swap":
        assert np.array_equal(res["global_p2"]["y"], (y_te + 50) % 100)
    elif transform == "cyclic_shift":
        assert np.array_equal(res["global_p2"]["y"], (y_te + 1) % 100)

    params = {
        "num_clients": 10, "num_classes": 100, "alpha": 0.1,
        "partition_seed": 42, "drift_transform": transform,
        "drift_clients": list(range(10)), "drift_seed": 1234,
        "noise_std": 0.25, "test_ratio": 1.0 / 3.0, "min_test": 10,
        "generator_version": 1,
    }
    cfg = save_dataset(
        tmp_path, res, params,
        log=lambda *a, **k: None,
        generator_name="generate_Cifar100_astra_drift.py",
        train_source="cifar100_official_train",
        test_source="cifar100_official_test",
    )
    assert cfg["num_classes"] == 100
    assert cfg["strict_phase"] is True
    assert cfg["indices"]["train_source"] == "cifar100_official_train"
    assert cfg["indices"]["test_source"] == "cifar100_official_test"
    assert (tmp_path / "test" / "global_phase1.npz").exists()
    assert (tmp_path / "test" / "global_phase2.npz").exists()
