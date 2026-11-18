#!/usr/bin/env python3
"""
generate_Cifar100_astra_drift.py — CIFAR-100 particionado como no ASTRA-FL + Concept Drift do PUMA.

Espelha exatamente o protocolo sem vazamento de `generate_Cifar10_astra_drift.py` para 100 classes:
- ASTRA (estático): particiona SOMENTE o TREINO OFICIAL do CIFAR-100 (50k, 100 classes)
  via Dirichlet(alpha) por classe, seed de partição fixa, sem balanceamento/tamanho mínimo.
- PUMA (temporal): phase2 (label_swap 50 pares i<->(i+50)%100 | cyclic_shift mod 100 |
  class_shift | gaussian_noise) aplicada apenas em `drift_clients`.
- Sem vazamento: testes por cliente e teste global (`global_phase1.npz`, `global_phase2.npz`)
  vêm SOMENTE do TESTE OFICIAL (10k); treinos (inclusive redraw do `class_shift`) vêm SOMENTE
  do TREINO OFICIAL (50k).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src/
for _p in (parent_dir, current_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset.generate_Cifar10_astra_drift import (  # noqa: E402
    DRIFT_TRANSFORMS,
    GENERATOR_VERSION,
    build_astra_drift_dataset,
    dataset_is_current,
    default_output_name as _base_default_output_name,
    save_dataset,
)
from utils.drift_schedule import parse_drift_clients  # noqa: E402


def default_output_name(alpha, partition_seed, drift_transform, drift_clients, num_clients):
    return _base_default_output_name(
        alpha, partition_seed, drift_transform, drift_clients, num_clients, prefix="Cifar100"
    )


def load_cifar100_normalized(raw_dir):
    """Carrega CIFAR-100 oficial (50k treino + 10k teste) normalizado em [-1, 1] como float32 [N,3,32,32]."""
    import torch
    import torchvision

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR100(root=str(raw_dir), train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=str(raw_dir), train=False, download=True)

    def _norm(data):
        t = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().float().div(255.0)
        return t.sub_(0.5).div_(0.5).numpy()

    return (
        _norm(trainset.data), np.asarray(trainset.targets, dtype=np.int64),
        _norm(testset.data), np.asarray(testset.targets, dtype=np.int64),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-100 com partição ASTRA-FL + Concept Drift PUMA")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--drift-transform", default="none", choices=DRIFT_TRANSFORMS)
    parser.add_argument("--drift-clients", default="none",
                        help="IDs PUMA base 0: 'all', '0,1', '0-4' ou 'none'")
    parser.add_argument("--drift-seed", type=int, default=1234)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--test-ratio", type=float, default=1.0 / 3.0,
                        help="tamanho do teste por cliente relativo ao treino (0.25/0.75 = 1/3)")
    parser.add_argument("--min-test", type=int, default=10)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--no-hardlink", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--output-dir", default="",
                        help="nome da pasta em src/dataset/ (padrão: automático por configuração)")
    parser.add_argument("--raw-dir", default=str(current_dir / "Cifar100" / "rawdata"))
    parser.add_argument("--print-name", action="store_true",
                        help="só imprime o nome do diretório de saída e sai")
    args = parser.parse_args(argv)

    drift_clients = parse_drift_clients(args.drift_clients, args.num_clients)
    if args.drift_transform == "none":
        drift_clients = set()
    elif not drift_clients:
        raise SystemExit("--drift-transform exige --drift-clients")

    name = args.output_dir or default_output_name(
        args.dirichlet_alpha, args.partition_seed, args.drift_transform, drift_clients, args.num_clients
    )
    if args.print_name:
        print(name)
        return name
    out_dir = current_dir / name

    params = {
        "num_clients": args.num_clients,
        "num_classes": 100,
        "alpha": args.dirichlet_alpha,
        "partition_seed": args.partition_seed,
        "drift_transform": args.drift_transform,
        "drift_clients": sorted(drift_clients),
        "drift_seed": args.drift_seed,
        "noise_std": args.noise_std,
        "test_ratio": args.test_ratio,
        "min_test": args.min_test,
        "generator_version": GENERATOR_VERSION,
    }

    if args.skip_if_exists and dataset_is_current(out_dir, params):
        print(f"[INFO] Dataset já gerado e atualizado: {out_dir}")
        return name

    print(f"[ASTRA CIFAR-100] {args.num_clients} clientes, alpha={args.dirichlet_alpha}, seed={args.partition_seed}")
    if args.drift_transform != "none":
        print(f"[DRIFT] transform={args.drift_transform}, clientes(base 0)={sorted(drift_clients)}")

    X_train, y_train, X_test, y_test = load_cifar100_normalized(args.raw_dir)
    result = build_astra_drift_dataset(
        X_train, y_train, X_test, y_test,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        partition_seed=args.partition_seed,
        drift_transform=args.drift_transform,
        drift_clients=drift_clients,
        drift_seed=args.drift_seed,
        noise_std=args.noise_std,
        test_ratio=args.test_ratio,
        min_test=args.min_test,
        num_classes=100,
        allow_empty=args.allow_empty,
    )
    save_dataset(
        out_dir,
        result,
        params,
        hardlink=not args.no_hardlink,
        generator_name="generate_Cifar100_astra_drift.py",
        train_source="cifar100_official_train",
        test_source="cifar100_official_test",
    )
    print(f"[OK] Dataset salvo em {out_dir}")
    return name


if __name__ == "__main__":
    main()
