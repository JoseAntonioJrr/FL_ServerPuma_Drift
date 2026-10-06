#!/usr/bin/env python3
"""
generate_Cifar100_drift.py — Gerador de Datasets Temporais para Federated Concept Drift (CIFAR-100)

Gera partições estáticas pré e pós-drift:
- Phase 1 (Pre-Drift):  train/{idx}_phase1.npz, test/{idx}_phase1.npz
- Phase 2 (Post-Drift): train/{idx}_phase2.npz, test/{idx}_phase2.npz
Além de arquivos legados {idx}.npz (apontando para phase1).

Suporta:
- Real Concept Drift (label_swap, cyclic_shift)
- Dynamic Non-IID / Label Shift (class_shift)
- Virtual Drift / Covariate Shift (gaussian_noise)
"""

import os
import sys
import random
import copy
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import ujson

# Garante resolução de imports independentemente de onde o script é chamado
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src/
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from dataset.utils.dataset_utils import separate_data, split_data


def apply_pairwise_label_swap(y_array):
    """
    Swap semântico de 50 pares de classes em CIFAR-100:
    i <-> (i + 50) % 100 para i in [0, 49]
    """
    y_drift = np.copy(y_array)
    for i in range(50):
        pair_a = i
        pair_b = i + 50
        mask_a = (y_array == pair_a)
        mask_b = (y_array == pair_b)
        y_drift[mask_a] = pair_b
        y_drift[mask_b] = pair_a
    return y_drift


def apply_cyclic_shift(y_array, shift=1, num_classes=100):
    """Deslocamento circular dos rótulos em 100 classes: y' = (y + shift) % 100"""
    return (np.copy(y_array) + shift) % num_classes


def apply_gaussian_noise(x_array, std=0.25):
    """Adiciona ruído Gaussiano nas imagens normalizadas em [-1, 1]"""
    noise = np.random.normal(0, std, size=x_array.shape).astype(np.float32)
    x_noisy = np.clip(x_array + noise, -1.0, 1.0)
    return x_noisy


def save_temporal_files(config_path: Path, train_path: Path, test_path: Path,
                        train_p1, test_p1, train_p2, test_p2,
                        num_clients, num_classes, statistic_p1, statistic_p2,
                        niid, balance, partition, drift_type, drift_clients):
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    print("\n--- Salvando arquivos temporais em disco (CIFAR-100) ---")
    for idx in range(num_clients):
        # Phase 1
        with open(train_path / f"{idx}_phase1.npz", 'wb') as f:
            np.savez_compressed(f, data=train_p1[idx])
        with open(test_path / f"{idx}_phase1.npz", 'wb') as f:
            np.savez_compressed(f, data=test_p1[idx])

        # Phase 2
        with open(train_path / f"{idx}_phase2.npz", 'wb') as f:
            np.savez_compressed(f, data=train_p2[idx])
        with open(test_path / f"{idx}_phase2.npz", 'wb') as f:
            np.savez_compressed(f, data=test_p2[idx])

        # Arquivos padrão (retrocompatibilidade apontando para phase1)
        with open(train_path / f"{idx}.npz", 'wb') as f:
            np.savez_compressed(f, data=train_p1[idx])
        with open(test_path / f"{idx}.npz", 'wb') as f:
            np.savez_compressed(f, data=test_p1[idx])

        print(f"Cliente {idx}: salvos phase1 e phase2 ({'COM DRIFT' if idx in drift_clients else 'ESTÁVEL'})")

    config = {
        'dataset': 'Cifar100_drift',
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'drift_type': drift_type,
        'drift_clients': list(drift_clients),
        'statistic_phase1': statistic_p1,
        'statistic_phase2': statistic_p2
    }

    with open(config_path, 'w') as f:
        ujson.dump(config, f, indent=4)
    print(f"Metadados salvos em: {config_path}\n")


def generate_drift_dataset(dir_path: Path, num_clients=2, niid=True, balance=False,
                           partition="dir", drift_type="label_swap", drift_clients=(0,)):
    random.seed(1)
    np.random.seed(1)

    dir_path.mkdir(parents=True, exist_ok=True)
    config_path = dir_path / "config.json"
    train_path = dir_path / "train"
    test_path = dir_path / "test"

    raw_data_dir = current_dir / "Cifar100" / "rawdata"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("Baixando/Carregando CIFAR-100 original...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=str(raw_data_dir), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=str(raw_data_dir), train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    num_classes = len(set(dataset_label))

    print(f"Total de amostras: {len(dataset_image)}, Classes: {num_classes}")
    print(f"Configuração FL: {num_clients} clientes, Non-IID={niid}, Partition={partition}")
    print(f"Configuração Drift: Tipo={drift_type}, Clientes afetados={list(drift_clients)}")

    # 1. GERAÇÃO DA FASE 1 (Pre-Drift)
    print("\n[Phase 1] Separando dados originais...")
    X_p1, y_p1, stat_p1 = separate_data(
        (dataset_image, dataset_label), num_clients, num_classes,
        niid, balance, partition, class_per_client=10
    )
    train_p1, test_p1 = split_data(X_p1, y_p1)

    # 2. GERAÇÃO DA FASE 2 (Post-Drift)
    print("\n[Phase 2] Aplicando Concept Drift...")
    train_p2 = copy.deepcopy(train_p1)
    test_p2 = copy.deepcopy(test_p1)
    stat_p2 = copy.deepcopy(stat_p1)

    if drift_type == "class_shift":
        # Dynamic Non-IID: Nova amostragem Dirichlet com seed diferente
        np.random.seed(42)
        random.seed(42)
        X_p2, y_p2, stat_p2 = separate_data(
            (dataset_image, dataset_label), num_clients, num_classes,
            niid, balance, partition, class_per_client=10
        )
        train_p2_fresh, test_p2_fresh = split_data(X_p2, y_p2)
        for c in range(num_clients):
            if c in drift_clients:
                train_p2[c] = train_p2_fresh[c]
                test_p2[c] = test_p2_fresh[c]
    else:
        for c in range(num_clients):
            if c in drift_clients:
                print(f"Aplicando transformação '{drift_type}' no Cliente {c}...")
                if drift_type == "label_swap":
                    train_p2[c]['y'] = apply_pairwise_label_swap(train_p2[c]['y'])
                    test_p2[c]['y'] = apply_pairwise_label_swap(test_p2[c]['y'])
                elif drift_type == "cyclic_shift":
                    train_p2[c]['y'] = apply_cyclic_shift(train_p2[c]['y'], num_classes=100)
                    test_p2[c]['y'] = apply_cyclic_shift(test_p2[c]['y'], num_classes=100)
                elif drift_type == "gaussian_noise":
                    train_p2[c]['x'] = apply_gaussian_noise(train_p2[c]['x'])
                    test_p2[c]['x'] = apply_gaussian_noise(test_p2[c]['x'])
                else:
                    raise ValueError(f"drift_type desconhecido: {drift_type}")

                # Atualiza estatística do cliente
                stat_p2[c] = []
                for i in np.unique(train_p2[c]['y']):
                    stat_p2[c].append((int(i), int(np.sum(train_p2[c]['y'] == i))))

    save_temporal_files(
        config_path, train_path, test_path,
        train_p1, test_p1, train_p2, test_p2,
        num_clients, num_classes, stat_p1, stat_p2,
        niid, balance, partition, drift_type, drift_clients
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gerador de CIFAR-100 com Concept Drift")
    parser.add_argument("niid", nargs="?", default="noniid", help="'noniid' ou 'iid'")
    parser.add_argument("balance", nargs="?", default="-", help="'balance' ou '-'")
    parser.add_argument("partition", nargs="?", default="dir", help="'dir', 'pat' ou '-'")
    parser.add_argument("clients", nargs="?", type=int, default=2, help="Número de clientes (padrão: 2)")

    parser.add_argument("--drift-type", type=str, default="label_swap",
                        choices=["label_swap", "cyclic_shift", "class_shift", "gaussian_noise"],
                        help="Tipo de drift a injetar")
    parser.add_argument("--drift-clients", type=str, default="0",
                        help="IDs dos clientes que sofrem drift (ex: '0' ou '0,1')")
    parser.add_argument("--output-dir", type=str, default="Cifar100_drift",
                        help="Nome da pasta de saída dentro de src/dataset/")

    args, unknown = parser.parse_known_args()

    is_niid = True if args.niid == "noniid" else False
    is_balance = True if args.balance == "balance" else False
    partition_mode = args.partition if args.partition != "-" else None
    num_c = args.clients

    if args.drift_clients.strip().lower() == "all":
        drift_c_set = set(range(num_c))
    else:
        drift_c_set = set(int(x.strip()) for x in args.drift_clients.split(",") if x.strip().isdigit())
        if not drift_c_set:
            drift_c_set = {0}

    target_dir = current_dir / args.output_dir

    generate_drift_dataset(
        dir_path=target_dir,
        num_clients=num_c,
        niid=is_niid,
        balance=is_balance,
        partition=partition_mode,
        drift_type=args.drift_type,
        drift_clients=drift_c_set
    )
