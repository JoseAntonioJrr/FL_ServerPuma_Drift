#!/usr/bin/env python3
"""
generate_Cifar10_astra_drift.py — CIFAR-10 particionado como no ASTRA-FL + Concept Drift do PUMA.

Separação de responsabilidades:
- ASTRA (estático): quais amostras do TREINO OFICIAL (50k) pertencem a cada um dos N clientes
  (Dirichlet(alpha) por classe, seed de partição fixa, sem balanceamento/tamanho mínimo).
  Isso define a phase1/train de cada cliente (partição inteira, sem split 75/25).
- PUMA (temporal): o conteúdo da phase2 (label_swap | cyclic_shift | class_shift | gaussian_noise)
  aplicado apenas em ``drift_clients``; o *quando* continua no client.py/server.py
  (abrupt / gradual / periódico).

Sem vazamento: testes (por cliente e global) vêm SOMENTE do TESTE OFICIAL (10k); treinos (inclusive
o redraw do class_shift) vêm SOMENTE do TREINO OFICIAL.

Formato de saída (contrato do PUMA, inalterado):
  <out>/train/{idx}_phase1.npz, {idx}_phase2.npz, {idx}.npz (legado = phase1)
  <out>/test/{idx}_phase1.npz,  {idx}_phase2.npz, {idx}.npz
  <out>/test/global_phase1.npz (teste oficial), global_phase2.npz (teste oficial transformado)
  <out>/config.json, <out>/partitions.json (formato ASTRA, chaves "1".."N")
  cada .npz: np.savez_compressed(f, data={'x': float32[N,3,32,32] em [-1,1], 'y': int64[N]})
IDs: puma_idx = astra_id - 1.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src/
for _p in (parent_dir, current_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset.utils.astra_partition import astra_dirichlet_partition  # noqa: E402
from dataset.utils.drift_transforms import (  # noqa: E402
    apply_pairwise_label_swap,
    apply_cyclic_shift,
    apply_gaussian_noise,
)
from utils.drift_schedule import parse_drift_clients  # noqa: E402

DRIFT_TRANSFORMS = ("none", "label_swap", "cyclic_shift", "class_shift", "gaussian_noise")
GENERATOR_VERSION = 1

# Namespaces de RNG (independentes do RNG global usado pela partição ASTRA)
_NS_CLIENT_TEST = 0
_NS_NOISE = 1
_NS_CLASS_SHIFT_TEST = 2
_GLOBAL_KEY = 10_000


# ---------------------------------------------------------------------------
# Núcleo puro (testável com dados sintéticos)
# ---------------------------------------------------------------------------


def format_drift_clients(clients, num_clients):
    clients = sorted(clients)
    if not clients:
        return "none"
    if clients == list(range(num_clients)):
        return "all"
    parts, start, prev = [], clients[0], clients[0]
    for c in clients[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        if c is not None:
            start = prev = c
    return "_".join(parts)


def default_output_name(alpha, partition_seed, drift_transform, drift_clients, num_clients):
    a = f"{alpha:g}"
    if drift_transform == "none":
        return f"Cifar10_astra_a{a}_s{partition_seed}"
    return (f"Cifar10_astra_a{a}_s{partition_seed}_{drift_transform}"
            f"_dc{format_drift_clients(drift_clients, num_clients)}")


def _label_stats(y):
    return [(int(c), int(np.sum(y == c))) for c in np.unique(y)]


def sample_matching_test(test_labels, train_labels, rng, test_ratio, min_test, num_classes=10):
    """Amostra índices do TESTE OFICIAL replicando a distribuição de rótulos do treino do cliente.

    Tamanho alvo = max(min_test, round(n_train * test_ratio)); alocação por maior resto;
    sem reposição dentro do cliente; limitado pela disponibilidade por classe.
    """
    test_labels = np.asarray(test_labels)
    train_labels = np.asarray(train_labels)
    n_train = len(train_labels)
    n_test = max(int(min_test), int(round(n_train * test_ratio)))

    if n_train == 0:
        props = np.full(num_classes, 1.0 / num_classes)
    else:
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        props = counts / counts.sum()

    raw = props * n_test
    alloc = np.floor(raw).astype(int)
    remainder = n_test - alloc.sum()
    if remainder > 0:
        order = np.argsort(-(raw - alloc), kind="stable")
        for k in order[:remainder]:
            alloc[k] += 1

    chosen = []
    for k in range(num_classes):
        if alloc[k] == 0:
            continue
        pool = np.where(test_labels == k)[0]
        take = min(alloc[k], len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    chosen = np.array(sorted(chosen), dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


def build_astra_drift_dataset(X_train, y_train, X_test, y_test, num_clients=20, alpha=0.1,
                              partition_seed=42, drift_transform="none", drift_clients=(),
                              drift_seed=1234, noise_std=0.25, test_ratio=1.0 / 3.0,
                              min_test=10, num_classes=10, allow_empty=False, log=print):
    """Constrói em memória todas as fases. Retorna dict com dados, índices e metadados.

    X_* já normalizados (float32, [N,3,32,32] em [-1,1]); y_* int64.
    """
    if drift_transform not in DRIFT_TRANSFORMS:
        raise ValueError(f"drift_transform desconhecido: {drift_transform}")
    drift_clients = set(int(c) for c in drift_clients)
    if drift_transform == "none":
        drift_clients = set()

    y_train = np.asarray(y_train).astype(np.int64)
    y_test = np.asarray(y_test).astype(np.int64)

    # 1. Partição ASTRA (treino oficial apenas)
    astra_part = astra_dirichlet_partition(y_train, num_clients, alpha, partition_seed, num_classes)
    train_idx_p1 = {aid - 1: np.asarray(idx, dtype=np.int64) for aid, idx in astra_part.items()}

    sizes = [len(train_idx_p1[c]) for c in range(num_clients)]
    empty = [c for c, n in enumerate(sizes) if n == 0]
    small = [c for c, n in enumerate(sizes) if 0 < n < 32]
    if small:
        log(f"[AVISO] Clientes com < 32 amostras de treino (partição ASTRA, sem reamostragem): {small}")
    if empty:
        msg = f"Clientes com 0 amostras de treino na partição ASTRA: {empty}"
        if not allow_empty:
            raise ValueError(msg + " (use --allow-empty para prosseguir)")
        log(f"[AVISO] {msg}")

    # 2. Teste por cliente (phase1) a partir do teste oficial
    test_idx_p1 = {}
    for c in range(num_clients):
        rng = np.random.default_rng([drift_seed, _NS_CLIENT_TEST, c])
        test_idx_p1[c] = sample_matching_test(y_test, y_train[train_idx_p1[c]], rng,
                                              test_ratio, min_test, num_classes)

    clients = {}
    for c in range(num_clients):
        clients[c] = {
            "train_p1": {"x": X_train[train_idx_p1[c]], "y": y_train[train_idx_p1[c]]},
            "test_p1": {"x": X_test[test_idx_p1[c]], "y": y_test[test_idx_p1[c]]},
            "train_idx_p1": train_idx_p1[c],
            "test_idx_p1": test_idx_p1[c],
            "phase2_identical": True,
        }

    # 3. Phase 2
    if drift_transform == "class_shift" and drift_clients:
        redraw = astra_dirichlet_partition(y_train, num_clients, alpha, drift_seed, num_classes)
        for c in sorted(drift_clients):
            idx2 = np.asarray(redraw[c + 1], dtype=np.int64)
            if len(idx2) == 0:
                log(f"[AVISO] class_shift: cliente {c} ficou com 0 amostras no redraw")
                if not allow_empty:
                    raise ValueError(f"class_shift deixou o cliente {c} vazio (use --allow-empty)")
            rng = np.random.default_rng([drift_seed, _NS_CLASS_SHIFT_TEST, c])
            tidx2 = sample_matching_test(y_test, y_train[idx2], rng, test_ratio, min_test, num_classes)
            if len(idx2) < len(train_idx_p1[c]):
                log(f"[AVISO] class_shift: cliente {c} phase2 ({len(idx2)}) < phase1 "
                    f"({len(train_idx_p1[c])}); o blending gradual usará menos amostras")
            clients[c].update({
                "train_p2": {"x": X_train[idx2], "y": y_train[idx2]},
                "test_p2": {"x": X_test[tidx2], "y": y_test[tidx2]},
                "train_idx_p2": idx2, "test_idx_p2": tidx2, "phase2_identical": False,
            })
    else:
        for c in sorted(drift_clients):
            tr, te = clients[c]["train_p1"], clients[c]["test_p1"]
            if drift_transform == "label_swap":
                tr2 = {"x": tr["x"], "y": apply_pairwise_label_swap(tr["y"])}
                te2 = {"x": te["x"], "y": apply_pairwise_label_swap(te["y"])}
            elif drift_transform == "cyclic_shift":
                tr2 = {"x": tr["x"], "y": apply_cyclic_shift(tr["y"], 1, num_classes)}
                te2 = {"x": te["x"], "y": apply_cyclic_shift(te["y"], 1, num_classes)}
            elif drift_transform == "gaussian_noise":
                r_tr = np.random.default_rng([drift_seed, _NS_NOISE, c, 0])
                r_te = np.random.default_rng([drift_seed, _NS_NOISE, c, 1])
                tr2 = {"x": apply_gaussian_noise(tr["x"], noise_std, rng=r_tr), "y": tr["y"]}
                te2 = {"x": apply_gaussian_noise(te["x"], noise_std, rng=r_te), "y": te["y"]}
            else:  # pragma: no cover
                raise ValueError(drift_transform)
            clients[c].update({"train_p2": tr2, "test_p2": te2, "phase2_identical": False,
                               "train_idx_p2": clients[c]["train_idx_p1"],
                               "test_idx_p2": clients[c]["test_idx_p1"]})

    for c in range(num_clients):
        if clients[c]["phase2_identical"]:
            clients[c].update({"train_p2": clients[c]["train_p1"], "test_p2": clients[c]["test_p1"],
                               "train_idx_p2": clients[c]["train_idx_p1"],
                               "test_idx_p2": clients[c]["test_idx_p1"]})

    # 4. Teste global (teste oficial completo)
    global_p1 = {"x": X_test, "y": y_test}
    global_identical = True
    if drift_clients and drift_transform == "label_swap":
        global_p2, global_identical = {"x": X_test, "y": apply_pairwise_label_swap(y_test)}, False
    elif drift_clients and drift_transform == "cyclic_shift":
        global_p2, global_identical = {"x": X_test, "y": apply_cyclic_shift(y_test, 1, num_classes)}, False
    elif drift_clients and drift_transform == "gaussian_noise":
        r = np.random.default_rng([drift_seed, _NS_NOISE, _GLOBAL_KEY])
        global_p2, global_identical = {"x": apply_gaussian_noise(X_test, noise_std, rng=r), "y": y_test}, False
    else:  # none / class_shift: P(y|x) inalterado
        global_p2 = global_p1

    return {
        "clients": clients,
        "global_p1": global_p1,
        "global_p2": global_p2,
        "global_identical": global_identical,
        "astra_partition": {str(k): [int(i) for i in v] for k, v in astra_part.items()},
        "drift_clients": sorted(drift_clients),
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def _save_npz(path, data):
    with open(path, "wb") as f:
        np.savez_compressed(f, data={"x": np.asarray(data["x"], dtype=np.float32),
                                     "y": np.asarray(data["y"], dtype=np.int64)})


def _link_or_copy(src, dst, hardlink=True):
    if os.path.lexists(dst):
        os.remove(dst)
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def params_signature(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def save_dataset(out_dir, result, params, hardlink=True, log=print):
    out_dir = Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    num_clients = len(result["clients"])

    for c in range(num_clients):
        cl = result["clients"][c]
        for split, d in (("train", train_dir), ("test", test_dir)):
            p1 = d / f"{c}_phase1.npz"
            _save_npz(p1, cl[f"{split}_p1"])
            if cl["phase2_identical"]:
                _link_or_copy(p1, d / f"{c}_phase2.npz", hardlink)
            else:
                _save_npz(d / f"{c}_phase2.npz", cl[f"{split}_p2"])
            _link_or_copy(p1, d / f"{c}.npz", hardlink)  # legado = phase1
        log(f"Cliente {c} (ASTRA client_{c + 1:03d}): train={len(cl['train_p1']['y'])} "
            f"test={len(cl['test_p1']['y'])} {'COM DRIFT' if not cl['phase2_identical'] else 'ESTÁVEL'}")

    g1 = test_dir / "global_phase1.npz"
    _save_npz(g1, result["global_p1"])
    if result["global_identical"]:
        _link_or_copy(g1, test_dir / "global_phase2.npz", hardlink)
    else:
        _save_npz(test_dir / "global_phase2.npz", result["global_p2"])

    with open(out_dir / "partitions.json", "w") as f:
        json.dump(result["astra_partition"], f)

    files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.npz"))
    manifest = {f: _sha256(out_dir / f) for f in files}
    manifest["partitions.json"] = _sha256(out_dir / "partitions.json")

    versions = {"numpy": np.__version__}
    try:
        import torch
        import torchvision
        versions.update({"torch": torch.__version__, "torchvision": torchvision.__version__})
    except ImportError:
        pass

    cl = result["clients"]
    config = {
        "dataset": out_dir.name,
        "generator": "generate_Cifar10_astra_drift.py",
        "generator_version": GENERATOR_VERSION,
        "params": params,
        "params_signature": params_signature(params),
        "partition_source": "astra",
        "num_clients": num_clients,
        "num_classes": params["num_classes"],
        "non_iid": True,
        "balance": False,
        "partition": "astra_dir",
        "alpha": params["alpha"],
        "drift_type": params["drift_transform"],
        "drift_clients": result["drift_clients"],
        "strict_phase": True,
        "client_test_source": "official_test",
        "id_map": {str(c): str(c + 1) for c in range(num_clients)},
        "num_samples": {
            "train_phase1": [int(len(cl[c]["train_p1"]["y"])) for c in range(num_clients)],
            "test_phase1": [int(len(cl[c]["test_p1"]["y"])) for c in range(num_clients)],
            "train_phase2": [int(len(cl[c]["train_p2"]["y"])) for c in range(num_clients)],
            "test_phase2": [int(len(cl[c]["test_p2"]["y"])) for c in range(num_clients)],
        },
        "statistic_phase1": [_label_stats(cl[c]["train_p1"]["y"]) for c in range(num_clients)],
        "statistic_phase2": [_label_stats(cl[c]["train_p2"]["y"]) for c in range(num_clients)],
        "indices": {
            "train_phase1": {str(c): cl[c]["train_idx_p1"].tolist() for c in range(num_clients)},
            "train_phase2": {str(c): cl[c]["train_idx_p2"].tolist() for c in range(num_clients)},
            "test_phase1": {str(c): cl[c]["test_idx_p1"].tolist() for c in range(num_clients)},
            "test_phase2": {str(c): cl[c]["test_idx_p2"].tolist() for c in range(num_clients)},
            "train_source": "cifar10_official_train",
            "test_source": "cifar10_official_test",
        },
        "versions": versions,
        "manifest_sha256": manifest,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"Metadados salvos em: {out_dir / 'config.json'}")
    return config


def dataset_is_current(out_dir, params):
    cfg = Path(out_dir) / "config.json"
    if not cfg.exists():
        return False
    try:
        with open(cfg) as f:
            config = json.load(f)
    except ValueError:
        return False
    if config.get("params_signature") != params_signature(params):
        return False
    return all((Path(out_dir) / f).exists() for f in config.get("manifest_sha256", {}))


# ---------------------------------------------------------------------------
# Carregamento do CIFAR-10 oficial
# ---------------------------------------------------------------------------

def load_cifar10_normalized(raw_dir):
    """Retorna (X_train, y_train, X_test, y_test) com ToTensor + Normalize(0.5, 0.5) (igual ASTRA/PUMA)."""
    import torch
    import torchvision

    trainset = torchvision.datasets.CIFAR10(root=str(raw_dir), train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=str(raw_dir), train=False, download=True)

    def _norm(data):
        t = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().float().div(255.0)
        return t.sub_(0.5).div_(0.5).numpy()

    return (_norm(trainset.data), np.asarray(trainset.targets, dtype=np.int64),
            _norm(testset.data), np.asarray(testset.targets, dtype=np.int64))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 com partição ASTRA-FL + Concept Drift PUMA")
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
    parser.add_argument("--raw-dir", default=str(current_dir / "Cifar10" / "rawdata"))
    parser.add_argument("--print-name", action="store_true",
                        help="só imprime o nome do diretório de saída e sai")
    args = parser.parse_args(argv)

    drift_clients = parse_drift_clients(args.drift_clients, args.num_clients)
    if args.drift_transform == "none":
        drift_clients = set()
    elif not drift_clients:
        raise SystemExit("--drift-transform exige --drift-clients")

    name = args.output_dir or default_output_name(args.dirichlet_alpha, args.partition_seed,
                                                  args.drift_transform, drift_clients, args.num_clients)
    if args.print_name:
        print(name)
        return name
    out_dir = current_dir / name

    params = {
        "num_clients": args.num_clients, "num_classes": 10, "alpha": args.dirichlet_alpha,
        "partition_seed": args.partition_seed, "drift_transform": args.drift_transform,
        "drift_clients": sorted(drift_clients), "drift_seed": args.drift_seed,
        "noise_std": args.noise_std, "test_ratio": args.test_ratio, "min_test": args.min_test,
        "generator_version": GENERATOR_VERSION,
    }

    if args.skip_if_exists and dataset_is_current(out_dir, params):
        print(f"[INFO] Dataset já gerado e atualizado: {out_dir}")
        return name

    print(f"[ASTRA] {args.num_clients} clientes, alpha={args.dirichlet_alpha}, seed={args.partition_seed}")
    print(f"[DRIFT] transform={args.drift_transform}, clientes(base 0)={sorted(drift_clients)}")
    X_tr, y_tr, X_te, y_te = load_cifar10_normalized(args.raw_dir)
    result = build_astra_drift_dataset(
        X_tr, y_tr, X_te, y_te, num_clients=args.num_clients, alpha=args.dirichlet_alpha,
        partition_seed=args.partition_seed, drift_transform=args.drift_transform,
        drift_clients=drift_clients, drift_seed=args.drift_seed, noise_std=args.noise_std,
        test_ratio=args.test_ratio, min_test=args.min_test, allow_empty=args.allow_empty)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    save_dataset(out_dir, result, params, hardlink=not args.no_hardlink)
    print(f"[OK] Dataset salvo em {out_dir}")
    return name


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

