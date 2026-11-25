import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "system"))

from dataset.generate_Cifar10_astra_drift import (
    build_astra_drift_dataset,
    dataset_is_current,
    params_signature,
    save_dataset,
)
from dataset.utils.drift_transforms import (
    apply_cyclic_shift,
    apply_gaussian_noise,
    apply_pairwise_label_swap,
)
from utils.drift_schedule import (
    DriftPhase,
    parse_drift_clients,
    phase_for_round,
    resolve_drift_clients,
)

try:
    import torch
    from utils.data_utils import (
        _STRICT_CACHE,
        has_global_test,
        read_client_data,
        read_global_test,
    )
except ImportError:
    torch = None


def _make_synthetic_cifar(n_train=2000, n_test=500, seed=99):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-1.0, 1.0, size=(n_train, 3, 32, 32)).astype(np.float32)
    y_train = np.tile(np.arange(10), n_train // 10).astype(np.int64)
    rng.shuffle(y_train)
    X_test = rng.uniform(-1.0, 1.0, size=(n_test, 3, 32, 32)).astype(np.float32)
    y_test = np.tile(np.arange(10), n_test // 10).astype(np.int64)
    rng.shuffle(y_test)
    return X_train, y_train, X_test, y_test


class DriftTransformTests(unittest.TestCase):
    def test_label_swap_involution_and_cyclic_shift(self):
        y = np.arange(10, dtype=np.int64)
        swapped = apply_pairwise_label_swap(y)
        self.assertFalse(np.array_equal(swapped, y))
        self.assertTrue(np.array_equal(apply_pairwise_label_swap(swapped), y))

        shifted = apply_cyclic_shift(y, shift=1, num_classes=10)
        self.assertEqual(shifted.tolist(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    def test_gaussian_noise_clipped_and_deterministic(self):
        x = np.zeros((8, 3, 32, 32), dtype=np.float32)
        x[0, 0, 0, 0] = 0.99
        r1 = np.random.default_rng(123)
        r2 = np.random.default_rng(123)
        n1 = apply_gaussian_noise(x, std=0.25, rng=r1)
        n2 = apply_gaussian_noise(x, std=0.25, rng=r2)
        self.assertTrue(np.array_equal(n1, n2))
        self.assertFalse(np.array_equal(n1, x))
        self.assertGreaterEqual(float(n1.min()), -1.0)
        self.assertLessEqual(float(n1.max()), 1.0)

    def test_legacy_generator_reexports_transforms(self):
        import dataset.generate_Cifar10_drift as legacy
        y = np.array([0, 1, 2, 9], dtype=np.int64)
        self.assertTrue(np.array_equal(
            legacy.apply_pairwise_label_swap(y),
            apply_pairwise_label_swap(y),
        ))


class AstraDriftDatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.X_tr, self.y_tr, self.X_te, self.y_te = _make_synthetic_cifar()

    def test_no_leakage_and_transform_semantics(self):
        drift_clients = {0, 2, 5}
        for transform in ("label_swap", "cyclic_shift", "gaussian_noise", "class_shift"):
            res = build_astra_drift_dataset(
                self.X_tr, self.y_tr, self.X_te, self.y_te,
                num_clients=20, alpha=0.5, partition_seed=42,
                drift_transform=transform, drift_clients=drift_clients,
                drift_seed=1234, min_test=10, allow_empty=True, log=lambda *_: None,
            )
            for c in range(20):
                cl = res["clients"][c]
                # Índices de treino dentro do pool de treino, índices de teste dentro do pool de teste
                self.assertTrue(np.all((cl["train_idx_p1"] >= 0) & (cl["train_idx_p1"] < len(self.X_tr))))
                self.assertTrue(np.all((cl["train_idx_p2"] >= 0) & (cl["train_idx_p2"] < len(self.X_tr))))
                self.assertTrue(np.all((cl["test_idx_p1"] >= 0) & (cl["test_idx_p1"] < len(self.X_te))))
                self.assertTrue(np.all((cl["test_idx_p2"] >= 0) & (cl["test_idx_p2"] < len(self.X_te))))

                if c not in drift_clients:
                    self.assertTrue(cl["phase2_identical"])
                    self.assertTrue(np.array_equal(cl["train_p1"]["x"], cl["train_p2"]["x"]))
                    self.assertTrue(np.array_equal(cl["train_p1"]["y"], cl["train_p2"]["y"]))
                    self.assertTrue(np.array_equal(cl["test_p1"]["x"], cl["test_p2"]["x"]))
                    self.assertTrue(np.array_equal(cl["test_p1"]["y"], cl["test_p2"]["y"]))
                else:
                    self.assertFalse(cl["phase2_identical"])
                    if transform in ("label_swap", "cyclic_shift"):
                        self.assertTrue(np.array_equal(cl["train_p1"]["x"], cl["train_p2"]["x"]))
                        self.assertFalse(np.array_equal(cl["train_p1"]["y"], cl["train_p2"]["y"]))
                        self.assertFalse(np.array_equal(cl["test_p1"]["y"], cl["test_p2"]["y"]))
                    elif transform == "gaussian_noise":
                        self.assertTrue(np.array_equal(cl["train_p1"]["y"], cl["train_p2"]["y"]))
                        self.assertFalse(np.array_equal(cl["train_p1"]["x"], cl["train_p2"]["x"]))
                        self.assertFalse(np.array_equal(cl["test_p1"]["x"], cl["test_p2"]["x"]))

            if transform == "class_shift":
                self.assertTrue(res["global_identical"])
            else:
                self.assertFalse(res["global_identical"])

    def test_reproducibility_sha256_manifest_and_hardlink(self):
        params = {
            "num_clients": 20, "num_classes": 10, "alpha": 0.5,
            "partition_seed": 42, "drift_transform": "label_swap",
            "drift_clients": [0, 1], "drift_seed": 1234, "noise_std": 0.25,
            "test_ratio": 1.0 / 3.0, "min_test": 10, "generator_version": 1,
        }
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            res1 = build_astra_drift_dataset(
                self.X_tr, self.y_tr, self.X_te, self.y_te,
                num_clients=20, alpha=0.5, partition_seed=42,
                drift_transform="label_swap", drift_clients={0, 1},
                drift_seed=1234, min_test=10, allow_empty=True, log=lambda *_: None,
            )
            res2 = build_astra_drift_dataset(
                self.X_tr, self.y_tr, self.X_te, self.y_te,
                num_clients=20, alpha=0.5, partition_seed=42,
                drift_transform="label_swap", drift_clients={0, 1},
                drift_seed=1234, min_test=10, allow_empty=True, log=lambda *_: None,
            )
            cfg1 = save_dataset(d1, res1, params, hardlink=True, log=lambda *_: None)
            cfg2 = save_dataset(d2, res2, params, hardlink=True, log=lambda *_: None)
            self.assertEqual(cfg1["manifest_sha256"], cfg2["manifest_sha256"])
            self.assertEqual(cfg1["id_map"]["0"], "1")
            self.assertEqual(cfg1["id_map"]["19"], "20")
            self.assertTrue(dataset_is_current(d1, params))

            # Legado 0.npz == 0_phase1.npz via hardlink (mesmo inode)
            st_p1 = os.stat(Path(d1) / "train" / "0_phase1.npz")
            st_leg = os.stat(Path(d1) / "train" / "0.npz")
            self.assertEqual(st_p1.st_ino, st_leg.st_ino)
            # Cliente estável 2 tem phase2 hardlinked para phase1
            st_c2_p1 = os.stat(Path(d1) / "train" / "2_phase1.npz")
            st_c2_p2 = os.stat(Path(d1) / "train" / "2_phase2.npz")
            self.assertEqual(st_c2_p1.st_ino, st_c2_p2.st_ino)


@unittest.skipIf(torch is None, "PyTorch required")
class DataContractAndStrictPhaseTests(unittest.TestCase):
    def test_puma_contract_and_strict_phase_errors(self):
        X_tr, y_tr, X_te, y_te = _make_synthetic_cifar(n_train=1000, n_test=300)
        params = {
            "num_clients": 5, "num_classes": 10, "alpha": 0.5,
            "partition_seed": 42, "drift_transform": "label_swap",
            "drift_clients": [0], "drift_seed": 1234, "noise_std": 0.25,
            "test_ratio": 1.0 / 3.0, "min_test": 10, "generator_version": 1,
        }
        with tempfile.TemporaryDirectory() as root:
            ds_name = "Cifar10_astra_synth"
            out_dir = Path(root) / ds_name
            res = build_astra_drift_dataset(
                X_tr, y_tr, X_te, y_te,
                num_clients=5, alpha=0.5, partition_seed=42,
                drift_transform="label_swap", drift_clients={0},
                drift_seed=1234, min_test=10, allow_empty=True, log=lambda *_: None,
            )
            save_dataset(out_dir, res, params, hardlink=False, log=lambda *_: None)

            prev_root = os.environ.get("PUMA_DATASET_ROOT")
            prev_strict = os.environ.get("PUMA_STRICT_PHASE")
            os.environ["PUMA_DATASET_ROOT"] = root
            os.environ.pop("PUMA_STRICT_PHASE", None)
            _STRICT_CACHE.clear()
            try:
                for phase in ("phase1", "phase2", None):
                    X, y = read_client_data(ds_name, 0, is_train=True, phase=phase)
                    self.assertEqual(X.dtype, torch.float32)
                    self.assertEqual(y.dtype, torch.int64)
                    self.assertEqual(tuple(X.shape[1:]), (3, 32, 32))

                self.assertTrue(has_global_test(ds_name))
                gX1, gy1 = read_global_test(ds_name, "phase1")
                gX2, gy2 = read_global_test(ds_name, "phase2")
                self.assertEqual(gX1.shape[0], len(X_te))
                self.assertFalse(torch.equal(gy1, gy2))

                # Remover phase2 e garantir que modo estrito falha em vez de usar fallback
                (out_dir / "train" / "0_phase2.npz").unlink()
                with self.assertRaises(FileNotFoundError):
                    read_client_data(ds_name, 0, is_train=True, phase="phase2")
            finally:
                _STRICT_CACHE.clear()
                if prev_root is None:
                    os.environ.pop("PUMA_DATASET_ROOT", None)
                else:
                    os.environ["PUMA_DATASET_ROOT"] = prev_root
                if prev_strict is None:
                    os.environ.pop("PUMA_STRICT_PHASE", None)
                else:
                    os.environ["PUMA_STRICT_PHASE"] = prev_strict


class DriftScheduleTests(unittest.TestCase):
    def test_parse_and_resolve_drift_clients(self):
        self.assertEqual(parse_drift_clients("0,2,5", 10), {0, 2, 5})
        self.assertEqual(parse_drift_clients("0-3,7", 10), {0, 1, 2, 3, 7})
        self.assertEqual(parse_drift_clients("all", 4), {0, 1, 2, 3})
        self.assertEqual(resolve_drift_clients("all", [0, 1, 2]), {0, 1, 2})
        self.assertEqual(resolve_drift_clients("0-2", [0, 1, 2, 3]), {0, 1, 2})
        with self.assertRaises(ValueError):
            parse_drift_clients("0,20", 20)

    def test_abrupt_gradual_and_periodic_schedules(self):
        # Abrupto simples
        args = types.SimpleNamespace(drift=True, drift_type="abrupt", drift_round=5, drift_length=3, drift_freq=0)
        self.assertEqual(phase_for_round(4, args), DriftPhase("phase1", 0.0))
        self.assertEqual(phase_for_round(5, args), DriftPhase("phase2", 1.0))
        self.assertEqual(phase_for_round(6, args), DriftPhase("phase2", 1.0))
        self.assertEqual(phase_for_round(6, args, is_drifted_client=False), DriftPhase("phase1", 0.0))

        # Gradual simples: cliente e servidor alinhados vs hold legado
        args_g = types.SimpleNamespace(drift=True, drift_type="gradual", drift_round=5, drift_length=3, drift_freq=0)
        self.assertEqual(phase_for_round(4, args_g), DriftPhase("phase1", 0.0))
        self.assertEqual(phase_for_round(5, args_g), DriftPhase("blend", 0.25))
        self.assertEqual(phase_for_round(6, args_g), DriftPhase("blend", 0.50))
        self.assertEqual(phase_for_round(7, args_g), DriftPhase("blend", 0.75))
        self.assertEqual(phase_for_round(8, args_g), DriftPhase("phase2", 1.0))
        self.assertEqual(phase_for_round(6, args_g, legacy_gradual_server_hold=True), DriftPhase("phase1", 0.0))
        self.assertEqual(phase_for_round(8, args_g, legacy_gradual_server_hold=True), DriftPhase("phase2", 1.0))

        # Periódico abrupto (freq=5)
        args_pa = types.SimpleNamespace(drift=True, drift_type="abrupt", drift_round=10, drift_length=5, drift_freq=5)
        self.assertEqual([phase_for_round(r, args_pa).phase for r in range(1, 12)],
                         ["phase1"] * 4 + ["phase2"] * 5 + ["phase1"] * 2)

        # Periódico gradual (freq=3 -> r in {0.0, 0.5, 1.0}, depois {1.0, 0.5, 0.0})
        args_pg = types.SimpleNamespace(drift=True, drift_type="gradual", drift_round=10, drift_length=5, drift_freq=3)
        alphas = [phase_for_round(r, args_pg).alpha for r in range(1, 7)]
        self.assertEqual(alphas, [0.0, 0.5, 1.0, 1.0, 0.5, 0.0])

        # Sem drift
        args_off = types.SimpleNamespace(drift=False)
        self.assertEqual(phase_for_round(1, args_off), DriftPhase(None, 0.0))


@unittest.skipIf(torch is None, "PyTorch required")
class ClientAndServerIntegrationUnitTests(unittest.TestCase):
    def test_astra_train_transform_preserves_shape_and_bounds(self):
        from client import get_astra_train_transform
        tf = get_astra_train_transform()
        x = torch.full((3, 32, 32), 0.25, dtype=torch.float32)
        out = tf(x)
        self.assertEqual(tuple(out.shape), (3, 32, 32))
        self.assertEqual(out.dtype, torch.float32)
        self.assertGreaterEqual(float(out.min()), -1.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_server_global_eval_requires_global_test_and_loads_it(self):
        from server import FederatedLearningServer
        X_tr, y_tr, X_te, y_te = _make_synthetic_cifar(n_train=400, n_test=100)
        params = {
            "num_clients": 4, "num_classes": 10, "alpha": 0.5,
            "partition_seed": 42, "drift_transform": "label_swap",
            "drift_clients": [0], "drift_seed": 1234, "noise_std": 0.25,
            "test_ratio": 1.0 / 3.0, "min_test": 10, "generator_version": 1,
        }
        with tempfile.TemporaryDirectory() as root:
            ds_name = "Cifar10_astra_srvtest"
            out_dir = Path(root) / ds_name
            res = build_astra_drift_dataset(
                X_tr, y_tr, X_te, y_te,
                num_clients=4, alpha=0.5, partition_seed=42,
                drift_transform="label_swap", drift_clients={0},
                drift_seed=1234, min_test=10, allow_empty=True, log=lambda *_: None,
            )
            save_dataset(out_dir, res, params, hardlink=False, log=lambda *_: None)

            prev_root = os.environ.get("PUMA_DATASET_ROOT")
            prev_strict = os.environ.get("PUMA_STRICT_PHASE")
            os.environ["PUMA_DATASET_ROOT"] = root
            os.environ["PUMA_STRICT_PHASE"] = "1"
            _STRICT_CACHE.clear()
            try:
                args = types.SimpleNamespace(
                    model="cnn", dataset=ds_name, in_features=3, num_classes=10, dim=1600,
                    load_model=None, device="cpu", prune=0, paca=12, rank=8,
                    test_client_idx=0, batch_size=16, drift=True, global_eval="both",
                )
                srv = FederatedLearningServer(args)
                self.assertIsNotNone(srv.global_test_loader_p1)
                self.assertIsNotNone(srv.global_test_loader_p2)
                self.assertEqual(len(srv.global_test_loader_p1.dataset), len(X_te))

                (out_dir / "test" / "global_phase1.npz").unlink()
                with self.assertRaises(FileNotFoundError):
                    FederatedLearningServer(args)
            finally:
                _STRICT_CACHE.clear()
                if prev_root is None:
                    os.environ.pop("PUMA_DATASET_ROOT", None)
                else:
                    os.environ["PUMA_DATASET_ROOT"] = prev_root
                if prev_strict is None:
                    os.environ.pop("PUMA_STRICT_PHASE", None)
                else:
                    os.environ["PUMA_STRICT_PHASE"] = prev_strict


if __name__ == "__main__":
    unittest.main()
