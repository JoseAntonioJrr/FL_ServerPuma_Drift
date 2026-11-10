import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.utils.astra_partition import (
    astra_dirichlet_partition,
    astra_to_puma_id,
    astra_to_puma_ids,
    puma_to_astra_id,
)


def _literal_astra_prepare_data_loop(labels, num_clients, alpha, seed, num_classes=10):
    """Cópia literal da lógica de ASTRA-FL/scripts/setup/prepare_data.py."""
    np.random.seed(seed)
    labels = np.array(labels)
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

    return {str(i + 1): indices for i, indices in enumerate(client_partitions)}


class AstraPartitionTests(unittest.TestCase):
    def setUp(self):
        # 2000 amostras sintéticas, 10 classes em ordem determinística não-ordenada
        rng = np.random.default_rng(7)
        self.labels = np.tile(np.arange(10), 200)
        rng.shuffle(self.labels)

    def test_exact_equivalence_with_astra_prepare_data(self):
        for alpha in (0.1, 0.5):
            expected = _literal_astra_prepare_data_loop(self.labels, num_clients=20, alpha=alpha, seed=42)
            actual = astra_dirichlet_partition(self.labels, num_clients=20, alpha=alpha, seed=42)
            self.assertEqual(set(actual.keys()), set(range(1, 21)))
            for cid_str, exp_indices in expected.items():
                self.assertEqual(actual[int(cid_str)], exp_indices)

    def test_partition_invariants_and_determinism(self):
        p1 = astra_dirichlet_partition(self.labels, num_clients=20, alpha=0.1, seed=42)
        p2 = astra_dirichlet_partition(self.labels, num_clients=20, alpha=0.1, seed=42)
        self.assertEqual(p1, p2)

        all_indices = [idx for indices in p1.values() for idx in indices]
        self.assertEqual(len(all_indices), len(self.labels))
        self.assertEqual(len(set(all_indices)), len(self.labels))
        self.assertEqual(set(all_indices), set(range(len(self.labels))))

    def test_id_mapping_one_based_to_zero_based(self):
        astra_dict = {str(i): [i * 10] for i in range(1, 21)}
        puma_dict = astra_to_puma_ids(astra_dict)
        self.assertEqual(puma_dict[0], [10])
        self.assertEqual(puma_dict[19], [200])
        self.assertEqual(sorted(puma_dict.keys()), list(range(20)))
        self.assertEqual(astra_to_puma_id("client_001"), 0)
        self.assertEqual(astra_to_puma_id("20"), 19)
        self.assertEqual(puma_to_astra_id(0), 1)

    def test_id_mapping_rejects_missing_keys(self):
        incomplete = {str(i): [i] for i in range(1, 20) if i != 5}
        with self.assertRaises(ValueError):
            astra_to_puma_ids(incomplete)
        zero_based = {str(i): [i] for i in range(20)}
        with self.assertRaises(ValueError):
            astra_to_puma_ids(zero_based)


if __name__ == "__main__":
    unittest.main()

