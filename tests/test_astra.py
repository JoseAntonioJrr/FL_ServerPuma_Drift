import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'system'))
from astra import astra_round_config, astra_loss


class CurriculumTests(unittest.TestCase):
    def test_foundation_and_mentorship(self):
        alphas = [astra_round_config(t, 50)['alpha'] for t in (0, 10, 11, 12, 49)]
        self.assertAlmostEqual(alphas[0], 0.2)
        self.assertAlmostEqual(alphas[1], 0.16)
        self.assertEqual(alphas[2], 0)
        self.assertAlmostEqual(alphas[3], 0.152)
        self.assertEqual(alphas[4], 0)

    def test_invalid_config(self):
        for kwargs in ({'interval': 0}, {'temperature': 0}, {'mu': -1},
                       {'local_epochs': 0}, {'boot_rounds': -1}):
            with self.assertRaises(ValueError):
                astra_round_config(0, 50, **kwargs)


try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class ObjectiveTests(unittest.TestCase):
    def test_proximal_and_teacher_gradient(self):
        student = torch.nn.Linear(2, 2, bias=False)
        reference = {name: p.detach().clone() for name, p in student.named_parameters()}
        x = torch.tensor([[1., 2.], [2., -1.]])
        labels = torch.tensor([0, 1])
        with torch.no_grad():
            teacher_logits = student(x).detach().clone()
            student.weight[0, 0].add_(0.4)
        logits = student(x)
        loss = astra_loss(logits, labels, student, reference, 0.01, 0.2, 3., teacher_logits)
        expected = (F.cross_entropy(logits, labels) +
                    0.005 * (student.weight - reference['weight']).square().sum() +
                    0.2 * 9 * F.kl_div(F.log_softmax(logits / 3, dim=1),
                                       F.softmax(teacher_logits / 3, dim=1), reduction='batchmean'))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(student.weight.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_inactive_round_needs_no_teacher(self):
        student = torch.nn.Linear(2, 2)
        x = torch.ones(1, 2)
        labels = torch.tensor([1])
        reference = {name: p.detach().clone() for name, p in student.named_parameters()}
        loss = astra_loss(student(x), labels, student, reference, 0.01, alpha=0)
        loss.backward()
        self.assertIsNotNone(student.weight.grad)

    def test_puma_local_training_active_and_inactive(self):
        from unittest.mock import patch
        from torch.utils.data import DataLoader, TensorDataset
        from client import local_training

        loader = DataLoader(TensorDataset(
            torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]]),
            torch.tensor([0, 1, 0, 1])), batch_size=2)
        initial = torch.nn.Linear(2, 2)
        global_state = {name: p.detach().clone() for name, p in initial.named_parameters()}
        for round_index in (0, 11):
            model = torch.nn.Linear(2, 2)
            config = astra_round_config(round_index, 50, local_epochs=2)
            steps = 0
            original_step = torch.optim.SGD.step

            def counted_step(optimizer, *args, **kwargs):
                nonlocal steps
                steps += 1
                return original_step(optimizer, *args, **kwargs)

            with patch.object(torch.optim.SGD, 'step', counted_step):
                updated, personalized_acc = local_training(
                    model, global_state, 0, loader, loader, round_num=round_index + 1,
                    alaarg=0, astra_config=config)
            self.assertEqual(steps, 2 * len(loader))
            self.assertIsNone(personalized_acc)  # ASTRA bypasses FedALA
            self.assertFalse(torch.equal(updated['weight'], global_state['weight']))


if __name__ == '__main__':
    unittest.main()
