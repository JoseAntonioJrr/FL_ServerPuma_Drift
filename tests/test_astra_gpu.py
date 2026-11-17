"""Optional, tiny GPU smoke test; runs only when CUDA is available."""

import copy
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'system'))

try:
    import torch
    from astra import astra_loss, astra_round_config
    from model import SimpleModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None or not torch.cuda.is_available(), 'CUDA PyTorch required')
class AstraGpuTests(unittest.TestCase):
    def test_active_and_inactive_gpu_step(self):
        device = torch.device('cuda')
        torch.manual_seed(42)
        model = SimpleModel(in_features=3, num_classes=10, dim=1600).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(2, 3, 32, 32, device=device)
        labels = torch.tensor([1, 5], device=device)
        torch.cuda.reset_peak_memory_stats()

        # First pass initializes CUDA kernels; don't compare its time to a warm pass.
        for pass_index, round_index in enumerate((0, 0, 11)):
            config = astra_round_config(round_index, 50, local_epochs=1)
            reference = {name: p.detach().clone() for name, p in model.named_parameters()}
            teacher = copy.deepcopy(model).eval() if config['alpha'] else None
            if teacher is not None:
                teacher.requires_grad_(False)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(2):
                optimizer.zero_grad()
                logits = model(images)
                with torch.no_grad():
                    teacher_logits = teacher(images) if teacher is not None else None
                loss = astra_loss(logits, labels, model, reference, config['mu'],
                                  config['alpha'], config['temperature'], teacher_logits)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all().item()
                                    for p in model.parameters() if p.grad is not None))
                optimizer.step()
            torch.cuda.synchronize()
            if pass_index:
                print(f'ASTRA GPU round {round_index}: 2 warmed steps in '
                      f'{time.perf_counter() - started:.3f}s, '
                      f'peak allocated {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB')
            del teacher, reference


if __name__ == '__main__':
    unittest.main()
