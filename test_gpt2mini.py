"""Verificações pequenas: nenhuma carrega o GPT-2 completo ou baixa pesos."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

SCRIPT = Path(__file__).with_name('gpt2mini-default.py')
spec = importlib.util.spec_from_file_location('gpt2mini', SCRIPT)
gpt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gpt
spec.loader.exec_module(gpt)


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def model(self):
        return gpt.GPT(gpt.GPTconfigs(block_size=8, vocab_size=32,
                                     n_layer=2, n_head=2, n_embd=16))

    def test_device_priority(self):
        for cuda, mps, expected in ((True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(expected=expected):
                with mock.patch.object(torch.cuda, 'is_available', return_value=cuda), \
                     mock.patch.object(torch.backends.mps, 'is_available', return_value=mps) as mps_check:
                    self.assertEqual(gpt.select_device().type, expected)
                    if cuda:
                        mps_check.assert_not_called()

    def test_generation_clears_terminal(self):
        with mock.patch.object(gpt, 'clear_terminal') as clear, \
             mock.patch.object(gpt, 'select_device', return_value=torch.device('cpu')), \
             mock.patch.object(gpt, 'run_generation') as generate, \
             contextlib.redirect_stdout(io.StringIO()):
            gpt.main(['generate'])
        clear.assert_called_once_with()
        self.assertEqual(generate.call_args.args[0].device.type, 'cpu')

    def test_clear_terminal_uses_original_platform_command(self):
        for platform, command, shell in (('nt', 'cls', True), ('posix', 'clear', False)):
            with mock.patch.object(gpt.os, 'name', platform), \
                 mock.patch.object(gpt.subprocess, 'run') as run:
                gpt.clear_terminal()
                run.assert_called_once_with(command, shell=shell, check=False)

    def test_causality_and_backward(self):
        model = self.model()
        x = torch.randint(0, 32, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 32
        logits, loss = model(x, (x + 1) % 32)
        altered, _ = model(changed)
        torch.testing.assert_close(logits[:, :4], altered[:, :4])
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = model.configure_optimizers(0.1, 3e-4, torch.device('cpu'))
        before = model.lm_head.weight.detach().clone()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.lm_head.weight))

    def test_accumulation_matches_batch_mean(self):
        full, accumulated = self.model(), self.model()
        accumulated.load_state_dict(full.state_dict())
        x = torch.randint(0, 32, (2, 8))
        y = (x + 1) % 32
        full(x, y)[1].backward()
        for i in range(2):
            (accumulated(x[i:i+1], y[i:i+1])[1] / 2).backward()
        for a, b in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-4)

    def test_generation_excludes_padding_and_crops_context(self):
        class ForcedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = gpt.GPTconfigs(block_size=3)
                self.contexts = []

            def forward(self, x):
                self.contexts.append(x.tolist())
                scores = torch.full((1, x.shape[1], 50304), -100.0)
                scores[:, :, 7] = 0
                scores[:, :, 50303] = 100 # padding teria a maior probabilidade
                return scores, None

        model = ForcedLogits()
        result = gpt.generate(model, [1, 2, 3, 4], 3, top_k=1)
        self.assertEqual(result, [1, 2, 3, 4, 7, 7, 7])
        self.assertEqual(model.contexts[0], [[2, 3, 4]])
        self.assertEqual(model.contexts[-1], [[4, 7, 7]])
        self.assertTrue(model.training)

    def test_loader_wraps_and_rejects_short_data(self):
        loader = gpt.DataLoaderLite(2, 4, torch.arange(17))
        x, y = loader.next_batch()
        torch.testing.assert_close(y, x + 1)
        loader.next_batch()
        torch.testing.assert_close(loader.next_batch()[0], x)
        with self.assertRaises(ValueError):
            gpt.DataLoaderLite(2, 4, torch.arange(8))

    def test_learning_rate_endpoints(self):
        self.assertAlmostEqual(gpt.get_learnrate(0, 50, 5, 3e-4), 6e-5)
        self.assertAlmostEqual(gpt.get_learnrate(4, 50, 5, 3e-4), 3e-4)
        self.assertAlmostEqual(gpt.get_learnrate(49, 50, 5, 3e-4), 3e-5)

    def test_cli_training_resume_and_generation(self):
        # Integração real, mas com apenas uma camada de oito dimensões.
        with tempfile.TemporaryDirectory(prefix='.test-gpt-', dir=SCRIPT.parent) as directory:
            assert Path(directory).resolve().is_relative_to(SCRIPT.parent.resolve())
            checkpoint = Path(directory) / 'last.pt'

            def run(*args):
                result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            run('train', '--checkpoint', checkpoint, '--steps', 2, '--warmup-steps', 0,
                '--batch-size', 1, '--seq-len', 8, '--n-layer', 1,
                '--n-head', 2, '--n-embd', 8, '--block-size', 8,
                '--eval-every', 1, '--eval-batches', 1, '--save-every', 1)
            saved = gpt.load_checkpoint(checkpoint)
            self.assertEqual(saved['step'], 2)
            self.assertTrue(saved['optimizer']['state'])
            previous = saved['model']['lm_head.weight'].clone()
            output = run('train', '--checkpoint', checkpoint, '--resume', '--steps', 3,
                         '--eval-batches', 1)
            self.assertIn('Retomando', output)
            resumed = gpt.load_checkpoint(checkpoint)
            self.assertEqual(resumed['step'], 3)
            self.assertEqual(resumed['train_config']['seq_len'], 8)
            self.assertFalse(torch.equal(previous, resumed['model']['lm_head.weight']))
            output = run('generate', '--checkpoint', checkpoint, '--prompt', 'First',
                         '--max-new-tokens', 3, '--top-k', 1)
            self.assertIn('1> First', output)


if __name__ == '__main__':
    unittest.main()
