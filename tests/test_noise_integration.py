"""Cross-component checks for estimator alignment and optimizer semantics."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dust_qlora.adapters import adapter_state_dict, assign_lora_gradients
from dust_qlora.config import TrainConfig
from dust_qlora.noise import NoiseConfig, estimate_activation_error
from dust_qlora.runtime import load_runtime
from dust_qlora.training import collate, train


def tiny_config():
    config = TrainConfig()
    config.model.name = "tiny"
    config.model.device = "cpu"
    config.model.dtype = "float32"
    config.model.load_in_4bit = False
    config.model.tiny_vocab_size = 16
    config.model.tiny_hidden_size = 5
    config.model.tiny_num_layers = 2
    config.adapter.rank = 2
    config.adapter.alpha = 2.0
    config.adapter.last_n_layers = 2
    config.training.loss_chunk_size = 32
    config.training.max_grad_norm = 1000000.0
    return config


class NoiseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_runtime_estimate_aligns_with_masked_causal_gradient(self):
        torch.manual_seed(27)
        runtime = load_runtime(tiny_config())
        target = next(iter(runtime.adapters))
        wrapper = runtime.adapters[target]
        batch = collate([
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]},
            {"input_ids": [3, 2, 6], "labels": [-100, -100, 6]},
        ], "cpu")
        clean, captured, replay = runtime.capture(target, batch)
        outputs = []

        def retain_output(module, args, output):
            output.retain_grad()
            outputs.append(output)

        handle = wrapper.register_forward_hook(retain_output)
        try:
            runtime.losses(batch).sum().backward()
        finally:
            handle.remove()
        exact = outputs[0].grad.detach().clone()
        exact_b = wrapper.lora_B.grad.clone()
        runtime.model.zero_grad(set_to_none=True)

        def objective(delta):
            wrapper.perturbation = delta
            return runtime.losses(batch, replay)

        estimated, _ = estimate_activation_error(
            objective, tuple(exact.shape), NoiseConfig(pairs=1024, sigma=0.005, seed=19),
            "cpu", token_mask=batch["attention_mask"],
        )
        wrapper.perturbation = None
        cosine = torch.nn.functional.cosine_similarity(exact.flatten(), estimated.flatten(), dim=0)
        self.assertGreater(float(cosine), 0.95)
        # These positions have no direct loss, but do influence later targets.
        self.assertEqual(float(clean[0, 0]), 0.0)
        self.assertGreater(float(exact[0, :2].norm()), 0)
        self.assertGreater(float(estimated[0, :2].norm()), 0)
        self.assertTrue(torch.equal(estimated[1, 3:], torch.zeros_like(estimated[1, 3:])))
        _, estimated_b = assign_lora_gradients(wrapper, captured, estimated)
        self.assertGreater(float((estimated_b * exact_b).sum()), 0)
        with torch.no_grad():
            wrapper.lora_B.add_(estimated_b, alpha=-0.005)
            updated = runtime.losses(batch).sum()
        self.assertLess(float(updated), float(clean.sum()))
        self.assertTrue(all(parameter.grad is None for name, parameter in runtime.model.named_parameters()
                            if "lora_" not in name))

    def test_variable_length_microbatches_match_full_batch_exact_update(self):
        """An exact output-error oracle isolates accumulation from probe variance.

        This exercises train()'s real normalization, callback, capture and local
        contraction paths against independent full-batch autograd. SGD makes
        the gradient magnitude, not only its coordinate signs, observable.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"id": "one", "group_id": "one", "input_ids": [1, 2, 3],
                 "labels": [-100, -100, 3]},
                {"id": "three", "group_id": "three", "input_ids": [4, 5, 6, 7, 8],
                 "labels": [-100, -100, 6, 7, 8]},
            ]
            data = root / "train.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = tiny_config()
            config.model.replay_mode = "full"
            config.adapter.last_n_layers = 1
            config.data.train, config.data.validation = str(data), None
            config.training.output_dir = str(root / "micro")
            config.training.max_steps = 1
            config.training.gradient_accumulation_steps = 2
            config.training.shuffle = False
            config.training.learning_rate = 0.03
            config.training.weight_decay = 0.2
            torch.manual_seed(81)
            micro_runtime = load_runtime(config)
            with torch.no_grad():
                next(iter(micro_runtime.adapters.values())).lora_B.normal_(std=0.1)
            baseline = copy.deepcopy(config)
            baseline.training.method = "backprop"
            baseline.training.output_dir = str(root / "full")
            baseline.training.batch_size = 2
            baseline.training.gradient_accumulation_steps = 1
            full_runtime = copy.deepcopy(micro_runtime)
            full_runtime.config = baseline

            def exact_error(evaluate, shape, noise_config, device, **kwargs):
                with torch.enable_grad():
                    delta = torch.zeros(shape, device=device, requires_grad=True)
                    gradient, = torch.autograd.grad(evaluate(delta).sum(), delta)
                return gradient.detach(), {"exact_oracle": True}

            def sgd(parameters, **kwargs):
                return torch.optim.SGD(parameters, **kwargs)

            with redirect_stdout(io.StringIO()), \
                    patch("dust_qlora.training.torch.optim.AdamW", side_effect=sgd), \
                    patch("dust_qlora.training.estimate_activation_error", side_effect=exact_error):
                train(config, runtime=micro_runtime)
                train(baseline, runtime=full_runtime)
            expected = adapter_state_dict(full_runtime.model)
            actual = adapter_state_dict(micro_runtime.model)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=1e-7)

    def test_round_robin_skips_inactive_adam_momentum_and_weight_decay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}
            data = root / "train.jsonl"
            data.write_text(json.dumps(row) + "\n")
            config = tiny_config()
            config.data.train, config.data.validation = str(data), None
            config.training.output_dir = str(root / "run")
            config.training.max_steps = 3
            config.training.weight_decay = 0.4
            config.training.learning_rate = 0.01
            config.noise.pairs = 4
            torch.manual_seed(28)
            runtime = load_runtime(config)
            snapshots = [adapter_state_dict(runtime.model)]

            class TrackingAdamW(torch.optim.AdamW):
                def step(self, *args, **kwargs):
                    result = super().step(*args, **kwargs)
                    snapshots.append(adapter_state_dict(runtime.model))
                    return result

            with redirect_stdout(io.StringIO()), patch("dust_qlora.training.torch.optim.AdamW", TrackingAdamW):
                train(config, runtime=runtime)
            self.assertEqual(len(snapshots), 4)
            targets = list(runtime.adapters)
            for step in range(1, 4):
                current = targets[(step - 1) % 2]
                inactive = targets[step % 2]
                for factor in ("lora_A", "lora_B"):
                    key = f"base_model.model.{inactive}.{factor}.weight"
                    self.assertTrue(torch.equal(snapshots[step][key], snapshots[step - 1][key]))
                key = f"base_model.model.{current}.lora_B.weight"
                self.assertFalse(torch.equal(snapshots[step][key], snapshots[step - 1][key]))


if __name__ == "__main__":
    unittest.main()
