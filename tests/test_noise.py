"""Probe geometry, future credit, and actual finite-difference convergence."""

import unittest

import torch

from dust_qlora.adapters import LowRankLinear, assign_lora_gradients
from dust_qlora.noise import (
    NoiseConfig,
    causal_credits,
    estimate_activation_error,
    hadamard_rows,
    iter_activation_probes,
)


class NoiseTests(unittest.TestCase):
    def test_hadamard_power_of_two_rows_and_truncated_column_covariance(self):
        h8 = hadamard_rows(torch.arange(8), 8)
        self.assertTrue(torch.equal(h8 @ h8.T, 8 * torch.eye(8)))
        h5 = hadamard_rows(torch.arange(8), 5)
        self.assertTrue(torch.equal(h5.T @ h5, 8 * torch.eye(5)))
        # Truncated rows are intentionally not advertised as orthogonal.
        self.assertFalse(torch.equal(h5 @ h5.T, 5 * torch.eye(8)))

    def test_randomized_hadamard_cycles_remain_orthogonal_per_token(self):
        probes = torch.stack(list(iter_activation_probes((2, 3, 8), 8, seed=4, device="cpu")))
        for batch in range(2):
            for token in range(3):
                rows = probes[:, batch, token]
                self.assertTrue(torch.equal(rows @ rows.T, 8 * torch.eye(8)))
        odd_probes = torch.stack(list(iter_activation_probes((1, 1, 5), 8, seed=4, device="cpu")))
        rows = odd_probes[:, 0, 0]
        self.assertTrue(torch.equal(rows.T @ rows, 8 * torch.eye(5)))

    def test_seed_reproducibility_and_padding_mask(self):
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        def make(seed):
            return torch.stack(list(iter_activation_probes((1, 3, 5), 7, seed=seed, device="cpu", token_mask=mask)))
        self.assertTrue(torch.equal(make(4), make(4)))
        self.assertFalse(torch.equal(make(4), make(5)))
        self.assertTrue(torch.equal(make(4)[:, 0, 2], torch.zeros(7, 5)))

    def test_causal_credit_flows_through_masked_prefix_and_not_across_examples(self):
        diff = torch.tensor([[0.0, 0.0, 2.0, 3.0], [0.0, 7.0, 0.0, 0.0]])
        expected = torch.tensor([[5.0, 5.0, 5.0, 3.0], [7.0, 7.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(causal_credits(diff), expected))

    def test_complete_cycle_recovers_linear_output_gradient_without_width_factor(self):
        weight = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.25])
        def objective(delta):
            self.assertFalse(torch.is_grad_enabled())
            return (delta * weight).sum(-1)
        error, stats = estimate_activation_error(
            objective, (1, 1, 5), NoiseConfig(pairs=8, sigma=0.1, seed=77), "cpu"
        )
        self.assertTrue(torch.allclose(error[0, 0], weight, atol=2e-6))
        self.assertEqual(stats.forward_evaluations, 16)

    def test_causal_masked_estimator_aligns_with_exact_gradient_and_improves_loss(self):
        torch.manual_seed(2)
        base = (torch.randn(1, 4, 3) * 0.1).requires_grad_()
        weights = torch.tensor([0.6, -0.8, 0.3])
        mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
        targets = torch.tensor([[0.0, 0.0, 0.2, -0.3]])
        def losses(value):
            prediction = (value.cumsum(dim=1) * weights).sum(-1)
            return (prediction - targets).square() * mask / mask.sum()
        losses(base).sum().backward()
        exact = base.grad.clone()
        initial = base.detach()
        def objective(delta):
            return losses(initial + delta)
        estimated, _ = estimate_activation_error(
            objective, tuple(base.shape), NoiseConfig(pairs=512, sigma=0.01, seed=41), "cpu"
        )
        cosine = torch.nn.functional.cosine_similarity(exact.flatten(), estimated.flatten(), dim=0)
        self.assertGreater(float(cosine), 0.97)
        self.assertGreater(float(estimated[:, :2].norm()), 0)
        self.assertLess(float(losses(initial - 0.01 * estimated).sum()), float(losses(initial).sum()))

    def test_zero_B_adapter_receives_a_forward_only_update(self):
        torch.manual_seed(7)
        layer = LowRankLinear(torch.nn.Linear(4, 3), 2, 2)
        x = torch.randn(1, 1, 4)
        target = torch.tensor([[[0.2, -0.3, 0.1]]])
        initial = (layer(x) - target).square().sum()
        initial.backward()
        expected_b = layer.lora_B.grad.clone()
        layer.lora_A.grad = layer.lora_B.grad = None
        def objective(delta):
            layer.perturbation = delta
            return (layer(x) - target).square().sum(-1)
        estimated, _ = estimate_activation_error(
            objective, (1, 1, 3), NoiseConfig(pairs=4, sigma=0.02, seed=5), "cpu"
        )
        layer.perturbation = None
        ga, gb = assign_lora_gradients(layer, x, estimated)
        self.assertTrue(torch.equal(ga, torch.zeros_like(ga)))
        self.assertTrue(torch.allclose(gb, expected_b, atol=2e-5, rtol=2e-5))
        self.assertGreater(float(gb.norm()), 0)
        with torch.no_grad():
            layer.lora_B.add_(gb, alpha=-0.01)
            updated = (layer(x) - target).square().sum()
        self.assertLess(float(updated), float(initial.detach()))
        self.assertTrue(all(p.grad is None for p in layer.base.parameters()))

    def test_no_loss_credit_crosses_batch_sequences(self):
        def objective(delta):
            values = (delta.cumsum(dim=1) + 0.3).square().sum(-1)
            values[1].zero_()
            return values
        estimated, _ = estimate_activation_error(
            objective, (2, 3, 5), NoiseConfig(pairs=9, sigma=0.1, seed=17), "cpu"
        )
        self.assertGreater(float(estimated[0].norm()), 0)
        self.assertTrue(torch.equal(estimated[1], torch.zeros_like(estimated[1])))

    def test_paired_torch_randomness_is_common_and_global_rng_is_restored(self):
        torch.manual_seed(8)
        before = torch.get_rng_state().clone()
        draws = []
        def objective(delta):
            random_offset = torch.rand(())
            draws.append(float(random_offset))
            return delta.sum(-1) + random_offset
        estimate_activation_error(objective, (1, 1, 2), NoiseConfig(pairs=4, sigma=0.1), "cpu")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for index in range(0, 8, 2):
            self.assertEqual(draws[index], draws[index + 1])

    def test_nonfinite_loss_rejected_and_wrong_alignment_rejected(self):
        with self.assertRaises(FloatingPointError):
            estimate_activation_error(lambda _: torch.tensor([[float("nan")]]), (1, 1, 2), NoiseConfig(), "cpu")
        with self.assertRaises(ValueError):
            estimate_activation_error(lambda _: torch.zeros(1, 2), (1, 1, 2), NoiseConfig(), "cpu")


if __name__ == "__main__":
    unittest.main()
