"""Numerical and serialization checks for the actual adapter implementation."""

import unittest

import torch
from torch import nn

from dust_qlora.adapters import (
    LowRankLinear,
    adapter_parameters,
    adapter_state_dict,
    assign_lora_gradients,
    inject_lora,
    load_adapter_state_dict,
)


class FakeQuantizedLinear(nn.Module):
    """A frozen weight-only int8 layer; no trainable/dequantized base copy."""

    def __init__(self):
        super().__init__()
        self.in_features, self.out_features = 4, 3
        self.register_buffer("weight", torch.arange(12, dtype=torch.int8).reshape(3, 4))

    def forward(self, x):
        return x @ (self.weight.to(x.dtype) / 10).T


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_zero_B_is_exact_noop_but_output_noise_has_signal(self):
        base = nn.Linear(4, 3)
        x = torch.randn(2, 5, 4)
        expected = base(x).detach()
        layer = LowRankLinear(base, 2, 4)
        self.assertTrue(torch.equal(layer(x), expected))
        layer.perturbation = torch.ones(2, 5, 3) * 0.125
        self.assertTrue(torch.allclose(layer(x), expected + 0.125))
        self.assertFalse(any(p.requires_grad for p in base.parameters()))
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)

    def test_capture_is_detached_and_explicitly_cleared(self):
        layer = LowRankLinear(nn.Linear(4, 3), 2, 2)
        x = torch.randn(1, 3, 4, requires_grad=True)
        layer.capture_input = True
        layer(x)
        self.assertFalse(layer.last_input.requires_grad)
        self.assertTrue(torch.equal(layer.last_input, x))
        layer.clear_capture()
        self.assertIsNone(layer.last_input)
        self.assertIsNone(layer.perturbation)
        self.assertFalse(layer.capture_input)

    def test_local_gradients_match_backprop_on_causal_masked_objective(self):
        layer = LowRankLinear(nn.Linear(4, 3), 2, 3)
        with torch.no_grad():
            layer.lora_B.normal_(std=0.2)
        x = torch.randn(2, 4, 4)
        y = layer(x)
        y.retain_grad()
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        predicted = torch.tanh(y.cumsum(dim=1))
        loss = (predicted.square().sum(-1) * mask).sum() / mask.sum()
        loss.backward()
        expected_a, expected_b = layer.lora_A.grad.clone(), layer.lora_B.grad.clone()
        error = y.grad.clone()
        self.assertGreater(float(error[:, 0].norm()), 0)
        layer.lora_A.grad = layer.lora_B.grad = None
        ga, gb = assign_lora_gradients(layer, x, error, token_chunk_size=2)
        self.assertTrue(torch.allclose(ga, expected_a, rtol=2e-5, atol=2e-6))
        self.assertTrue(torch.allclose(gb, expected_b, rtol=2e-5, atol=2e-6))
        self.assertTrue(all(p.grad is None for p in layer.base.parameters()))
        assign_lora_gradients(layer, x, error, accumulate=True, token_chunk_size=3)
        self.assertTrue(torch.allclose(layer.lora_A.grad, 2 * expected_a, rtol=2e-5, atol=2e-6))
        self.assertTrue(torch.allclose(layer.lora_B.grad, 2 * expected_b, rtol=2e-5, atol=2e-6))

    def test_untouched_quantized_base_and_manual_update(self):
        base = FakeQuantizedLinear()
        original = base.weight.clone()
        layer = LowRankLinear(base, 2, 2)
        x = torch.randn(1, 4, 4)
        self.assertTrue(torch.equal(layer(x), base(x)))
        assign_lora_gradients(layer, x, torch.ones(1, 4, 3) / 4)
        optimizer = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.01)
        optimizer.step()
        self.assertTrue(torch.equal(original, base.weight))
        self.assertEqual(base.weight.dtype, torch.int8)
        self.assertGreater(float(layer.lora_B.detach().norm()), 0)

    def test_export_uses_standard_peft_names_and_strict_restore(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        layers = inject_lora(model, ["0"], 2, 4)
        with torch.no_grad():
            layers["0"].lora_B.fill_(0.1)
        state = adapter_state_dict(model)
        self.assertEqual(set(state), {
            "base_model.model.0.lora_A.weight", "base_model.model.0.lora_B.weight"
        })
        self.assertEqual(len(list(adapter_parameters(model))), 2)
        with torch.no_grad():
            layers["0"].lora_B.zero_()
        load_adapter_state_dict(model, state)
        self.assertTrue(torch.equal(layers["0"].lora_B.cpu(), state["base_model.model.0.lora_B.weight"]))
        with self.assertRaises(ValueError):
            load_adapter_state_dict(model, {})
        self.assertFalse(model[1].weight.requires_grad)

    def test_mixed_precision_forward_keeps_master_parameters_float32(self):
        layer = LowRankLinear(nn.Linear(4, 3, dtype=torch.bfloat16), 2, 2)
        x = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        self.assertEqual(layer(x).dtype, torch.bfloat16)
        self.assertEqual(layer.lora_A.dtype, torch.float32)
        self.assertEqual(layer.lora_B.dtype, torch.float32)

    def test_invalid_dimensions_and_nonfinite_errors_rejected(self):
        with self.assertRaises(ValueError):
            LowRankLinear(nn.Linear(4, 3), 0, 1)
        with self.assertRaises(ValueError):
            LowRankLinear(nn.Linear(4, 3), 1.5, 1)
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        with self.assertRaises(TypeError):
            inject_lora(model, ["0", "1"], 2, 2)
        self.assertTrue(model[0].weight.requires_grad)
        layer = LowRankLinear(nn.Linear(4, 3), 2, 2)
        with self.assertRaises(ValueError):
            assign_lora_gradients(layer, torch.ones(1, 2, 4), torch.ones(1, 3, 3))
        with self.assertRaises(FloatingPointError):
            assign_lora_gradients(layer, torch.ones(1, 2, 4), torch.full((1, 2, 3), float("nan")))


if __name__ == "__main__":
    unittest.main()
