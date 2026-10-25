"""Runtime equivalence tests: causal labels, bounded head, native decoder replay."""
from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
import warnings
from unittest.mock import patch

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None

if torch is not None:
    from dust_qlora.adapters import inject_lora
    from dust_qlora.config import TrainConfig
    from dust_qlora.runtime import LocalRuntime, load_runtime, select_adapter_targets
    from dust_qlora.training import collate


def config_for_tiny():
    config = TrainConfig()
    config.model.name = "tiny"
    config.model.device = "cpu"
    config.model.dtype = "float32"
    config.model.load_in_4bit = False
    config.model.tiny_vocab_size = 32
    config.model.tiny_hidden_size = 8
    config.model.tiny_num_layers = 3
    config.adapter.rank = 2
    config.adapter.alpha = 2.0
    config.adapter.last_n_layers = 2
    config.training.loss_chunk_size = 2
    return config


@unittest.skipIf(torch is None, "Install the training/test extra for torch runtime tests")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.config = config_for_tiny()
        self.runtime = load_runtime(self.config)
        self.batch = collate([
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [6, 7, 8], "labels": [-100, 7, 8]},
        ], "cpu")

    def test_last_layers_selected_and_base_frozen(self):
        self.assertEqual(list(self.runtime.adapters), ["model.layers.1.mlp.down_proj", "model.layers.2.mlp.down_proj"])
        trainable = [name for name, parameter in self.runtime.model.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(trainable), 4)
        self.assertTrue(all("lora_" in name for name in trainable))
        self.config.adapter.explicit_targets = ["model.layers.0.mlp.down_proj"]
        self.assertEqual(select_adapter_targets(self.runtime.model, self.config), self.config.adapter.explicit_targets)
        self.config.adapter.explicit_targets = ["missing"]
        with self.assertRaisesRegex(ValueError, "not found"):
            select_adapter_targets(self.runtime.model, self.config)

    def test_chunked_head_matches_full_cross_entropy_and_gradient(self):
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        chunked = self.runtime.losses_from_hidden(hidden, self.batch)
        self.assertEqual(tuple(chunked.shape), (2, 5))
        self.assertTrue(torch.equal(chunked[:, -1], torch.zeros(2)))
        self.assertEqual(float(chunked[0, 0].detach()), 0.0)
        chunked.sum().backward()
        chunked_gradient = hidden.grad.clone()
        reference = hidden.detach().clone().requires_grad_(True)
        logits = self.runtime.output_head(reference[:, :-1, :])
        direct = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 self.batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        direct.backward()
        torch.testing.assert_close(chunked.sum(), direct, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(chunked_gradient, reference.grad, rtol=1e-5, atol=1e-7)

    def test_suffix_matches_full_for_perturbed_and_clean_passes(self):
        target = list(self.runtime.adapters)[0]
        with torch.no_grad():
            clean, captured, replay = self.runtime.capture(target, self.batch)
            self.assertIsNotNone(replay)
            self.assertEqual(tuple(captured.shape), (2, 5, 16))
            torch.testing.assert_close(clean, self.runtime.losses(self.batch, replay))
            wrapper = self.runtime.adapters[target]
            wrapper.perturbation = torch.randn(2, 5, 8) * 0.1
            full = self.runtime.losses(self.batch)
            suffix = self.runtime.losses(self.batch, replay)
            torch.testing.assert_close(suffix, full, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(self.runtime.losses(self.batch, replay), suffix)
        self.assertGreater(self.runtime.suffix_forwards, 0)
        self.assertIn(target, self.runtime.metadata()["replay_verified_targets"])
        self.runtime.clear_interventions()

    def test_full_mode_skips_suffix_and_unsupported_strict_mode_fails(self):
        self.config.model.replay_mode = "full"
        _, _, replay = self.runtime.capture(next(iter(self.runtime.adapters)), self.batch)
        self.assertIsNone(replay)
        self.assertEqual(self.runtime.suffix_forwards, 0)
        self.config.model.replay_mode = "suffix"
        self.runtime.model.config.model_type = "unknown_architecture"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.runtime.capture(next(iter(self.runtime.adapters)), self.batch)

    def test_mismatched_suffix_falls_back_explicitly_and_strict_mode_refuses(self):
        target = next(iter(self.runtime.adapters))
        original = self.runtime.replay_hidden
        with patch.object(self.runtime, "replay_hidden", lambda cached: original(cached) + 0.1):
            # unittest.assertWarns traverses every loaded module's attributes;
            # Transformers' lazy optional imports can then require torchvision.
            with warnings.catch_warnings(record=True) as reported:
                warnings.simplefilter("always", RuntimeWarning)
                _, _, replay = self.runtime.capture(target, self.batch)
            self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                                and "full-forward probes" in str(item.message) for item in reported))
            self.assertIsNone(replay)
            self.assertEqual(self.runtime.metadata()["probe_execution"], "full_forward")
        runtime = load_runtime(self.config)
        runtime.config.model.replay_mode = "suffix"
        original = runtime.replay_hidden
        with patch.object(runtime, "replay_hidden", lambda cached: original(cached) + 0.1):
            with self.assertRaisesRegex(RuntimeError, "refusing to train"):
                runtime.capture(target, self.batch)


@unittest.skipIf(torch is None or importlib.util.find_spec("transformers") is None,
                 "Install transformers for native Qwen3.5 decoder verification")
class NativeQwenReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_runtime(self):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        torch.manual_seed(51)
        model_config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
        )
        model_config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(model_config).eval().requires_grad_(False)
        original_model = copy.deepcopy(model)
        config = config_for_tiny()
        config.model.name = "offline-tiny-qwen3.5"
        config.model.replay_mode = "suffix"
        config.adapter.last_n_layers = 3
        adapters = inject_lora(model, select_adapter_targets(model, config), rank=2, alpha=2.0)
        runtime = LocalRuntime(model, adapters, config, torch.device("cpu"), torch.float32)
        return runtime, original_model

    def batch(self):
        return collate([
            {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, 3, 4, 5, 6]},
            {"input_ids": [3, 2, 7, 8], "labels": [-100, -100, 7, 8]},
        ], "cpu")

    def test_mixed_deltanet_and_attention_suffix_matches_native_forward(self):
        runtime, _ = self.make_runtime()
        model, adapters = runtime.model, runtime.adapters
        batch = self.batch()
        with torch.no_grad():
            native = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"], use_cache=False)
            torch.testing.assert_close(runtime.losses(batch).sum(), native.loss, rtol=1e-5, atol=1e-6)
            for target, wrapper in adapters.items():
                clean, _, replay = runtime.capture(target, batch)
                self.assertIsNotNone(replay)
                torch.testing.assert_close(clean, runtime.losses(batch, replay), rtol=1e-5, atol=1e-6)
                wrapper.perturbation = torch.randn(2, 6, 32) * 0.07
                expected = runtime.losses(batch)
                for _ in range(2):
                    torch.testing.assert_close(runtime.losses(batch, replay), expected, rtol=1e-5, atol=1e-6)
                runtime.clear_interventions()
        self.assertEqual(set(adapters), runtime._verified_targets)

    @unittest.skipIf(importlib.util.find_spec("peft") is None, "Install PEFT for export compatibility test")
    def test_peft_export_reloads_with_matching_native_qwen_loss(self):
        from peft import PeftModel
        from dust_qlora.training import _write_adapter

        runtime, original_model = self.make_runtime()
        batch = self.batch()
        with torch.no_grad():
            for wrapper in runtime.adapters.values():
                wrapper.lora_B.normal_(std=0.07)
            expected = runtime.losses(batch).sum()
        with tempfile.TemporaryDirectory() as directory:
            _write_adapter(directory, runtime)
            loaded = PeftModel.from_pretrained(original_model, directory, is_trainable=False).eval()
            with torch.no_grad():
                actual = loaded(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                labels=batch["labels"], use_cache=False).loss
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_local_multimodal_checkpoint_loads_text_backbone_without_network(self):
        from pathlib import Path
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5VisionConfig

        _, original_text = self.make_runtime()
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                    out_hidden_size=32, patch_size=2, temporal_patch_size=2,
                                    spatial_merge_size=2, num_position_embeddings=16)
        model_config = Qwen3_5Config(text_config=original_text.config, vision_config=vision)
        original = Qwen3_5ForConditionalGeneration(model_config).eval()
        batch = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(Path(directory) / "base")
            config = config_for_tiny()
            config.model.name = str(Path(directory) / "base")
            config.model.local_files_only = True
            config.model.replay_mode = "suffix"
            config.model.attn_implementation = "eager"
            config.adapter.last_n_layers = 3
            runtime = load_runtime(config)
            self.assertTrue(all("language_model.layers." in name for name in runtime.adapters))
            self.assertTrue(all(not parameter.requires_grad for parameter in runtime.model.model.visual.parameters()))
            with torch.no_grad():
                expected = original(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                    labels=batch["labels"], use_cache=False).loss
                actual = runtime.losses(batch).sum()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                target = next(iter(runtime.adapters))
                clean, _, replay = runtime.capture(target, batch)
                self.assertIsNotNone(replay)
                torch.testing.assert_close(clean, runtime.losses(batch, replay), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
