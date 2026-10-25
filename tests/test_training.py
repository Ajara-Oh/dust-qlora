"""Offline end-to-end training, checkpoint identity, and resume equivalence."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from dust_qlora.adapters import adapter_state_dict
    from dust_qlora.config import TrainConfig, load_config
    from dust_qlora.runtime import load_runtime
    from dust_qlora.training import TokenDataset, evaluate, train


@unittest.skipIf(torch is None, "Install the training/test extra for offline trainer tests")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        for split, count in (("train", 5), ("validation", 2)):
            rows = []
            for index in range(count):
                tokens = [1, 2 + index, 3, 7, 8, 9][:5 + index % 2]
                rows.append({"id": f"{split}-{index}", "group_id": f"{split}-group-{index}",
                             "input_ids": tokens, "labels": [-100, -100] + tokens[2:]})
            (self.root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.config = TrainConfig()
        self.config.model.name = "tiny"
        self.config.model.device = "cpu"
        self.config.model.dtype = "float32"
        self.config.model.load_in_4bit = False
        self.config.model.tiny_vocab_size = 16
        self.config.model.tiny_hidden_size = 8
        self.config.model.tiny_num_layers = 2
        self.config.adapter.rank = 2
        self.config.adapter.alpha = 2.0
        self.config.adapter.last_n_layers = 2
        self.config.noise.pairs = 4
        self.config.noise.sigma = 0.1
        self.config.data.train = str(self.root / "train.jsonl")
        self.config.data.validation = str(self.root / "validation.jsonl")
        self.config.data.max_length = 16
        self.config.training.output_dir = str(self.root / "run")
        self.config.training.max_steps = 4
        self.config.training.learning_rate = 0.01
        self.config.training.gradient_accumulation_steps = 2
        self.config.training.loss_chunk_size = 2
        self.config.training.checkpoint_every = 2
        self.config.training.eval_every = 0
        self.config.training.eval_batches = 0

    def run_training(self, config, **kwargs):
        with redirect_stdout(io.StringIO()):
            return train(config, **kwargs)

    def test_noise_updates_only_adapters_and_writes_local_artifacts(self):
        torch.manual_seed(self.config.training.seed)
        runtime = load_runtime(self.config)
        before = {name: parameter.detach().clone() for name, parameter in runtime.model.named_parameters()}
        result = self.run_training(self.config, runtime=runtime)
        changed = []
        for name, parameter in runtime.model.named_parameters():
            if not torch.equal(parameter, before[name]):
                changed.append(name)
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in name for name in changed))
        self.assertTrue(any("lora_B" in name for name in changed))
        self.assertEqual(result["steps"], 4)
        self.assertIsNotNone(result["validation"]["loss"])
        output = Path(self.config.training.output_dir)
        self.assertTrue((output / "adapter" / "adapter_config.json").is_file())
        self.assertTrue((output / "checkpoint-000004" / "training_state.pt").is_file())
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 4)
        self.assertTrue(all(row["probe_replay"] == ["suffix", "suffix"] for row in metrics))
        self.assertTrue(all(row["full_forwards"] == 2 for row in metrics))
        self.assertTrue(all(row["suffix_forwards"] >= 16 for row in metrics))
        self.assertTrue(all(wrapper.perturbation is None for wrapper in runtime.adapters.values()))

    def test_resume_is_equivalent_to_uninterrupted_training(self):
        for method in ("noise", "backprop"):
            with self.subTest(method=method):
                full = copy.deepcopy(self.config)
                full.training.method = method
                full.training.output_dir = str(self.root / f"full-{method}")
                torch.manual_seed(full.training.seed)
                full_runtime = load_runtime(full)
                self.run_training(full, runtime=full_runtime)
                expected = adapter_state_dict(full_runtime.model)

                partial = copy.deepcopy(full)
                partial.training.max_steps = 2
                partial.training.output_dir = str(self.root / f"partial-{method}")
                result = self.run_training(partial)
                resumed = copy.deepcopy(partial)
                resumed.training.max_steps = 4
                torch.manual_seed(resumed.training.seed)
                runtime = load_runtime(resumed)
                self.run_training(resumed, resume=result["checkpoint"], runtime=runtime)
                actual = adapter_state_dict(runtime.model)
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_resume_rejects_changed_dataset_and_learning_semantics(self):
        self.config.training.max_steps = 2
        result = self.run_training(self.config)
        changed = copy.deepcopy(self.config)
        changed.training.max_steps = 3
        changed.training.learning_rate *= 2
        with self.assertRaisesRegex(ValueError, "config_fingerprint"):
            self.run_training(changed, resume=result["checkpoint"])
        changed = copy.deepcopy(self.config)
        changed.training.max_steps = 3
        data = Path(changed.data.train)
        data.write_text(data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
            self.run_training(changed, resume=result["checkpoint"])

    def test_tokenization_identity_and_manifest_are_checked(self):
        manifest = {
            "model_name": "different-model", "revision": None, "resolved_revision": None,
            "chat_template_sha256": "test-template", "template_kwargs": {},
            "splits": {split: {"sha256": hashlib.sha256((self.root / f"{split}.jsonl").read_bytes()).hexdigest()}
                       for split in ("train", "validation")},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "model_name"):
            self.run_training(self.config)
        manifest["model_name"] = "tiny"
        path.write_text(json.dumps(manifest))
        self.config.training.max_steps = 2
        result = self.run_training(self.config)
        manifest["chat_template_sha256"] = "changed-template"
        path.write_text(json.dumps(manifest))
        self.config.training.max_steps = 3
        with self.assertRaisesRegex(ValueError, "tokenization_fingerprint"):
            self.run_training(self.config, resume=result["checkpoint"])

    def test_validation_evaluates_base_and_saved_adapter(self):
        baseline = evaluate(self.config)
        result = self.run_training(self.config)
        trained = evaluate(self.config, checkpoint=result["adapter"])
        self.assertEqual(trained["dataset_sha256"], baseline["dataset_sha256"])
        self.assertEqual(trained["supervised_tokens"], baseline["supervised_tokens"])
        self.assertNotEqual(trained["loss"], baseline["loss"])
        self.assertIsNone(baseline["checkpoint"])
        wrong = copy.deepcopy(self.config)
        wrong.adapter.alpha *= 2
        with self.assertRaisesRegex(ValueError, "lora_alpha"):
            evaluate(wrong, checkpoint=result["adapter"])

    def test_data_rejects_truncation_wrong_labels_and_session_leakage(self):
        path = self.root / "bad.jsonl"
        path.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 3, 3]}) + "\n")
        with self.assertRaisesRegex(ValueError, "unshifted"):
            TokenDataset(path, max_length=8)
        path.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}) + "\n")
        with self.assertRaisesRegex(ValueError, "never silently truncate"):
            TokenDataset(path, max_length=2)
        self.config.data.validation = self.config.data.train
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.run_training(self.config)

    def test_config_rejects_unknown_keys_and_unsupported_probe(self):
        path = self.root / "bad.toml"
        path.write_text('[noise]\nparis = 8\n')
        with self.assertRaisesRegex(ValueError, "Unknown keys"):
            load_config(path)
        path.write_text('[noise]\nprobe = "gaussian"\n')
        with self.assertRaisesRegex(ValueError, "probe"):
            load_config(path)

    def test_overwrite_archives_old_run_and_refuses_unrecognized_contents(self):
        self.config.training.max_steps = 2
        self.run_training(self.config)
        self.config.training.overwrite = True
        result = self.run_training(self.config)
        previous = Path(result["previous_run"])
        self.assertTrue((previous / "checkpoint-000002" / "training_state.pt").exists())
        self.assertTrue(Path(result["checkpoint"]).exists())
        (Path(self.config.training.output_dir) / "unrecognized.txt").write_text("keep this file")
        with self.assertRaisesRegex(ValueError, "not a recognized training run"):
            self.run_training(self.config)

    def test_older_checkpoint_requires_new_output_when_later_progress_exists(self):
        self.run_training(self.config)
        old = str(Path(self.config.training.output_dir) / "checkpoint-000002")
        self.config.training.max_steps = 5
        with self.assertRaisesRegex(ValueError, "progress after checkpoint step 2"):
            self.run_training(self.config, resume=old)
        self.config.training.output_dir = str(self.root / "resumed-earlier")
        self.config.training.max_steps = 3
        result = self.run_training(self.config, resume=old)
        self.assertEqual(result["steps"], 3)
        # A crash can leave newer metric rows even without a completed checkpoint.
        output = Path(self.config.training.output_dir)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": 4}) + "\n")
        with self.assertRaisesRegex(ValueError, "progress after checkpoint step 3"):
            self.run_training(self.config, resume=result["checkpoint"])


if __name__ == "__main__":
    unittest.main()
