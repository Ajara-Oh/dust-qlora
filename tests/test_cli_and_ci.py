from pathlib import Path
import json

import yaml

from dust_qlora.cli import main


def test_workflows_cannot_trigger_on_push_or_pull_request():
    workflows = Path(__file__).resolve().parents[1] / ".github/workflows"
    assert list(workflows.glob("*.yml"))
    for path in workflows.glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        assert set(workflow["on"]) == {"workflow_dispatch"}
        assert workflow["permissions"] == {"contents": "read"}


def test_offline_cli_smoke_creates_real_adapter_and_checkpoint(tmp_path):
    path = tmp_path / "smoke"
    assert main(["smoke", "--output-dir", str(path), "--steps", "2"]) == 0
    summary = json.loads((path / "run/summary.json").read_text())
    assert summary["steps"] == 2
    assert Path(summary["checkpoint"]).is_dir()
    assert (Path(summary["adapter"]) / "adapter_model.safetensors").is_file()
    assert summary["validation"]["supervised_tokens"] > 0


def test_prepare_cli_accepts_both_synthetic_sources(tmp_path):
    fixtures = Path(__file__).parent / "fixtures/transcripts"
    dest = tmp_path / "prepared"
    assert main(["prepare", "--input", str(fixtures), "--output-dir", str(dest),
                 "--validation-fraction", "0"]) == 0
    rows = [json.loads(line) for line in (dest / "train.jsonl").read_text().splitlines()]
    assert {row["source"] for row in rows} == {"pi", "claude"}
