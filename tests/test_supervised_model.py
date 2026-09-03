import importlib
from pathlib import Path
import subprocess
import sys

import torch

from scripts.sentinel_utils import FEATURES, LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_supervised_artifacts_round_trip(tmp_path):
    model_defs = importlib.import_module("scripts.model_defs")
    model = model_defs.IDSNet(len(FEATURES), len(LABELS), hidden=16, dropout=0.0)
    model_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "preprocess.joblib"
    metadata = {"features": FEATURES, "classes": LABELS}

    model_defs.save_supervised_artifacts(model, metadata, model_path, metadata_path)
    loaded, loaded_metadata = model_defs.load_supervised_artifacts(model_path, metadata_path)

    assert loaded(torch.zeros(2, len(FEATURES))).shape == (2, len(LABELS))
    assert loaded_metadata["features"] == FEATURES
    assert loaded_metadata["classes"] == LABELS


def test_loaded_supervised_model_preserves_predictions(tmp_path):
    model_defs = importlib.import_module("scripts.model_defs")
    torch.manual_seed(42)
    model = model_defs.IDSNet(len(FEATURES), len(LABELS), hidden=16, dropout=0.0)
    model.eval()
    values = torch.arange(2 * len(FEATURES), dtype=torch.float32).reshape(2, -1)
    expected = model(values)
    model_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "preprocess.joblib"

    model_defs.save_supervised_artifacts(
        model,
        {"features": FEATURES, "classes": LABELS},
        model_path,
        metadata_path,
    )
    loaded, _ = model_defs.load_supervised_artifacts(model_path, metadata_path)

    torch.testing.assert_close(loaded(values), expected)


def test_supervised_training_cli_completes(tmp_path):
    generator = importlib.import_module("scripts.00_generate_synthetic_network_data")
    generator.generate_dataset(600, tmp_path / "data" / "synthetic_flows.csv", seed=42)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "02_train_supervised_ids.py"),
            "--epochs",
            "0",
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models" / "supervised_ids.pt").stat().st_size > 0
