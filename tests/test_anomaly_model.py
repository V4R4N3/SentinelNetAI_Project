import importlib

import numpy as np
import pytest
import torch

from scripts.sentinel_utils import FEATURES


def test_threshold_uses_requested_benign_percentile():
    model_defs = importlib.import_module("scripts.model_defs")
    errors = np.array([1.0, 2.0, 3.0, 4.0])

    assert model_defs.select_benign_threshold(errors, 75.0) == pytest.approx(3.25)


def test_anomaly_score_increases_above_threshold():
    model_defs = importlib.import_module("scripts.model_defs")

    below = model_defs.normalized_anomaly_score(0.5, threshold=1.0)
    above = model_defs.normalized_anomaly_score(2.0, threshold=1.0)

    assert 0 <= below < above <= 1


def test_autoencoder_artifact_round_trip(tmp_path):
    model_defs = importlib.import_module("scripts.model_defs")
    torch.manual_seed(42)
    model = model_defs.AutoEncoder(len(FEATURES))
    model.eval()
    values = torch.arange(2 * len(FEATURES), dtype=torch.float32).reshape(2, -1)
    expected = model(values)
    model_path = tmp_path / "autoencoder.pt"
    metadata_path = tmp_path / "anomaly_preprocess.joblib"

    model_defs.save_autoencoder_artifact(
        model,
        {"features": FEATURES, "threshold": 1.25, "threshold_percentile": 95.0},
        model_path,
        metadata_path,
    )
    loaded, metadata = model_defs.load_autoencoder_artifact(model_path, metadata_path)

    torch.testing.assert_close(loaded(values), expected)
    assert metadata["threshold"] == 1.25
