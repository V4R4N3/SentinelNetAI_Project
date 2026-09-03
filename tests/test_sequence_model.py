import importlib

import numpy as np
import torch

from scripts.sentinel_utils import FEATURES, LABELS


def test_sequence_windows_preserve_temporal_shape():
    model_defs = importlib.import_module("scripts.model_defs")
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    labels = np.arange(10)

    sequences, targets = model_defs.make_sequences(
        values,
        labels,
        timestamps=np.arange(10),
        window=4,
    )

    assert sequences.shape == (6, 4, 6)
    np.testing.assert_array_equal(sequences[0], values[:4])
    np.testing.assert_array_equal(targets, labels[4:])


def test_gru_forward_shape():
    model_defs = importlib.import_module("scripts.model_defs")
    model = model_defs.SequenceGRU(input_dim=len(FEATURES), hidden_dim=16, classes=len(LABELS))

    assert model(torch.zeros(3, 8, len(FEATURES))).shape == (3, len(LABELS))


def test_sequence_artifact_round_trip(tmp_path):
    model_defs = importlib.import_module("scripts.model_defs")
    torch.manual_seed(42)
    model = model_defs.SequenceGRU(input_dim=len(FEATURES), hidden_dim=16, classes=len(LABELS))
    model.eval()
    values = torch.arange(3 * 8 * len(FEATURES), dtype=torch.float32).reshape(3, 8, -1)
    expected = model(values)
    model_path = tmp_path / "sequence_gru.pt"
    metadata_path = tmp_path / "sequence_preprocess.joblib"

    model_defs.save_sequence_artifact(
        model,
        {"features": FEATURES, "classes": LABELS, "window": 8},
        model_path,
        metadata_path,
    )
    loaded, metadata = model_defs.load_sequence_artifact(model_path, metadata_path)

    torch.testing.assert_close(loaded(values), expected)
    assert metadata["window"] == 8
