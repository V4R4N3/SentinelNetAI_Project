import importlib

import pandas as pd

from scripts.sentinel_utils import FEATURES


generator = importlib.import_module("scripts.00_generate_synthetic_network_data")
profiler = importlib.import_module("scripts.01_profile_dataset")


def test_generate_dataset_is_deterministic(tmp_path):
    first = generator.generate_dataset(100, tmp_path / "a.csv", seed=42)
    second = generator.generate_dataset(100, tmp_path / "b.csv", seed=42)

    assert first.read_bytes() == second.read_bytes()


def test_different_seed_changes_generated_dataset(tmp_path):
    first = generator.generate_dataset(100, tmp_path / "a.csv", seed=42)
    second = generator.generate_dataset(100, tmp_path / "b.csv", seed=43)

    assert first.read_bytes() != second.read_bytes()


def test_generated_dataset_has_expected_schema(tmp_path):
    path = generator.generate_dataset(60, tmp_path / "flows.csv", seed=42)
    dataframe = pd.read_csv(path)

    assert len(dataframe) == 60
    assert set(FEATURES).issubset(dataframe.columns)
    assert set(dataframe["label"]).issubset(set(generator.LABELS))
    assert not dataframe.isna().any().any()


def test_profile_dataset_reports_rows_labels_and_missing_values(tmp_path):
    path = generator.generate_dataset(60, tmp_path / "flows.csv", seed=42)

    report = profiler.profile_dataset(path)

    assert report["rows"] == 60
    assert sum(report["label_counts"].values()) == 60
    assert all(count == 0 for count in report["missing_values"].values())
