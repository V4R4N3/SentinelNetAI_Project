from pathlib import Path
import importlib
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def flow_csv(tmp_path):
    generator = importlib.import_module("scripts.00_generate_synthetic_network_data")
    return generator.generate_dataset(600, tmp_path / "flows.csv", seed=42)


@pytest.fixture
def sample_df(flow_csv):
    return pd.read_csv(flow_csv)
