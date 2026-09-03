import numpy as np
import pytest

from scripts import sentinel_utils


FEATURES = sentinel_utils.FEATURES


def test_validate_rejects_missing_feature(sample_df):
    with pytest.raises(ValueError, match="missing required columns"):
        sentinel_utils.validate_flow_dataframe(sample_df.drop(columns=[FEATURES[0]]))


def test_validate_rejects_non_finite_feature(sample_df):
    sample_df.loc[0, FEATURES[0]] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        sentinel_utils.validate_flow_dataframe(sample_df)


def test_scaler_is_fit_on_training_rows_only(flow_csv):
    bundle = sentinel_utils.prepare_split(str(flow_csv), test_size=0.25, seed=42)
    raw_train = bundle.dataframe.loc[bundle.train_indices, FEATURES].to_numpy()

    np.testing.assert_allclose(bundle.scaler.mean_, raw_train.mean(axis=0), rtol=1e-5)
    assert set(bundle.train_indices).isdisjoint(bundle.test_indices)


def test_prepare_split_is_reproducible(flow_csv):
    first = sentinel_utils.prepare_split(str(flow_csv), test_size=0.25, seed=42)
    second = sentinel_utils.prepare_split(str(flow_csv), test_size=0.25, seed=42)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
