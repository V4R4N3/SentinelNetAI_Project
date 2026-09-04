import importlib
from pathlib import Path


def test_pipeline_dry_run_has_required_order():
    pipeline = importlib.import_module("scripts.09_run_pipeline")

    commands = pipeline.build_pipeline_commands(profile="final", seed=42)
    scripts = [command[1] for command in commands]

    assert scripts == [
        "scripts/00_generate_synthetic_network_data.py",
        "scripts/01_profile_dataset.py",
        "scripts/02_train_supervised_ids.py",
        "scripts/03_train_autoencoder_anomaly.py",
        "scripts/04_train_sequence_gru.py",
        "scripts/05_run_streaming_detector.py",
        "scripts/06_generate_incident_report.py",
        "scripts/07_response_simulator.py",
        "scripts/08_validate_deliverables.py",
    ]


def test_quick_profile_reduces_training_work():
    pipeline = importlib.import_module("scripts.09_run_pipeline")

    quick = pipeline.build_pipeline_commands(profile="quick", seed=7)
    final = pipeline.build_pipeline_commands(profile="final", seed=7)

    assert int(quick[0][3]) < int(final[0][3])
    assert int(quick[2][3]) < int(final[2][3])
    assert quick[0][-1] == final[0][-1] == "7"


def test_execution_log_ends_with_one_newline(tmp_path):
    pipeline = importlib.import_module("scripts.09_run_pipeline")
    path = tmp_path / "execution_log.txt"
    path.write_text("stage complete\n\n")

    pipeline.normalize_log_ending(path)

    assert path.read_bytes().endswith(b"stage complete\n")
    assert not path.read_bytes().endswith(b"\n\n")
