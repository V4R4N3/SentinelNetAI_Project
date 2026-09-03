import importlib
import json


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_validator_reports_missing_required_artifact(tmp_path):
    validator = importlib.import_module("scripts.08_validate_deliverables")

    errors = validator.validate_deliverables(tmp_path)

    assert any("supervised_metrics.json" in error for error in errors)


def test_incident_report_contains_required_sections(tmp_path):
    reporting = importlib.import_module("scripts.06_generate_incident_report")
    outputs = tmp_path / "outputs"
    write_json(
        outputs / "supervised_metrics.json",
        {
            "macro_f1": 0.91,
            "weighted_f1": 0.94,
            "classes": ["Benign", "DDoS"],
            "confusion_matrix": [[9, 1], [2, 8]],
        },
    )
    write_json(
        outputs / "anomaly_metrics.json",
        {
            "autoencoder_threshold": 0.5,
            "threshold_percentile": 95,
            "autoencoder_report": {"1": {"f1-score": 0.8}},
            "isolation_forest_report": {"1": {"f1-score": 0.75}},
        },
    )
    write_json(
        outputs / "sequence_metrics.json",
        {"macro_f1": 0.2, "weighted_f1": 0.5, "model_type": "PyTorch GRU"},
    )
    alert = {
        "timestamp": "2026-01-01T00:00:00",
        "src_ip": "10.10.1.5",
        "dst_ip": "10.10.1.10",
        "predicted_threat": "DDoS",
        "risk_score": 88,
        "mitre_technique": "T1498 Network Denial of Service",
        "risk_components": {"supervised": 40, "anomaly": 20, "sequence": 8, "telemetry": 8, "sensor": 4, "asset": 8},
    }
    write_json(outputs / "fusion_summary.json", {"alerts": 1, "top_alerts": [alert]})
    (outputs / "stream_alerts.jsonl").write_text(json.dumps(alert) + "\n")

    report = reporting.generate_incident_report(outputs)

    for heading in ["Model Comparison", "Alert Investigations", "False Positives", "Limitations"]:
        assert heading in report
