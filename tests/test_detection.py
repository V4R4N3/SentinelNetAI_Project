import importlib


def example_row():
    return {
        "timestamp": "2026-01-01T00:00:00",
        "src_ip": "10.10.1.5",
        "dst_ip": "10.10.1.10",
        "dst_port": 22,
        "bytes_out": 5000,
        "packets_per_sec": 20.0,
        "unique_dst_ports_5m": 2,
        "failed_conn_5m": 60,
        "beacon_score": 0.1,
        "suricata_alert_count": 3,
        "asset_criticality": 5,
    }


def example_evidence(detection):
    return detection.DetectionEvidence(
        predicted_threat="BruteForce",
        supervised_confidence=0.8,
        anomaly_score=0.5,
        sequence_confidence=0.4,
        heuristic_score=0.7,
        model_probabilities={"Benign": 0.2, "BruteForce": 0.8},
        heuristic_probabilities={"Benign": 0.3, "BruteForce": 0.7},
    )


def test_risk_is_bounded_and_decomposable():
    detection = importlib.import_module("scripts.detection")

    result = detection.calculate_risk(example_evidence(detection), example_row())

    assert result.total == 72
    assert result.total == sum(result.components.values())
    assert set(result.components) == {
        "supervised",
        "anomaly",
        "sequence",
        "telemetry",
        "sensor",
        "asset",
    }


def test_risk_clamps_out_of_range_inputs():
    detection = importlib.import_module("scripts.detection")
    evidence = example_evidence(detection)
    evidence.supervised_confidence = 5.0
    evidence.anomaly_score = -1.0
    row = example_row()
    row["suricata_alert_count"] = 100
    row["asset_criticality"] = 100

    result = detection.calculate_risk(evidence, row)

    assert 0 <= result.total <= 100


def test_alert_contains_analyst_evidence():
    detection = importlib.import_module("scripts.detection")

    alert = detection.build_alert(
        example_row(),
        example_evidence(detection),
        detection.calculate_risk(example_evidence(detection), example_row()),
    )

    assert {
        "confidence",
        "risk_score",
        "mitre_technique",
        "evidence",
        "risk_components",
    } <= alert.keys()
    assert alert["risk_score"] == sum(alert["risk_components"].values())
    assert alert["mitre_technique"] == "T1110 Brute Force"
