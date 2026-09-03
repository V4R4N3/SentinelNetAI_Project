from collections import deque
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

try:
    from .model_defs import (
        load_autoencoder_artifact,
        load_sequence_artifact,
        load_supervised_artifacts,
        normalized_anomaly_score,
        reconstruction_errors,
    )
    from .sentinel_utils import FEATURES, validate_flow_dataframe
except ImportError:
    from model_defs import (
        load_autoencoder_artifact,
        load_sequence_artifact,
        load_supervised_artifacts,
        normalized_anomaly_score,
        reconstruction_errors,
    )
    from sentinel_utils import FEATURES, validate_flow_dataframe


ATTACK_MAP = {
    'PortScan': ('Discovery', 'T1046 Network Service Discovery'),
    'DDoS': ('Impact', 'T1498 Network Denial of Service'),
    'BruteForce': ('Credential Access', 'T1110 Brute Force'),
    'Exfiltration': ('Exfiltration', 'T1041 Exfiltration Over C2 Channel'),
    'C2Beacon': ('Command and Control', 'T1071 Application Layer Protocol'),
    'Benign': ('None', 'None'),
}


@dataclass
class DetectionEvidence:
    predicted_threat: str
    supervised_confidence: float
    anomaly_score: float
    sequence_confidence: float
    heuristic_score: float
    model_probabilities: dict
    heuristic_probabilities: dict
    predicted_confidence: float = None


@dataclass
class RiskBreakdown:
    total: int
    components: dict


@dataclass
class DetectorBundle:
    supervised_model: object
    supervised_metadata: dict
    autoencoder: object
    anomaly_metadata: dict
    isolation_forest: object
    sequence_model: object = None
    sequence_metadata: dict = None
    history: deque = None

    def __post_init__(self):
        if self.history is None:
            window = self.sequence_metadata.get('window', 8) if self.sequence_metadata else 8
            self.history = deque(maxlen=window)

    @classmethod
    def load(cls, models_dir):
        models_dir = Path(models_dir)
        supervised_model, supervised_metadata = load_supervised_artifacts(
            models_dir / 'supervised_ids.pt', models_dir / 'preprocess.joblib'
        )
        autoencoder, anomaly_metadata = load_autoencoder_artifact(
            models_dir / 'autoencoder.pt', models_dir / 'anomaly_preprocess.joblib'
        )
        isolation_forest = joblib.load(models_dir / 'isolation_forest.joblib')
        sequence_model = None
        sequence_metadata = None
        sequence_path = models_dir / 'sequence_gru.pt'
        sequence_metadata_path = models_dir / 'sequence_preprocess.joblib'
        if sequence_path.exists() and sequence_metadata_path.exists():
            sequence_model, sequence_metadata = load_sequence_artifact(
                sequence_path, sequence_metadata_path
            )
        return cls(
            supervised_model=supervised_model,
            supervised_metadata=supervised_metadata,
            autoencoder=autoencoder,
            anomaly_metadata=anomaly_metadata,
            isolation_forest=isolation_forest,
            sequence_model=sequence_model,
            sequence_metadata=sequence_metadata,
        )


def _clamp(value):
    return float(np.clip(value, 0.0, 1.0))


def telemetry_heuristic(row):
    scores = {
        'Benign': 0.4,
        'PortScan': 0.02,
        'DDoS': 0.02,
        'BruteForce': 0.02,
        'Exfiltration': 0.02,
        'C2Beacon': 0.02,
    }
    if row['unique_dst_ports_5m'] > 30 or row['failed_conn_5m'] > 20:
        scores['PortScan'] += 0.7
    if row['packets_per_sec'] > 400 or row['burst_score'] > 0.75:
        scores['DDoS'] += 0.7
    if row['failed_conn_5m'] > 50 and row['dst_port'] in [22, 21, 3389, 445, 25]:
        scores['BruteForce'] += 0.75
    if row['bytes_out'] > 2_000_000 and row['outbound_ratio'] > 0.85:
        scores['Exfiltration'] += 0.8
    if row['beacon_score'] > 0.7:
        scores['C2Beacon'] += 0.8
    total = sum(scores.values())
    return {label: value / total for label, value in scores.items()}


def _probability_map(model, scaled_values, classes):
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(scaled_values, dtype=torch.float32)), dim=1
        )[0].numpy()
    return {label: float(probabilities[index]) for index, label in enumerate(classes)}


def _highest_threat(probabilities):
    threats = {label: score for label, score in probabilities.items() if label != 'Benign'}
    label = max(threats, key=threats.get)
    return label, float(threats[label])


def score_flow(row, bundle):
    raw_values = np.asarray([[float(row[feature]) for feature in FEATURES]], dtype=np.float32)
    supervised_scaled = bundle.supervised_metadata['scaler'].transform(raw_values)
    model_probabilities = _probability_map(
        bundle.supervised_model,
        supervised_scaled,
        bundle.supervised_metadata['classes'],
    )
    supervised_label, supervised_confidence = _highest_threat(model_probabilities)

    anomaly_scaled = bundle.anomaly_metadata['scaler'].transform(raw_values)
    error = reconstruction_errors(bundle.autoencoder, anomaly_scaled)[0]
    autoencoder_score = normalized_anomaly_score(
        error, bundle.anomaly_metadata['threshold']
    )
    isolation_score = 0.75 if bundle.isolation_forest.predict(anomaly_scaled)[0] == -1 else 0.0
    anomaly_score = max(autoencoder_score, isolation_score)

    heuristic_probabilities = telemetry_heuristic(row)
    heuristic_label, heuristic_score = _highest_threat(heuristic_probabilities)

    sequence_label = None
    sequence_confidence = 0.0
    if bundle.sequence_model is not None:
        window = bundle.sequence_metadata['window']
        if len(bundle.history) == window:
            history = np.asarray(bundle.history, dtype=np.float32)
            sequence_scaled = bundle.sequence_metadata['scaler'].transform(history)
            with torch.no_grad():
                sequence_probs = torch.softmax(
                    bundle.sequence_model(torch.tensor(sequence_scaled[None, :, :])), dim=1
                )[0].numpy()
            sequence_map = {
                label: float(sequence_probs[index])
                for index, label in enumerate(bundle.sequence_metadata['classes'])
            }
            sequence_label, sequence_confidence = _highest_threat(sequence_map)
        bundle.history.append(raw_values[0])

    candidates = [
        (supervised_confidence, supervised_label),
        (heuristic_score, heuristic_label),
    ]
    if sequence_label is not None:
        candidates.append((sequence_confidence, sequence_label))
    predicted_confidence, predicted_threat = max(candidates)
    return DetectionEvidence(
        predicted_threat=predicted_threat,
        supervised_confidence=supervised_confidence,
        anomaly_score=anomaly_score,
        sequence_confidence=sequence_confidence,
        heuristic_score=heuristic_score,
        model_probabilities=model_probabilities,
        heuristic_probabilities=heuristic_probabilities,
        predicted_confidence=predicted_confidence,
    )


def calculate_risk(evidence, row):
    components = {
        'supervised': round(45 * _clamp(evidence.supervised_confidence)),
        'anomaly': round(20 * _clamp(evidence.anomaly_score)),
        'sequence': round(10 * _clamp(evidence.sequence_confidence)),
        'telemetry': round(10 * _clamp(evidence.heuristic_score)),
        'sensor': round(5 * _clamp(float(row.get('suricata_alert_count', 0)) / 3.0)),
        'asset': round(10 * _clamp(float(row.get('asset_criticality', 1)) / 5.0)),
    }
    return RiskBreakdown(total=min(100, sum(components.values())), components=components)


def build_alert(row, evidence, risk):
    tactic, technique = ATTACK_MAP[evidence.predicted_threat]
    predicted_confidence = evidence.predicted_confidence
    if predicted_confidence is None:
        predicted_confidence = max(
            evidence.model_probabilities.get(evidence.predicted_threat, 0.0),
            evidence.heuristic_probabilities.get(evidence.predicted_threat, 0.0),
        )
    return {
        'timestamp': row['timestamp'],
        'src_ip': row['src_ip'],
        'dst_ip': row['dst_ip'],
        'predicted_threat': evidence.predicted_threat,
        'confidence': round(predicted_confidence, 4),
        'risk_score': risk.total,
        'risk_components': risk.components,
        'mitre_tactic': tactic,
        'mitre_technique': technique,
        'evidence': {
            'anomaly_score': round(evidence.anomaly_score, 4),
            'sequence_confidence': round(evidence.sequence_confidence, 4),
            'telemetry_score': round(evidence.heuristic_score, 4),
            'model_probabilities': {
                label: round(score, 4) for label, score in evidence.model_probabilities.items()
            },
            'dst_port': int(row['dst_port']),
            'bytes_out': int(row['bytes_out']),
            'packets_per_sec': float(row['packets_per_sec']),
            'unique_dst_ports_5m': int(row['unique_dst_ports_5m']),
            'failed_conn_5m': int(row['failed_conn_5m']),
            'beacon_score': float(row['beacon_score']),
        },
    }


def run_streaming_detector(input_path, models_dir, limit, medium_threshold):
    dataframe = pd.read_csv(input_path).head(limit)
    validate_flow_dataframe(dataframe)
    bundle = DetectorBundle.load(models_dir)
    alerts = []
    for _, row in dataframe.iterrows():
        evidence = score_flow(row, bundle)
        risk = calculate_risk(evidence, row)
        if risk.total >= medium_threshold:
            alerts.append(build_alert(row, evidence, risk))
    return len(dataframe), alerts
