#!/usr/bin/env python3
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detection import DetectorBundle
from sentinel_utils import FEATURES


REQUIRED_OUTPUTS = (
    'data_profile.json',
    'supervised_metrics.json',
    'anomaly_metrics.json',
    'sequence_metrics.json',
    'stream_alerts.jsonl',
    'fusion_summary.json',
    'incident_report.md',
    'response_plan_lab_only.json',
)
REQUIRED_MODELS = (
    'supervised_ids.pt',
    'preprocess.joblib',
    'autoencoder.pt',
    'anomaly_preprocess.joblib',
    'isolation_forest.joblib',
    'sequence_gru.pt',
    'sequence_preprocess.joblib',
)


def _load_json(path, errors):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f'{path}: invalid JSON ({exc})')
        return None


def _validate_model_artifacts(models, errors):
    if any(not (models / name).exists() for name in REQUIRED_MODELS):
        return
    try:
        bundle = DetectorBundle.load(models)
        expected_features = list(FEATURES)
        metadata_items = (
            ('supervised', bundle.supervised_metadata),
            ('anomaly', bundle.anomaly_metadata),
            ('sequence', bundle.sequence_metadata),
        )
        for name, metadata in metadata_items:
            if metadata.get('features') != expected_features:
                raise ValueError(f'{name} feature order does not match detector features')
            scaler = metadata.get('scaler')
            if getattr(scaler, 'n_features_in_', None) != len(expected_features):
                raise ValueError(f'{name} scaler dimension is incompatible')
        if bundle.supervised_metadata.get('classes') != bundle.sequence_metadata.get('classes'):
            raise ValueError('supervised and sequence class orders differ')
        if bundle.supervised_model.config.get('in_dim') != len(expected_features):
            raise ValueError('supervised model input dimension is incompatible')
        if bundle.autoencoder.config.get('input_dim') != len(expected_features):
            raise ValueError('autoencoder input dimension is incompatible')
        if bundle.sequence_model.config.get('input_dim') != len(expected_features):
            raise ValueError('sequence model input dimension is incompatible')
        if float(bundle.anomaly_metadata.get('threshold', 0)) <= 0:
            raise ValueError('anomaly threshold must be positive')
        if getattr(bundle.isolation_forest, 'n_features_in_', None) != len(expected_features):
            raise ValueError('Isolation Forest input dimension is incompatible')
    except Exception as exc:
        errors.append(f'{models}: model artifacts are not loadable or compatible ({exc})')


def validate_deliverables(project_root):
    root = Path(project_root)
    outputs = root / 'outputs'
    models = root / 'models'
    errors = []
    for name in REQUIRED_OUTPUTS:
        path = outputs / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f'missing or empty required output: {path}')
    for name in REQUIRED_MODELS:
        path = models / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f'missing or empty required model: {path}')
    _validate_model_artifacts(models, errors)

    metric_requirements = {
        'supervised_metrics.json': {'macro_f1', 'weighted_f1', 'confusion_matrix'},
        'anomaly_metrics.json': {'autoencoder_threshold', 'autoencoder_report', 'isolation_forest_report'},
        'sequence_metrics.json': {'macro_f1', 'weighted_f1', 'model_type'},
    }
    for name, keys in metric_requirements.items():
        path = outputs / name
        if path.exists() and path.stat().st_size:
            value = _load_json(path, errors)
            if value is not None:
                missing = sorted(keys - set(value))
                if missing:
                    errors.append(f'{path}: missing keys {missing}')

    alerts = []
    alert_path = outputs / 'stream_alerts.jsonl'
    if alert_path.exists() and alert_path.stat().st_size:
        for line_number, line in enumerate(alert_path.read_text().splitlines(), start=1):
            try:
                alert = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f'{alert_path}:{line_number}: invalid JSON ({exc})')
                continue
            required = {'risk_score', 'risk_components', 'predicted_threat', 'mitre_technique'}
            if required - set(alert):
                errors.append(f'{alert_path}:{line_number}: incomplete alert contract')
            elif alert['risk_score'] != sum(alert['risk_components'].values()):
                errors.append(f'{alert_path}:{line_number}: risk components do not sum to total')
            alerts.append(alert)
        if not alerts:
            errors.append(f'{alert_path}: no alerts available for analyst investigation')

    summary_path = outputs / 'fusion_summary.json'
    if summary_path.exists() and summary_path.stat().st_size:
        summary = _load_json(summary_path, errors)
        if summary is not None and summary.get('alerts') != len(alerts):
            errors.append(f'{summary_path}: alert count does not match stream_alerts.jsonl')

    response_path = outputs / 'response_plan_lab_only.json'
    if response_path.exists() and response_path.stat().st_size:
        response = _load_json(response_path, errors)
        if response is not None:
            for index, action in enumerate(response.get('actions', []), start=1):
                if action.get('mode') != 'dry-run' or action.get('human_approval_required') is not True:
                    errors.append(f'{response_path}: action {index} violates dry-run approval contract')

    report_path = outputs / 'incident_report.md'
    if report_path.exists() and report_path.stat().st_size:
        report = report_path.read_text()
        for heading in ('Model Comparison', 'Alert Investigations', 'False Positives', 'Limitations'):
            if heading not in report:
                errors.append(f'{report_path}: missing section {heading}')
    return errors


def main():
    errors = validate_deliverables(Path.cwd())
    if errors:
        print('[!] deliverable validation failed')
        for error in errors:
            print(f'- {error}')
        raise SystemExit(1)
    print('[+] all required deliverables are present and internally consistent')


if __name__ == '__main__':
    main()
