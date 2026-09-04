#!/usr/bin/env python3
import argparse
import datetime
import json
from pathlib import Path


def load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def load_alerts(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metric(value):
    return 'not available' if value is None else f'{value:.4f}'


def _false_positive_summary(supervised):
    matrix = supervised.get('confusion_matrix', [])
    classes = supervised.get('classes', [])
    if not matrix or len(matrix) != len(classes):
        return ['- Confusion-matrix evidence is not available.']
    lines = []
    for column, label in enumerate(classes):
        false_positives = sum(matrix[row][column] for row in range(len(matrix)) if row != column)
        lines.append(f'- {label}: {false_positives} false-positive predictions in the held-out test set.')
    return lines


def generate_incident_report(outputs_dir):
    outputs = Path(outputs_dir)
    supervised = load_json(outputs / 'supervised_metrics.json', {})
    anomaly = load_json(outputs / 'anomaly_metrics.json', {})
    sequence = load_json(outputs / 'sequence_metrics.json', {})
    fusion = load_json(outputs / 'fusion_summary.json', {'alerts': 0, 'top_alerts': []})
    alerts = load_alerts(outputs / 'stream_alerts.jsonl')
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()

    lines = [
        '# SentinelNet AI Defense Fabric - Incident and Model Report',
        '',
        f'Generated: {generated}',
        '',
        '## Executive Summary',
        '',
        f"SentinelNet processed network-flow telemetry with supervised, anomaly, sequence, and rule-based evidence. The streaming stage produced {fusion.get('alerts', len(alerts))} prioritized alerts for analyst review. All response actions are recommendations only.",
        '',
        '## Model Comparison',
        '',
        '| Model | Macro F1 | Weighted F1 | Role |',
        '|---|---:|---:|---|',
        f"| Residual MLP | {_metric(supervised.get('macro_f1'))} | {_metric(supervised.get('weighted_f1'))} | Multiclass flow classification |",
        f"| Autoencoder | {_metric(anomaly.get('autoencoder_report', {}).get('macro avg', {}).get('f1-score'))} | {_metric(anomaly.get('autoencoder_report', {}).get('weighted avg', {}).get('f1-score'))} | Reconstruction anomaly detection |",
        f"| Isolation Forest | {_metric(anomaly.get('isolation_forest_report', {}).get('macro avg', {}).get('f1-score'))} | {_metric(anomaly.get('isolation_forest_report', {}).get('weighted avg', {}).get('f1-score'))} | Tree-based anomaly detection |",
        f"| Sequence GRU | {_metric(sequence.get('macro_f1'))} | {_metric(sequence.get('weighted_f1'))} | Next-flow temporal classification |",
        '',
        f"The autoencoder threshold was {anomaly.get('autoencoder_threshold', 'not available')} at the {anomaly.get('threshold_percentile', 'not available')}th percentile of held-out benign training reconstruction errors.",
        '',
        '## Alert Summary',
        '',
        f"- Processed flows: {fusion.get('processed', 'not available')}",
        f"- Alerts at or above the medium threshold: {fusion.get('alerts', len(alerts))}",
        f"- Medium risk threshold: {fusion.get('medium_threshold', 'not available')}",
        '',
        '## Alert Investigations',
        '',
    ]
    selected = fusion.get('top_alerts', alerts)[:3]
    if not selected:
        lines.append('- No alerts were available for investigation.')
    for index, alert in enumerate(selected, start=1):
        components = ', '.join(
            f'{name}={value}' for name, value in alert.get('risk_components', {}).items()
        )
        lines.extend([
            f"### Alert {index}: {alert['predicted_threat']}",
            '',
            f"- Flow: {alert['src_ip']} -> {alert['dst_ip']} at {alert['timestamp']}",
            f"- Risk: {alert['risk_score']}/100 ({components})",
            f"- ATT&CK: {alert['mitre_technique']}",
            '- Analyst decision: validate the raw telemetry and sensor context before containment.',
            '',
        ])
    lines.extend([
        '## False Positives',
        '',
        'False positives are legitimate flows incorrectly assigned to an attack class. The held-out supervised confusion matrix gives the following review counts:',
        '',
        *_false_positive_summary(supervised),
        '',
        'Synthetic class patterns are intentionally distinct, so very high supervised scores may overstate performance on real networks. Thresholds must be revalidated on authorized operational telemetry.',
        '',
        '## Analyst Recommendations',
        '',
        '- Validate high-risk alerts against raw Zeek or Suricata context before containment.',
        '- Review risk components instead of treating a single model probability as operational risk.',
        '- Track false positives and data drift before changing thresholds.',
        '- Keep every response dry-run until an authorized analyst approves the action.',
        '',
        '## Limitations',
        '',
        '- The model was trained in a controlled lab and needs validation on real authorized telemetry.',
        '- The response simulator is not a production SOAR system.',
        '- High confidence does not remove the need for analyst review.',
        '- Dataset quality and label quality directly affect model reliability.',
        '- Synthetic flow labels are sampled independently, so next-flow GRU performance may remain close to the majority-class baseline.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', default='outputs')
    args = parser.parse_args()
    outputs = Path(args.outputs_dir)
    outputs.mkdir(exist_ok=True)
    report_path = outputs / 'incident_report.md'
    report_path.write_text(generate_incident_report(outputs))
    print(f'[+] wrote {report_path}')


if __name__ == '__main__':
    main()
