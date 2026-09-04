#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from detection import run_streaming_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/synthetic_flows.csv')
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    medium_threshold = round(100 * config['thresholds']['medium'])
    processed, alerts = run_streaming_detector(
        args.input,
        args.models_dir,
        args.limit,
        medium_threshold,
        config['fusion_weights'],
    )
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    alert_path = output_dir / 'stream_alerts.jsonl'
    with alert_path.open('w') as stream:
        for alert in alerts:
            stream.write(json.dumps(alert) + '\n')
    summary = {
        'processed': processed,
        'alerts': len(alerts),
        'medium_threshold': medium_threshold,
        'risk_formula': config['fusion_weights'],
        'top_alerts': sorted(alerts, key=lambda item: item['risk_score'], reverse=True)[:10],
    }
    (output_dir / 'fusion_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[+] processed={processed} alerts={len(alerts)} output={alert_path}')


if __name__ == '__main__':
    main()
