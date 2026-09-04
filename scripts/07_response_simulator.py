#!/usr/bin/env python3
import argparse
import ipaddress
import json
from pathlib import Path


LAB_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')
)


def parse_address(address):
    try:
        return ipaddress.ip_address(address)
    except ValueError:
        return None


def is_lab_private(address):
    parsed = parse_address(address)
    if parsed is None:
        return False
    return any(parsed in network for network in LAB_NETWORKS)


def recommend_action(alert, medium_threshold, high_threshold):
    risk = int(alert['risk_score'])
    source = parse_address(alert['src_ip'])
    destination = parse_address(alert['dst_ip'])
    source_is_lab_private = source is not None and any(source in network for network in LAB_NETWORKS)
    destination_is_valid = destination is not None
    if source is None or destination is None:
        action = 'escalate_to_tier2'
        reason = 'Malformed source or destination address requires analyst data-quality review.'
    elif risk >= high_threshold and source_is_lab_private:
        action = 'recommend_isolate_lab_host'
        reason = 'High fused risk on an RFC1918 lab source; analyst validation required before isolation.'
    elif risk >= medium_threshold:
        action = 'escalate_to_tier2'
        reason = 'Elevated fused risk requires deeper telemetry review by a Tier 2 analyst.'
    else:
        action = 'monitor'
        reason = 'Risk does not justify containment; retain evidence and monitor for recurrence.'
    if risk >= high_threshold and source is not None and not source_is_lab_private:
        action = 'escalate_to_tier2'
        reason = 'Source is outside approved RFC1918 lab ranges; no isolation is recommended.'
    return {
        'src_ip': alert['src_ip'],
        'dst_ip': alert['dst_ip'],
        'threat': alert['predicted_threat'],
        'risk_score': risk,
        'recommended_action': action,
        'mode': 'dry-run',
        'human_approval_required': True,
        'source_is_lab_private': source_is_lab_private,
        'destination_is_valid': destination_is_valid,
        'reason': reason,
    }


def build_response_plan(alerts, medium_threshold, high_threshold):
    selected = sorted(alerts, key=lambda item: item['risk_score'], reverse=True)[:20]
    return {
        'safety': 'Dry-run recommendations only. Do not execute on production networks.',
        'thresholds': {'medium': medium_threshold, 'high': high_threshold},
        'actions': [
            recommend_action(alert, medium_threshold, high_threshold) for alert in selected
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/stream_alerts.jsonl')
    parser.add_argument('--out', default='outputs/response_plan_lab_only.json')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    input_path = Path(args.input)
    alerts = []
    if input_path.exists():
        alerts = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    config = json.loads(Path(args.config).read_text())
    medium_threshold = round(100 * config['thresholds']['medium'])
    high_threshold = round(100 * config['thresholds']['high'])
    plan = build_response_plan(alerts, medium_threshold, high_threshold)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2))
    print(f'[+] wrote {output_path}')


if __name__ == '__main__':
    main()
