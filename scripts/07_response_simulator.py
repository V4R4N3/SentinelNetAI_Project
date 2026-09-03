#!/usr/bin/env python3
import argparse
import ipaddress
import json
from pathlib import Path


LAB_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')
)


def is_lab_private(address):
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(parsed in network for network in LAB_NETWORKS)


def recommend_action(alert):
    risk = int(alert['risk_score'])
    source_is_lab_private = is_lab_private(alert['src_ip'])
    if risk >= 85 and source_is_lab_private:
        action = 'recommend_isolate_lab_host'
        reason = 'High fused risk on an RFC1918 lab source; analyst validation required before isolation.'
    elif risk >= 70:
        action = 'escalate_to_tier2'
        reason = 'Elevated fused risk requires deeper telemetry review by a Tier 2 analyst.'
    else:
        action = 'monitor'
        reason = 'Risk does not justify containment; retain evidence and monitor for recurrence.'
    if risk >= 85 and not source_is_lab_private:
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
        'reason': reason,
    }


def build_response_plan(alerts):
    selected = sorted(alerts, key=lambda item: item['risk_score'], reverse=True)[:20]
    return {
        'safety': 'Dry-run recommendations only. Do not execute on production networks.',
        'actions': [recommend_action(alert) for alert in selected],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/stream_alerts.jsonl')
    parser.add_argument('--out', default='outputs/response_plan_lab_only.json')
    args = parser.parse_args()

    input_path = Path(args.input)
    alerts = []
    if input_path.exists():
        alerts = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    plan = build_response_plan(alerts)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2))
    print(f'[+] wrote {output_path}')


if __name__ == '__main__':
    main()
