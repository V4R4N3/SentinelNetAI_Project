# SentinelNet AI Defense Fabric - Incident and Model Report

Generated: 2026-09-04T09:26:48.469087+00:00

## Executive Summary

SentinelNet processed network-flow telemetry with supervised, anomaly, sequence, and rule-based evidence. The streaming stage produced 384 prioritized alerts for analyst review. All response actions are recommendations only.

## Model Comparison

| Model | Macro F1 | Weighted F1 | Role |
|---|---:|---:|---|
| Residual MLP | 0.9987 | 0.9993 | Multiclass flow classification |
| Autoencoder | 0.9595 | 0.9613 | Reconstruction anomaly detection |
| Isolation Forest | 0.7942 | 0.8112 | Tree-based anomaly detection |
| Sequence GRU | 0.1223 | 0.1795 | Next-flow temporal classification |

The autoencoder threshold was 0.054191410541534424 at the 95.0th percentile of held-out benign training reconstruction errors.

## Alert Summary

- Processed flows: 1000
- Alerts at or above the medium threshold: 384
- Medium risk threshold: 65

## Alert Investigations

### Alert 1: DDoS

- Flow: 10.10.4.137 -> 10.10.7.146 at 2026-01-01T00:04:20
- Risk: 88/100 (supervised=45, anomaly=20, sequence=2, telemetry=6, sensor=5, asset=10)
- ATT&CK: T1498 Network Denial of Service
- Analyst decision: validate the raw telemetry and sensor context before containment.

### Alert 2: DDoS

- Flow: 10.10.2.189 -> 10.10.8.164 at 2026-01-01T00:12:06
- Risk: 88/100 (supervised=45, anomaly=20, sequence=2, telemetry=6, sensor=5, asset=10)
- ATT&CK: T1498 Network Denial of Service
- Analyst decision: validate the raw telemetry and sensor context before containment.

### Alert 3: DDoS

- Flow: 10.10.3.161 -> 10.10.5.44 at 2026-01-01T00:18:35
- Risk: 88/100 (supervised=45, anomaly=20, sequence=2, telemetry=6, sensor=5, asset=10)
- ATT&CK: T1498 Network Denial of Service
- Analyst decision: validate the raw telemetry and sensor context before containment.

## False Positives

False positives are legitimate flows incorrectly assigned to an attack class. The held-out supervised confusion matrix gives the following review counts:

- Benign: 0 false-positive predictions in the held-out test set.
- BruteForce: 2 false-positive predictions in the held-out test set.
- C2Beacon: 0 false-positive predictions in the held-out test set.
- DDoS: 0 false-positive predictions in the held-out test set.
- Exfiltration: 0 false-positive predictions in the held-out test set.
- PortScan: 0 false-positive predictions in the held-out test set.

Synthetic class patterns are intentionally distinct, so very high supervised scores may overstate performance on real networks. Thresholds must be revalidated on authorized operational telemetry.

## Analyst Recommendations

- Validate high-risk alerts against raw Zeek or Suricata context before containment.
- Review risk components instead of treating a single model probability as operational risk.
- Track false positives and data drift before changing thresholds.
- Keep every response dry-run until an authorized analyst approves the action.

## Limitations

- The model was trained in a controlled lab and needs validation on real authorized telemetry.
- The response simulator is not a production SOAR system.
- High confidence does not remove the need for analyst review.
- Dataset quality and label quality directly affect model reliability.
- Synthetic flow labels are sampled independently, so next-flow GRU performance may remain close to the majority-class baseline.
