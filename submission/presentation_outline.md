# SentinelNet Presentation Outline

## Slide 1 Project and Defensive Problem

- SentinelNet AI Defense Fabric
- Problem: prioritize suspicious network flows without allowing a single opaque score to trigger containment.
- Scope: controlled lab prototype using synthetic flow telemetry.

## Slide 2 System Architecture

- Data generation and profiling.
- Supervised, anomaly, and temporal model paths.
- Transparent evidence fusion.
- ATT&CK context and dry-run response.

## Slide 3 Dataset and Features

- 12,000 rows, six labels, 22 columns, no missing values.
- Fourteen numeric features used for modeling.
- Examples: rates, volume, outbound ratio, scan behavior, authentication failures, DNS entropy, beaconing, sensor alerts, and asset criticality.

## Slide 4 Supervised Residual MLP

- 14 inputs, hidden width 96, two residual blocks, six outputs.
- Stratified 75/25 split; scaler fitted on training data only.
- Macro F1 0.9987 and weighted F1 0.9993 on 3,000 test flows.
- Interpretation: strong fit to intentionally distinct synthetic classes, not production generalization.

## Slide 5 Anomaly Detection

- Autoencoder trained only on benign training flows.
- Threshold selected at the 95th percentile of held-out benign training reconstruction errors.
- Autoencoder macro F1 0.9595; Isolation Forest macro F1 0.7942.
- Role: supporting evidence for unfamiliar or off-baseline behavior.

## Slide 6 Sequence GRU and Honest Evaluation

- Previous eight chronological flows predict the next flow label.
- Windows do not cross the train/test boundary.
- Macro F1 0.1223 and weighted F1 0.1795.
- The generator lacks campaign-level temporal dependencies, so the weak result is expected and explicitly reported.

## Slide 7 Fusion and ATT&CK Mapping

- 100-point score: supervised 45, anomaly 20, sequence 10, telemetry 10, sensor 5, asset 10.
- 1,000 flows processed; 384 alerts at score 65 or higher.
- Example mappings: T1498 DDoS, T1110 Brute Force, T1041 Exfiltration, T1071 C2, T1046 Network Service Discovery.

## Slide 8 Alert Investigation

- Show three alerts: DDoS, BruteForce, and Exfiltration.
- Explain source and destination, model confidence, telemetry indicators, risk decomposition, ATT&CK context, and analyst decision.
- Emphasize that the alert contains evidence, not just a verdict.

## Slide 9 Safety and Response

- Recommendations only; no network commands execute.
- Human approval required for every action.
- Isolation recommendations limited to private lab sources.
- Operational action remains an analyst decision.

## Slide 10 Limitations and Next Steps

- Synthetic and separable data can inflate supervised performance.
- Sequence data lacks realistic campaigns.
- Real telemetry introduces drift, imbalance, noise, privacy, and adversarial behavior.
- Next step: authorized Zeek or Suricata validation with calibrated thresholds and drift monitoring.

## Slide 11 Reproducibility and Deliverables

- One-command final pipeline.
- Versioned models, metrics, alerts, report, response plan, and execution log.
- 33 automated tests plus a deliverable consistency validator.
- End with the claim: SentinelNet is a defensible, reproducible lab prototype with explicit safety boundaries.

