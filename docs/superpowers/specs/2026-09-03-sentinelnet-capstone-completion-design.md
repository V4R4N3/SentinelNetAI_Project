# SentinelNet Capstone Completion Design

## Purpose

Complete the AICS-109 SentinelNet AI Defense Fabric capstone within two days (approximately 20 focused hours) while maximizing rubric coverage, preserving technical honesty, and preparing the student to defend every major design decision.

## Success Criteria

The work is complete when:

- Every rubric-required output exists, is valid, and comes from a reproducible final run.
- The supervised, anomaly, and sequence model families are accurately implemented and described.
- Saved models and preprocessing artifacts can be loaded after training.
- Streaming detection incorporates trained-model evidence rather than relying only on heuristics.
- Alert fusion, risk scoring, thresholds, and MITRE ATT&CK mappings are documented.
- Response recommendations remain dry-run, private-lab-only, proportional, and subject to human approval.
- Critical data, model, detector, and response behavior has automated test coverage.
- The capstone report, screenshots, execution log, and source tree agree with the final results.
- The student can explain the architecture, metrics, limitations, safety choices, and three investigated alerts without depending on generated prose.

## Scope

### Required

- Reproducible synthetic-data generation and profiling.
- Data validation and leakage-safe preprocessing.
- Residual MLP supervised classifier with classification metrics and confusion matrix.
- Autoencoder and Isolation Forest anomaly detection with documented threshold selection.
- A genuine CPU-friendly PyTorch GRU sequence classifier if it can be stabilized within the allocated block.
- Honest fallback to an accurately named rolling-window MLP if the GRU cannot be stabilized without risking submission completion.
- Streaming inference that loads saved artifacts and combines model, anomaly, telemetry, Suricata-count, and asset-criticality evidence.
- Documented risk scoring and configurable thresholds.
- Safe dry-run response recommendations.
- Tests, final artifacts, execution evidence, completed report, defense notes, and mock-defense preparation.

### Excluded

- Production deployment or production SOAR integration.
- Public-network scanning or active containment.
- A web dashboard.
- Full live Zeek or Suricata deployment.
- Large public-dataset retraining unless all required deliverables finish early.
- Broad refactoring unrelated to rubric completion.

## Delivery Strategy

Use a rubric-first and evidence-first sequence. Establish a working baseline immediately, improve the highest-risk technical gaps next, and reserve protected time for final execution, reporting, and defense rehearsal. Each two-hour block must leave a testable artifact. Enhancements stop when they threaten the required submission.

## Architecture

The pipeline has seven bounded stages:

1. Generate labelled synthetic network-flow telemetry.
2. Validate and profile the dataset.
3. Train supervised, anomaly, and sequence detectors with saved preprocessing metadata.
4. Load detector artifacts for streaming inference.
5. Fuse detector confidence, anomaly evidence, telemetry indicators, sensor alerts, and asset criticality into prioritized alerts.
6. Produce analyst-facing reports and safe response recommendations.
7. Package metrics, models, logs, screenshots, reports, tests, and defense notes as submission evidence.

Each stage communicates through explicit CSV, JSON, JSONL, Joblib, or PyTorch artifacts. A failure must identify the missing or incompatible artifact rather than silently substituting fabricated results.

## Model Design

### Supervised Detector

Retain the residual PyTorch MLP as the multiclass flow classifier. Use a deterministic stratified train/test split, fit scaling only on training data, account for class imbalance, and report macro F1, weighted F1, per-class performance, and a confusion matrix.

### Anomaly Detectors

Train the autoencoder and Isolation Forest only on benign training samples. Select the autoencoder threshold from benign validation reconstruction error, record the percentile and value, and report benign-versus-anomalous performance. Do not select thresholds from the final test labels.

### Sequence Detector

Use chronological rolling windows and a compact PyTorch GRU that predicts the current or next flow class according to one documented contract. Fit preprocessing on training-period observations only and keep train/test windows separated chronologically. If this cannot be made reliable in the allocated time, preserve the rolling-window MLP under an accurate name and document the GRU as future work; never ship a text placeholder as a trained GRU.

## Detection and Fusion

The streaming detector loads preprocessing and model artifacts produced by training. It generates separate evidence values for supervised confidence, anomaly score, sequence confidence when available, telemetry rules, Suricata alert count, and asset criticality.

The final risk score must be deterministic, bounded from 0 to 100, and decomposable so an analyst can see why an alert received its priority. Heuristics remain supporting evidence, not a hidden replacement for trained models. Medium and high thresholds come from configuration. Every non-benign alert includes timestamp, endpoints, predicted threat, confidence, risk, MITRE tactic and technique, and human-readable evidence.

## Response Safety

The response component produces recommendations only. It never changes firewall rules, isolates a host, scans a target, or executes a network action. Recommendations must validate addresses, distinguish private lab addresses from external documentation/test ranges, apply proportional actions by risk, and state that human approval is required. Misleading execution flags must be removed or made incapable of side effects.

## Testing and Verification

Automated tests cover:

- Deterministic dataset generation and schema validation.
- Leakage-safe preprocessing and expected feature shape.
- Model serialization and loading.
- Sequence construction without boundary leakage.
- Risk-score bounds and evidence decomposition.
- Known benign and attack-like detector cases.
- Response behavior for low, medium, and high risk.
- Rejection of malformed or missing artifacts.

The final verification runs the complete pipeline from data generation through response planning, validates all JSON/JSONL files, loads every saved model, and checks that reported metrics match generated outputs.

## Evidence and Report

The final submission contains:

- `outputs/data_profile.json`
- `outputs/supervised_metrics.json`
- `outputs/anomaly_metrics.json`
- `outputs/sequence_metrics.json`
- `outputs/stream_alerts.jsonl`
- `outputs/fusion_summary.json`
- `outputs/incident_report.md`
- `outputs/response_plan_lab_only.json`
- All real saved model and preprocessing artifacts
- Source code and automated tests
- A terminal execution log and selected screenshots
- A completed editable capstone report and PDF export
- A rubric-to-evidence index
- Defense notes and a short presentation outline

The report covers the lab environment, data, feature engineering, all model families, metrics, confusion matrix, false positives, alert fusion, ATT&CK mapping, response safety, limitations, reproduction commands, and output evidence. It must explicitly acknowledge synthetic-data limitations and any sequence-model fallback.

## Collaboration and Learning Protocol

Work in two-hour milestones. The assistant implements and verifies focused changes, then explains the result and identifies the exact evidence created. The student runs or inspects the important command, checks the output, and explains the purpose, inputs, algorithm, metric, and limitation in their own words. Each milestone adds concise answers to the defense notes.

No section is considered defense-ready until the student can explain:

- What the component does.
- Why that approach was selected.
- What evidence shows it works.
- What can cause it to fail.
- How it would need to change for production use.

## Schedule

### Day 1

- Hours 1-2: environment, baseline run, output inventory, and execution logging.
- Hours 3-4: deterministic data generation, validation, profiling, and preprocessing tests.
- Hours 5-6: supervised and anomaly training, evaluation, threshold correction, and false-positive review.
- Hours 7-8: genuine GRU implementation and evaluation, with an explicit fallback decision at the end of the block.
- Hours 9-10: integrated training run, artifact loading checks, model comparison, and oral review.

### Day 2

- Hours 11-13: trained-model streaming inference, evidence fusion, ATT&CK mapping, and detector tests.
- Hours 14-15: response safety hardening and response tests.
- Hours 16-17: clean final pipeline run, artifact validation, terminal log, screenshots, and evidence index.
- Hours 18-19: capstone report completion and PDF export.
- Hour 20: clean reproduction check, rubric audit, presentation outline, and mock oral defense.

## Risk Controls

- Preserve a working baseline before structural changes.
- Time-box the GRU and use the documented honest fallback if necessary.
- Do not add optional dashboards, deployments, or public datasets before all required evidence exists.
- Keep generated outputs separate from source and make regeneration commands explicit.
- Treat suspiciously perfect synthetic-data metrics as a limitation requiring leakage checks and discussion.
- Reserve the final four hours for evidence, reporting, and defense; do not consume them with optional model tuning.

## Decision Record

The approved approach is rubric-first completion rather than starter-code submission or production redesign. Technical truthfulness takes priority over impressive naming: a model or integration is claimed only when its saved artifact, loading path, test, metric, and explanation all exist.
