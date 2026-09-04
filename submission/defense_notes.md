# SentinelNet Defense Notes

## Opening Statement

SentinelNet is a reproducible lab prototype that turns network-flow telemetry into analyst-reviewable alerts. It combines a supervised residual MLP, two anomaly detectors, a GRU sequence model, telemetry rules, sensor context, and asset criticality. The output is a transparent 100-point risk score, an ATT&CK mapping, and a dry-run response recommendation that always requires human approval.

## What We Built

- A deterministic generator for 12,000 synthetic flows across six labels.
- Leakage-safe preprocessing fitted only on training data.
- A residual MLP for flow classification.
- An autoencoder and Isolation Forest for anomaly evidence.
- A real PyTorch GRU that predicts a flow label from the preceding eight flows.
- A fusion engine with an auditable risk decomposition.
- ATT&CK tactic and technique mappings for the five attack classes.
- A response simulator that never executes containment.
- A validator, 38 automated tests, and a single-command final pipeline.

## Architecture Walkthrough

1. `00_generate_synthetic_network_data.py` creates deterministic telemetry with seed 42.
2. `01_profile_dataset.py` validates schema, missing data, and class distribution.
3. `02_train_supervised_ids.py` trains the residual MLP on a stratified 75/25 split.
4. `03_train_autoencoder_anomaly.py` trains anomaly models only on benign training data and selects the autoencoder threshold from held-out benign training validation data.
5. `04_train_sequence_gru.py` creates chronological windows that do not cross the split boundary.
6. `05_run_streaming_detector.py` loads every trained artifact and fuses its evidence.
7. `06_generate_incident_report.py` summarizes metrics and investigated alerts.
8. `07_response_simulator.py` produces dry-run, human-approved recommendations.
9. `08_validate_deliverables.py` fails if required artifacts are absent or inconsistent.
10. `09_run_pipeline.py` reproduces the full run and records versions, commands, and timings.

## Evidence To Quote

- Dataset: 12,000 rows, 22 columns, no missing values.
- Supervised test set: 3,000 rows; macro F1 0.9987; weighted F1 0.9993.
- Supervised confusion matrix: two PortScan samples were predicted as BruteForce; every other test sample was correct.
- Autoencoder anomaly macro F1: 0.9595; threshold 0.05419 at the 95th percentile of held-out benign training errors.
- Isolation Forest anomaly macro F1: 0.7942.
- GRU sequence macro F1: 0.1223; weighted F1 0.1795.
- Streaming integration demonstration: 1,000 flows processed and 384 alerts at risk score 65 or higher. This stream overlaps the training corpus and is not an independent evaluation set.
- Verification: 38 tests passed and the deliverable validator passed.

## Why The GRU Result Is Low

The synthetic generator samples labels independently rather than creating multi-flow attack campaigns. A model that only sees the previous eight flows therefore has little legitimate temporal signal for predicting the next label. The low score is evidence that the evaluation is honest, not that the checkpoint is fake. In production-oriented future work, the dataset should model sessions, campaigns, beacon intervals, scan bursts, and exfiltration stages before temporal performance is used operationally.

## Why The Supervised Result Is High

The synthetic attack classes were designed with distinct feature distributions. That makes the classification problem much easier than a real network where behaviors overlap, labels are noisy, and traffic changes over time. The result proves the implementation can learn the generated patterns; it does not prove production readiness.

## Risk Score Defense

The maximum score is 100 points: supervised 45, anomaly 20, sequence 10, telemetry 10, sensor 5, and asset criticality 10. This keeps the primary classifier influential without allowing one probability to become the entire operational decision. Every alert exposes its component values so an analyst can challenge the score.

## Safety Defense

- The response output uses `mode: dry-run`.
- Every recommendation sets `human_approval_required: true`.
- Host-isolation recommendations are limited to RFC1918 source addresses.
- Public, documentation-range, or malformed source addresses are escalated rather than isolated.
- No firewall, endpoint, or SOAR command is executed.

## Likely Questions

**Why use three model families?** The supervised model recognizes known labeled patterns, anomaly models provide evidence for deviations from benign behavior, and the GRU tests whether prior flows add temporal context. Fusion lets the system expose agreement and disagreement.

**How did you prevent leakage?** The train/test split occurs before scaler fitting. The scaler is fitted on training features only. The anomaly threshold comes from a held-out subset of benign training data, never from the test set.

**Why is ATT&CK mapping useful?** It converts a model label into an investigation vocabulary. The mapping is contextual metadata, not proof that the technique occurred; the analyst must validate raw telemetry.

**What would you change first for production?** Replace synthetic flows with authorized Zeek or Suricata telemetry, create campaign-aware temporal labels, calibrate probabilities and thresholds, measure alert precision under realistic base rates, monitor drift, and integrate approval-gated case management.

**What is the strongest engineering control?** The validator and deterministic pipeline make claims reproducible, while the dry-run response boundary prevents the prototype from causing network changes.

**Why use macro F1 instead of accuracy alone?** The classes are imbalanced. Macro F1 gives each class equal influence, while weighted F1 and accuracy can be dominated by the larger Benign class.

**How was the anomaly threshold selected?** It is the 95th percentile of reconstruction errors from a held-out subset of benign training data. The test set was not used to select it.

**What does the confusion matrix show?** Of 3,000 supervised test flows, two actual PortScan flows were predicted as BruteForce. Every other sample was classified correctly, creating two BruteForce false positives.

**Why keep both the autoencoder and Isolation Forest?** They use different mechanisms. Their agreement strengthens anomaly evidence, while disagreement tells the analyst that the deviation is model-dependent.

**Are the model probabilities calibrated?** No production calibration claim is made. Calibration against authorized telemetry and realistic base rates is future work.

**Why is the alert threshold 65?** It is a lab policy threshold that requires multiple evidence contributions. It has not been optimized for a production cost function and must be recalibrated per environment.

**Can SentinelNet detect a zero-day?** The anomaly paths may flag behavior outside the benign baseline, but the system cannot identify or attribute an unknown exploit from flow features alone.

**What happens if a model artifact is missing or inconsistent?** The detector cannot silently substitute a fake model, and the deliverable validator fails when required artifacts, counts, schemas, or safety invariants are missing or inconsistent.

**How is reproducibility demonstrated?** The data seed, model seeds, command sequence, package versions, timings, model metadata, and outputs are recorded. The final pipeline rebuilds the evidence set from one command.

**Is the 1,000-flow stream an independent evaluation?** No. It is an end-to-end integration demonstration over the first 1,000 generated rows, which overlap the model-development corpus. Generalization claims come only from the held-out model test partitions.

**What did automated testing cover?** Tests cover deterministic generation, schema validation, train-only preprocessing, checkpoint reloads, anomaly threshold selection, sequence windows, fusion arithmetic, confidence semantics, response safety, reporting, validation, and pipeline command construction.

## Five Minute Demo

1. Show `README.md` and the single final pipeline command.
2. Open `outputs/execution_log.txt` to prove environment, commands, timings, and successful stages.
3. Open `outputs/supervised_metrics.json`, `outputs/anomaly_metrics.json`, and `outputs/sequence_metrics.json` to compare the models honestly.
4. Open one line from `outputs/stream_alerts.jsonl` and explain the six risk components.
5. Open `outputs/response_plan_lab_only.json` and point out dry-run mode and human approval.
6. Run `.venv/bin/python scripts/08_validate_deliverables.py` if a live verification is requested.

## Statements Not To Overclaim

- Do not call SentinelNet production ready.
- Do not imply the synthetic metrics generalize to real networks.
- Do not hide the weak sequence-model result.
- Do not present an ATT&CK mapping as confirmed adversary attribution.
- Do not say the response simulator performs containment.
