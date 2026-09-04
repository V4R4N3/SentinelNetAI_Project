# AICS-109 - AI for Advanced Network Defense

## Project: SentinelNet AI Defense Fabric

This is a safe defensive AI network defense project for the AI-Driven Cybersecurity and Digital Forensics Fellowship. It includes synthetic data generation, model training, streaming detection, alert fusion, and lab-only response simulation.

Run:

```bash
python scripts/00_generate_synthetic_network_data.py --rows 12000 --seed 42
python scripts/01_profile_dataset.py
python scripts/02_train_supervised_ids.py --epochs 8 --seed 42
python scripts/03_train_autoencoder_anomaly.py --epochs 8 --seed 42 --threshold-percentile 95
python scripts/04_train_sequence_gru.py --epochs 6 --window 8 --seed 42
python scripts/05_run_streaming_detector.py --input data/synthetic_flows.csv --limit 200
python scripts/06_generate_incident_report.py
python scripts/07_response_simulator.py
python scripts/08_validate_deliverables.py
```

For a reproducible end-to-end run with a terminal log:

```bash
python scripts/09_run_pipeline.py --profile quick --seed 42
python scripts/09_run_pipeline.py --profile final --seed 42
```

The quick profile is for development checks. Use the final profile for submitted metrics and evidence. The sequence stage is a genuine PyTorch GRU that predicts each flow label from the preceding eight flows. Because the synthetic generator samples labels independently, weak next-flow performance is an expected limitation rather than evidence that the supervised flow classifier failed.

Safety: response is dry-run by default. Do not use this project to scan or disrupt networks.

## Submission Package

- `submission/SentinelNet_Capstone_Report.pdf` - final eight-page report.
- `submission/SentinelNet_Capstone_Report.docx` - editable report based on the supplied AICS-109 template.
- `submission/defense_notes.md` - evidence-backed oral defense notes and likely questions.
- `submission/presentation_outline.md` - eleven-slide presentation structure.
- `submission/final_checklist.md` - automated gates and final human upload checks.
