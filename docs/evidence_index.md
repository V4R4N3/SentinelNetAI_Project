# SentinelNet Evidence Index

| Assessment component | Evidence |
|---|---|
| Lab completion | `outputs/execution_log.txt`, final terminal screenshots, and generated artifacts |
| Data handling | `outputs/data_profile.json`, `data/synthetic_flows.csv`, and `tests/test_data_pipeline.py` |
| Supervised model | `models/supervised_ids.pt`, `models/preprocess.joblib`, and `outputs/supervised_metrics.json` |
| Anomaly models | `models/autoencoder.pt`, `models/isolation_forest.joblib`, `models/anomaly_preprocess.joblib`, and `outputs/anomaly_metrics.json` |
| Sequence model | `models/sequence_gru.pt`, `models/sequence_preprocess.joblib`, and `outputs/sequence_metrics.json` |
| Detection and fusion | `outputs/stream_alerts.jsonl`, `outputs/fusion_summary.json`, and `tests/test_detection.py` |
| ATT&CK mapping | Alert fields in `outputs/stream_alerts.jsonl` and investigations in `outputs/incident_report.md` |
| Response safety | `outputs/response_plan_lab_only.json` and `tests/test_response_simulator.py` |
| Capstone analysis | `submission/SentinelNet_Capstone_Report.docx` and `submission/SentinelNet_Capstone_Report.pdf` |
| Oral defense | `submission/defense_notes.md` and `submission/presentation_outline.md` |
