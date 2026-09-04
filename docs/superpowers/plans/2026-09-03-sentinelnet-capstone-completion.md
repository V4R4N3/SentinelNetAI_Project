# SentinelNet Capstone Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce a reproducible, rubric-complete SentinelNet capstone with honest AI models, trained-model alert fusion, safe response recommendations, validated evidence, and oral-defense preparation within 20 focused hours.

**Architecture:** Keep the numbered scripts as student-facing CLI entry points while moving shared model and inference contracts into focused modules. Every training stage saves both its model and the metadata needed for inference; the streaming stage consumes those artifacts and emits decomposable risk evidence. Tests and a deliverable validator provide fast feedback before the final report and defense package are generated.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, PyTorch, Joblib, pytest, Markdown, JSON/JSONL, DOCX/PDF.

**Spec:** `docs/superpowers/specs/2026-09-03-sentinelnet-capstone-completion-design.md`

## Global Constraints

- Finish within two days and approximately 20 focused hours.
- Required rubric artifacts take priority over optional tuning or integrations.
- All network response behavior remains dry-run and performs no system or network mutation.
- Fit preprocessing and thresholds without using final test labels.
- Use deterministic seeds and record experiment configuration in metric files.
- Never describe the rolling-window MLP or a text placeholder as a trained GRU.
- Reserve the final four hours for final evidence, report completion, and oral-defense preparation.
- Preserve the numbered CLI workflow documented in `README.md`.

---

### Task 1: Reproducible Data and Test Harness

**Files:**

- Modify: `requirements.txt`
- Modify: `scripts/00_generate_synthetic_network_data.py`
- Modify: `scripts/01_profile_dataset.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data_pipeline.py`

**Interfaces:**

- Produces: `generate_dataset(rows: int, output: Path, seed: int) -> Path`
- Produces: `profile_dataset(path: Path) -> dict`
- Produces: deterministic CSV schema consumed by all later training tasks.

- [x] **Step 1: Add pytest and write failing deterministic-data tests**

```python
def test_generate_dataset_is_deterministic(tmp_path):
    first = generate_dataset(100, tmp_path / "a.csv", seed=42)
    second = generate_dataset(100, tmp_path / "b.csv", seed=42)
    assert first.read_bytes() == second.read_bytes()

def test_generated_dataset_has_expected_schema(tmp_path):
    path = generate_dataset(60, tmp_path / "flows.csv", seed=42)
    df = pd.read_csv(path)
    assert set(FEATURES).issubset(df.columns)
    assert set(df["label"]).issubset(set(LABELS))
    assert not df.isna().any().any()
```

- [x] **Step 2: Run tests and confirm they fail because callable generation does not exist**

Run: `pytest tests/test_data_pipeline.py -v`

Expected: collection or import failure for `generate_dataset` and `profile_dataset`.

- [x] **Step 3: Extract callable generation/profiling functions and add `--seed`**

Use local `random.Random(seed)` values so repeated calls do not depend on process-global state. Keep CLI defaults at 12,000 rows and seed 42. Add `pytest>=8.0` to `requirements.txt`.

- [x] **Step 4: Run focused tests and CLI smoke checks**

Run: `pytest tests/test_data_pipeline.py -v`

Run: `python scripts/00_generate_synthetic_network_data.py --rows 120 --seed 42 --out /tmp/sentinel-smoke.csv`

Expected: all tests pass and the smoke CSV contains 120 rows plus its header.

- [x] **Step 5: Commit the independently working data stage**

```bash
git add requirements.txt scripts/00_generate_synthetic_network_data.py scripts/01_profile_dataset.py tests
git commit -m "test: make dataset generation reproducible"
```

### Task 2: Leakage-Safe Preprocessing Contract

**Files:**

- Modify: `scripts/sentinel_utils.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**

- Produces: `seed_everything(seed: int) -> None`
- Produces: `validate_flow_dataframe(df: pd.DataFrame) -> None`
- Produces: `prepare_split(path: str, test_size: float, seed: int) -> SplitBundle`
- `SplitBundle` contains train/test arrays, labels, fitted scaler, label encoder, features, row indices, and seed.

- [x] **Step 1: Write failing validation and scaler tests**

```python
def test_validate_rejects_missing_feature(sample_df):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_flow_dataframe(sample_df.drop(columns=[FEATURES[0]]))

def test_scaler_is_fit_on_training_rows_only(flow_csv):
    bundle = prepare_split(str(flow_csv), test_size=0.25, seed=42)
    raw_train = bundle.dataframe.loc[bundle.train_indices, FEATURES].to_numpy()
    np.testing.assert_allclose(bundle.scaler.mean_, raw_train.mean(axis=0), rtol=1e-5)
    assert set(bundle.train_indices).isdisjoint(bundle.test_indices)
```

- [x] **Step 2: Verify focused failures**

Run: `pytest tests/test_preprocessing.py -v`

Expected: failures for missing interfaces.

- [x] **Step 3: Implement the explicit preprocessing bundle**

Validate columns, numeric finiteness, known labels, feature order, and minimum class counts. Preserve `split_scaled()` as a compatibility wrapper around `prepare_split()`.

- [x] **Step 4: Verify preprocessing and existing script imports**

Run: `pytest tests/test_preprocessing.py -v`

Run: `python -m py_compile scripts/*.py`

Expected: tests pass and all scripts compile.

- [x] **Step 5: Commit the preprocessing contract**

```bash
git add scripts/sentinel_utils.py tests/test_preprocessing.py
git commit -m "feat: validate flows and prevent preprocessing leakage"
```

### Task 3: Shared Supervised Model and Loadable Artifacts

**Files:**

- Create: `scripts/model_defs.py`
- Modify: `scripts/02_train_supervised_ids.py`
- Modify: `scripts/05_run_streaming_detector.py`
- Create: `tests/test_supervised_model.py`

**Interfaces:**

- Produces: `IDSNet(in_dim: int, classes: int, hidden: int = 96, dropout: float = 0.15)`
- Produces: `load_supervised_artifacts(model_path: Path, preprocess_path: Path) -> tuple[IDSNet, dict]`
- Produces: `models/supervised_ids.pt`, `models/preprocess.joblib`, and `outputs/supervised_metrics.json`.

- [x] **Step 1: Write a failing round-trip test**

```python
def test_supervised_artifacts_round_trip(tmp_path):
    model = IDSNet(14, 6)
    model_path = tmp_path / "model.pt"
    meta_path = tmp_path / "preprocess.joblib"
    save_supervised_artifacts(model, make_test_metadata(), model_path, meta_path)
    loaded, metadata = load_supervised_artifacts(model_path, meta_path)
    assert loaded(torch.zeros(2, 14)).shape == (2, 6)
    assert metadata["features"] == FEATURES
```

- [x] **Step 2: Confirm the test fails because architecture/loading are duplicated or absent**

Run: `pytest tests/test_supervised_model.py -v`

Expected: import failure for shared artifact helpers.

- [x] **Step 3: Centralize architecture and record experiment metadata**

Save the state dictionary with architecture configuration and include seed, split size, class counts, epochs, features, macro F1, weighted F1, classification report, confusion matrix, and training duration in the metrics JSON.

- [x] **Step 4: Train a short model and verify reload**

Run: `python scripts/02_train_supervised_ids.py --epochs 2 --seed 42`

Run: `pytest tests/test_supervised_model.py -v`

Expected: metrics are generated and a new process can load the model.

- [x] **Step 5: Commit the supervised model contract**

```bash
git add scripts/model_defs.py scripts/02_train_supervised_ids.py scripts/05_run_streaming_detector.py tests/test_supervised_model.py
git commit -m "feat: make supervised IDS artifacts loadable"
```

### Task 4: Defensible Anomaly Training and Thresholds

**Files:**

- Modify: `scripts/model_defs.py`
- Modify: `scripts/03_train_autoencoder_anomaly.py`
- Create: `tests/test_anomaly_model.py`

**Interfaces:**

- Produces: `AutoEncoder(input_dim: int)`
- Produces: `reconstruction_errors(model, values) -> np.ndarray`
- Produces: `select_benign_threshold(errors: np.ndarray, percentile: float) -> float`
- Produces: `models/autoencoder.pt`, `models/isolation_forest.joblib`, `models/anomaly_preprocess.joblib`, and `outputs/anomaly_metrics.json`.

- [x] **Step 1: Write failing threshold and serialization tests**

```python
def test_threshold_uses_requested_benign_percentile():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    assert select_benign_threshold(errors, 75.0) == pytest.approx(3.25)

def test_anomaly_score_increases_above_threshold():
    assert normalized_anomaly_score(2.0, threshold=1.0) > normalized_anomaly_score(0.5, threshold=1.0)
```

- [x] **Step 2: Verify the tests fail on missing helpers**

Run: `pytest tests/test_anomaly_model.py -v`

- [x] **Step 3: Split benign training data into fit and validation subsets**

Fit the autoencoder on benign-fit samples, derive the threshold from benign-validation errors at the configured percentile, evaluate once on the held-out test set, and save scaler, feature list, threshold, percentile, architecture, and seed.

- [x] **Step 4: Train and verify both anomaly artifacts**

Run: `python scripts/03_train_autoencoder_anomaly.py --epochs 2 --seed 42 --threshold-percentile 95`

Run: `pytest tests/test_anomaly_model.py -v`

Expected: tests pass; both detectors and anomaly preprocessing metadata load successfully.

- [x] **Step 5: Commit anomaly training**

```bash
git add scripts/model_defs.py scripts/03_train_autoencoder_anomaly.py tests/test_anomaly_model.py
git commit -m "feat: select anomaly thresholds without test leakage"
```

### Task 5: Genuine GRU Sequence Model

**Files:**

- Modify: `scripts/model_defs.py`
- Rewrite: `scripts/04_train_sequence_gru.py`
- Create: `tests/test_sequence_model.py`

**Interfaces:**

- Produces: `SequenceGRU(input_dim: int, hidden_dim: int, classes: int, layers: int = 1)`
- Produces: `make_sequences(values, labels, timestamps, window) -> tuple[np.ndarray, np.ndarray]`
- Produces: `models/sequence_gru.pt`, `models/sequence_preprocess.joblib`, and `outputs/sequence_metrics.json`.

- [x] **Step 1: Write failing shape and artifact tests**

```python
def test_sequence_windows_preserve_temporal_shape():
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    labels = np.arange(10)
    x, y = make_sequences(values, labels, np.arange(10), window=4)
    assert x.shape == (6, 4, 6)
    np.testing.assert_array_equal(y, labels[4:])

def test_gru_forward_shape():
    model = SequenceGRU(input_dim=14, hidden_dim=16, classes=6)
    assert model(torch.zeros(3, 8, 14)).shape == (3, 6)
```

- [x] **Step 2: Verify failures against the current flattened MLP implementation**

Run: `pytest tests/test_sequence_model.py -v`

- [x] **Step 3: Implement chronological splitting and the compact GRU**

Sort by timestamp, choose the chronological boundary first, fit scaling on pre-boundary rows, build train and test windows without crossing the boundary, and train with seeded PyTorch. Save model configuration, window length, feature order, scaler, label encoder, metrics, confusion matrix, and training duration.

- [x] **Step 4: Apply the two-hour decision gate**

Run: `python scripts/04_train_sequence_gru.py --epochs 2 --window 8 --seed 42`

Run: `pytest tests/test_sequence_model.py -v`

Expected: a binary PyTorch checkpoint loads and runs inference. If training is unstable or runtime threatens required outputs, restore the working sequence MLP under `sequence_model.joblib`, rename its metric type accurately, and remove the fake `.pt` checkpoint.

- [x] **Step 5: Commit the truthful sequence implementation**

```bash
git add scripts/model_defs.py scripts/04_train_sequence_gru.py tests/test_sequence_model.py
git commit -m "feat: train a real temporal sequence detector"
```

### Task 6: Trained-Model Streaming Detection and Risk Fusion

**Files:**

- Create: `scripts/detection.py`
- Rewrite: `scripts/05_run_streaming_detector.py`
- Modify: `config.json`
- Create: `tests/test_detection.py`

**Interfaces:**

- Produces: `DetectorBundle.load(models_dir: Path) -> DetectorBundle`
- Produces: `score_flow(row: pd.Series, bundle: DetectorBundle) -> DetectionEvidence`
- Produces: `calculate_risk(evidence: DetectionEvidence, row: pd.Series) -> RiskBreakdown`
- Produces: `outputs/stream_alerts.jsonl` and `outputs/fusion_summary.json`.

- [x] **Step 1: Write failing risk and alert-contract tests**

```python
def test_risk_is_bounded_and_decomposable():
    result = calculate_risk(example_evidence(), example_attack_row())
    assert 0 <= result.total <= 100
    assert result.total == sum(result.components.values())

def test_alert_contains_analyst_evidence(detector_bundle, attack_row):
    alert = detect_row(attack_row, detector_bundle)
    assert {"confidence", "risk_score", "mitre_technique", "evidence", "risk_components"} <= alert.keys()
```

- [x] **Step 2: Confirm failures on the heuristic-only detector**

Run: `pytest tests/test_detection.py -v`

- [x] **Step 3: Implement artifact loading and fixed fusion weights**

Use a 100-point decomposition: supervised confidence 45, normalized anomaly evidence 20, sequence confidence 10, telemetry heuristic 10, Suricata alert count 5, and asset criticality 10. Clamp every normalized input to `[0, 1]`, round components consistently, and expose all components in each alert. Load medium/high thresholds from `config.json`.

- [x] **Step 4: Run detector tests and a 200-row integration sample**

Run: `pytest tests/test_detection.py -v`

Run: `python scripts/05_run_streaming_detector.py --input data/synthetic_flows.csv --limit 200`

Expected: alerts contain model and risk evidence; summary counts match JSONL lines.

- [x] **Step 5: Commit model-based fusion**

```bash
git add config.json scripts/detection.py scripts/05_run_streaming_detector.py tests/test_detection.py
git commit -m "feat: fuse trained detector evidence into risk scores"
```

### Task 7: Safe Response Recommendations

**Files:**

- Refactor: `scripts/07_response_simulator.py`
- Create: `tests/test_response_simulator.py`

**Interfaces:**

- Produces: `recommend_action(alert: dict) -> dict`
- Produces: `build_response_plan(alerts: list[dict]) -> dict`
- Produces: `outputs/response_plan_lab_only.json` with no execution capability.

- [x] **Step 1: Write failing proportionality and safety tests**

```python
@pytest.mark.parametrize(("risk", "action"), [(40, "monitor"), (75, "escalate_to_tier2"), (90, "recommend_isolate_lab_host")])
def test_response_is_proportional(risk, action):
    result = recommend_action(make_alert(risk=risk, src_ip="10.10.1.5"))
    assert result["recommended_action"] == action
    assert result["mode"] == "dry-run"
    assert result["human_approval_required"] is True

def test_external_source_is_never_recommended_for_isolation():
    result = recommend_action(make_alert(risk=99, src_ip="8.8.8.8"))
    assert result["recommended_action"] == "escalate_to_tier2"
```

- [x] **Step 2: Verify the current script fails the explicit safety contract**

Run: `pytest tests/test_response_simulator.py -v`

- [x] **Step 3: Remove misleading execution behavior and implement pure recommendations**

Accept input/output paths, validate alert fields and IP addresses, distinguish private lab sources, add reason and human-approval fields, and perform no subprocess, firewall, or network calls.

- [x] **Step 4: Run tests and generate the response plan**

Run: `pytest tests/test_response_simulator.py -v`

Run: `python scripts/07_response_simulator.py`

Expected: every action is dry-run and includes a safety rationale.

- [x] **Step 5: Commit safe response logic**

```bash
git add scripts/07_response_simulator.py tests/test_response_simulator.py
git commit -m "feat: enforce dry-run lab response recommendations"
```

### Task 8: Reports and Deliverable Validation

**Files:**

- Modify: `scripts/06_generate_incident_report.py`
- Create: `scripts/08_validate_deliverables.py`
- Create: `tests/test_reporting.py`
- Create: `docs/evidence_index.md`

**Interfaces:**

- Produces: `generate_incident_report(outputs_dir: Path) -> str`
- Produces: `validate_deliverables(project_root: Path) -> list[str]`, returning validation errors.
- Produces: an incident report with metrics, top alerts, three alert investigations, limitations, and analyst recommendations.

- [x] **Step 1: Write failing report and validation tests**

```python
def test_validator_reports_missing_required_artifact(tmp_path):
    errors = validate_deliverables(tmp_path)
    assert any("supervised_metrics.json" in error for error in errors)

def test_incident_report_contains_required_sections(populated_outputs):
    report = generate_incident_report(populated_outputs)
    for heading in ["Model Comparison", "Alert Investigations", "False Positives", "Limitations"]:
        assert heading in report
```

- [x] **Step 2: Verify tests fail on the minimal report generator**

Run: `pytest tests/test_reporting.py -v`

- [x] **Step 3: Implement evidence-rich reporting and validation**

Validate required files, JSON syntax, JSONL syntax, non-empty model checkpoints, expected metric keys, alert counts, dry-run response fields, and report headings. Map every assessment component to its exact evidence path in `docs/evidence_index.md`.

- [x] **Step 4: Generate and validate current artifacts**

Run: `python scripts/06_generate_incident_report.py`

Run: `python scripts/08_validate_deliverables.py`

Run: `pytest tests/test_reporting.py -v`

Expected: validator exits zero only when all required artifacts are present and internally consistent.

- [x] **Step 5: Commit reporting and evidence validation**

```bash
git add scripts/06_generate_incident_report.py scripts/08_validate_deliverables.py tests/test_reporting.py docs/evidence_index.md
git commit -m "feat: validate capstone evidence and enrich reporting"
```

### Task 9: Reproducible Final Pipeline and Evidence Run

**Files:**

- Create: `scripts/09_run_pipeline.py`
- Modify: `README.md`
- Create: `tests/test_pipeline_cli.py`
- Generate: `outputs/execution_log.txt`
- Generate: required files under `models/` and `outputs/`.

**Interfaces:**

- Produces: `run_stage(name: str, command: list[str], log) -> None`
- Produces: a single documented command that executes all required stages and records each command, exit code, elapsed time, and key output.

- [x] **Step 1: Write a failing dry-run command-order test**

```python
def test_pipeline_dry_run_has_required_order():
    commands = build_pipeline_commands(rows=12000, seed=42, epochs=8)
    scripts = [command[1] for command in commands]
    assert scripts == [
        "scripts/00_generate_synthetic_network_data.py",
        "scripts/01_profile_dataset.py",
        "scripts/02_train_supervised_ids.py",
        "scripts/03_train_autoencoder_anomaly.py",
        "scripts/04_train_sequence_gru.py",
        "scripts/05_run_streaming_detector.py",
        "scripts/06_generate_incident_report.py",
        "scripts/07_response_simulator.py",
        "scripts/08_validate_deliverables.py",
    ]
```

- [x] **Step 2: Verify the orchestrator test fails**

Run: `pytest tests/test_pipeline_cli.py -v`

- [x] **Step 3: Implement fail-fast orchestration and document commands**

Use `sys.executable`, stream output to both terminal and `outputs/execution_log.txt`, stop on the first nonzero exit, and record environment/package versions. Add quick and final profiles so tests use small data while the official run uses 12,000 rows and final epochs.

- [x] **Step 4: Run all tests, then the official pipeline**

Run: `pytest -q`

Run: `python scripts/09_run_pipeline.py --profile final --seed 42`

Run: `python scripts/08_validate_deliverables.py`

Expected: tests pass, the complete run exits zero, and validation reports every required artifact as valid.

- [x] **Step 5: Commit the reproducible final run**

```bash
git add README.md scripts/09_run_pipeline.py tests/test_pipeline_cli.py models outputs
git commit -m "build: generate reproducible capstone evidence"
```

### Task 10: Capstone Report and Oral Defense Package

**Files:**

- Source: `../AICS109_Capstone_Report_Template.docx`
- Create: `submission/SentinelNet_Capstone_Report.docx`
- Create: `submission/SentinelNet_Capstone_Report.pdf`
- Create: `submission/defense_notes.md`
- Create: `submission/presentation_outline.md`
- Create: `submission/final_checklist.md`

**Interfaces:**

- Consumes: final metrics, model metadata, alerts, response plan, execution log, evidence index, and actual environment versions.
- Produces: the final written submission and a concise defense package whose claims exactly match generated evidence.

- [x] **Step 1: Extract final facts and populate the supplied report structure**

Complete all ten report sections and both appendices. Include a model comparison table, confusion matrix, false-positive analysis, three alert investigations, fusion formula, ATT&CK mappings, response rationale, required limitation statements, commands, and evidence paths.

- [x] **Step 2: Render and inspect the report**

Render the DOCX to page images and PDF using the bundled document runtime. Inspect every page for clipping, broken tables, missing content, inconsistent headings, and awkward page breaks. Correct and rerender until clean.

- [x] **Step 3: Build defense notes from evidence**

For each component, answer: purpose, input, method, metric, strongest result, limitation, safety concern, and production improvement. Include concise answers to the 15 likely defense questions from the approved design discussion.

- [x] **Step 4: Run final rubric and reproduction audit**

Run: `pytest -q`

Run: `python scripts/08_validate_deliverables.py`

Check that every value quoted in the report exists in the final JSON evidence, every model name is accurate, every response is dry-run, and every rubric row maps to a file or oral-defense item.

- [x] **Step 5: Commit the final submission package**

```bash
git add submission docs/evidence_index.md
git commit -m "docs: complete SentinelNet capstone submission"
```

## Execution Order and Timeboxes

| Time | Tasks | Exit condition |
|---|---|---|
| Day 1, hours 1-2 | Task 1 | Deterministic data and tests pass |
| Day 1, hours 3-4 | Task 2 | Validated leakage-safe split passes |
| Day 1, hours 5-6 | Tasks 3-4 | Supervised and anomaly artifacts reload |
| Day 1, hours 7-8 | Task 5 | Real GRU works or honest fallback is recorded |
| Day 1, hours 9-10 | Tasks 3-5 integration | Three model families produce comparable metrics |
| Day 2, hours 11-13 | Task 6 | Model-based alerts and fusion tests pass |
| Day 2, hours 14-15 | Task 7 | Dry-run response safety tests pass |
| Day 2, hours 16-17 | Tasks 8-9 | Official run and deliverable validation pass |
| Day 2, hours 18-19 | Task 10 report | DOCX/PDF visually verified |
| Day 2, hour 20 | Task 10 defense | Rubric audit and mock-defense package complete |

## Stop Rules

- At the end of the GRU block, use the honest MLP fallback if the real GRU is not stable and loadable.
- Do not tune models after the official final run begins unless a rubric-blocking defect is found.
- Do not add live sensors, public datasets, dashboards, deployment, or active response before final validation passes.
- If any report claim cannot be traced to generated evidence, remove or correct the claim.
