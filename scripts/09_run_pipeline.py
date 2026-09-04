#!/usr/bin/env python3
import argparse
import datetime
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys
import time


PROFILES = {
    'quick': {'rows': 1200, 'supervised_epochs': 2, 'anomaly_epochs': 2, 'sequence_epochs': 2, 'limit': 400},
    'final': {'rows': 12000, 'supervised_epochs': 8, 'anomaly_epochs': 8, 'sequence_epochs': 6, 'limit': 1000},
}


def build_pipeline_commands(profile, seed):
    values = PROFILES[profile]
    python = sys.executable
    return [
        [python, 'scripts/00_generate_synthetic_network_data.py', '--rows', str(values['rows']), '--seed', str(seed)],
        [python, 'scripts/01_profile_dataset.py'],
        [python, 'scripts/02_train_supervised_ids.py', '--epochs', str(values['supervised_epochs']), '--seed', str(seed)],
        [python, 'scripts/03_train_autoencoder_anomaly.py', '--epochs', str(values['anomaly_epochs']), '--seed', str(seed), '--threshold-percentile', '95'],
        [python, 'scripts/04_train_sequence_gru.py', '--epochs', str(values['sequence_epochs']), '--window', '8', '--seed', str(seed)],
        [python, 'scripts/05_run_streaming_detector.py', '--input', 'data/synthetic_flows.csv', '--limit', str(values['limit'])],
        [python, 'scripts/06_generate_incident_report.py'],
        [python, 'scripts/07_response_simulator.py'],
        [python, 'scripts/08_validate_deliverables.py'],
    ]


def clean_generated_artifacts():
    for directory_name in ('models', 'outputs'):
        directory = Path(directory_name)
        directory.mkdir(exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and path.name != '.gitkeep':
                path.unlink()


def write_environment(log):
    log.write('# SentinelNet pipeline execution log\n')
    log.write(f"started_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    log.write(f'python={platform.python_version()}\n')
    log.write(f'platform={platform.platform()}\n')
    for package in ('pandas', 'numpy', 'scikit-learn', 'torch', 'joblib', 'pytest'):
        log.write(f'{package}={importlib.metadata.version(package)}\n')
    log.write('\n')
    log.flush()


def run_stage(name, command, log):
    rendered = ' '.join(command)
    started = time.perf_counter()
    heading = f'## {name}\ncommand={rendered}\n'
    print(heading, end='')
    log.write(heading)
    log.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end='')
        log.write(line)
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    trailer = f'exit_code={return_code}\nelapsed_seconds={elapsed:.3f}\n\n'
    print(trailer, end='')
    log.write(trailer)
    log.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=sorted(PROFILES), default='quick')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    clean_generated_artifacts()
    log_path = Path('outputs/execution_log.txt')
    commands = build_pipeline_commands(args.profile, args.seed)
    with log_path.open('w') as log:
        write_environment(log)
        for command in commands:
            run_stage(Path(command[1]).stem, command, log)
    print(f'[+] pipeline profile={args.profile} complete; log={log_path}')


if __name__ == '__main__':
    main()
