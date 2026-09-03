#!/usr/bin/env python3
"""Train a GRU to predict a flow label from the preceding flow window."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_defs import SequenceGRU, make_sequences, save_sequence_artifact
from sentinel_utils import FEATURES, save_json, seed_everything, validate_flow_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(1)

    dataframe = pd.read_csv('data/synthetic_flows.csv')
    validate_flow_dataframe(dataframe)
    dataframe = dataframe.sort_values('timestamp').reset_index(drop=True)
    boundary = int(len(dataframe) * 0.75)
    train_frame = dataframe.iloc[:boundary]
    test_frame = dataframe.iloc[boundary:]

    label_encoder = LabelEncoder().fit(train_frame['label'].astype(str))
    unseen = sorted(set(test_frame['label'].astype(str)) - set(label_encoder.classes_))
    if unseen:
        raise ValueError(f"test period contains unseen labels: {', '.join(unseen)}")

    scaler = StandardScaler().fit(train_frame[FEATURES].astype('float32').values)
    train_values = scaler.transform(train_frame[FEATURES].astype('float32').values)
    test_values = scaler.transform(test_frame[FEATURES].astype('float32').values)
    train_labels = label_encoder.transform(train_frame['label'].astype(str))
    test_labels = label_encoder.transform(test_frame['label'].astype(str))
    X_train, y_train = make_sequences(
        train_values, train_labels, train_frame['timestamp'].values, args.window
    )
    X_test, y_test = make_sequences(
        test_values, test_labels, test_frame['timestamp'].values, args.window
    )

    model = SequenceGRU(
        input_dim=len(FEATURES),
        hidden_dim=args.hidden,
        classes=len(label_encoder.classes_),
    )
    counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
    weights = counts.max() / np.maximum(counts, 1)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long()),
        batch_size=args.batch,
        shuffle=True,
    )

    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for values, targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(values), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(values)
        print(f'epoch={epoch + 1} gru_loss={total / len(X_train):.5f}')

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    class_ids = np.arange(len(label_encoder.classes_))
    metrics = {
        'classes': label_encoder.classes_.tolist(),
        'macro_f1': float(f1_score(y_test, predictions, average='macro')),
        'weighted_f1': float(f1_score(y_test, predictions, average='weighted')),
        'classification_report': classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        'confusion_matrix': confusion_matrix(y_test, predictions, labels=class_ids).tolist(),
        'window': args.window,
        'model_type': 'PyTorch GRU next-flow sequence classifier',
        'sequence_contract': 'predict each flow label from the preceding window of flows',
        'split_strategy': 'chronological 75/25; windows do not cross the split boundary',
        'features': FEATURES,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'train_sequences': len(X_train),
        'test_sequences': len(X_test),
        'architecture': model.config,
        'training_seconds': round(time.perf_counter() - started, 3),
    }
    save_json(metrics, 'outputs/sequence_metrics.json')
    metadata = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': FEATURES,
        'classes': label_encoder.classes_.tolist(),
        'window': args.window,
        'sequence_contract': metrics['sequence_contract'],
    }
    save_sequence_artifact(
        model,
        metadata,
        'models/sequence_gru.pt',
        'models/sequence_preprocess.joblib',
    )
    print(json.dumps({
        'sequence_macro_f1': metrics['macro_f1'],
        'sequence_weighted_f1': metrics['weighted_f1'],
    }, indent=2))


if __name__ == '__main__':
    main()
