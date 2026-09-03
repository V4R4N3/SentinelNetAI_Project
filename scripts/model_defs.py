from pathlib import Path

import joblib
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, values):
        return self.activation(values + self.net(values))


class IDSNet(nn.Module):
    def __init__(self, in_dim, classes, hidden=96, dropout=0.15):
        super().__init__()
        self.config = {
            'in_dim': in_dim,
            'classes': classes,
            'hidden': hidden,
            'dropout': dropout,
        }
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, values):
        return self.net(values)


def save_supervised_artifacts(model, metadata, model_path, metadata_path):
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'state_dict': model.state_dict(), 'architecture': model.config},
        model_path,
    )
    joblib.dump(metadata, metadata_path)


def load_supervised_artifacts(model_path, metadata_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = IDSNet(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, joblib.load(metadata_path)
