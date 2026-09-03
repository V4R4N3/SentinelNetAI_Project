from pathlib import Path
from dataclasses import dataclass
import random
import pandas as pd, numpy as np, json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

FEATURES=['duration','packets','bytes_out','bytes_in','bytes_per_sec','packets_per_sec','outbound_ratio','unique_dst_ports_5m','failed_conn_5m','dns_entropy','beacon_score','burst_score','suricata_alert_count','asset_criticality']
LABELS=['Benign','PortScan','DDoS','BruteForce','Exfiltration','C2Beacon']


@dataclass
class SplitBundle:
    dataframe: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    features: list
    train_indices: np.ndarray
    test_indices: np.ndarray
    seed: int


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def validate_flow_dataframe(df):
    required = FEATURES + ['label']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    numeric = df[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError('feature data contains missing or non-finite values')
    unknown = sorted(set(df['label'].astype(str)) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown labels: {', '.join(unknown)}")

def load_xy(path='data/synthetic_flows.csv'):
    df=pd.read_csv(path).reset_index(drop=True)
    validate_flow_dataframe(df)
    X=df[FEATURES].astype('float32')
    y=df['label'].astype(str)
    le=LabelEncoder(); y_enc=le.fit_transform(y)
    return df, X, y_enc, le

def prepare_split(path='data/synthetic_flows.csv', test_size=0.25, seed=42):
    df,X,y,le=load_xy(path)
    indices=np.arange(len(df))
    train_indices,test_indices=train_test_split(
        indices,test_size=test_size,random_state=seed,stratify=y
    )
    scaler=StandardScaler().fit(X.iloc[train_indices].values)
    return SplitBundle(
        dataframe=df,
        X_train=scaler.transform(X.iloc[train_indices].values).astype('float32'),
        X_test=scaler.transform(X.iloc[test_indices].values).astype('float32'),
        y_train=y[train_indices],
        y_test=y[test_indices],
        scaler=scaler,
        label_encoder=le,
        features=list(FEATURES),
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
    )


def split_scaled(path='data/synthetic_flows.csv'):
    bundle=prepare_split(path)
    return (
        bundle.dataframe,bundle.X_train,bundle.X_test,bundle.y_train,
        bundle.y_test,bundle.scaler,bundle.label_encoder
    )

def save_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2))
