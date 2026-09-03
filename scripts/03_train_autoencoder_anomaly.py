#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np, torch, joblib
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from model_defs import AutoEncoder, reconstruction_errors, select_benign_threshold, save_autoencoder_artifact
from sentinel_utils import prepare_split, save_json, FEATURES, seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--threshold-percentile',type=float,default=95.0); args=ap.parse_args()
    Path('models').mkdir(exist_ok=True); Path('outputs').mkdir(exist_ok=True)
    seed_everything(args.seed)
    bundle=prepare_split(seed=args.seed)
    Xtr,Xte,ytr,yte=bundle.X_train,bundle.X_test,bundle.y_train,bundle.y_test
    scaler,le=bundle.scaler,bundle.label_encoder
    benign_train=Xtr[ytr==le.transform(['Benign'])[0]]
    benign_fit,benign_validation=train_test_split(benign_train,test_size=0.2,random_state=args.seed)
    device='cpu'; model=AutoEncoder(Xtr.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.MSELoss(); dl=DataLoader(TensorDataset(torch.tensor(benign_fit)),batch_size=256,shuffle=True)
    started=time.perf_counter()
    for e in range(args.epochs):
        total=0; model.train()
        for (xb,) in dl:
            xb=xb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),xb); loss.backward(); opt.step(); total+=loss.item()*len(xb)
        print(f'epoch={e+1} ae_loss={total/len(benign_fit):.5f}')
    validation_err=reconstruction_errors(model,benign_validation)
    threshold=select_benign_threshold(validation_err,args.threshold_percentile)
    err=reconstruction_errors(model,Xte)
    pred_anom=(err>threshold).astype(int); true_anom=(yte!=le.transform(['Benign'])[0]).astype(int)
    iso=IsolationForest(n_estimators=200, contamination=0.05, random_state=args.seed).fit(benign_fit)
    iso_pred=(iso.predict(Xte)==-1).astype(int)
    metrics={'autoencoder_threshold':threshold,'threshold_percentile':args.threshold_percentile,'threshold_source':'held-out benign training validation set','autoencoder_report':classification_report(true_anom,pred_anom,output_dict=True,zero_division=0),'isolation_forest_report':classification_report(true_anom,iso_pred,output_dict=True,zero_division=0),'features':FEATURES,'seed':args.seed,'epochs':args.epochs,'benign_fit_rows':len(benign_fit),'benign_validation_rows':len(benign_validation),'test_rows':len(Xte),'architecture':model.config,'training_seconds':round(time.perf_counter()-started,3)}
    save_json(metrics,'outputs/anomaly_metrics.json')
    metadata={'scaler':scaler,'features':FEATURES,'threshold':threshold,'threshold_percentile':args.threshold_percentile,'architecture':model.config}
    save_autoencoder_artifact(model,metadata,'models/autoencoder.pt','models/anomaly_preprocess.joblib'); joblib.dump(iso,'models/isolation_forest.joblib')
    print(json.dumps({'ae_threshold':threshold},indent=2))
if __name__=='__main__': main()
