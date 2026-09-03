#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from model_defs import IDSNet, save_supervised_artifacts
from sentinel_utils import prepare_split, save_json, FEATURES, seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    Path('models').mkdir(exist_ok=True); Path('outputs').mkdir(exist_ok=True)
    seed_everything(args.seed)
    bundle=prepare_split(seed=args.seed)
    Xtr,Xte,ytr,yte=bundle.X_train,bundle.X_test,bundle.y_train,bundle.y_test
    scaler,le=bundle.scaler,bundle.label_encoder
    device='cpu'
    model=IDSNet(Xtr.shape[1], len(le.classes_)).to(device)
    counts=np.bincount(ytr); weights=(counts.max()/np.maximum(counts,1)).astype('float32')
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr).long()), batch_size=args.batch, shuffle=True)
    started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total=0
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb)
        print(f'epoch={epoch+1} loss={total/len(Xtr):.4f}')
    model.eval()
    with torch.no_grad():
        logits=model(torch.tensor(Xte).to(device)); prob=torch.softmax(logits,dim=1).cpu().numpy(); pred=prob.argmax(1)
    metrics={'classes': le.classes_.tolist(), 'macro_f1': float(f1_score(yte,pred,average='macro')), 'weighted_f1': float(f1_score(yte,pred,average='weighted')), 'classification_report': classification_report(yte,pred,target_names=le.classes_,output_dict=True), 'confusion_matrix': confusion_matrix(yte,pred).tolist(), 'features': FEATURES, 'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch, 'train_rows': len(Xtr), 'test_rows': len(Xte), 'class_counts': np.bincount(ytr).tolist(), 'architecture': model.config, 'training_seconds': round(time.perf_counter()-started,3)}
    save_json(metrics,'outputs/supervised_metrics.json')
    save_supervised_artifacts(model, {'scaler':scaler,'label_encoder':le,'features':FEATURES,'classes':le.classes_.tolist()}, 'models/supervised_ids.pt', 'models/preprocess.joblib')
    print(json.dumps({'macro_f1':metrics['macro_f1'],'weighted_f1':metrics['weighted_f1']},indent=2))
if __name__=='__main__': main()
