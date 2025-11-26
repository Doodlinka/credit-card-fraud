import common, config
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, Checkpoint, LoadInitState, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.metrics import fbeta_score, make_scorer
import torch.optim as optim
import joblib
import random, os
from NN_definition import FraudDetectorModule
import torch.nn as nn

def set_seed(seed=config.TRAIN_SEED):
    """Sets the seed for reproducibility across all libraries."""
    # 1. Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you use multi-GPU
    
    # 4. CU-DNN (The tricky part)
    # This forces the GPU to use the exact same algorithm every time,
    # even if a faster one exists. It might slow down training slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

# for consistency's sake
set_seed()


df = pd.read_csv("v132_creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = RobustScaler()
columns_to_scale = ['Time', 'Amount']
X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=config.TT_SPLIT_SEED)

X_train = X_train.to_numpy().astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

TRAIN = True
if TRAIN:
    X_train_tune, X_val, y_train_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    CONFIG = {
        'lr': 0.0020, # fixed: 0.0020 with the scheduler, 0.0010 without
        'hidden_units': 50,      # (record: 50 - 0.8710)
        'dropout': 0.3,           # <--- Overfitting control
        'weight_decay': 0.0001,   # <--- L2 Regularization
        'pos_weight_mult': 1.5,   # <--- Increase to boost Recall (1.0 = Default Balanced)
        'max_epochs': 50,         # <--- Give it time to learn
        'batch_size': 1024,       # <--- Speed vs Stability
        'patience': 10,
    }

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    base_pos_weight = torch.tensor(n_neg / n_pos)
    final_pos_weight = base_pos_weight * CONFIG['pos_weight_mult']

    cp = Checkpoint(
        monitor='valid_auc_best', 
        fn_prefix='best_model_'
    )
    load_best = LoadInitState(checkpoint=cp)

    net = NeuralNetBinaryClassifier(
        module=FraudDetectorModule,
        module__num_features=X_train.shape[1],
        module__num_hidden_first=CONFIG['hidden_units'],
        module__dropout_rate=CONFIG['dropout'],
        
        criterion=nn.BCEWithLogitsLoss,
        criterion__pos_weight=final_pos_weight,
        
        optimizer=optim.AdamW,
        optimizer__lr=CONFIG['lr'],
        optimizer__weight_decay=CONFIG['weight_decay'],
        
        max_epochs=CONFIG['max_epochs'],
        batch_size=CONFIG['batch_size'],
        
        callbacks=[
            EpochScoring(scoring='roc_auc', lower_is_better=False, name='valid_auc'),
            cp,
            load_best,
            LRScheduler(
                policy=ReduceLROnPlateau, 
                monitor='valid_loss', 
                mode='min', 
                patience=3, 
                factor=0.5
            ),
            EarlyStopping(monitor='valid_auc', patience=CONFIG['patience'], lower_is_better=False),
        ],
        
        device='cuda',
        verbose=1
    )

    print(f"Training with: {CONFIG}")
    net.fit(X_train_tune, y_train_tune)
    joblib.dump(net, "one_mlp.joblib")
else:
    net = joblib.load("one_mlp.joblib")

# this returns 2 columns, take the one for class 1
y_proba_nn = net.predict_proba(X_test)[:, 1]
# make sure it's horizontal
fraud_probs_nn = y_proba_nn.flatten()
# 3. Find Threshold (Use your existing function!)
common.print_best_threshold(y_test, fraud_probs_nn)