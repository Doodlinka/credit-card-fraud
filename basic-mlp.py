import common, config
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from skorch import NeuralNetBinaryClassifier
import torch.optim as optim
import joblib


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

class FraudDetectorModule(nn.Module):
    def __init__(self, num_features=30):
        super(FraudDetectorModule, self).__init__()
        self.activation = nn.ReLU()

        self.layer1 = nn.Linear(num_features, 60)
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(60, 30)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        return x

TRAIN = False
if TRAIN:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor(n_neg / n_pos) 

    scikit_model = NeuralNetBinaryClassifier(
        module=FraudDetectorModule,
        module__num_features=X_train.shape[1],
        
        # class_weight = balanced
        criterion=nn.BCEWithLogitsLoss,
        criterion__pos_weight=pos_weight,

        optimizer=optim.AdamW,
        lr=0.001,
        max_epochs=20,
        batch_size=1024,
        
        device='cuda',
    )
    
    print("Training...")
    scikit_model.fit(X_train, y_train)
    joblib.dump(scikit_model, "one_mlp.joblib")
else:
    scikit_model = joblib.load("one_mlp.joblib")

# this returns 2 columns, take the one for class 1
y_proba_nn = scikit_model.predict_proba(X_test)[:, 1]
# make sure it's horizontal
fraud_probs_nn = y_proba_nn.flatten()
# 3. Find Threshold (Use your existing function!)
common.print_best_threshold(y_test, fraud_probs_nn)