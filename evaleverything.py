import common, joblib, config
import numpy as np
import pandas as pd
from NN_definition import FraudDetectorModule
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("v132_creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=config.TT_SPLIT_SEED)

scaler = RobustScaler()
scaler.fit(X[['Time', 'Amount']])
columns_to_scale = ['Time', 'Amount']
X_train_scaled = X_train.copy(True)
X_test_scaled = X_test.copy(True)
X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

X_train_scaled = X_train_scaled.to_numpy().astype(np.float32)
X_test_scaled = X_test_scaled.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

gb_ensemble = joblib.load("miracle-based dart/lgbm_dart_ensemble.joblib")
nn_ensemble = joblib.load("one-mlp/mlp-ensemble.joblib")

all_fraud_probs = []
for model in gb_ensemble:
    y_pred = model.predict_proba(X_test)
    all_fraud_probs.append(y_pred[:, 1])
for model in nn_ensemble:
    y_pred = model.predict_proba(X_test_scaled)
    all_fraud_probs.append(y_pred[:, 1])
combined_fraud_probs = np.average(np.array(all_fraud_probs), axis=0)

common.print_best_threshold(y_test, combined_fraud_probs)