import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import common

df = pd.read_csv("v132_creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = common.load_dataset()

TRAIN = True

if TRAIN:
    # this only works if you go in there and comment a bunch
    # of shit unrelated to RF out
    ensemble = common.crossvalidate(
        RandomForestClassifier,
        X_train,
        y_train,
        {"class_weight": "balanced"},
        {},
        []
    )
    joblib.dump(ensemble, "random_forest_ensemble.joblib")
else:
    ensemble = joblib.load("random_forest_ensemble.joblib")

all_fraud_probs = []
for model in ensemble:
    y_pred = model.predict_proba(X_test)
    all_fraud_probs.append(y_pred[:, 1])
avg_fraud_probs = np.mean(np.array(all_fraud_probs), axis=0)

common.print_best_threshold(y_test, avg_fraud_probs)
