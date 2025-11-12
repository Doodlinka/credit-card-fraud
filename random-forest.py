import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import joblib
import common

df = pd.read_csv("v132_creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42)

TRAIN = False

if TRAIN:
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, "random_forest_model.joblib")
else:
    model = joblib.load("random_forest_model.joblib")

y_pred = model.predict_proba(X_test)
fraud_probs = y_pred[:, 1]
common.print_best_threshold(y_test, fraud_probs)
