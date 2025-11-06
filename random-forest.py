import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import joblib

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

BETA = 10 # cost of missing a fraud compared to false alarm

precisions, recalls, thresholds = precision_recall_curve(y_test, fraud_probs)
# 1e-10 just in case of 0 division
f_beta_scores = (1 + BETA**2) * (precisions * recalls) / ((BETA**2 * precisions) + recalls + 1e-10)

best_score_index = np.argmax(f_beta_scores)
best_threshold = thresholds[best_score_index]

print(f"--- F{BETA}-Score Optimization ---")
print(f"Best F{BETA}-Score: {f_beta_scores[best_score_index]:.4f}")
print(f"Optimal Threshold: {best_threshold:.4f}")

y_final_pred = (fraud_probs > best_threshold).astype(int)

print(f"--- Classification Report for threshold {best_threshold}---")
print(classification_report(y_test, y_final_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_final_pred))
