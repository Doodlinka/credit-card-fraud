import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import fbeta_score, make_scorer


def load_dataset():
    df = pd.read_csv("v132_creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42)
    
    return X_train, X_test, y_train, y_test

BETA = 50

my_scorer = make_scorer(fbeta_score, greater_is_better=True, beta=BETA)

def print_best_threshold(y_test, fraud_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, fraud_probs)
    # 1e-10 just in case of 0 division
    # yeah, i'm still usig my own, because that one can't mix yes/no and probabilities
    # precisions and recalls are added AND MULTIPILED element-wise here
    f_beta_scores = (1 + BETA**2) * (precisions * recalls) / ((BETA**2 * precisions) + recalls + 1e-10)

    best_score_index = np.argmax(f_beta_scores)
    best_threshold = thresholds[best_score_index]

    print(f"--- F{BETA}-Score Optimization ---")
    print(f"Best F{BETA}-Score: {f_beta_scores[best_score_index]:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")

    y_final_pred = (fraud_probs > best_threshold).astype(int)

    print(f"--- Classification Report for threshold {best_threshold:.4f}---")
    print(classification_report(y_test, y_final_pred))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_final_pred))