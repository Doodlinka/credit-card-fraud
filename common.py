import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import fbeta_score, make_scorer
import config

def load_dataset():
    df = pd.read_csv("v132_creditcard.csv")

    X = df.drop("Class", axis=1)
    # print("avg tansaction amouont: ", np.mean(X["Amount"]))
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=config.TT_SPLIT_SEED)
    
    return X_train, X_test, y_train, y_test


# turns out beta SQUARED is the ratio of fn cost to fp cost
# i guesstimated it to be 50
BETA = 7.1

my_scorer = make_scorer(fbeta_score, greater_is_better=True, beta=BETA)

def my_lgbm_format_scorer(y_true, y_pred_probs):
    # THRESHOLD HERE
    y_pred = (y_pred_probs > 0.1).astype(int)
    # now i can use it    
    score = fbeta_score(y_true, y_pred, beta=BETA)
    # the required format (name, score, higher_is_better
    return (f'f{BETA}', score, True)

def crossvalidate(model_constructor, X_train, y_train, static_parameters, tuned_parameters, callbacks):
    skf = StratifiedKFold(n_splits=config.CV_COUNT, shuffle=True, random_state=config.CV_SPLIT_SEED)
    best_tree_counts = []
    fold_scores = []
    ensemble = []
    
    for fold_n, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        print(f"--- Fold {fold_n + 1}/{config.CV_COUNT} ---")
        
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = model_constructor(**(static_parameters | tuned_parameters))

        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric=my_lgbm_format_scorer,
            callbacks=callbacks
        )
        ensemble.append(model)

        best_trees = model.best_iteration_
        best_score = model.best_score_['valid_0'][f'f{BETA}']
        
        best_tree_counts.append(best_trees)
        fold_scores.append(best_score)
        
        print(f"Best Trees: {best_trees}, F{BETA}-Score: {best_score:.4f}")


    avg_trees = np.mean(best_tree_counts)
    avg_score = np.mean(fold_scores)

    print("\n--- CV Results ---")
    print(f"Tested Params: {tuned_parameters}")
    print(f"Average F2-Score across {config.CV_COUNT} folds: {avg_score:.4f}")
    print(f"Average optimal tree count: {avg_trees:.0f}")
    return ensemble
    

def print_best_threshold(y_test, fraud_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, fraud_probs)
    # 1e-10 just in case of 0 division
    # yeah, i'm still using my own, because that one can't mix yes/no and probabilities
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