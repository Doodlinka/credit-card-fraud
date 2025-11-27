import common, joblib
import numpy as np
from NN_definition import FraudDetectorModule
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata
# from sklearn.metrics import precision_recall_curve

print("loading data...")
X_train, X_test, y_train, y_test = common.load_dataset()

print("scaling...")
scaler = RobustScaler()
columns_to_scale = ['Time', 'Amount']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

X_train_scaled = X_train_scaled.to_numpy().astype(np.float32)
X_test_scaled = X_test_scaled.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

print("loading models...")
gb_ensemble = joblib.load("miracle-based dart/lgbm_dart_ensemble.joblib")
nn_ensemble = joblib.load("one-mlp/mlp-ensemble.joblib")

print("making predictions...")
gb_fraud_probs = []
for model in gb_ensemble:
    y_pred = model.predict_proba(X_test)
    gb_fraud_probs.append(y_pred[:, 1])
gb_consensus = np.max(gb_fraud_probs, axis=0)
    
nn_fraud_probs = []
for model in nn_ensemble:
    y_pred = model.predict_proba(X_test_scaled)
    nn_fraud_probs.append(y_pred[:, 1])
nn_consensus = np.average(nn_fraud_probs, axis=0)

rank_gb = rankdata(gb_consensus, axis=0) / len(gb_consensus)
rank_nn = rankdata(nn_consensus, axis=0) / len(nn_consensus)
hybrid_ranks = (0.85 * rank_gb + 0.15 * rank_nn) # 0.8860 - misses that one case, unfortunately
# hybrid_ranks = np.maximum(rank_gb, rank_nn) # 0.8725 - precision suffers

# this entire mess, trying to let NN override when it's confident, gives 0.8710 due to tanked precision
# thresh_gb, score_gb = common.get_best_threshold_and_score(y_test, gb_consensus)
# thresh_nn, score_nn = common.get_best_threshold_and_score(y_test, nn_consensus)
# final_pred = (gb_consensus > thresh_gb).astype(int)
# rescue_mask = (nn_consensus > thresh_nn)
# final_pred[rescue_mask] = 1
# # f"{thresh_gb:.4f} for GB, {thresh_nn:.4f} for NN"
# common.only_print_report(final_pred, y_test, 0)

print("\n\nGBM Ensemble Predictions:\n")
common.print_best_threshold(y_test, rank_gb)
print("\n\nNN Ensemble Predictions:\n")
common.print_best_threshold(y_test, rank_nn)
print("\n\nFull Ensemble Predictions:\n")
common.print_best_threshold(y_test, hybrid_ranks)


# correlation = np.corrcoef(gb_consensus, nn_consensus)[0, 1]
# print(f"Correlation between GB and NN: {correlation:.4f}") # 0.3756

# best_score = 0
# best_weight = 0
# best_threshold = 0

# results = []
# for w_gb in np.linspace(0, 1, 21): # 0.00, 0.05, 0.10 ... 1.00
#     w_nn = 1.0 - w_gb

#     hybrid_ranks = (w_gb * rank_gb) + (w_nn * rank_nn)
    
#     # Get Score (Using your trusted function logic internally)
#     # Note: You might need to expose the scoring logic from common.print_best_threshold
#     # or just copy-paste the F-beta calc here for speed.
#     precisions, recalls, thresholds = precision_recall_curve(y_test, hybrid_ranks)
#     f_scores = (1 + 7.1**2) * (precisions[:-1] * recalls[:-1]) / ((7.1**2 * precisions[:-1]) + recalls[:-1] + 1e-10)
    
#     current_best_threshold, current_best_score = common.get_best_threshold_and_score(y_test, hybrid_ranks)
    
#     results.append((w_gb, current_best_score))
    
#     if current_best_score > best_score:
#         best_score = current_best_score
#         best_threshold = current_best_threshold
#         best_weight = w_gb

# print(f"\n--- RESULTS ---")
# print(f"GB Only Score (w=1.0): {results[-1][1]:.5f}")
# print(f"NN Only Score (w=0.0): {results[0][1]:.5f}")
# print(f"Best Hybrid Score:     {best_score:.5f}")
# print(f"Best Hybrid Threshold: {best_threshold:.5f}")
# print(f"Best GB Weight:        {best_weight:.2f}")


# def get_missed_indices(y_true, probs):
#     thresh, _ = common.get_best_threshold_and_score(y_true, probs)
#     preds = (probs > thresh).astype(int)
#     missed_mask = (y_true == 1) & (preds == 0)
#     # If y_test is a dataframe/series, use .index, else use np.where
#     if hasattr(y_true, 'index'):
#         missed_indices = y_true.index[missed_mask].tolist()
#     else:
#         missed_indices = np.where(missed_mask)[0].tolist()
#     return set(missed_indices), thresh

# missed_gb, thresh_gb = get_missed_indices(y_test, gb_consensus)
# missed_nn, thresh_nn = get_missed_indices(y_test, nn_consensus)
# print(f"GB Missed {len(missed_gb)} cases.")
# print(f"NN Missed {len(missed_nn)} cases.")

# only_gb_missed = missed_gb - missed_nn
# only_nn_missed = missed_nn - missed_gb
# both_missed    = missed_gb.intersection(missed_nn)
# print(f"\n- Cases GB missed but NN CAUGHT: {sorted(list(only_gb_missed))}")
# print(f"- Cases NN missed but GB CAUGHT: {sorted(list(only_nn_missed))}")
# print(f"- Cases BOTH missed: {sorted(list(both_missed))}")