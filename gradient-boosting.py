import joblib
from lightgbm import LGBMClassifier, early_stopping
import common
import numpy as np
import config
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = common.load_dataset()

TRAIN = True

if TRAIN:
    
    static_parameters = {
        "class_weight": 'balanced',
        "random_state": config.TRAIN_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    # gbdt record: 0.9260
    # other parameters either had no effect or are best left default
    tuned_parameters = {
        "learning_rate": 0.05,
        "colsample_bytree": 0.7,
        "num_leaves": 40,
        # "min_sum_hessian_in_leaf": 2e-3,
    }
    # this is super important too
    callbacks = [early_stopping(stopping_rounds=50, verbose=False)]
    
    # split the dataset again so early stopping has stuff to check when to stop
    # X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=config.ES_SPLIT_SEED, stratify=y_train
    # )
    # that's one model
    # best_model = LGBMClassifier(**(static_parameters | tuned_parameters))
    # best_model.fit(X_train_final,
    # y_train_final,
    # eval_set=[(X_val_final, y_val_final)],
    # eval_metric=common.my_lgbm_format_scorer,
    # callbacks=callbacks)
    
    ensemble = common.crossvalidate(LGBMClassifier, X_train, y_train, static_parameters, tuned_parameters, callbacks)
    
#     # joblib.dump(best_model, f"lightgbm_model_{config.TRAIN_SEED}.joblib")
    joblib.dump(ensemble, "lgbm_ensemble.joblib")
    print(f"Saved ensemble of {config.CV_COUNT} models with following params:")
    print(ensemble[0].get_params())
    
else:
    ensemble = joblib.load("lgbm_ensemble.joblib")
    print(f"Loaded ensemble of {config.CV_COUNT} models with following params:")
    print(ensemble[0].get_params())

all_fraud_probs = []
for model in ensemble:
    y_pred = model.predict_proba(X_test)
    all_fraud_probs.append(y_pred[:, 1])
avg_fraud_probs = np.mean(np.array(all_fraud_probs), axis=0)

common.print_best_threshold(y_test, avg_fraud_probs)
