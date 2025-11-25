import joblib
from lightgbm import LGBMClassifier, early_stopping
import common
import numpy as np
import config
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = common.load_dataset()

TRAIN = False

if TRAIN:
    
    # static_parameters = {
    #     "class_weight": 'balanced',
    #     "random_state": config.TRAIN_SEED,
    #     "n_jobs": -1,
    #     "verbose": -1,
    # }
    
    # # record: 0.8748 (12/28) (with CV seed 50)
    # tuned_parameters = {
    #     # "num_estimators": 100,
    #     "learning_rate": 0.06, # fixed 0.06
    #     "colsample_bytree": 0.3, # fixed 0.3
    #     "num_leaves": 56, # fixed 56
    # }
    # # this is super important too
    # callbacks = [early_stopping(stopping_rounds=50, verbose=False)]
    
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
    
    static_parameters = {
        "force_col_wise": True,
        'class_weight': 'balanced',
        'random_state': config.TRAIN_SEED, 
        }
    tuned_parameters = {
        'boosting_type': 'dart', # fixed - tyring dart
        'n_estimators': 600, # fixed - 575 or 600?
        'learning_rate': 0.1, # fixed - 0.1
        'num_leaves': 31, # fixed - 31
        'colsample_bytree': 1,
        'subsample': 1.0,
        'importance_type': 'split',
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'subsample_for_bin': 200000,
        'subsample_freq': 0
    }
    callbacks = []
    
    ensemble = common.crossvalidate(LGBMClassifier, X_train, y_train, static_parameters, tuned_parameters, callbacks)
    
#     # joblib.dump(best_model, f"lightgbm_model_{config.TRAIN_SEED}.joblib")
    joblib.dump(ensemble, "lgbm_dart_ensemble.joblib")
    print(f"Saved ensemble of {config.CV_COUNT} models with following params:")
    print(ensemble[0].get_params())
    
else:
    ensemble = joblib.load("lgbm_dart_ensemble.joblib")
    print(f"Loaded ensemble of {config.CV_COUNT} models with following params:")
    print(ensemble[0].get_params())

all_fraud_probs = []
for model in ensemble:
    y_pred = model.predict_proba(X_test)
    all_fraud_probs.append(y_pred[:, 1])
avg_fraud_probs = np.max(np.array(all_fraud_probs), axis=0)

common.print_best_threshold(y_test, avg_fraud_probs)
