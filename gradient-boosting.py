import joblib
from lightgbm import LGBMClassifier
import common
from sklearn.model_selection import RandomizedSearchCV
import config

X_train, X_test, y_train, y_test = common.load_dataset()

TRAIN = True

if TRAIN:
    param_dist = {
        'n_estimators': [200, 500, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'num_leaves': [10, 20, 31, 50, 70],
        'max_depth': [10, 20, 40, -1],
        'boosting_type': ['gbdt', 'dart'],
        'max_bin': [128, 256, 512],
    }
    
    
    
    base_model = LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1)
        
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=config.CANDIDATE_AMOUNT, 
        cv=config.CV_COUNT, 
        scoring=common.my_scorer,
        random_state=config.TRAIN_SEED, 
        n_jobs=-1,
        verbose=2
    )
    
    # apparently the override doesn't go inside the function
    # print("|...." * 6 + "|")
    # old_print = print
    # def my_print(_):
    #     old_print("#", end="")
    # print = my_print
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # print = old_print
    
    joblib.dump(best_model, f"lightgbm_model_{config.TRAIN_SEED}.joblib")
    print(f"best parameters: {random_search.best_params_}")
    print(f"Best F{common.BETA}-score during search: {random_search.best_score_:.4f}")
    
else:
    best_model = joblib.load(f"lightgbm_model._{config.TRAIN_SEED}joblib")

y_pred = best_model.predict_proba(X_test)
fraud_probs = y_pred[:, 1]
common.print_best_threshold(y_test, fraud_probs)
